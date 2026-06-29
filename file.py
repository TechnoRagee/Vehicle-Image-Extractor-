"""
Car Image Pipeline
==================
TWO modes:

MODE 1 — BROCHURE PDFs:
    Put PDFs in the Brochures folder like:
        Brochures/
            tata/
                tiago.pdf
            toyota/
                fortuner.pdf
    Run:  python file.py --mode brochure

MODE 2 — Scrape from CarWale using models.csv:
    Run:  python file.py --mode scrape

Output:
    Car_images/
        tata/
            tiago/
                tiago.webp
        toyota/
            fortuner/
                fortuner.webp

Requirements:
    pip install requests beautifulsoup4 pillow opencv-python-headless rembg
    PDF mode also needs poppler:
        Windows: https://github.com/oschwartz10612/poppler-windows
                 (download, extract, add bin/ to system PATH)
        Mac:     brew install poppler
        Linux:   sudo apt install poppler-utils
"""

import os, re, sys, csv, time, random, logging, argparse, subprocess, shutil
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import uuid

from rembg import remove
import requests
from bs4 import BeautifulSoup
from PIL import Image
import cv2
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR        = Path(__file__).parent.resolve()
DEFAULT_CSV       = SCRIPT_DIR / "models.csv"
DEFAULT_OUTPUT    = SCRIPT_DIR / "Car_images"
DEFAULT_BROCHURES = SCRIPT_DIR / "Brochures"

# ── Canvas / image settings ───────────────────────────────────────────────────
BG_THRESHOLD  = 215
MORPH_CLOSE   = 13
MORPH_OPEN    = 3
EDGE_BLUR     = 7

# Final output canvas — output is ALWAYS exactly this size.
TARGET_W = 600
TARGET_H = 450

# Output format + extension (kept in sync so skip-checks match what we write).
OUT_EXT     = "webp"     # change to "png" if you really need lossless+alpha
WEBP_QUALITY = 85

# ── Scrape settings ───────────────────────────────────────────────────────────
REQUEST_DELAY = (0.8, 2.0)
MAX_RETRIES   = 3
TIMEOUT       = 20
MAX_WORKERS   = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
    "Connection": "keep-alive",
}

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(SCRIPT_DIR / "pipeline.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

session = requests.Session()
session.headers.update(HEADERS)


def http_get(url: str) -> Optional[requests.Response]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                return r
        except Exception as e:
            log.debug(f"Attempt {attempt} error: {e}")
        time.sleep(random.uniform(*REQUEST_DELAY))
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def remove_background(img_pil: Image.Image) -> Image.Image:
    """Remove light/white/studio background. Returns RGBA image."""
    arr  = np.array(img_pil.convert("RGB"))
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Threshold: bright pixels = potential background
    _, bg_thresh = cv2.threshold(gray, BG_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Flood-fill from all four edges to isolate CONNECTED background
    flood = bg_thresh.copy()
    step  = 3
    for x in range(0, w, step):
        if flood[0, x]   == 255: cv2.floodFill(flood, None, (x, 0),   128)
        if flood[h-1, x] == 255: cv2.floodFill(flood, None, (x, h-1), 128)
    for y in range(0, h, step):
        if flood[y, 0]   == 255: cv2.floodFill(flood, None, (0, y),   128)
        if flood[y, w-1] == 255: cv2.floodFill(flood, None, (w-1, y), 128)

    fg_mask = (flood != 128).astype(np.uint8)

    # Morphological cleanup
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE, MORPH_CLOSE))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN,  MORPH_OPEN))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k_close)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  k_open)

    # Keep only the largest connected component (the car)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fg_mask = (labels == largest).astype(np.uint8)

    # Feather mask edges for smooth cutout
    alpha = (fg_mask * 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (EDGE_BLUR, EDGE_BLUR), 0)

    rgba = np.dstack([arr, alpha])
    return Image.fromarray(rgba.astype(np.uint8), "RGBA")


def classify_image(img: Image.Image) -> str:
    """
    Simple classifier.
    Returns: exterior, interior or others
    """
    w, h = img.size
    ratio = w / h

    arr = np.array(img.convert("RGB"))

    # Mean brightness
    brightness = arr.mean()

    # Contrast = std-dev of grayscale luminance.
    # (Previously referenced an undefined name `CONTRAST`, which crashed
    #  with NameError for any image that wasn't classified "exterior".)
    contrast = np.array(img.convert("L")).std()

    # Exterior images are usually wider.
    if ratio > 1.35 and brightness > 90:
        return "exterior"

    # Interior images are usually darker and flatter.
    if brightness < 140 and contrast < 65:
        return "interior"

    return "others"


def save_car_image(img_pil: Image.Image, out_dir: Path, label: str):
    """Full pipeline: remove bg → crop → fit into 600x450 canvas → save WebP.

    Output dimensions are ALWAYS exactly TARGET_W x TARGET_H.
    """
    img = img_pil.convert("RGB")

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    car_rgba = Image.open(BytesIO(remove(buffer.getvalue()))).convert("RGBA")

    # Sanity check: if bg removal was too aggressive, fall back to original
    alpha_arr = np.array(car_rgba)[:, :, 3]
    if (alpha_arr > 30).mean() < 0.10:
        log.warning(f"  ! {label}: bg removal too aggressive, using original")
        car_rgba = img.convert("RGBA")

    bbox = car_rgba.getbbox()
    if bbox is None:
        raise ValueError("No foreground detected after background removal")

    car_rgba = car_rgba.crop(bbox)
    w, h = car_rgba.size

    # Fit inside the canvas without upscaling.
    scale = min(TARGET_W / w, TARGET_H / h, 1.0)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    car = car_rgba.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    x = (TARGET_W - new_w) // 2
    y = (TARGET_H - new_h) // 2
    canvas.paste(car, (x, y), car)

    category = classify_image(car)
    save_folder = out_dir / category
    save_folder.mkdir(parents=True, exist_ok=True)

    safe_label = label.replace("/", "_")
    out_file = save_folder / f"{safe_label}_{uuid.uuid4().hex[:8]}.{OUT_EXT}"

    if OUT_EXT == "webp":
        canvas.save(str(out_file), "WEBP", quality=WEBP_QUALITY, method=6)
    else:
        canvas.save(str(out_file), "PNG", optimize=True)

    log.info(
        f"✓ {label} → {out_file} "
        f"({canvas.size[0]}x{canvas.size[1]}, {out_file.stat().st_size // 1024} KB)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1 — BROCHURE PDF
# ═══════════════════════════════════════════════════════════════════════════════

def extract_car_from_pdf(pdf_path: Path, tmp_dir: Path) -> Optional[Image.Image]:
    """
    Rasterise every page, pick the one with the most car content,
    crop away titles and text, and return a PIL Image.
    """
    if not shutil.which("pdftoppm"):
        log.error("pdftoppm not found!")
        log.error("Windows: download poppler from https://github.com/oschwartz10612/poppler-windows")
        log.error("         extract it and add the bin/ folder to your system PATH")
        return None

    prefix = str(tmp_dir / "pg")
    try:
        subprocess.run(
            ["pdftoppm", "-png", "-r", "600", str(pdf_path), prefix],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        log.error(f"pdftoppm failed for {pdf_path.name}: {e}")
        return None

    pages = sorted(tmp_dir.glob("pg-*.png")) + sorted(tmp_dir.glob("pg*.png"))
    if not pages:
        log.warning(f"No pages found for {pdf_path.name}")
        return None

    # Pick page with the most non-white content in the middle region
    best_img, best_score = None, -1
    for pg in pages:
        img  = Image.open(pg).convert("RGB")
        gray = np.array(img.convert("L"))
        h, w = gray.shape
        mid  = gray[int(h*0.15):int(h*0.75), int(w*0.1):int(w*0.9)]
        score = (mid < 230).mean()
        if score > best_score:
            best_score = score
            best_img   = img

    if best_img is None:
        return None

    # Crop to the non-white content bounding box (+ padding)
    arr = np.array(best_img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    mask = gray < 245
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return best_img

    pad = 60
    x1 = max(0, xs.min() - pad)
    y1 = max(0, ys.min() - pad)
    x2 = min(arr.shape[1], xs.max() + pad)
    y2 = min(arr.shape[0], ys.max() + pad)

    return best_img.crop((x1, y1, x2, y2))


def run_brochure_mode(brochure_dir: Path, out_dir: Path):
    log.info(f"\n{'='*55}")
    log.info(f"  MODE: BROCHURE PDF")
    log.info(f"  Brochures : {brochure_dir}")
    log.info(f"  Output    : {out_dir}")
    log.info(f"{'='*55}\n")

    # First run: create example structure and explain
    if not brochure_dir.exists():
        (brochure_dir / "tata").mkdir(parents=True, exist_ok=True)
        (brochure_dir / "toyota").mkdir(parents=True, exist_ok=True)
        log.info("Created Brochures/ folder. Add your PDFs like this:\n")
        log.info("    Brochures/")
        log.info("      tata/")
        log.info("        tiago.pdf")
        log.info("        nexon.pdf")
        log.info("      toyota/")
        log.info("        fortuner.pdf\n")
        log.info("Then run:  python file.py --mode brochure")
        return

    pdfs = sorted(brochure_dir.rglob("*.pdf"))
    if not pdfs:
        log.warning(f"No PDF files found in {brochure_dir}")
        log.info("Add PDFs like:  Brochures/tata/tiago.pdf")
        return

    log.info(f"Found {len(pdfs)} PDF(s):\n")
    for p in pdfs:
        log.info(f"  {p.relative_to(brochure_dir)}")
    log.info("")

    results = []
    tmp_dir = SCRIPT_DIR / "_tmp_pages"

    for pdf in pdfs:
        # brand = parent folder name, model = pdf filename (no extension)
        brand = pdf.parent.name.lower().replace(" ", "-")
        model = pdf.stem.lower().replace(" ", "-")
        label = f"{brand}/{model}"

        # Output: Car_images/<category>/<brand>_<model>_xxxx.webp
        model_folder = out_dir / brand / model

        if model_folder.exists() and any(model_folder.rglob(f"*.{OUT_EXT}")):
            log.info(f"  ✓ SKIP  {label}  (already exists)")
            results.append({
                "brand": brand, "model": model,
                "status": "skip", "reason": "", "file": str(model_folder),
            })
            continue

        log.info(f"  → Processing: {label}  [{pdf.name}]")
        tmp_dir.mkdir(exist_ok=True)

        try:
            img = extract_car_from_pdf(pdf, tmp_dir)
            if img is None:
                raise ValueError("Could not extract image from PDF")
            save_car_image(img, model_folder, label)
            results.append({
                "brand": brand, "model": model,
                "status": "ok", "reason": "", "file": str(model_folder),
            })
        except Exception as e:
            log.error(f"  ✗ {label}: {e}")
            results.append({"brand": brand, "model": model,
                            "status": "fail", "reason": str(e), "file": ""})
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    _write_summary(results, out_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2 — SCRAPE CarWale / CarDekho
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_carwale_image(url: str) -> Optional[str]:
    r = http_get(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    og   = soup.find("meta", property="og:image")
    if og and og.get("content"):
        return re.sub(r'/\d+x\d+/', '/1920x1080/', og["content"])
    for img in soup.find_all("img", src=True):
        if "aeplcdn" in img["src"]:
            return img["src"]
    return None


def scrape_cardekho_image(url: str) -> Optional[str]:
    r = http_get(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    og   = soup.find("meta", property="og:image")
    if og and og.get("content"):
        return og["content"]
    for img in soup.find_all("img", src=True):
        if "stimg.cardekho.com" in img["src"]:
            return img["src"]
    return None


def download_image(img_url: str) -> Optional[Image.Image]:
    ref = ("https://www.carwale.com/" if "aeplcdn" in img_url
           else "https://www.cardekho.com/")
    try:
        r = requests.get(img_url,
                         headers={**HEADERS, "Referer": ref},
                         timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        img = Image.open(BytesIO(r.content)).convert("RGB")
        return img if img.width >= 200 and img.height >= 100 else None
    except Exception:
        return None


def process_one_model(row: dict, out_dir: Path) -> dict:
    brand      = row["brand_slug"].strip()
    model      = row["model_slug"].strip()
    brand_name = row["brand_name"].strip()
    model_name = row["model_name"].strip()
    label      = f"{brand_name} {model_name}"

    result   = {"brand": brand_name, "model": model_name,
                "status": "fail", "reason": "", "file": ""}
    # Output: Car_images/<category>/<brand>_<model>_xxxx.webp
    model_folder = out_dir / brand / model

    if model_folder.exists() and any(model_folder.rglob(f"*.{OUT_EXT}")):
        log.info(f"✓ SKIP {label}")
        result.update(status="skip", file=str(model_folder))
        return result

    log.info(f"  → {label}")
    time.sleep(random.uniform(*REQUEST_DELAY))

    source  = row.get("model_source", "carwale").strip().lower()
    url     = row["model_url"].strip()
    img_url = (scrape_carwale_image(url) if source == "carwale"
               else scrape_cardekho_image(url))

    if not img_url:
        result["reason"] = "no image URL found on page"
        log.warning(f"  ✗ {label}: no image URL found")
        return result

    img = download_image(img_url)
    if img is None:
        result["reason"] = "image download failed"
        log.warning(f"  ✗ {label}: download failed")
        return result

    try:
        save_car_image(img, model_folder, label)
        result.update(status="ok", file=str(model_folder))
    except Exception as e:
        result["reason"] = str(e)
        log.error(f"  ✗ {label}: {e}")

    return result


def read_csv_safe(csv_path: Path) -> list:
    """Read CSV handling Windows (\\r\\n) and BOM (utf-8-sig) safely."""
    rows = []
    for encoding in ["utf-8-sig", "utf-8", "cp1252"]:
        try:
            with open(csv_path, newline="", encoding=encoding) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                # Strip hidden \r from all values and keys
                rows = [
                    {k.strip().strip("\r"): v.strip().strip("\r")
                     for k, v in row.items()}
                    for row in rows
                ]
            if rows:
                log.info(f"  Read {len(rows)} rows (encoding: {encoding})")
                break
        except Exception:
            continue
    return rows


def run_scrape_mode(csv_path: Path, out_dir: Path, args):
    log.info(f"\n{'='*55}")
    log.info(f"  MODE: SCRAPE (CarWale / CarDekho)")
    log.info(f"  CSV    : {csv_path}")
    log.info(f"  Output : {out_dir}")
    log.info(f"{'='*55}\n")

    if not csv_path.exists():
        log.error(f"models.csv not found at: {csv_path}")
        log.error("Put models.csv in the SAME folder as file.py")
        log.error(f"Expected location: {csv_path}")
        return

    rows = read_csv_safe(csv_path)

    if not rows:
        log.error("CSV is empty or could not be read")
        return

    # Show available columns to help debug
    log.info(f"  Columns found: {list(rows[0].keys())}")

    # Apply filters
    if args.brand:
        rows = [r for r in rows if r.get("brand_slug", "") == args.brand]
        log.info(f"  Filtered by brand: {args.brand}")
    if args.model:
        rows = [r for r in rows if r.get("model_slug", "") == args.model]
        log.info(f"  Filtered by model: {args.model}")
    if args.limit:
        rows = rows[:args.limit]
        log.info(f"  Limited to first {args.limit} models")

    log.info(f"\n  Total to process: {len(rows)}")
    log.info(f"  Workers: {args.workers}\n")

    if len(rows) == 0:
        log.error("No rows to process after filtering. Check your CSV columns.")
        return

    results = []
    if args.workers == 1:
        for row in rows:
            results.append(process_one_model(row, out_dir))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one_model, row, out_dir): row
                       for row in rows}
            for fut in as_completed(futures):
                results.append(fut.result())

    _write_summary(results, out_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY + MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def _write_summary(results: list, out_dir: Path):
    ok   = sum(1 for r in results if r["status"] == "ok")
    skip = sum(1 for r in results if r["status"] == "skip")
    fail = sum(1 for r in results if r["status"] == "fail")

    log.info(f"\n{'─'*55}")
    log.info(f"  ✅  Done — {ok} saved  |  {skip} skipped  |  {fail} failed")
    log.info(f"      Output : {out_dir}/")

    if not results:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "pipeline_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    log.info(f"      Log    : {log_path}")

    if fail > 0:
        log.info(f"\n  Failed models:")
        for r in results:
            if r["status"] == "fail":
                log.info(f"    ✗ {r['brand']} / {r['model']}: {r.get('reason','')}")


def main():
    p = argparse.ArgumentParser(
        description="Car Image Pipeline — brochure PDF or CarWale scrape"
    )
    p.add_argument(
        "--mode",
        choices=["brochure", "scrape"],
        default="scrape",
        help=(
            "brochure = extract from local PDF files  |  "
            "scrape   = download from CarWale (default)"
        )
    )
    p.add_argument("--brand",     default=None,
                   help="Filter to one brand slug, e.g. tata")
    p.add_argument("--model",     default=None,
                   help="Filter to one model slug, e.g. tiago")
    p.add_argument("--workers",   type=int, default=MAX_WORKERS,
                   help="Parallel download workers (scrape mode only)")
    p.add_argument("--limit",     type=int, default=None,
                   help="Process only the first N models (for testing)")
    p.add_argument("--brochures", default=str(DEFAULT_BROCHURES),
                   help="Path to Brochures/ folder (brochure mode)")
    p.add_argument("--csv",       default=str(DEFAULT_CSV),
                   help="Path to models.csv (scrape mode)")
    p.add_argument("--output",    default=str(DEFAULT_OUTPUT),
                   help="Output folder (default: Car_images/)")
    args = p.parse_args()

    out_dir = Path(args.output)

    if args.mode == "brochure":
        run_brochure_mode(Path(args.brochures), out_dir)
    else:
        run_scrape_mode(Path(args.csv), out_dir, args)


if __name__ == "__main__":
    main()