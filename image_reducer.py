"""
Image Size Reducer
==================
Standalone. Shrink already-extracted car images to fixed dimensions.
No rembg / no scraping — just resize + recompress.

Fits each image inside a W x H box (no upscaling), centers it on a
canvas of EXACTLY W x H, and saves as WebP (or PNG). Folder tree is
preserved in the output.

Usage:
    pip install pillow

    # default: 600x450 webp, quality 85
    python reduce_size.py --input Brochures --output Car_images_small

    # custom dimensions
    python reduce_size.py --input raw_imgs --output out --width 800 --height 600

    # cap file size (drops webp quality until it fits)
    python reduce_size.py --input raw_imgs --output out --max-kb 150

    # white background instead of transparent, save as png
    python reduce_size.py --input raw_imgs --output out --no-alpha --format png

    # single file
    python reduce_size.py --input image_005.png --output out
"""

import sys, csv, logging, argparse
from pathlib import Path
from typing import Optional

from PIL import Image

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_W       = 600
DEFAULT_H       = 450
DEFAULT_QUALITY = 85
DEFAULT_FORMAT  = "webp"          # webp | png
IMG_EXTS        = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

# ── Logging ──────────────────────────────────────────────────────────────────--
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE
# ═══════════════════════════════════════════════════════════════════════════════

def reduce_image(img: Image.Image, w: int, h: int,
                 keep_alpha: bool = True) -> Image.Image:
    """Fit img into w x h (no upscale), centered on canvas. Output is exactly w x h."""
    src = img.convert("RGBA") if keep_alpha else img.convert("RGB")

    # Trim transparent / empty border if any, so the car fills the frame.
    bbox = src.getbbox()
    if bbox:
        src = src.crop(bbox)

    iw, ih = src.size
    scale = min(w / iw, h / ih, 1.0)          # never upscale
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    src = src.resize((nw, nh), Image.LANCZOS)

    mode = "RGBA" if keep_alpha else "RGB"
    bg   = (0, 0, 0, 0) if keep_alpha else (255, 255, 255)
    canvas = Image.new(mode, (w, h), bg)
    pos = ((w - nw) // 2, (h - nh) // 2)
    canvas.paste(src, pos, src if keep_alpha else None)
    return canvas


def save_reduced(canvas: Image.Image, out_file: Path,
                 quality: int = DEFAULT_QUALITY,
                 max_kb: Optional[int] = None) -> int:
    """Save canvas. If max_kb set + webp, drop quality until it fits. Returns KB."""
    is_webp = out_file.suffix.lower() == ".webp"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    q = quality
    while True:
        if is_webp:
            canvas.save(str(out_file), "WEBP", quality=q, method=6)
        else:
            canvas.convert("RGB").save(str(out_file), "PNG", optimize=True)
        kb = out_file.stat().st_size // 1024
        if max_kb is None or kb <= max_kb or not is_webp or q <= 30:
            return kb
        q -= 10                                # too big → recompress harder


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def collect_files(in_path: Path) -> list:
    if in_path.is_file():
        return [in_path] if in_path.suffix.lower() in IMG_EXTS else []
    return sorted(p for p in in_path.rglob("*") if p.suffix.lower() in IMG_EXTS)


def run(args):
    in_path  = Path(args.input)
    out_dir  = Path(args.output)
    fmt      = args.format.lower()
    keep_alpha = (not args.no_alpha) and fmt == "webp"   # png path forces RGB below if no-alpha

    log.info(f"\n{'='*55}")
    log.info(f"  IMAGE SIZE REDUCER")
    log.info(f"  Input   : {in_path}")
    log.info(f"  Output  : {out_dir}")
    log.info(f"  Target  : {args.width}x{args.height}  ({fmt}, q={args.quality}, max_kb={args.max_kb})")
    log.info(f"{'='*55}\n")

    if not in_path.exists():
        log.error(f"Input not found: {in_path}")
        return

    files = collect_files(in_path)
    if not files:
        log.error(f"No images found in {in_path}")
        log.info(f"  Supported: {', '.join(sorted(IMG_EXTS))}")
        return

    log.info(f"Found {len(files)} image(s)\n")

    base = in_path.parent if in_path.is_file() else in_path
    results = []
    for f in files:
        rel = f.relative_to(base).with_suffix(f".{fmt}")
        out = out_dir / rel
        try:
            with Image.open(f) as im:
                im.load()
                before_kb = f.stat().st_size // 1024
                before_dim = im.size
                canvas = reduce_image(im, args.width, args.height, keep_alpha)
            kb = save_reduced(canvas, out, args.quality, args.max_kb)
            log.info(
                f"  ✓ {f.name}  {before_dim[0]}x{before_dim[1]} ({before_kb} KB)"
                f"  →  {canvas.size[0]}x{canvas.size[1]} ({kb} KB)  {out}"
            )
            results.append({"file": str(f), "out": str(out),
                            "status": "ok", "reason": ""})
        except Exception as e:
            log.error(f"  ✗ {f.name}: {e}")
            results.append({"file": str(f), "out": "",
                            "status": "fail", "reason": str(e)})

    _summary(results, out_dir)


def _summary(results: list, out_dir: Path):
    ok   = sum(1 for r in results if r["status"] == "ok")
    fail = sum(1 for r in results if r["status"] == "fail")
    log.info(f"\n{'─'*55}")
    log.info(f"  ✅  Done — {ok} reduced  |  {fail} failed")

    if not results:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "reduce_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    log.info(f"      Log : {log_path}")
    if fail:
        log.info("\n  Failed:")
        for r in results:
            if r["status"] == "fail":
                log.info(f"    ✗ {r['file']}: {r['reason']}")


def main():
    p = argparse.ArgumentParser(description="Reduce car images to fixed dimensions.")
    p.add_argument("--input",   required=True, help="image file OR folder (recursed)")
    p.add_argument("--output",  default="Car_images_small", help="output folder")
    p.add_argument("--width",   type=int, default=DEFAULT_W,       help="output width")
    p.add_argument("--height",  type=int, default=DEFAULT_H,       help="output height")
    p.add_argument("--quality", type=int, default=DEFAULT_QUALITY, help="webp quality 1-100")
    p.add_argument("--max-kb",  type=int, default=None, dest="max_kb",
                   help="cap file size; drops quality until it fits (webp only)")
    p.add_argument("--format",  choices=["webp", "png"], default=DEFAULT_FORMAT,
                   help="output format")
    p.add_argument("--no-alpha", action="store_true",
                   help="white background instead of transparent")
    run(p.parse_args())


if __name__ == "__main__":
    main()