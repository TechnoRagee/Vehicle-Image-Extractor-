#!/usr/bin/env python3
"""
yolo_pipeline.py
================
PDF -> pages -> extract images -> YOLO car-detect -> crop -> bg-remove -> save

Pipeline:
    PDF
     |  convert pages (PyMuPDF)
     v
    Extract all raster images from each page
     |  YOLO (COCO pretrained, class 'car')
     v
    Is there a car?  -- no --> skip
     | yes
     v
    Crop car using bounding box
     v
    Background removal (rembg)
     v
    Save -> Car_images/<brand>/<model>/image_XXX.png

Requirements:
    pip install pymupdf pillow numpy rembg ultralytics

Usage:
    python yolo_pipeline.py --pdf-dir Brochures --out Car_images
    python yolo_pipeline.py --pdf-dir Brochures --out Car_images --conf 0.35 --pad 12
"""

import argparse
import io
import os
import sys

import fitz
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # avoid DecompressionBombError on big brochure scans

try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

COCO_CAR_CLASSES = {2: "car", 5: "bus", 7: "truck"}  # COCO ids; car=2


def safe_name(s):
    for ch in '<>:"/\\|?*':
        s = s.replace(ch, "_")
    return s.strip().lower()


def load_model(weights="yolov8n.pt"):
    return YOLO(weights)


def pixmap_to_pil(pix):
    """Convert a fitz.Pixmap to a PIL RGB image, stripping alpha/CMYK first."""
    if pix.colorspace is None or pix.colorspace.n not in (1, 3):
        pix = fitz.Pixmap(fitz.csRGB, pix)
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)
    png_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def extract_page_images(doc, page):
    """Yield PIL images for every embedded raster image on a page."""
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            pix = fitz.Pixmap(doc, xref)
            pil_img = pixmap_to_pil(pix)
            pix = None
            yield pil_img
        except Exception as e:
            print(f"  skip xref {xref}: {e}")
            continue


def detect_car_box(model, pil_img, conf=0.35, classes=(2,)):
    """Run YOLO; return best car bounding box (x1,y1,x2,y2) or None."""
    arr = np.array(pil_img)
    results = model.predict(arr, conf=conf, classes=list(classes), verbose=False)
    best_box, best_conf = None, -1.0
    for r in results:
        if r.boxes is None:
            continue
        for box, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            if c > best_conf:
                best_conf = c
                best_box = box
    return best_box, best_conf


def crop_with_padding(pil_img, box, pad=10):
    w, h = pil_img.size
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)
    return pil_img.crop((x1, y1, x2, y2))


def remove_background(pil_img):
    if not REMBG_AVAILABLE:
        print("  WARNING: rembg not installed, skipping bg removal")
        return pil_img.convert("RGBA")
    buf = io.BytesIO()
    pil_img.save(buf, "PNG")
    out_bytes = rembg_remove(buf.getvalue())
    return Image.open(io.BytesIO(out_bytes)).convert("RGBA")


def process_pdf(pdf_path, out_dir, model, conf=0.35, pad=10, min_size=100):
    os.makedirs(out_dir, exist_ok=True)
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  ERROR opening PDF: {e}")
        return 0, 0

    saved, skipped = 0, 0
    counter = 1

    for pno in range(doc.page_count):
        page = doc[pno]
        for pil_img in extract_page_images(doc, page):
            if pil_img.width < min_size or pil_img.height < min_size:
                skipped += 1
                continue

            box, score = detect_car_box(model, pil_img, conf=conf)
            if box is None:
                skipped += 1
                continue

            cropped = crop_with_padding(pil_img, box, pad=pad)
            result = remove_background(cropped)

            fname = f"image_{counter:03d}.png"
            fpath = os.path.join(out_dir, fname)
            result.save(fpath, "PNG", optimize=True)
            print(f"  saved {fname}  (car conf={score:.2f})")

            saved += 1
            counter += 1

    doc.close()
    if saved == 0:
        print("  WARNING: no cars detected in this PDF.")
    return saved, skipped


def find_pdfs(pdf_dir):
    """brand_model.pdf naming convention -> (brand, model, path)."""
    out = []
    for fname in sorted(os.listdir(pdf_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        stem = os.path.splitext(fname)[0]
        if "_" in stem:
            brand, model = stem.split("_", 1)
        else:
            brand, model = "unknown", stem
        out.append((safe_name(brand), safe_name(model), os.path.join(pdf_dir, fname)))
    return out


def main():
    p = argparse.ArgumentParser(description="PDF -> YOLO car extraction pipeline")
    p.add_argument("--pdf-dir", required=True, help="Folder of brand_model.pdf files")
    p.add_argument("--out", default="Car_images", help="Output root folder")
    p.add_argument("--weights", default="yolov8n.pt", help="YOLO weights (pretrained or custom)")
    p.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    p.add_argument("--pad", type=int, default=10, help="Padding (px) around car bbox")
    p.add_argument("--min-size", type=int, default=100, help="Skip images smaller than this (px)")
    args = p.parse_args()

    if not os.path.isdir(args.pdf_dir):
        print(f"ERROR: --pdf-dir not found: {args.pdf_dir}")
        sys.exit(1)

    print(f"Loading YOLO model: {args.weights}")
    model = load_model(args.weights)

    pdfs = find_pdfs(args.pdf_dir)
    if not pdfs:
        print(f"No PDFs found in {args.pdf_dir}")
        return

    total_ok, total_skip = 0, 0
    for brand, model_name, pdf_path in pdfs:
        out_dir = os.path.join(args.out, brand, model_name)
        print(f"\n[{brand}/{model_name}] <- {os.path.basename(pdf_path)}")
        ok, skip = process_pdf(pdf_path, out_dir, model,
                               conf=args.conf, pad=args.pad, min_size=args.min_size)
        total_ok += ok
        total_skip += skip

    print(f"\nDone — {total_ok} saved, {total_skip} skipped -> {args.out}/")


if __name__ == "__main__":
    main()