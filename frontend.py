#!/usr/bin/env python3
"""
Car Image Extractor + Reducer — Premium GUI
============================================
Drag & drop PDFs, set brand/model, extract → remove background → resize → export.

Requirements:
    pip install pymupdf Pillow numpy
    pip install rembg   (optional — AI background removal)

Usage:
    python file.py
"""

import hashlib
import io
import os
import threading
from collections import deque
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import fitz
import numpy as np
from PIL import Image

try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


# ── Colours ───────────────────────────────────────────────────────────────────
BG       = "#0D1117"   # page background
SURFACE  = "#161B22"   # card / panel surface
SURFACE2 = "#21262D"   # slightly lighter card
BORDER   = "#30363D"   # hairline border
CYAN     = "#00D2FF"   # primary accent — headlight blue
CYAN_DIM = "#0A7FA0"   # dimmed accent
RED      = "#FF4757"   # danger / remove
GREEN    = "#3FB950"   # success
AMBER    = "#D29922"   # warning
FG       = "#E6EDF3"   # primary text
FG2      = "#8B949E"   # secondary / muted text
FG3      = "#484F58"   # very muted / placeholder


# ── Backend — background removal ──────────────────────────────────────────────

def strip_white_bg(png_bytes, tol=15):
    im  = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    arr = np.array(im)
    rgb = arr[:, :, :3].astype(np.int16)
    white = np.all(rgb >= (255 - tol), axis=2)
    h, w  = white.shape
    visited = np.zeros((h, w), bool)
    q = deque()
    for r in range(h):
        for c in [0, w - 1]:
            if white[r, c] and not visited[r, c]:
                visited[r, c] = True; q.append((r, c))
    for c in range(w):
        for r in [0, h - 1]:
            if white[r, c] and not visited[r, c]:
                visited[r, c] = True; q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<h and 0<=nc<w and white[nr,nc] and not visited[nr,nc]:
                visited[nr,nc] = True; q.append((nr,nc))
    arr[visited, 3] = 0
    out = Image.fromarray(arr, "RGBA")
    b   = io.BytesIO(); out.save(b, "PNG"); return b.getvalue()


def remove_bg(png_bytes, use_rembg, tol=15):
    if use_rembg and REMBG_AVAILABLE:
        try:
            return rembg_remove(png_bytes)
        except Exception:
            pass
    return strip_white_bg(png_bytes, tol)


# ── Backend — size reducer ────────────────────────────────────────────────────

def reduce_image(img, w, h, keep_alpha=True):
    """Fit img into w x h (no upscale), centered on canvas. Output is exactly w x h."""
    src = img.convert("RGBA") if keep_alpha else img.convert("RGB")
    bbox = src.getbbox()
    if bbox:
        src = src.crop(bbox)               # trim transparent / empty border
    iw, ih = src.size
    scale = min(w / iw, h / ih, 1.0)       # never upscale
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    src = src.resize((nw, nh), Image.LANCZOS)
    mode = "RGBA" if keep_alpha else "RGB"
    bg   = (0, 0, 0, 0) if keep_alpha else (255, 255, 255)
    canvas = Image.new(mode, (w, h), bg)
    canvas.paste(src, ((w - nw) // 2, (h - nh) // 2), src if keep_alpha else None)
    return canvas


def save_canvas(canvas, path, out_fmt="webp", quality=85, max_kb=None):
    """Save canvas. If max_kb set + webp, drop quality until it fits. Returns KB."""
    if out_fmt == "webp":
        q = quality
        while True:
            canvas.save(path, "WEBP", quality=q, method=6)
            kb = os.path.getsize(path) // 1024
            if max_kb is None or kb <= max_kb or q <= 30:
                return kb
            q -= 10
    else:  # png keeps alpha + full quality
        canvas.save(path, "PNG", optimize=True)
        return os.path.getsize(path) // 1024


# ── Backend — PDF extraction ──────────────────────────────────────────────────

def extract_from_pdf(pdf_path, out_dir, min_size, do_rembg, bg_tol, log_fn, progress_fn=None,
                     resize=False, target_w=600, target_h=450, out_fmt="webp",
                     quality=85, max_kb=None):
    os.makedirs(out_dir, exist_ok=True)
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log_fn(f"  ERROR opening PDF: {e}"); return 0, 0

    seen    = set()
    saved   = 0
    skipped = 0
    counter = 1
    total_pages = doc.page_count

    for pno in range(total_pages):
        page = doc[pno]
        if progress_fn:
            progress_fn(pno + 1, total_pages)

        for img in page.get_images(full=True):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
            except Exception as e:
                log_fn(f"  skip xref {xref}: {e}"); continue

            try:
                if pix.colorspace is None or pix.colorspace.n != 1 and pix.colorspace.n != 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                if pix.alpha:
                    pix = fitz.Pixmap(pix, 0)

                png_bytes = pix.tobytes("png")

            except Exception as e:
                log_fn(f"  skip xref {xref}: {e}")
                pix = None
                skipped += 1
                continue

            src_w, src_h = pix.width, pix.height
            pix = None
            digest = hashlib.md5(png_bytes).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)

            if do_rembg or bg_tol > 0:
                png_bytes = remove_bg(png_bytes, do_rembg, bg_tol)

            ext   = out_fmt if resize else "png"
            fname = f"image_{counter:03d}.{ext}"
            fpath = os.path.join(out_dir, fname)

            try:
                if resize:
                    im = Image.open(io.BytesIO(png_bytes))
                    canvas = reduce_image(im, target_w, target_h, keep_alpha=True)
                    kb = save_canvas(canvas, fpath, out_fmt, quality, max_kb)
                    log_fn(f"  saved  {fname}  {src_w}x{src_h} -> {target_w}x{target_h} ({kb} KB)")
                else:
                    with open(fpath, "wb") as f:
                        f.write(png_bytes)
                    log_fn(f"  saved  {fname}  ({src_w}x{src_h})")
            except Exception as e:
                log_fn(f"  ERROR saving {fname}: {e}"); continue

            saved += 1; counter += 1

    doc.close()
    if progress_fn:
        progress_fn(total_pages, total_pages)
    if saved == 0:
        log_fn("  WARNING: no images found above minimum size.")
    return saved, skipped


def safe_name(s):
    for ch in '<>:"/\\|?*':
        s = s.replace(ch, "_")
    return s.strip().lower()


# ── Custom widgets ────────────────────────────────────────────────────────────

class Tooltip:
    """Simple hover tooltip."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text   = text
        self.tip    = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _=None):
        x, y, *_ = self.widget.bbox("insert") if hasattr(self.widget, "bbox") else (0, 0, 0, 0)
        x += self.widget.winfo_rootx() + 20
        y += self.widget.winfo_rooty() + 20
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text,
                 bg=SURFACE2, fg=FG2, relief="flat",
                 font=("Segoe UI", 8), padx=8, pady=4).pack()

    def hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class IconButton(tk.Button):
    """Flat button with hover effect."""
    def __init__(self, parent, text, command, bg=SURFACE2,
                 fg=FG, hover_bg=BORDER, **kw):
        super().__init__(parent, text=text, command=command,
                         bg=bg, fg=fg, activebackground=hover_bg,
                         activeforeground=FG, relief="flat",
                         cursor="hand2", bd=0, **kw)
        self._bg  = bg
        self._hbg = hover_bg
        self.bind("<Enter>", lambda _: self.config(bg=self._hbg))
        self.bind("<Leave>", lambda _: self.config(bg=self._bg))


class CyanButton(tk.Button):
    """Primary action button — cyan accent."""
    def __init__(self, parent, text, command, **kw):

        kw.setdefault("font", ("Segoe UI", 10, "bold"))

        super().__init__(
            parent,
            text=text,
            command=command,
            bg=CYAN,
            fg=BG,
            activebackground=CYAN_DIM,
            activeforeground=FG,
            relief="flat",
            cursor="hand2",
            bd=0,
            **kw
        )

        self.bind("<Enter>", lambda _: self.config(bg=CYAN_DIM))
        self.bind("<Leave>", lambda _: self.config(bg=CYAN))


class PlaceholderEntry(tk.Entry):
    """Entry with placeholder text that clears on focus."""
    def __init__(self, parent, placeholder, **kw):
        super().__init__(parent, **kw)
        self.placeholder = placeholder
        self._active_fg  = kw.get("fg", FG)
        self._show_placeholder()
        self.bind("<FocusIn>",  self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)

    def _show_placeholder(self):
        self.insert(0, self.placeholder)
        self.config(fg=FG3)

    def _on_focus_in(self, _=None):
        if self.get() == self.placeholder:
            self.delete(0, "end")
            self.config(fg=self._active_fg)

    def _on_focus_out(self, _=None):
        if not self.get().strip():
            self._show_placeholder()

    def real_value(self):
        v = self.get().strip()
        return "" if v == self.placeholder else v


# ── Main application ──────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Car Image Extractor + Reducer")
        self.geometry("980x760")
        self.minsize(820, 620)
        self.configure(bg=BG)
        self.entries        = []
        self._job_running   = False
        self._setup_style()
        self._build_ui()

    # ── ttk style ─────────────────────────────────────────────────────────────
    def _setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Vertical.TScrollbar",
                         troughcolor=BG, background=BORDER,
                         arrowcolor=FG2, borderwidth=0, relief="flat")
        style.configure("TProgressbar",
                         troughcolor=SURFACE2, background=CYAN,
                         borderwidth=0, thickness=3)
        style.configure(
                "Thin.Horizontal.TProgressbar",
                troughcolor=SURFACE2,
                background=CYAN,
                thickness=2
                        )
        # dark combobox
        style.configure("Dark.TCombobox",
                        fieldbackground=SURFACE2, background=SURFACE2,
                        foreground=FG, arrowcolor=CYAN, bordercolor=BORDER,
                        relief="flat")

    # ── UI build ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top bar ───────────────────────────────────────────────────────────
        topbar = tk.Frame(self, bg=SURFACE, height=52)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)

        tk.Label(topbar,
                 text="  ◈  CAR IMAGE EXTRACTOR",
                 font=("Segoe UI", 13, "bold"),
                 fg=CYAN, bg=SURFACE).pack(side="left", padx=16, pady=14)

        tk.Label(topbar,
                 text="Extract · Remove Background · Resize · Export",
                 font=("Segoe UI", 9),
                 fg=FG2, bg=SURFACE).pack(side="left", padx=4)

        # rembg badge
        badge_color = GREEN if REMBG_AVAILABLE else AMBER
        badge_text  = "AI BG  ✓" if REMBG_AVAILABLE else "AI BG  ✗"
        tk.Label(topbar, text=badge_text,
                 font=("Segoe UI", 8, "bold"),
                 fg=badge_color, bg=SURFACE).pack(side="right", padx=20)

        # thin cyan accent line under topbar
        tk.Frame(self, bg=CYAN, height=2).pack(fill="x")

        # ── Main area — two columns ───────────────────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=0, pady=0)

        # Left column — PDF list
        left = tk.Frame(body, bg=BG, width=560)
        left.pack(side="left", fill="both", expand=True, padx=(16,8), pady=12)
        left.pack_propagate(False)

        # Right column — settings + log
        right = tk.Frame(body, bg=BG, width=340)
        right.pack(side="right", fill="both", padx=(8,16), pady=12)
        right.pack_propagate(False)

        self._build_left(left)
        self._build_right(right)

    # ── LEFT column ───────────────────────────────────────────────────────────
    def _build_left(self, parent):
        # Section label
        self._section_label(parent, "PDF BROCHURES")

        # Drop zone
        drop = tk.Frame(parent, bg=SURFACE, height=72,
                        highlightbackground=BORDER,
                        highlightthickness=1)
        drop.pack(fill="x", pady=(0, 8))
        drop.pack_propagate(False)

        drop_inner = tk.Frame(drop, bg=SURFACE)
        drop_inner.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(drop_inner,
                 text="⊕  Add PDF Brochures",
                 font=("Segoe UI", 10, "bold"),
                 fg=CYAN, bg=SURFACE).pack(side="left", padx=6)
        tk.Label(drop_inner,
                 text="—  click Browse or drag files here",
                 font=("Segoe UI", 9),
                 fg=FG2, bg=SURFACE).pack(side="left")
        CyanButton(drop_inner, "Browse PDFs",
                   command=self._browse,
                   padx=14, pady=5).pack(side="left", padx=12)

        # Hover glow on drop zone
        drop.bind("<Enter>",
                  lambda _: drop.config(highlightbackground=CYAN))
        drop.bind("<Leave>",
                  lambda _: drop.config(highlightbackground=BORDER))

        # Column headers
        hdr = tk.Frame(parent, bg=BG)
        hdr.pack(fill="x", pady=(4, 2))
        for text, width in [("PDF File", 28), ("Brand", 13), ("Model", 13), ("", 3)]:
            tk.Label(hdr, text=text, width=width, anchor="w",
                     font=("Segoe UI", 7, "bold"),
                     fg=FG3, bg=BG).pack(side="left", padx=2)

        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=(0, 4))

        # Scrollable PDF rows
        wrap = tk.Frame(parent, bg=BG)
        wrap.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(wrap, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(wrap, orient="vertical",
                           command=self.canvas.yview)
        self.list_frame = tk.Frame(self.canvas, bg=BG)
        self.list_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.list_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Empty state label
        self.empty_label = tk.Label(
            self.list_frame,
            text="No PDFs added yet.\nClick Browse above to get started.",
            font=("Segoe UI", 9), fg=FG3, bg=BG,
            justify="center")
        self.empty_label.pack(pady=40)

        # Overall progress (hidden until running)
        self.overall_frame = tk.Frame(parent, bg=BG)
        self.overall_frame.pack(fill="x", pady=(6, 0))
        self.overall_bar = ttk.Progressbar(
            self.overall_frame, style="TProgressbar", mode="determinate")
        self.overall_label = tk.Label(
            self.overall_frame, text="", font=("Segoe UI", 8),
            fg=FG2, bg=BG)

    # ── RIGHT column ──────────────────────────────────────────────────────────
    def _build_right(self, parent):
        # ── Output folder ──────────────────────────────────────────────────
        self._section_label(parent, "OUTPUT")

        out_card = tk.Frame(parent, bg=SURFACE,
                            highlightbackground=BORDER, highlightthickness=1)
        out_card.pack(fill="x", pady=(0, 10))

        out_inner = tk.Frame(out_card, bg=SURFACE)
        out_inner.pack(fill="x", padx=12, pady=10)

        tk.Label(out_inner, text="Folder", font=("Segoe UI", 8, "bold"),
                 fg=FG2, bg=SURFACE).pack(anchor="w")

        out_row = tk.Frame(out_inner, bg=SURFACE)
        out_row.pack(fill="x", pady=(4, 0))

        self.out_var = tk.StringVar(value="Car_images")
        out_entry = tk.Entry(out_row, textvariable=self.out_var,
                             bg=SURFACE2, fg=FG, insertbackground=CYAN,
                             relief="flat", font=("Segoe UI", 9),
                             highlightbackground=BORDER,
                             highlightthickness=1)
        out_entry.pack(side="left", fill="x", expand=True, ipady=5, padx=(0, 6))

        IconButton(out_row, "…", command=self._browse_out,
                   bg=SURFACE2, hover_bg=BORDER,
                   font=("Segoe UI", 9), padx=8, pady=4).pack(side="left")

        tk.Label(out_inner,
                 text="Car_images / brand / model / image_001.webp",
                 font=("Segoe UI", 7), fg=FG3, bg=SURFACE).pack(anchor="w", pady=(4, 0))

        # ── Options ────────────────────────────────────────────────────────
        self._section_label(parent, "OPTIONS")

        opt_card = tk.Frame(parent, bg=SURFACE,
                            highlightbackground=BORDER, highlightthickness=1)
        opt_card.pack(fill="x", pady=(0, 10))
        opt_inner = tk.Frame(opt_card, bg=SURFACE)
        opt_inner.pack(fill="x", padx=12, pady=10)

        # Min size
        size_row = tk.Frame(opt_inner, bg=SURFACE)
        size_row.pack(fill="x", pady=(0, 8))
        tk.Label(size_row, text="Minimum image size",
                 font=("Segoe UI", 9), fg=FG, bg=SURFACE).pack(side="left")
        self.min_var = tk.IntVar(value=150)
        tk.Spinbox(size_row, from_=0, to=2000,
                   textvariable=self.min_var, width=5,
                   bg=SURFACE2, fg=FG, buttonbackground=SURFACE2,
                   relief="flat", font=("Segoe UI", 9),
                   insertbackground=CYAN).pack(side="right")
        tk.Label(size_row, text="px", font=("Segoe UI", 8),
                 fg=FG2, bg=SURFACE).pack(side="right", padx=(0, 4))

        tk.Frame(opt_inner, bg=BORDER, height=1).pack(fill="x", pady=4)

        # BG tolerance
        tol_row = tk.Frame(opt_inner, bg=SURFACE)
        tol_row.pack(fill="x", pady=(0, 8))
        tk.Label(tol_row, text="BG removal tolerance",
                 font=("Segoe UI", 9), fg=FG, bg=SURFACE).pack(side="left")
        self.tol_var = tk.IntVar(value=15)
        tk.Spinbox(tol_row, from_=0, to=50,
                   textvariable=self.tol_var, width=5,
                   bg=SURFACE2, fg=FG, buttonbackground=SURFACE2,
                   relief="flat", font=("Segoe UI", 9),
                   insertbackground=CYAN).pack(side="right")
        tk.Label(tol_row, text="0–50", font=("Segoe UI", 8),
                 fg=FG2, bg=SURFACE).pack(side="right", padx=(0, 4))

        tk.Frame(opt_inner, bg=BORDER, height=1).pack(fill="x", pady=4)

        # AI rembg toggle
        rembg_row = tk.Frame(opt_inner, bg=SURFACE)
        rembg_row.pack(fill="x")

        self.rembg_var = tk.BooleanVar(value=False)
        cb = tk.Checkbutton(rembg_row,
                            text="AI background removal",
                            variable=self.rembg_var,
                            state="normal" if REMBG_AVAILABLE else "disabled",
                            fg=FG if REMBG_AVAILABLE else FG3,
                            bg=SURFACE, selectcolor=SURFACE2,
                            activebackground=SURFACE,
                            activeforeground=FG,
                            font=("Segoe UI", 9),
                            cursor="hand2" if REMBG_AVAILABLE else "arrow")
        cb.pack(side="left")

        status_text  = "rembg ready" if REMBG_AVAILABLE else "pip install rembg"
        status_color = GREEN if REMBG_AVAILABLE else AMBER
        tk.Label(rembg_row, text=status_text,
                 font=("Segoe UI", 7), fg=status_color,
                 bg=SURFACE).pack(side="right")

        # ── Resize / reducer ────────────────────────────────────────────────
        self._section_label(parent, "RESIZE")

        rz_card = tk.Frame(parent, bg=SURFACE,
                           highlightbackground=BORDER, highlightthickness=1)
        rz_card.pack(fill="x", pady=(0, 10))
        rz_inner = tk.Frame(rz_card, bg=SURFACE)
        rz_inner.pack(fill="x", padx=12, pady=10)

        # toggle
        rz_top = tk.Frame(rz_inner, bg=SURFACE)
        rz_top.pack(fill="x", pady=(0, 8))
        self.resize_var = tk.BooleanVar(value=True)
        tk.Checkbutton(rz_top, text="Resize to fixed canvas",
                       variable=self.resize_var,
                       fg=FG, bg=SURFACE, selectcolor=SURFACE2,
                       activebackground=SURFACE, activeforeground=FG,
                       font=("Segoe UI", 9), cursor="hand2").pack(side="left")

        # dimensions  W × H
        dim_row = tk.Frame(rz_inner, bg=SURFACE)
        dim_row.pack(fill="x", pady=(0, 8))
        tk.Label(dim_row, text="Output size", font=("Segoe UI", 9),
                 fg=FG, bg=SURFACE).pack(side="left")
        self.h_var = tk.IntVar(value=450)
        tk.Spinbox(dim_row, from_=1, to=10000, textvariable=self.h_var, width=5,
                   bg=SURFACE2, fg=FG, buttonbackground=SURFACE2, relief="flat",
                   font=("Segoe UI", 9), insertbackground=CYAN).pack(side="right")
        tk.Label(dim_row, text="×", font=("Segoe UI", 9),
                 fg=FG2, bg=SURFACE).pack(side="right", padx=4)
        self.w_var = tk.IntVar(value=600)
        tk.Spinbox(dim_row, from_=1, to=10000, textvariable=self.w_var, width=5,
                   bg=SURFACE2, fg=FG, buttonbackground=SURFACE2, relief="flat",
                   font=("Segoe UI", 9), insertbackground=CYAN).pack(side="right")

        # format + max kb
        fmt_row = tk.Frame(rz_inner, bg=SURFACE)
        fmt_row.pack(fill="x")
        tk.Label(fmt_row, text="Format", font=("Segoe UI", 9),
                 fg=FG, bg=SURFACE).pack(side="left")
        self.fmt_var = tk.StringVar(value="webp")
        ttk.Combobox(fmt_row, textvariable=self.fmt_var, values=["webp", "png"],
                     width=6, state="readonly",
                     style="Dark.TCombobox").pack(side="left", padx=6)
        self.maxkb_var = tk.StringVar(value="")
        tk.Entry(fmt_row, textvariable=self.maxkb_var, width=6,
                 bg=SURFACE2, fg=FG, insertbackground=CYAN, relief="flat",
                 font=("Segoe UI", 9), highlightbackground=BORDER,
                 highlightthickness=1).pack(side="right", ipady=3)
        tk.Label(fmt_row, text="Max KB", font=("Segoe UI", 8),
                 fg=FG2, bg=SURFACE).pack(side="right", padx=(0, 4))

        # ── Action button ──────────────────────────────────────────────────
        self._section_label(parent, "ACTION")

        self.run_btn = CyanButton(
            parent, "▶   Extract + Resize",
            command=self._run,
            padx=0, pady=12,
            font=("Segoe UI", 12, "bold"))
        self.run_btn.pack(fill="x", pady=(0, 4))

        btn_row = tk.Frame(parent, bg=BG)
        btn_row.pack(fill="x", pady=(0, 10))
        IconButton(btn_row, "Clear all",
                   command=self._clear_all,
                   bg=SURFACE, hover_bg=SURFACE2,
                   font=("Segoe UI", 9), fg=FG2,
                   padx=10, pady=6).pack(side="left")
        IconButton(btn_row, "Open output folder",
                   command=self._open_output,
                   bg=SURFACE, hover_bg=SURFACE2,
                   font=("Segoe UI", 9), fg=FG2,
                   padx=10, pady=6).pack(side="right")

        # ── Log console ────────────────────────────────────────────────────
        self._section_label(parent, "LOG")

        log_wrap = tk.Frame(parent, bg=SURFACE,
                            highlightbackground=BORDER,
                            highlightthickness=1)
        log_wrap.pack(fill="both", expand=True)

        self.log_box = tk.Text(
            log_wrap, bg=SURFACE, fg="#3FB950",
            font=("Consolas", 8), relief="flat",
            state="disabled", wrap="word",
            insertbackground=CYAN,
            selectbackground=SURFACE2,
            padx=10, pady=8)
        log_sb = ttk.Scrollbar(log_wrap, orient="vertical",
                               command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=log_sb.set)
        self.log_box.pack(side="left", fill="both", expand=True)
        log_sb.pack(side="right", fill="y")

        # Tag colours for the log
        self.log_box.tag_configure("ok",   foreground=GREEN)
        self.log_box.tag_configure("warn", foreground=AMBER)
        self.log_box.tag_configure("err",  foreground=RED)
        self.log_box.tag_configure("dim",  foreground=FG2)
        self.log_box.tag_configure("head", foreground=CYAN,
                                   font=("Consolas", 8, "bold"))

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(parent, textvariable=self.status_var,
                 font=("Segoe UI", 8), fg=FG2, bg=BG,
                 anchor="w").pack(fill="x", pady=(4, 0))

    # ── Helper: section label ─────────────────────────────────────────────────
    def _section_label(self, parent, text):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", pady=(8, 4))
        tk.Label(row, text=text,
                 font=("Segoe UI", 7, "bold"),
                 fg=FG3, bg=BG).pack(side="left")
        tk.Frame(row, bg=BORDER, height=1).pack(
            side="left", fill="x", expand=True, padx=(6, 0), pady=6)

    # ── Logging ───────────────────────────────────────────────────────────────
    def _log(self, msg, tag="ok"):
        def _do():
            self.log_box.configure(state="normal")
            self.log_box.insert("end", msg + "\n", tag)
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.after(0, _do)

    def _log_head(self, msg):
        self._log(msg, tag="head")

    def _set_status(self, text):
        self.after(0, lambda: self.status_var.set(text))

    # ── Browse ────────────────────────────────────────────────────────────────
    def _browse(self):
        paths = filedialog.askopenfilenames(
            title="Select PDF brochures",
            filetypes=[("PDF files", "*.pdf")])
        for p in paths:
            self._add_row(p)

    def _browse_out(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.out_var.set(d)

    def _open_output(self):
        path = self.out_var.get().strip() or "Car_images"
        if os.path.isdir(path):
            os.startfile(path) if os.name == "nt" else os.system(f'open "{path}"')
        else:
            messagebox.showinfo("Not found",
                f"Folder '{path}' doesn't exist yet.\nRun extraction first.")

    # ── PDF row management ────────────────────────────────────────────────────
    def _add_row(self, pdf_path):
        if any(e["pdf"] == pdf_path for e in self.entries):
            return

        # Hide empty state
        self.empty_label.pack_forget()

        fname = os.path.basename(pdf_path)
        row   = tk.Frame(self.list_frame, bg=SURFACE,
                         highlightbackground=BORDER,
                         highlightthickness=1)
        row.pack(fill="x", pady=2, padx=0)

        # Thin left accent
        tk.Frame(row, bg=CYAN, width=3).pack(side="left", fill="y")

        inner = tk.Frame(row, bg=SURFACE)
        inner.pack(fill="x", padx=8, pady=6)

        # Filename
        disp = fname[:36] + "…" if len(fname) > 36 else fname
        tk.Label(inner, text=disp, width=27, anchor="w",
                 font=("Segoe UI", 9), fg=FG, bg=SURFACE,
                 cursor="arrow").grid(row=0, column=0, padx=(0,6))

        # Brand entry
        brand_e = PlaceholderEntry(
            inner, "brand",
            bg=SURFACE2, fg=FG, insertbackground=CYAN,
            relief="flat", font=("Segoe UI", 9), width=12,
            highlightbackground=BORDER, highlightthickness=1)
        brand_e.grid(row=0, column=1, padx=3, ipady=4)

        # Model entry
        model_e = PlaceholderEntry(
            inner, "model",
            bg=SURFACE2, fg=FG, insertbackground=CYAN,
            relief="flat", font=("Segoe UI", 9), width=12,
            highlightbackground=BORDER, highlightthickness=1)
        model_e.grid(row=0, column=2, padx=3, ipady=4)

        # Per-row mini progress
        prog = ttk.Progressbar(
            inner,
            style="TProgressbar",
            mode="determinate",
            length=60
        )
        prog.grid(row=0, column=3, padx=(6, 2))

        # Remove button
        def _remove(r=row, p=pdf_path):
            r.destroy()
            self.entries = [e for e in self.entries if e["pdf"] != p]
            if not self.entries:
                self.empty_label.pack(pady=40)

        rm = tk.Button(inner, text="✕",
                       command=_remove,
                       bg=SURFACE, fg=RED,
                       activebackground=SURFACE2, activeforeground=RED,
                       relief="flat", font=("Segoe UI", 9, "bold"),
                       cursor="hand2", bd=0, padx=4)
        rm.grid(row=0, column=4, padx=(2, 0))

        # Tooltip with full path
        Tooltip(inner.grid_slaves(row=0, column=0)[0], pdf_path)

        self.entries.append({
            "pdf":         pdf_path,
            "brand_entry": brand_e,
            "model_entry": model_e,
            "row":         row,
            "progress":    prog,
        })

    def _clear_all(self):
        for e in self.entries:
            e["row"].destroy()
        self.entries.clear()
        self.empty_label.pack(pady=40)

    # ── Extraction ────────────────────────────────────────────────────────────
    def _run(self):
        if self._job_running:
            return
        if not self.entries:
            messagebox.showwarning("No PDFs", "Add at least one PDF first.")
            return

        # Validate brand + model
        for e in self.entries:
            b = e["brand_entry"].real_value()
            m = e["model_entry"].real_value()
            if not b or not m:
                fname = os.path.basename(e["pdf"])
                messagebox.showwarning(
                    "Missing info",
                    f"Enter brand and model for:\n{fname}")
                return

        # Validate resize dims
        if self.resize_var.get():
            try:
                if int(self.w_var.get()) < 1 or int(self.h_var.get()) < 1:
                    raise ValueError
            except Exception:
                messagebox.showerror("Bad size", "Output width and height must be positive integers.")
                return

        self._job_running = True
        self.run_btn.configure(state="disabled", text="⏳  Running…")
        self._set_status("Extracting…")
        threading.Thread(target=self._run_thread, daemon=True).start()

    def _run_thread(self):
        out_root  = self.out_var.get().strip() or "Car_images"
        min_size  = self.min_var.get()
        bg_tol    = self.tol_var.get()
        do_rembg  = self.rembg_var.get()

        # resize params
        do_resize = self.resize_var.get()
        target_w  = int(self.w_var.get())
        target_h  = int(self.h_var.get())
        out_fmt   = self.fmt_var.get()
        mk        = self.maxkb_var.get().strip()
        max_kb    = int(mk) if mk.isdigit() else None
        quality   = 85

        total_ok  = 0
        total_skip = 0
        n = len(self.entries)

        rz = f"{target_w}x{target_h} {out_fmt}" if do_resize else "no resize"
        self._log_head(f"\n── Starting → {out_root}/  ({rz}) ──")

        # Show overall bar
        def show_overall():
            self.overall_bar.pack(fill="x")
            self.overall_label.pack(anchor="e", pady=(2, 0))
        self.after(0, show_overall)

        for idx, e in enumerate(self.entries):
            brand = safe_name(e["brand_entry"].real_value())
            model = safe_name(e["model_entry"].real_value())
            out_dir = os.path.join(out_root, brand, model)
            fname = os.path.basename(e["pdf"])

            self._log(f"\n[{idx+1}/{n}]  {brand} / {model}  ←  {fname}", "head")

            # Reset this row's progress bar
            prog = e["progress"]
            self.after(0, lambda p=prog: p.configure(value=0))

            def make_progress(p=prog):
                def _fn(current, total):
                    pct = int(current / total * 100) if total else 100
                    self.after(0, lambda: p.configure(value=pct))
                return _fn

            saved, skipped = extract_from_pdf(
                e["pdf"], out_dir, min_size, do_rembg, bg_tol,
                self._log, make_progress(),
                resize=do_resize, target_w=target_w, target_h=target_h,
                out_fmt=out_fmt, quality=quality, max_kb=max_kb)

            total_ok    += saved
            total_skip  += skipped
            tag = "ok" if saved > 0 else "warn"
            self._log(f"  → {saved} saved, {skipped} skipped", tag)

            # Mark row green/amber
            accent_color = GREEN if saved > 0 else AMBER
            self.after(0, lambda r=e["row"], c=accent_color:
                       r.configure(highlightbackground=c))

            # Update overall bar
            pct = int((idx + 1) / n * 100)
            self.after(0, lambda v=pct: self.overall_bar.configure(value=v))
            self.after(
                    0,
                    lambda v=pct, i=idx + 1: self.overall_label.configure(
                        text=f"{i}/{n} files — {v}%"
                    )
                )

        # ── Done ──────────────────────────────────────────────────────────
        self._log_head(f"\n── Done — {total_ok} images saved, "
                       f"{total_skip} skipped ──")
        self._set_status(f"Done — {total_ok} saved to '{out_root}/'")

        def _finish():
            self.run_btn.configure(state="normal", text="▶   Extract + Resize")
            self._job_running = False
            messagebox.showinfo(
                "Extraction complete",
                f"{total_ok} images saved.\n"
                f"Output: {out_root}/")
        self.after(0, _finish)


if __name__ == "__main__":
    App().mainloop()