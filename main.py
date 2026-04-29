# main.py
"""
AI Image Detector — GUI
Drag and drop an image to run it through all three detection layers.

Dependencies (all standard except tkinterdnd2):
    pip install tkinterdnd2 pillow

Layers:
  1. Metadata analysis  — checks EXIF data for missing/suspicious fields
  2. Pixel analysis     — checks noise, frequency, edges, texture, colour
  3. CNN analysis       — trained EfficientNet-B0 (cnn_ai_detector.pth)
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, font
from PIL import Image, ImageTk

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False
    print("Warning: tkinterdnd2 not installed. Drag and drop disabled.")
    print("Install with: pip install tkinterdnd2")

from Layers.metadata import extract_metadata, analyze_metadata
from Layers.pixel_analysis import get_pixel_analysis_results
from Layers.ml_model import AIImageDetector


# ── Palette ────────────────────────────────────────────────────────────────────
BG          = "#0d0f14"
BG_CARD     = "#13161e"
BG_DROP     = "#0f1219"
BORDER      = "#1e2330"
ACCENT      = "#4f8ef7"
ACCENT_DIM  = "#1e3a6e"
RED         = "#f75a5a"
RED_DIM     = "#6e1e1e"
GREEN       = "#4fd98e"
GREEN_DIM   = "#1e6e3a"
YELLOW      = "#f7c94f"
TEXT        = "#e8eaf0"
TEXT_DIM    = "#5a6070"
TEXT_MID    = "#9098a8"
FONT_MONO   = ("Consolas", 10)
FONT_BODY   = ("Segoe UI", 10)
FONT_HEAD   = ("Segoe UI Semibold", 11)
FONT_BIG    = ("Segoe UI Semibold", 28)
FONT_LABEL  = ("Segoe UI", 9)


# ── Backend logic (same as original main.py) ───────────────────────────────────
class Detector:
    def __init__(self):
        self.cnn = None

    def load_cnn(self):
        MODEL_PATH = r"C:\Users\Dylan\.vscode\AI Image Detector\layers\cnn_ai_detector.pth"
        self.cnn = AIImageDetector(model_path=MODEL_PATH, confidence_threshold=0.65)

    def run_metadata_layer(self, image_path):
        metadata = extract_metadata(image_path)
        flags    = analyze_metadata(metadata)
        ai_score = 0
        if not metadata.get('has_exif'):     ai_score += 1
        if metadata.get('camera_model') == 'Unknown': ai_score += 1
        if metadata.get('software') != 'Unknown':     ai_score += 1
        return {
            'has_exif':     metadata.get('has_exif', False),
            'camera_model': metadata.get('camera_model', 'Unknown'),
            'software':     metadata.get('software', 'Unknown'),
            'datetime':     metadata.get('datetime', 'Unknown'),
            'flags':        flags,
            'ai_score':     ai_score,
            'likely_ai':    ai_score >= 2,
        }

    def run_pixel_layer(self, image_path):
        return get_pixel_analysis_results(image_path)

    def run_cnn_layer(self, image_path):
        return self.cnn.predict(image_path) if self.cnn else None

    def combine_results(self, meta, pixel, cnn):
        votes_ai, votes_real, total = 0, 0, 0
        if meta['likely_ai']: votes_ai += 1
        else: votes_real += 1
        total += 1
        if pixel and pixel.get('likely_ai'): votes_ai += 1
        else: votes_real += 1
        total += 1
        if cnn:
            weight = 1 if cnn.get('uncertain') else 2
            if cnn['is_ai_generated']: votes_ai += weight
            else: votes_real += weight
            total += weight
        is_ai      = votes_ai > votes_real
        confidence = votes_ai / total if is_ai else votes_real / total
        return {
            'verdict':    'AI-Generated' if is_ai else 'Real Photo',
            'is_ai':      is_ai,
            'confidence': confidence,
            'votes_ai':   votes_ai,
            'votes_real': votes_real,
            'total':      total,
        }

    def analyze(self, image_path):
        meta    = self.run_metadata_layer(image_path)
        pixel   = self.run_pixel_layer(image_path)
        cnn     = self.run_cnn_layer(image_path)
        verdict = self.combine_results(meta, pixel, cnn)
        return meta, pixel, cnn, verdict


# ── GUI ────────────────────────────────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root  = root
        self.det   = Detector()
        self._photo = None   # keep reference to avoid GC

        root.title("AI Image Detector")
        root.configure(bg=BG)
        root.geometry("860x740")
        root.minsize(700, 600)
        root.resizable(True, True)

        self._build_ui()
        self._set_status("Loading CNN model...", color=YELLOW)
        threading.Thread(target=self._load_model, daemon=True).start()

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg=BG, pady=20)
        hdr.pack(fill="x", padx=30)
        tk.Label(hdr, text="AI IMAGE DETECTOR", bg=BG, fg=TEXT,
                 font=("Segoe UI Semibold", 16)).pack(side="left")
        self.status_lbl = tk.Label(hdr, text="", bg=BG, fg=YELLOW,
                                   font=FONT_LABEL)
        self.status_lbl.pack(side="right", padx=4)

        # Drop zone
        drop_frame = tk.Frame(self.root, bg=BG, padx=30)
        drop_frame.pack(fill="x")

        self.drop_zone = tk.Frame(drop_frame, bg=BG_DROP,
                                  highlightbackground=BORDER,
                                  highlightthickness=1, pady=30)
        self.drop_zone.pack(fill="x")

        self.drop_icon = tk.Label(self.drop_zone, text="⬇", bg=BG_DROP,
                                  fg=ACCENT_DIM, font=("Segoe UI", 32))
        self.drop_icon.pack()
        self.drop_label = tk.Label(self.drop_zone,
                                   text="Drag & drop an image here",
                                   bg=BG_DROP, fg=TEXT_MID, font=FONT_BODY)
        self.drop_label.pack(pady=(6, 2))
        tk.Label(self.drop_zone, text="or", bg=BG_DROP,
                 fg=TEXT_DIM, font=FONT_LABEL).pack()

        browse_btn = tk.Button(self.drop_zone, text="Browse file",
                               bg=ACCENT_DIM, fg=ACCENT,
                               activebackground=ACCENT, activeforeground=BG,
                               font=FONT_BODY, bd=0, padx=16, pady=6,
                               cursor="hand2", command=self._browse)
        browse_btn.pack(pady=(6, 4))

        # Register drag and drop
        if DND_AVAILABLE:
            self.drop_zone.drop_target_register(DND_FILES)
            self.drop_zone.dnd_bind('<<Drop>>', self._on_drop)
            for w in [self.drop_icon, self.drop_label]:
                w.drop_target_register(DND_FILES)
                w.dnd_bind('<<Drop>>', self._on_drop)

        # Preview + results side by side
        body = tk.Frame(self.root, bg=BG, padx=30, pady=16)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # Left — image preview
        left = tk.Frame(body, bg=BG_CARD,
                        highlightbackground=BORDER, highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        tk.Label(left, text="PREVIEW", bg=BG_CARD, fg=TEXT_DIM,
                 font=("Segoe UI Semibold", 8)).pack(anchor="w", padx=12, pady=(10, 0))

        self.preview_lbl = tk.Label(left, bg=BG_CARD, fg=TEXT_DIM,
                                    text="No image loaded",
                                    font=FONT_LABEL)
        self.preview_lbl.pack(expand=True)

        # Right — results
        right = tk.Frame(body, bg=BG_CARD,
                         highlightbackground=BORDER, highlightthickness=1)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        tk.Label(right, text="ANALYSIS", bg=BG_CARD, fg=TEXT_DIM,
                 font=("Segoe UI Semibold", 8)).grid(row=0, column=0,
                                                      sticky="w", padx=12, pady=(10, 0))

        # Scrollable text area
        txt_frame = tk.Frame(right, bg=BG_CARD)
        txt_frame.grid(row=1, column=0, sticky="nsew", padx=1, pady=(4, 1))
        txt_frame.rowconfigure(0, weight=1)
        txt_frame.columnconfigure(0, weight=1)

        self.result_text = tk.Text(txt_frame, bg=BG_CARD, fg=TEXT,
                                   font=FONT_MONO, bd=0, padx=12, pady=8,
                                   wrap="word", state="disabled",
                                   selectbackground=ACCENT_DIM,
                                   insertbackground=ACCENT,
                                   spacing1=2, spacing3=2)
        self.result_text.grid(row=0, column=0, sticky="nsew")

        sb = tk.Scrollbar(txt_frame, command=self.result_text.yview, bg=BG_CARD,
                          troughcolor=BG_CARD, bd=0, width=8)
        sb.grid(row=0, column=1, sticky="ns")
        self.result_text.config(yscrollcommand=sb.set)

        # Configure text tags for colours
        self.result_text.tag_config("head",    foreground=ACCENT,  font=("Consolas", 10, "bold"))
        self.result_text.tag_config("ok",      foreground=GREEN)
        self.result_text.tag_config("warn",    foreground=RED)
        self.result_text.tag_config("yellow",  foreground=YELLOW)
        self.result_text.tag_config("dim",     foreground=TEXT_DIM)
        self.result_text.tag_config("mid",     foreground=TEXT_MID)
        self.result_text.tag_config("verdict_ai",   foreground=RED,   font=("Consolas", 13, "bold"))
        self.result_text.tag_config("verdict_real", foreground=GREEN, font=("Consolas", 13, "bold"))

        # Bottom verdict bar
        self.verdict_bar = tk.Frame(self.root, bg=BG_CARD,
                                    highlightbackground=BORDER, highlightthickness=1)
        self.verdict_bar.pack(fill="x", padx=30, pady=(0, 20))

        self.verdict_lbl = tk.Label(self.verdict_bar, text="—",
                                    bg=BG_CARD, fg=TEXT_DIM,
                                    font=("Segoe UI Semibold", 18), pady=12)
        self.verdict_lbl.pack(side="left", padx=20)

        self.conf_lbl = tk.Label(self.verdict_bar, text="",
                                 bg=BG_CARD, fg=TEXT_DIM,
                                 font=FONT_BODY, pady=12)
        self.conf_lbl.pack(side="right", padx=20)

    # ── Model loading ──────────────────────────────────────────────────────────
    def _load_model(self):
        try:
            self.det.load_cnn()
            self.root.after(0, lambda: self._set_status("Ready", color=GREEN))
        except Exception as e:
            self.root.after(0, lambda: self._set_status(f"CNN failed: {e}", color=RED))

    # ── File handling ──────────────────────────────────────────────────────────
    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff *.jfif"),
                       ("All files", "*.*")]
        )
        if path:
            self._run_analysis(path)

    def _on_drop(self, event):
        path = event.data.strip().strip('{}')   # tkinterdnd2 wraps paths in {} on Windows
        if os.path.isfile(path):
            self._run_analysis(path)

    # ── Analysis ───────────────────────────────────────────────────────────────
    def _run_analysis(self, image_path):
        self._set_status("Analyzing...", color=YELLOW)
        self._show_preview(image_path)
        self._clear_results()
        self._append("Analyzing image...\n", "dim")
        threading.Thread(target=self._analyze_thread,
                         args=(image_path,), daemon=True).start()

    def _analyze_thread(self, image_path):
        try:
            meta, pixel, cnn, verdict = self.det.analyze(image_path)
            self.root.after(0, lambda: self._show_results(image_path, meta, pixel, cnn, verdict))
        except Exception as e:
            self.root.after(0, lambda: self._append(f"\nError: {e}\n", "warn"))
            self.root.after(0, lambda: self._set_status("Error", color=RED))

    # ── Results display ────────────────────────────────────────────────────────
    def _show_results(self, image_path, meta, pixel, cnn, verdict):
        self._clear_results()

        fn = os.path.basename(image_path)
        self._append(f"  {fn}\n", "head")
        self._append("─" * 44 + "\n", "dim")

        # Layer 1 — Metadata
        self._append("\n  LAYER 1 · METADATA\n", "head")
        self._row("EXIF data",     "Yes" if meta['has_exif'] else "No",
                  ok=meta['has_exif'])
        self._row("Camera model",  meta['camera_model'],
                  ok=meta['camera_model'] != 'Unknown')
        self._row("Software",      meta['software'],
                  ok=meta['software'] == 'Unknown')
        self._row("DateTime",      meta['datetime'])
        if meta['flags']:
            for flag in meta['flags']:
                self._append(f"{flag}\n", "warn")
        result_tag = "warn" if meta['likely_ai'] else "ok"
        self._append(f"\n  Result: ", "mid")
        self._append(f"{'Likely AI' if meta['likely_ai'] else 'Likely Real'}",
                     result_tag)
        self._append(f"  ({meta['ai_score']}/3)\n", "dim")

        # Layer 2 — Pixel
        self._append("\n  LAYER 2 · PIXEL ANALYSIS\n", "head")
        if pixel:
            checks = ['noise', 'frequency', 'edges', 'texture', 'color']
            for check in checks:
                susp = pixel[check].get('suspicious', False)
                self._row(check.capitalize(),
                          "suspicious" if susp else "ok",
                          ok=not susp)
            result_tag = "warn" if pixel['likely_ai'] else "ok"
            self._append(f"\n  Result: ", "mid")
            self._append(f"{'Likely AI' if pixel['likely_ai'] else 'Likely Real'}",
                         result_tag)
            self._append(f"  ({pixel['suspicion_score']:.0%})\n", "dim")
        else:
            self._append("  Could not read image.\n", "warn")

        # Layer 3 — CNN
        self._append("\n  LAYER 3 · CNN (ViT)\n", "head")
        if cnn:
            self._row("Probability AI",   f"{cnn['probability_ai']:.2%}",
                      ok=not cnn['is_ai_generated'])
            self._row("Probability Real", f"{cnn['probability_real']:.2%}",
                      ok=not cnn['is_ai_generated'])
            if cnn.get('uncertain'):
                self._append("  ⚠  Low confidence\n", "yellow")
            result_tag = "warn" if cnn['is_ai_generated'] else "ok"
            self._append(f"\n  Result: ", "mid")
            self._append(f"{'Likely AI' if cnn['is_ai_generated'] else 'Likely Real'}",
                         result_tag)
            self._append(f"  ({cnn['confidence']:.2%})\n", "dim")
        else:
            self._append("  CNN layer unavailable.\n", "warn")

        # Verdict
        self._append("\n" + "─" * 44 + "\n", "dim")
        self._append("  FINAL VERDICT\n", "head")
        vtag = "verdict_ai" if verdict['is_ai'] else "verdict_real"
        self._append(f"  {'AI-GENERATED' if verdict['is_ai'] else 'REAL PHOTO'}\n", vtag)
        self._append(f"  Confidence {verdict['confidence']:.0%}  "
                     f"(votes — AI {verdict['votes_ai']}  "
                     f"Real {verdict['votes_real']})\n", "dim")

        # Update verdict bar
        if verdict['is_ai']:
            self.verdict_lbl.config(text="AI-Generated", fg=RED)
        else:
            self.verdict_lbl.config(text="Real Photo", fg=GREEN)
        self.conf_lbl.config(
            text=f"{verdict['confidence']:.0%} confidence  ·  "
                 f"AI {verdict['votes_ai']} / Real {verdict['votes_real']}",
            fg=TEXT_MID
        )

        self._set_status("Done", color=GREEN)

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _show_preview(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((220, 220), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._photo = photo
            self.preview_lbl.config(image=photo, text="")
        except Exception:
            self.preview_lbl.config(image="", text="Could not load preview")

    def _set_status(self, text, color=TEXT_DIM):
        self.status_lbl.config(text=text, fg=color)

    def _clear_results(self):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.config(state="disabled")
        self.verdict_lbl.config(text="—", fg=TEXT_DIM)
        self.conf_lbl.config(text="")

    def _append(self, text, tag=None):
        self.result_text.config(state="normal")
        if tag:
            self.result_text.insert("end", text, tag)
        else:
            self.result_text.insert("end", text)
        self.result_text.config(state="disabled")
        self.result_text.see("end")

    def _row(self, label, value, ok=None):
        self._append(f"  {label:<18}", "dim")
        tag = ("ok" if ok else "warn") if ok is not None else "mid"
        self._append(f"{value}\n", tag)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    app = App(root)
    root.mainloop()