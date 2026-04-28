"""
Virtual H&E Stain Generator — Desktop GUI
Browse an unstained tissue image, generate the H&E stained version, save the result.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
import sys
from pathlib import Path

import app_inference as infer

# ── Constants ──────────────────────────────────────────────────────────────────
APP_TITLE      = "Virtual H&E Stain Generator"
APP_VERSION    = "1.0.0"
WINDOW_W       = 900
WINDOW_H       = 600
PREVIEW_SIZE   = 280
CKPT_FILENAME  = "ws-epoch=27-val_ssim=0.712.ckpt"
SUPPORTED_EXTS = [("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                  ("All files", "*.*")]


# ── App ────────────────────────────────────────────────────────────────────────

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.resizable(False, False)

        # Apply native Windows theme
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")

        # State
        self._queue         = queue.Queue()
        self._gen           = None
        self._device        = None
        self._input_image   = None
        self._output_image  = None
        self._input_tk      = None
        self._output_tk     = None
        self._tta_var       = tk.BooleanVar(value=False)
        self._status_var    = tk.StringVar(value="Loading model…")
        self._is_busy       = False

        # Build UI
        self._build_menu()
        toolbar = self._build_toolbar()
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(8, 0))

        panels = self._build_image_panels()
        panels.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        actions = self._build_action_bar()
        actions.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 6))

        status = self._build_status_bar()
        status.pack(side=tk.BOTTOM, fill=tk.X)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start model loading in background
        self._run_btn.config(state=tk.DISABLED)
        self._save_btn.config(state=tk.DISABLED)
        self._start_model_loading()
        self.after(100, self._poll_queue)

    # ── Widget builders ────────────────────────────────────────────────────────

    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image…", command=self._browse_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Output…", command=self._save_output, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)
        self.bind("<Control-o>", lambda e: self._browse_image())
        self.bind("<Control-s>", lambda e: self._save_output())

    def _build_toolbar(self) -> ttk.Frame:
        bar = ttk.Frame(self)

        ttk.Button(bar, text="Browse Image…", command=self._browse_image).pack(side=tk.LEFT, padx=(0, 8))

        self._path_label = ttk.Label(bar, text="No image selected", foreground="gray",
                                     width=38, anchor=tk.W)
        self._path_label.pack(side=tk.LEFT, padx=(0, 16))

        ttk.Checkbutton(bar, text="Use TTA (+quality, slower)", variable=self._tta_var).pack(side=tk.LEFT, padx=(0, 16))

        self._device_label = ttk.Label(bar, text="Device: detecting…", foreground="gray")
        self._device_label.pack(side=tk.RIGHT)

        return bar

    def _build_image_panels(self) -> ttk.Frame:
        outer = ttk.Frame(self)

        # Left panel — input
        left_frame = ttk.LabelFrame(outer, text="Input Image (Unstained)", padding=6)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        self._input_panel = tk.Label(
            left_frame,
            text="Browse an image\nor drag and drop",
            width=PREVIEW_SIZE,
            height=PREVIEW_SIZE // 16,
            bg="#f0f0f0",
            relief=tk.SUNKEN,
            fg="gray",
        )
        self._input_panel.pack(expand=True)

        # Right panel — output
        right_frame = ttk.LabelFrame(outer, text="H&E Stained Output", padding=6)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))

        self._output_panel = tk.Label(
            right_frame,
            text="Output will appear here\nafter generation",
            width=PREVIEW_SIZE,
            height=PREVIEW_SIZE // 16,
            bg="#f0f0f0",
            relief=tk.SUNKEN,
            fg="gray",
        )
        self._output_panel.pack(expand=True)

        return outer

    def _build_action_bar(self) -> ttk.Frame:
        bar = ttk.Frame(self)
        inner = ttk.Frame(bar)
        inner.pack(anchor=tk.CENTER)

        self._run_btn = ttk.Button(
            inner,
            text="Generate H&E Stain",
            command=self._run_inference,
            width=24,
        )
        self._run_btn.pack(side=tk.LEFT, padx=(0, 12))

        self._save_btn = ttk.Button(
            inner,
            text="Save Output",
            command=self._save_output,
            width=16,
        )
        self._save_btn.pack(side=tk.LEFT)

        return bar

    def _build_status_bar(self) -> ttk.Frame:
        bar = ttk.Frame(self, relief=tk.SUNKEN)
        ttk.Separator(bar, orient=tk.HORIZONTAL).pack(fill=tk.X)
        ttk.Label(bar, textvariable=self._status_var, anchor=tk.W, padding=(6, 2)).pack(fill=tk.X)
        return bar

    # ── Model loading ──────────────────────────────────────────────────────────

    def _resolve_checkpoint_path(self) -> Path:
        # Determine base directory (works both in dev and packaged exe)
        base = Path(getattr(sys, '_MEIPASS', Path(__file__).parent))
        script_dir = Path(__file__).parent

        candidates = [
            script_dir / "checkpoints_weakly_supervised" / CKPT_FILENAME,
            script_dir / "checkpoints_registered_v1" / CKPT_FILENAME,
            script_dir / "checkpoints" / CKPT_FILENAME,
            script_dir / "checkpoints_v7" / CKPT_FILENAME,
            base / "checkpoints" / CKPT_FILENAME,
        ]

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"Model checkpoint not found.\n\n"
            f"Please place '{CKPT_FILENAME}' in:\n"
            f"  {script_dir / 'checkpoints'}\n\n"
            f"Download the checkpoint from the project repository."
        )

    def _start_model_loading(self):
        self._status_var.set("Loading model… (first launch may take 20-30 seconds)")
        t = threading.Thread(target=self._load_model_thread, daemon=True)
        t.start()

    def _load_model_thread(self):
        try:
            ckpt_path = self._resolve_checkpoint_path()
            device = infer.get_device()
            gen = infer.load_model(str(ckpt_path), device)
            self._queue.put(("model_loaded", gen, device, str(ckpt_path)))
        except Exception as e:
            self._queue.put(("model_error", str(e)))

    # ── Queue polling ──────────────────────────────────────────────────────────

    def _poll_queue(self):
        try:
            while True:
                msg = self._queue.get_nowait()
                kind = msg[0]

                if kind == "model_loaded":
                    _, gen, device, ckpt_path = msg
                    self._on_model_loaded(gen, device, ckpt_path)

                elif kind == "model_error":
                    _, error = msg
                    self._on_model_error(error)

                elif kind == "infer_progress":
                    _, text = msg
                    self._status_var.set(text)

                elif kind == "infer_done":
                    _, result = msg
                    self._on_inference_done(result)

                elif kind == "infer_error":
                    _, error = msg
                    self._on_inference_error(error)

        except queue.Empty:
            pass

        self.after(100, self._poll_queue)

    # ── Model-load callbacks ───────────────────────────────────────────────────

    def _on_model_loaded(self, gen, device: str, ckpt_path: str):
        self._gen = gen
        self._device = device

        device_label = f"Device: {'CUDA (GPU)' if device == 'cuda' else 'CPU'}"
        self._device_label.config(text=device_label, foreground="green" if device == "cuda" else "orange")
        self._status_var.set(f"Model ready  |  {device_label}  |  SSIM 0.2696")

        if self._input_image is not None:
            self._run_btn.config(state=tk.NORMAL)

    def _on_model_error(self, message: str):
        self._status_var.set("Model failed to load — see error dialog")
        messagebox.showerror("Model Load Error", message)

    # ── Image I/O ──────────────────────────────────────────────────────────────

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select Unstained Image",
            filetypes=SUPPORTED_EXTS,
        )
        if path:
            self._load_input_image(path)

    def _load_input_image(self, path: str):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not open image:\n{e}")
            return

        self._input_image = img
        self._display_image(self._input_panel, img, save_ref="input")

        # Truncate path for display
        p = Path(path)
        display = str(p) if len(str(p)) <= 40 else f"…{str(p)[-37:]}"
        self._path_label.config(text=display, foreground="black")

        # Clear output
        self._output_image = None
        self._clear_panel(self._output_panel, "Output will appear here\nafter generation")
        self._save_btn.config(state=tk.DISABLED)

        if self._gen is not None:
            self._run_btn.config(state=tk.NORMAL)

    def _save_output(self):
        if self._output_image is None:
            return

        path = filedialog.asksaveasfilename(
            title="Save Stained Image",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("All files", "*.*")],
        )
        if path:
            try:
                self._output_image.save(path)
                self._status_var.set(f"Saved: {Path(path).name}")
                messagebox.showinfo("Saved", f"Output saved to:\n{path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save image:\n{e}")

    # ── Image display ──────────────────────────────────────────────────────────

    def _display_image(self, panel: tk.Label, pil_img: Image.Image, save_ref: str = ""):
        # Aspect-ratio-preserving resize to fit PREVIEW_SIZE x PREVIEW_SIZE
        img = pil_img.copy()
        img.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.LANCZOS)

        tk_img = ImageTk.PhotoImage(img)
        panel.config(image=tk_img, text="", bg="#ffffff", width=PREVIEW_SIZE, height=PREVIEW_SIZE)
        panel.image = tk_img  # keep reference to prevent GC

        if save_ref == "input":
            self._input_tk = tk_img
        elif save_ref == "output":
            self._output_tk = tk_img

    def _clear_panel(self, panel: tk.Label, placeholder_text: str):
        panel.config(image="", text=placeholder_text, bg="#f0f0f0",
                     fg="gray", width=PREVIEW_SIZE, height=PREVIEW_SIZE // 16)
        panel.image = None

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run_inference(self):
        if self._is_busy or self._input_image is None or self._gen is None:
            return

        self._is_busy = True
        self._run_btn.config(state=tk.DISABLED)
        self._save_btn.config(state=tk.DISABLED)

        use_tta = self._tta_var.get()
        if use_tta:
            self._status_var.set("Running TTA inference (8x passes — takes ~30s on CPU)…")
        else:
            self._status_var.set("Running inference…")

        t = threading.Thread(target=self._inference_thread, args=(use_tta,), daemon=True)
        t.start()

    def _inference_thread(self, use_tta: bool):
        def progress(msg: str):
            self._queue.put(("infer_progress", msg))

        try:
            result = infer.run_inference(
                self._gen,
                self._input_image,
                self._device,
                use_tta=use_tta,
                progress_callback=progress,
            )
            self._queue.put(("infer_done", result))
        except Exception as e:
            self._queue.put(("infer_error", str(e)))

    def _on_inference_done(self, result: Image.Image):
        self._output_image = result
        self._display_image(self._output_panel, result, save_ref="output")
        self._save_btn.config(state=tk.NORMAL)
        self._run_btn.config(state=tk.NORMAL)
        self._is_busy = False
        self._status_var.set("Done — 256×256 RGB  |  SSIM 0.2696 (TTA)  |  Click 'Save Output' to export")

    def _on_inference_error(self, message: str):
        self._run_btn.config(state=tk.NORMAL)
        self._is_busy = False
        self._status_var.set("Inference failed — see error dialog")
        messagebox.showerror("Inference Error", f"An error occurred:\n{message}")

    # ── Misc ──────────────────────────────────────────────────────────────────

    def _show_about(self):
        device_str = self._device or "detecting…"
        messagebox.showinfo(
            "About",
            f"{APP_TITLE}  v{APP_VERSION}\n\n"
            f"Converts unstained tissue images to virtual H&E stained images\n"
            f"using a WGAN-GP generative adversarial network.\n\n"
            f"Model:      v7  (epoch 22)\n"
            f"SSIM:       0.2696 (with TTA)\n"
            f"Device:     {device_str}\n"
            f"Checkpoint: {CKPT_FILENAME}\n\n"
            f"Check 'Use TTA' for slightly better quality (8× slower).",
        )

    def _on_close(self):
        self.destroy()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
