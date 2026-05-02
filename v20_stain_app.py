"""
Virtual H&E Stain Generator - v20 desktop app.

Runs the ConvNeXt-Base + UNet v20 model on a single unstained image.

Default checkpoint search order:
  1. models/v20_fixed_model.pth
  2. best checkpoints/v20_fixed/*.pth by SSIM in filename
  3. models/v20_model.pth
  4. best checkpoints/v20/*.pth by SSIM in filename

For a Python run:
  python v20_stain_app.py

For an executable, see build_v20_exe.bat. A truly standalone executable is
OS-specific and will be large because it must bundle PyTorch and the weights.
"""

from __future__ import annotations

import argparse
import os
import queue
import re
import sys
import threading
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageTk

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as exc:  # pragma: no cover - GUI import guard
    raise RuntimeError("Tkinter is required for the desktop app.") from exc

try:
    import torch
    import torch.nn as nn
    import segmentation_models_pytorch as smp
except Exception as exc:  # pragma: no cover - dependency import guard
    raise RuntimeError(
        "Missing ML dependencies. Install torch, timm, and "
        "segmentation_models_pytorch before running this app."
    ) from exc


APP_TITLE = "Virtual H&E Stain Generator"
PREVIEW_SIZE = 360
DEFAULT_SIZE = 1024
SUPPORTED_EXTS = [
    ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
    ("All files", "*.*"),
]


def app_root() -> Path:
    """Return app root for source runs and PyInstaller bundles."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def bundle_root() -> Path:
    """Return PyInstaller extraction root when available."""
    return Path(getattr(sys, "_MEIPASS", app_root())).resolve()


def parse_ssim_from_name(path: Path) -> float:
    match = re.search(r"ssim([0-9.]+)", path.name)
    if not match:
        return -1.0
    try:
        return float(match.group(1).rstrip("."))
    except ValueError:
        return -1.0


def best_checkpoint_in(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    checkpoints = sorted(directory.glob("*.pth"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: (parse_ssim_from_name(p), p.stat().st_mtime))


def checkpoint_candidates() -> list[Path]:
    roots = [app_root(), bundle_root(), Path.cwd()]
    candidates: list[Path] = []
    seen: set[Path] = set()

    for root in roots:
        direct = [
            root / "models" / "v20_fixed_model.pth",
            root / "models" / "v20_model.pth",
        ]
        dynamic = [
            best_checkpoint_in(root / "checkpoints" / "v20_fixed"),
            best_checkpoint_in(root / "checkpoints" / "v20"),
        ]
        for path in [direct[0], dynamic[0], direct[1], dynamic[1]]:
            if path is None:
                continue
            resolved = path.resolve()
            if resolved not in seen and resolved.exists():
                candidates.append(resolved)
                seen.add(resolved)
    return candidates


def default_checkpoint() -> Path | None:
    candidates = checkpoint_candidates()
    return candidates[0] if candidates else None


class ConvNeXtUNet(nn.Module):
    """v20 architecture. Weights are loaded from checkpoint, so no download."""

    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="tu-convnext_base.clip_laion2b",
            encoder_weights=None,
            in_channels=3,
            classes=3,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def load_state_dict(path: Path, device: torch.device) -> dict:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)

    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "state"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint is not a state dict: {path}")
    return state


def load_model(path: Path, device: torch.device) -> ConvNeXtUNet:
    model = ConvNeXtUNet().to(device)
    state = load_state_dict(path, device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"Warning: missing keys: {len(missing)}")
        if unexpected:
            print(f"Warning: unexpected keys: {len(unexpected)}")
    model.eval()
    return model


def pil_to_tensor(image: Image.Image, size: int) -> tuple[torch.Tensor, Image.Image]:
    image_rgb = image.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(image_rgb).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor, image_rgb


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.squeeze(0).detach().cpu().clamp(0.0, 1.0)
    arr = (arr.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def infer_once(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.inference_mode():
        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                return model(x).float().cpu()
        return model(x).cpu()


def infer_tta(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    progress: Callable[[str], None] | None = None,
) -> torch.Tensor:
    """4-flip TTA: original, horizontal, vertical, both."""
    transforms = [
        ("original", (), ()),
        ("horizontal", (3,), (3,)),
        ("vertical", (2,), (2,)),
        ("both", (2, 3), (2, 3)),
    ]
    outputs = []
    for i, (name, flip_in, flip_out) in enumerate(transforms, start=1):
        if progress:
            progress(f"TTA {i}/4: {name}")
        batch = torch.flip(x, flip_in) if flip_in else x
        pred = infer_once(model, batch, device)
        if flip_out:
            pred = torch.flip(pred, flip_out)
        outputs.append(pred)
    return torch.stack(outputs, dim=0).mean(dim=0)


def make_preview(image: Image.Image, size: int = PREVIEW_SIZE) -> ImageTk.PhotoImage:
    img = image.copy()
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return ImageTk.PhotoImage(canvas)


class StainApp(tk.Tk):
    def __init__(self, checkpoint: Path | None, size: int):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("980x680")
        self.minsize(900, 620)

        self.queue: queue.Queue = queue.Queue()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: ConvNeXtUNet | None = None
        self.input_image: Image.Image | None = None
        self.output_image: Image.Image | None = None
        self.input_preview: ImageTk.PhotoImage | None = None
        self.output_preview: ImageTk.PhotoImage | None = None
        self.output_size = tk.IntVar(value=size)
        self.use_tta = tk.BooleanVar(value=True)
        self.checkpoint_var = tk.StringVar(value=str(checkpoint) if checkpoint else "")
        self.status_var = tk.StringVar(value="Ready")
        self.busy = False

        self._build_ui()
        self.after(100, self._poll_queue)
        if checkpoint:
            self._load_model_async()
        else:
            self.status_var.set("Select a v20 checkpoint to load.")

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(root)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Checkpoint").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.checkpoint_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=8
        )
        ttk.Button(top, text="Browse", command=self._browse_checkpoint).pack(side=tk.LEFT)
        ttk.Button(top, text="Load", command=self._load_model_async).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        controls = ttk.Frame(root)
        controls.pack(fill=tk.X, pady=(10, 8))
        ttk.Button(controls, text="Open Image", command=self._open_image).pack(side=tk.LEFT)
        self.run_button = ttk.Button(
            controls, text="Generate Stain", command=self._run_inference
        )
        self.run_button.pack(side=tk.LEFT, padx=8)
        self.save_button = ttk.Button(
            controls, text="Save Output", command=self._save_output, state=tk.DISABLED
        )
        self.save_button.pack(side=tk.LEFT)
        ttk.Checkbutton(controls, text="Better quality TTA", variable=self.use_tta).pack(
            side=tk.LEFT, padx=(18, 6)
        )
        ttk.Label(controls, text="Size").pack(side=tk.LEFT)
        ttk.Combobox(
            controls,
            textvariable=self.output_size,
            values=[512, 768, 1024],
            width=7,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(controls, text=f"Device: {self.device}").pack(side=tk.RIGHT)

        panels = ttk.Frame(root)
        panels.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(panels, text="Unstained Input", padding=8)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self.input_label = ttk.Label(left, text="Open an image", anchor=tk.CENTER)
        self.input_label.pack(fill=tk.BOTH, expand=True)

        right = ttk.LabelFrame(panels, text="Virtual H&E Output", padding=8)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        self.output_label = ttk.Label(right, text="Output appears here", anchor=tk.CENTER)
        self.output_label.pack(fill=tk.BOTH, expand=True)

        ttk.Separator(root).pack(fill=tk.X, pady=(8, 4))
        ttk.Label(root, textvariable=self.status_var, anchor=tk.W).pack(fill=tk.X)

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.run_button.config(state=state)

    def _browse_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            title="Select v20 checkpoint",
            filetypes=[("PyTorch weights", "*.pth *.pt"), ("All files", "*.*")],
        )
        if path:
            self.checkpoint_var.set(path)

    def _load_model_async(self) -> None:
        if self.busy:
            return
        path = Path(self.checkpoint_var.get()).expanduser()
        if not path.exists():
            messagebox.showerror("Checkpoint missing", f"Checkpoint not found:\n{path}")
            return
        self._set_busy(True)
        self.status_var.set(f"Loading model on {self.device}...")
        threading.Thread(target=self._load_model_worker, args=(path,), daemon=True).start()

    def _load_model_worker(self, path: Path) -> None:
        try:
            model = load_model(path, self.device)
            self.queue.put(("model_loaded", model, path))
        except Exception as exc:
            self.queue.put(("error", f"Model load failed:\n{exc}"))

    def _open_image(self) -> None:
        path = filedialog.askopenfilename(title="Open unstained image", filetypes=SUPPORTED_EXTS)
        if not path:
            return
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))
            return
        self.input_image = image
        self.output_image = None
        self.save_button.config(state=tk.DISABLED)
        self.input_preview = make_preview(image)
        self.input_label.config(image=self.input_preview, text="")
        self.output_label.config(image="", text="Output appears here")
        self.status_var.set(f"Loaded image: {Path(path).name} ({image.width}x{image.height})")

    def _run_inference(self) -> None:
        if self.busy:
            return
        if self.model is None:
            messagebox.showwarning("Model not loaded", "Load a v20 checkpoint first.")
            return
        if self.input_image is None:
            messagebox.showwarning("No image", "Open an unstained image first.")
            return
        self._set_busy(True)
        self.status_var.set("Preparing image...")
        threading.Thread(target=self._infer_worker, daemon=True).start()

    def _infer_worker(self) -> None:
        try:
            assert self.model is not None
            assert self.input_image is not None
            size = int(self.output_size.get())
            tensor, resized = pil_to_tensor(self.input_image, size)
            self.queue.put(("status", f"Running inference at {size}x{size}..."))
            if self.use_tta.get():
                pred = infer_tta(
                    self.model,
                    tensor,
                    self.device,
                    progress=lambda msg: self.queue.put(("status", msg)),
                )
            else:
                pred = infer_once(self.model, tensor, self.device)
            output = tensor_to_pil(pred)
            self.queue.put(("inference_done", resized, output))
        except Exception as exc:
            self.queue.put(("error", f"Inference failed:\n{exc}"))

    def _save_output(self) -> None:
        if self.output_image is None:
            return
        path = filedialog.asksaveasfilename(
            title="Save virtual H&E image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif"), ("JPEG", "*.jpg")],
        )
        if not path:
            return
        try:
            self.output_image.save(path)
            self.status_var.set(f"Saved output: {path}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self.queue.get_nowait()
                kind = msg[0]
                if kind == "model_loaded":
                    _, model, path = msg
                    self.model = model
                    self._set_busy(False)
                    self.status_var.set(f"Loaded model: {path.name}")
                elif kind == "inference_done":
                    _, resized, output = msg
                    self.input_image = resized
                    self.output_image = output
                    self.input_preview = make_preview(resized)
                    self.output_preview = make_preview(output)
                    self.input_label.config(image=self.input_preview, text="")
                    self.output_label.config(image=self.output_preview, text="")
                    self.save_button.config(state=tk.NORMAL)
                    self._set_busy(False)
                    self.status_var.set("Inference complete.")
                elif kind == "status":
                    self.status_var.set(msg[1])
                elif kind == "error":
                    self._set_busy(False)
                    self.status_var.set("Error")
                    messagebox.showerror("Error", msg[1])
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Virtual H&E v20 desktop app")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to v20 .pth")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, choices=[512, 768, 1024])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint if args.checkpoint else default_checkpoint()
    app = StainApp(checkpoint=checkpoint, size=args.size)
    app.mainloop()


if __name__ == "__main__":
    main()
