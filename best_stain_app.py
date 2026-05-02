"""
Final Virtual H&E staining app.

Default high-quality mode:
  CLAHE-trained balanced ensemble = 0.20*v20 + 0.60*v22A + 0.20*v21B, each with 4-flip TTA.

Run GUI:
  python best_stain_app.py

Run CLI:
  python best_stain_app.py --input image.tif --output virtual_he.png

Notes:
- CLAHE was used during registration to estimate optical flow. The saved model
  inputs are RGB warped images, so this app does not apply CLAHE preprocessing.
- v21B uses the Hibou-B architecture. The Hugging Face model code/config must be
  available in the local cache unless the app is packaged with that cache.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
from PIL import Image, ImageTk

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Tkinter is required for GUI mode.") from exc

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import segmentation_models_pytorch as smp
    from transformers import AutoModel
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing ML dependencies. Install torch, timm, transformers, and "
        "segmentation_models_pytorch before running this app."
    ) from exc


APP_TITLE = "Virtual H&E Final Ensemble"
PREVIEW_SIZE = 360
DEFAULT_SIZE = 1024
HIBOU_MODEL_ID = "histai/hibou-b"
SUPPORTED_EXTS = [
    ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
    ("All files", "*.*"),
]

DEFAULT_WEIGHTS = {
    "best_ssim": (0.30, 0.50, 0.20),
    "balanced": (0.20, 0.60, 0.20),
}

MODE_LABELS = {
    "best_ssim": "Best SSIM ensemble (v20/v22A/v21B)",
    "balanced": "Balanced ensemble (default)",
    "v22_tta": "Single v22A TTA",
    "v20_tta": "Legacy v20 TTA",
}


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def bundle_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", app_root())).resolve()


def roots() -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for root in (app_root(), bundle_root(), Path.cwd()):
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


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


def first_existing(candidates: list[Path | None]) -> Path | None:
    for path in candidates:
        if path is not None and path.exists():
            return path.resolve()
    return None


def default_model_paths() -> dict[str, Path | None]:
    candidates = {"v20": [], "v22": [], "v21b": []}
    for root in roots():
        candidates["v20"].extend(
            [
                root / "models" / "v20_fixed_model.pth",
                best_checkpoint_in(root / "checkpoints" / "v20_fixed"),
                root / "models" / "v20_model.pth",
                best_checkpoint_in(root / "checkpoints" / "v20"),
            ]
        )
        candidates["v22"].extend(
            [
                root / "models" / "v22a_content_quality_top1000_ft_model.pth",
                best_checkpoint_in(root / "checkpoints" / "v22a_content_quality_top1000_ft"),
            ]
        )
        candidates["v21b"].extend(
            [
                root / "models" / "v21b_hibou_content_quality_ft_model.pth",
                best_checkpoint_in(root / "checkpoints" / "v21b_hibou_content_quality_ft"),
            ]
        )
    return {key: first_existing(value) for key, value in candidates.items()}


def default_hibou_model_ref() -> str:
    env_path = os.environ.get("HIBOU_MODEL_PATH", "").strip()
    if env_path:
        expanded = Path(env_path).expanduser()
        if expanded.exists():
            return str(expanded.resolve())

    for root in roots():
        for candidate in (
            root / "models" / "hibou-b",
            root / "hibou-b",
            root / "hf_models" / "hibou-b",
        ):
            if candidate.exists():
                return str(candidate.resolve())
    return HIBOU_MODEL_ID


class ConvNeXtUNet(nn.Module):
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


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class UpFuseBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fuse = conv_block(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class InputPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.s1024 = conv_block(3, 24)
        self.s512 = DownBlock(24, 32)
        self.s256 = DownBlock(32, 64)
        self.s128 = DownBlock(64, 96)
        self.s64 = DownBlock(96, 128)
        self.s32 = DownBlock(128, 160)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s1024 = self.s1024(x)
        s512 = self.s512(s1024)
        s256 = self.s256(s512)
        s128 = self.s128(s256)
        s64 = self.s64(s128)
        s32 = self.s32(s64)
        return {
            "s1024": s1024,
            "s512": s512,
            "s256": s256,
            "s128": s128,
            "s64": s64,
            "s32": s32,
        }


class HibouBInputSkipUNet(nn.Module):
    def __init__(self, hf_model_id: str = HIBOU_MODEL_ID):
        super().__init__()
        self.hibou = AutoModel.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
            local_files_only=True,
        )
        for param in self.hibou.parameters():
            param.requires_grad = False
        self.hibou.eval()

        hidden = int(self.hibou.config.hidden_size)
        self.num_register_tokens = int(getattr(self.hibou.config, "num_register_tokens", 4))
        self.register_buffer("hibou_mean", torch.tensor([0.7068, 0.5755, 0.7220]).view(1, 3, 1, 1))
        self.register_buffer("hibou_std", torch.tensor([0.1950, 0.2316, 0.1816]).view(1, 3, 1, 1))

        self.input_pyramid = InputPyramid()
        self.hibou_proj = nn.Sequential(
            nn.Conv2d(hidden, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.up32 = UpFuseBlock(512, 160, 256)
        self.up64 = UpFuseBlock(256, 128, 192)
        self.up128 = UpFuseBlock(192, 96, 128)
        self.up256 = UpFuseBlock(128, 64, 96)
        self.up512 = UpFuseBlock(96, 32, 64)
        self.up1024 = UpFuseBlock(64, 24, 32)
        self.head = nn.Conv2d(32, 3, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        self.hibou.eval()
        return self

    def encode_hibou(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.hibou_mean.to(dtype=x.dtype)) / self.hibou_std.to(dtype=x.dtype)
        with torch.no_grad():
            out = self.hibou(pixel_values=x, return_dict=True)
        tokens = out.last_hidden_state[:, 1 + self.num_register_tokens :, :]
        side = int(tokens.shape[1] ** 0.5)
        if side * side != tokens.shape[1]:
            raise RuntimeError(f"Unexpected Hibou token count: {tokens.shape[1]}")
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.input_pyramid(x)
        h = self.hibou_proj(self.encode_hibou(x))
        x = self.up32(h, skips["s32"])
        x = self.up64(x, skips["s64"])
        x = self.up128(x, skips["s128"])
        x = self.up256(x, skips["s256"])
        x = self.up512(x, skips["s512"])
        x = self.up1024(x, skips["s1024"])
        return torch.sigmoid(self.head(x))


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


def strip_hibou_buffers_for_legacy_state(state: dict) -> dict:
    # Older v21B state files did not include app-local normalization buffers.
    return {k: v for k, v in state.items() if k not in {"hibou_mean", "hibou_std"}}


def load_convnext(path: Path, device: torch.device) -> ConvNeXtUNet:
    model = ConvNeXtUNet().to(device)
    model.load_state_dict(load_state_dict(path, device), strict=True)
    model.eval()
    return model


def load_hibou(path: Path, device: torch.device) -> HibouBInputSkipUNet:
    model = HibouBInputSkipUNet(default_hibou_model_ref()).to(device)
    state = strip_hibou_buffers_for_legacy_state(load_state_dict(path, device))
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [key for key in missing if key not in {"hibou_mean", "hibou_std"}]
    if missing or unexpected:
        raise RuntimeError(f"v21B load mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model


@dataclass
class ModelPaths:
    v20: Path | None
    v22: Path | None
    v21b: Path | None


@dataclass
class ModelBundle:
    v20: nn.Module | None = None
    v22: nn.Module | None = None
    v21b: nn.Module | None = None


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
    label: str = "model",
) -> torch.Tensor:
    transforms = [
        ("original", (), ()),
        ("horizontal", (3,), (3,)),
        ("vertical", (2,), (2,)),
        ("both", (2, 3), (2, 3)),
    ]
    outputs = []
    for i, (name, flip_in, flip_out) in enumerate(transforms, start=1):
        if progress:
            progress(f"{label}: TTA {i}/4 {name}")
        batch = torch.flip(x, flip_in) if flip_in else x
        pred = infer_once(model, batch, device)
        if flip_out:
            pred = torch.flip(pred, flip_out)
        outputs.append(pred)
    return torch.stack(outputs, dim=0).mean(dim=0)


def run_model(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    use_tta: bool,
    progress: Callable[[str], None] | None,
    label: str,
) -> torch.Tensor:
    if use_tta:
        return infer_tta(model, x, device, progress, label)
    if progress:
        progress(f"{label}: inference")
    return infer_once(model, x, device)


def run_pipeline(
    bundle: ModelBundle,
    x: torch.Tensor,
    device: torch.device,
    mode: str,
    use_tta: bool = True,
    progress: Callable[[str], None] | None = None,
) -> torch.Tensor:
    if mode == "v20_tta":
        if bundle.v20 is None:
            raise RuntimeError("v20 model is not loaded")
        return run_model(bundle.v20, x, device, use_tta, progress, "v20").clamp(0, 1)
    if mode == "v22_tta":
        if bundle.v22 is None:
            raise RuntimeError("v22A model is not loaded")
        return run_model(bundle.v22, x, device, use_tta, progress, "v22A").clamp(0, 1)

    if mode not in DEFAULT_WEIGHTS:
        raise ValueError(f"Unknown mode: {mode}")
    if bundle.v20 is None or bundle.v22 is None or bundle.v21b is None:
        raise RuntimeError("Best ensemble requires v20, v22A, and v21B models")
    w20, w22, w21b = DEFAULT_WEIGHTS[mode]
    p20 = run_model(bundle.v20, x, device, use_tta, progress, "v20")
    p22 = run_model(bundle.v22, x, device, use_tta, progress, "v22A")
    p21b = run_model(bundle.v21b, x, device, use_tta, progress, "v21B")
    return (w20 * p20 + w22 * p22 + w21b * p21b).clamp(0, 1)


def load_bundle(paths: ModelPaths, device: torch.device, mode: str, progress=None) -> ModelBundle:
    bundle = ModelBundle()
    needs_v20 = mode in {"best_ssim", "balanced", "v20_tta"}
    needs_v22 = mode in {"best_ssim", "balanced", "v22_tta"}
    needs_v21b = mode in {"best_ssim", "balanced"}
    if needs_v20:
        if paths.v20 is None:
            raise FileNotFoundError("Missing v20 weights")
        if progress:
            progress(f"Loading v20: {paths.v20.name}")
        bundle.v20 = load_convnext(paths.v20, device)
    if needs_v22:
        if paths.v22 is None:
            raise FileNotFoundError("Missing v22A weights")
        if progress:
            progress(f"Loading v22A: {paths.v22.name}")
        bundle.v22 = load_convnext(paths.v22, device)
    if needs_v21b:
        if paths.v21b is None:
            raise FileNotFoundError("Missing v21B weights")
        if progress:
            progress(f"Loading v21B/Hibou: {paths.v21b.name}")
        bundle.v21b = load_hibou(paths.v21b, device)
    return bundle


def make_preview(image: Image.Image, size: int = PREVIEW_SIZE) -> ImageTk.PhotoImage:
    img = image.copy()
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return ImageTk.PhotoImage(canvas)


class BestStainApp(tk.Tk):
    def __init__(self, size: int, mode: str):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1080x740")
        self.minsize(980, 660)
        defaults = default_model_paths()
        self.paths = ModelPaths(defaults["v20"], defaults["v22"], defaults["v21b"])
        self.queue: queue.Queue = queue.Queue()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bundle: ModelBundle | None = None
        self.loaded_mode: str | None = None
        self.input_image: Image.Image | None = None
        self.output_image: Image.Image | None = None
        self.input_preview: ImageTk.PhotoImage | None = None
        self.output_preview: ImageTk.PhotoImage | None = None
        self.output_size = tk.IntVar(value=size)
        self.use_tta = tk.BooleanVar(value=True)
        self.mode_var = tk.StringVar(value=mode)
        self.v20_var = tk.StringVar(value=str(self.paths.v20 or ""))
        self.v22_var = tk.StringVar(value=str(self.paths.v22 or ""))
        self.v21b_var = tk.StringVar(value=str(self.paths.v21b or ""))
        self.status_var = tk.StringVar(value="Ready")
        self.busy = False
        self._build_ui()
        self.after(100, self._poll_queue)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        top = ttk.LabelFrame(root, text="Model weights", padding=8)
        top.pack(fill=tk.X)
        self._path_row(top, "v20", self.v20_var, lambda: self._browse_weight(self.v20_var), 0)
        self._path_row(top, "v22A", self.v22_var, lambda: self._browse_weight(self.v22_var), 1)
        self._path_row(top, "v21B", self.v21b_var, lambda: self._browse_weight(self.v21b_var), 2)

        controls = ttk.Frame(root)
        controls.pack(fill=tk.X, pady=(10, 8))
        ttk.Button(controls, text="Load Models", command=self._load_models_async).pack(side=tk.LEFT)
        ttk.Button(controls, text="Open Image", command=self._open_image).pack(side=tk.LEFT, padx=(8, 0))
        self.run_button = ttk.Button(controls, text="Generate Stain", command=self._run_inference)
        self.run_button.pack(side=tk.LEFT, padx=8)
        self.save_button = ttk.Button(controls, text="Save Output", command=self._save_output, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT)

        ttk.Label(controls, text="Mode").pack(side=tk.LEFT, padx=(18, 4))
        ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            values=list(MODE_LABELS.keys()),
            width=12,
            state="readonly",
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(controls, text="TTA4", variable=self.use_tta).pack(side=tk.LEFT, padx=(12, 6))
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

    def _path_row(self, parent, label, var, command, row):
        ttk.Label(parent, text=label, width=6).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky=tk.EW, padx=6)
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, sticky=tk.E)
        parent.columnconfigure(1, weight=1)

    def _browse_weight(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            title="Select model weights",
            filetypes=[("PyTorch weights", "*.pth *.pt"), ("All files", "*.*")],
        )
        if path:
            var.set(path)

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        self.run_button.config(state=tk.DISABLED if busy else tk.NORMAL)

    def _current_paths(self) -> ModelPaths:
        def to_path(value: str) -> Path | None:
            value = value.strip()
            return Path(value).expanduser() if value else None

        return ModelPaths(to_path(self.v20_var.get()), to_path(self.v22_var.get()), to_path(self.v21b_var.get()))

    def _load_models_async(self) -> None:
        if self.busy:
            return
        self._set_busy(True)
        mode = self.mode_var.get()
        paths = self._current_paths()
        self.status_var.set(f"Loading {MODE_LABELS.get(mode, mode)}...")
        threading.Thread(target=self._load_models_worker, args=(paths, mode), daemon=True).start()

    def _load_models_worker(self, paths: ModelPaths, mode: str) -> None:
        try:
            bundle = load_bundle(
                paths,
                self.device,
                mode,
                progress=lambda msg: self.queue.put(("status", msg)),
            )
            self.queue.put(("models_loaded", bundle, mode))
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
        if self.input_image is None:
            messagebox.showwarning("No image", "Open an unstained image first.")
            return
        if self.bundle is None or self.loaded_mode != self.mode_var.get():
            messagebox.showwarning("Models not loaded", "Load models for the selected mode first.")
            return
        self._set_busy(True)
        threading.Thread(target=self._infer_worker, daemon=True).start()

    def _infer_worker(self) -> None:
        try:
            assert self.input_image is not None
            assert self.bundle is not None
            size = int(self.output_size.get())
            tensor, resized = pil_to_tensor(self.input_image, size)
            mode = self.mode_var.get()
            pred = run_pipeline(
                self.bundle,
                tensor,
                self.device,
                mode,
                use_tta=bool(self.use_tta.get()),
                progress=lambda msg: self.queue.put(("status", msg)),
            )
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
                if kind == "models_loaded":
                    _, bundle, mode = msg
                    self.bundle = bundle
                    self.loaded_mode = mode
                    self._set_busy(False)
                    self.status_var.set(f"Loaded: {MODE_LABELS.get(mode, mode)}")
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


def cli_run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    paths = ModelPaths(args.v20, args.v22, args.v21b)
    bundle = load_bundle(paths, device, args.mode, progress=print)
    image = Image.open(args.input).convert("RGB")
    tensor, _ = pil_to_tensor(image, args.size)
    pred = run_pipeline(bundle, tensor, device, args.mode, use_tta=not args.no_tta, progress=print)
    output = tensor_to_pil(pred)
    output.save(args.output)
    print(f"Saved {args.output}")


def write_default_manifest(path: Path) -> None:
    defaults = default_model_paths()
    canonical = app_root() / "best_model_manifest.json"
    if canonical.exists():
        data = json.loads(canonical.read_text())
    else:
        data = {
            "name": "Virtual H&E final weighted ensemble",
            "default_mode": "balanced",
            "best_ssim_weights": {"v20": 0.30, "v22A": 0.50, "v21B": 0.20},
            "balanced_weights": {"v20": 0.20, "v22A": 0.60, "v21B": 0.20},
        }
    data["default_mode"] = "balanced"
    data["detected_models"] = {key: str(value) if value else None for key, value in defaults.items()}
    data["hibou_model_ref"] = default_hibou_model_ref()
    path.write_text(json.dumps(data, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    defaults = default_model_paths()
    parser = argparse.ArgumentParser(description="Final Virtual H&E ensemble app")
    parser.add_argument("--input", type=Path, default=None, help="Input image for CLI mode")
    parser.add_argument("--output", type=Path, default=None, help="Output image for CLI mode")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, choices=[512, 768, 1024])
    parser.add_argument("--mode", choices=list(MODE_LABELS.keys()), default="balanced")
    parser.add_argument("--no-tta", action="store_true", help="Disable 4-flip TTA")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--v20", type=Path, default=defaults["v20"])
    parser.add_argument("--v22", type=Path, default=defaults["v22"])
    parser.add_argument("--v21b", type=Path, default=defaults["v21b"])
    parser.add_argument("--write-manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_manifest:
        write_default_manifest(args.write_manifest)
        return
    if args.input or args.output:
        if not args.input or not args.output:
            raise SystemExit("--input and --output must be provided together")
        cli_run(args)
        return
    app = BestStainApp(size=args.size, mode=args.mode)
    app.mainloop()


if __name__ == "__main__":
    main()
