"""
Inference backend for the Virtual H&E Stain Generator GUI.
No tkinter imports here — safe to call from background threads.
"""

import sys
from pathlib import Path
import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from src.models.gan import UNetGenerator, PatchGANDiscriminator
from src.training.lightning_module_v11 import GANModuleV11


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(ckpt_path: str, device: str):
    """
    Load UNetGenerator from checkpoint.
    Uses pretrained=False to avoid internet download (checkpoint has trained weights).
    Uses strict=False to handle Lightning version mismatches.
    Returns generator in inference mode.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Place the .ckpt file in the checkpoints/ folder."
        )

    gen = UNetGenerator(pretrained=False).to(device)
    disc = PatchGANDiscriminator().to(device)
    module = GANModuleV11(gen, disc, lambda_l1=300, lambda_gp=10, lambda_struct=20, lambda_percept=10)

    checkpoint = torch.load(str(ckpt_path), map_location=device)
    module.load_state_dict(checkpoint['state_dict'], strict=False)

    gen = module.gen
    gen.train(False)   # switch to inference mode (equivalent to gen.eval())
    return gen


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    """
    Convert PIL image to model input tensor [1, 3, 256, 256] in [-1, 1].
    """
    img = pil_img.convert("RGB").resize((256, 256), Image.BILINEAR)
    tensor = TF.to_tensor(img)       # [3, 256, 256] in [0, 1]
    tensor = tensor * 2.0 - 1.0     # [3, 256, 256] in [-1, 1]
    return tensor.unsqueeze(0)       # [1, 3, 256, 256]


def postprocess_output(output_tensor: torch.Tensor) -> Image.Image:
    """
    Convert model output tensor [1, 3, 256, 256] in [-1, 1] to PIL Image.
    """
    out = output_tensor.squeeze(0).cpu()
    out = (out + 1.0) / 2.0
    out = out.clamp(0.0, 1.0)
    out_np = (out.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(out_np)


def _infer_plain(gen, tensor: torch.Tensor, device: str) -> torch.Tensor:
    with torch.no_grad():
        return gen(tensor.to(device))


def _infer_tta(gen, tensor: torch.Tensor, device: str, progress_callback=None) -> torch.Tensor:
    """8-rotation TTA: 4 angles x 2 h-flip states, predictions averaged."""
    predictions = []
    step = 0

    with torch.no_grad():
        for angle in [0, 90, 180, 270]:
            for hflip in [False, True]:
                step += 1
                if progress_callback:
                    progress_callback(f"TTA step {step}/8 (angle={angle}, flip={hflip})...")

                batch = tensor.clone()

                if angle != 0:
                    batch = TF.rotate(batch.squeeze(0), angle).unsqueeze(0)

                if hflip:
                    batch = torch.flip(batch, [-1])

                pred = gen(batch.to(device))

                if hflip:
                    pred = torch.flip(pred, [-1])

                if angle != 0:
                    pred = TF.rotate(pred.squeeze(0), -angle).unsqueeze(0)

                predictions.append(pred.cpu())

    return torch.stack(predictions).mean(dim=0)


def run_inference(
    gen,
    pil_img: Image.Image,
    device: str,
    use_tta: bool = False,
    progress_callback=None,
) -> Image.Image:
    """
    Run inference on a single PIL image.
    progress_callback(message: str) called with status strings.
    Returns stained PIL Image (256x256 RGB).
    """
    tensor = preprocess_image(pil_img)

    if use_tta:
        output = _infer_tta(gen, tensor, device, progress_callback)
    else:
        if progress_callback:
            progress_callback("Running inference...")
        output = _infer_plain(gen, tensor, device)

    return postprocess_output(output)
