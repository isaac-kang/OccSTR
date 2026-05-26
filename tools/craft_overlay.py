"""Light wrapper around CRAFT (clovaai/CRAFT-pytorch) for generating
character region-score heatmap overlays on 32x128 resized inputs.

CRAFT lives unmodified under OccSTR/third_party/craft. This module:
  - prepends third_party/craft to sys.path so its `craft.py` / `basenet`
    relative imports resolve
  - loads a CRAFT checkpoint (default: craft_ic15_20k.pth)
  - exposes `make_overlay_png_b64(pil_img)` returning a base64 PNG string
"""

import base64
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_HERE = Path(__file__).resolve().parent
_CRAFT_DIR = _HERE.parent / 'third_party' / 'craft'
if str(_CRAFT_DIR) not in sys.path:
    sys.path.insert(0, str(_CRAFT_DIR))

from craft import CRAFT  # noqa: E402

DEFAULT_CKPT = _HERE.parent / 'weights' / 'pretrained' / 'craft' / 'craft_ic15_20k.pth'

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


def _strip_module(state):
    if next(iter(state)).startswith('module.'):
        return {k[len('module.'):]: v for k, v in state.items()}
    return state


def load_craft(ckpt_path=None, device='cuda'):
    ckpt_path = str(ckpt_path or DEFAULT_CKPT)
    model = CRAFT(pretrained=False, freeze=False)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(_strip_module(state))
    model.eval().to(device)
    return model


@torch.inference_mode()
def region_heatmap(model, pil_img, target_hw=(32, 128), device='cuda'):
    """Forward a PIL image through CRAFT at target_hw, return the region
    (character) score map upsampled to target_hw as float32 in roughly [0,1]."""
    h, w = target_hw
    img = pil_img.convert('RGB').resize((w, h), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32)  # HxWx3, RGB, [0,255]
    arr = (arr - _MEAN) / _STD
    x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    out, _ = model(x)  # [1, H/2, W/2, 2]; ch 0 region, ch 1 affinity
    region = out[0, :, :, 0].float().cpu().numpy()
    region = np.clip(region, 0.0, 1.0)
    region_t = torch.from_numpy(region)[None, None]
    region_t = F.interpolate(region_t, size=target_hw, mode='bilinear',
                              align_corners=False)
    return region_t[0, 0].numpy()


def _jet(values):
    """Apply jet colormap. Uses matplotlib if available; else a fallback."""
    try:
        from matplotlib import colormaps
        cmap = colormaps['jet']
        return (cmap(values)[..., :3] * 255.0).astype(np.float32)
    except Exception:
        v = np.clip(values, 0.0, 1.0)
        r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
        return np.stack([r, g, b], axis=-1) * 255.0


def make_overlay_png_b64(model, pil_img, target_hw=(32, 128), alpha=0.55,
                         device='cuda'):
    """Run CRAFT on `pil_img` resized to `target_hw` and return a base64-encoded
    PNG of the input image with the region heatmap overlaid (jet colormap)."""
    heat = region_heatmap(model, pil_img, target_hw=target_hw, device=device)
    h, w = target_hw
    img = np.asarray(pil_img.convert('RGB').resize((w, h), Image.BICUBIC),
                     dtype=np.float32)
    colors = _jet(heat)
    blended = (1 - alpha) * img + alpha * colors
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')
