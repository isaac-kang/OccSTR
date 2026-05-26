"""Probe SVTRv2 Stage-1 (first 6 Conv blocks) shallow feature.

Forward an image through SVTRv2, capture the encoder's stage-0 block output
*before* the stage 0→1 downsample, and report its shape + summary stats.

Usage:
  python tools/probe_shallow_feat.py \
      --config weights/pretrained/svtrv2_smtr_gtc_rctc/config.yml \
      --ckpt   weights/pretrained/svtrv2_smtr_gtc_rctc/best.pth \
      --image  path/to/img.png
"""

import argparse
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import torch
from PIL import Image

from openrec.modeling import build_model
from openrec.postprocess import build_post_process
from openrec.preprocess.resize import RecTVResize
from tools.engine.config import Config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--image_shape', default='32,128')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(args.config).cfg

    post_process = build_post_process(cfg['PostProcess'], cfg['Global'])
    cfg['Architecture']['Decoder']['out_channels'] = post_process.get_character_num()

    model = build_model(cfg['Architecture'])
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval().to(args.device)

    encoder = model.encoder
    print(f'Encoder: {type(encoder).__name__}')
    print(f'  pope (patch embed) → dims[0]={encoder.stages[0].blocks[0].mlp.fc1.in_features}')
    print(f'  num stages (incl. Feat2D tail): {len(encoder.stages)}')
    print(f'  stage[0].blocks: {len(encoder.stages[0].blocks)} blocks '
          f'({type(encoder.stages[0].blocks[0]).__name__} × N)')

    # Capture output of last ConvBlock of stage 0, BEFORE stages[0].downsample.
    captured = {}

    def hook(module, inp, out):
        captured['feat'] = out.detach()

    handle = encoder.stages[0].blocks[-1].register_forward_hook(hook)

    H, W = [int(x) for x in args.image_shape.split(',')]
    resizer = RecTVResize(image_shape=[H, W], padding=True)
    img = Image.open(args.image).convert('RGB')
    x = resizer({'image': img})['image'].unsqueeze(0).to(args.device)
    print(f'\nInput tensor: {tuple(x.shape)}  (range [{x.min():.3f}, {x.max():.3f}])')

    with torch.inference_mode():
        _ = encoder(x)
    handle.remove()

    feat = captured['feat']
    print(f'\n=== Stage-1 shallow feature (after stages[0].blocks, before downsample) ===')
    print(f'  shape : {tuple(feat.shape)}    # [B, C, H, W]')
    print(f'  dtype : {feat.dtype}')
    print(f'  mean  : {feat.mean().item():+.4f}')
    print(f'  std   : {feat.std().item():.4f}')
    print(f'  min   : {feat.min().item():+.4f}')
    print(f'  max   : {feat.max().item():+.4f}')
    print(f'  |.|_2 : {feat.norm().item():.2f}')


if __name__ == '__main__':
    main()
