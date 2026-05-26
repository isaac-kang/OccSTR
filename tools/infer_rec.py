"""Single-image inference for OccSTR.

Usage:
  python tools/infer_rec.py \
      --config weights/pretrained/svtrv2_smtr_gtc_rctc/config.yml \
      --ckpt   weights/pretrained/svtrv2_smtr_gtc_rctc/best.pth \
      --image  path/to/img.png [path/to/img2.png ...]
"""

import argparse
import os
import sys
from pathlib import Path

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
    p.add_argument('--config', required=True, help='Architecture/PostProcess config.yml')
    p.add_argument('--ckpt', required=True, help='Path to .pth')
    p.add_argument('--image', nargs='+', required=True, help='One or more image paths')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--image_shape', default='32,128', help='H,W for RecTVResize')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(args.config).cfg

    post_process = build_post_process(cfg['PostProcess'], cfg['Global'])
    char_num = post_process.get_character_num()
    cfg['Architecture']['Decoder']['out_channels'] = char_num

    model = build_model(cfg['Architecture'])
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'[warn] {len(missing)} missing keys, e.g. {missing[:3]}')
    if unexpected:
        print(f'[warn] {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}')

    model.eval().to(args.device)

    H, W = [int(x) for x in args.image_shape.split(',')]
    resizer = RecTVResize(image_shape=[H, W], padding=True)

    for img_path in args.image:
        if not Path(img_path).exists():
            print(f'[skip] {img_path}: not found')
            continue
        img = Image.open(img_path).convert('RGB')
        tensor = resizer({'image': img})['image'].unsqueeze(0).to(args.device)
        with torch.inference_mode():
            preds = model(tensor)
        out = post_process(preds)
        # GTC: out = [gtc, ctc]; each is list of (text, score)
        if isinstance(out, list) and len(out) == 2:
            gtc, ctc = out
            gtc_text, gtc_score = gtc[0]
            ctc_text, ctc_score = ctc[0]
            print(f'{img_path}\tGTC="{gtc_text}" ({gtc_score:.3f})\tCTC="{ctc_text}" ({ctc_score:.3f})')
        else:
            text, score = out[0]
            print(f'{img_path}\t"{text}" ({score:.3f})')


if __name__ == '__main__':
    main()
