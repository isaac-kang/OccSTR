#!/usr/bin/env python3
"""Generate MLLM OCR predictions for a single LMDB dataset and save as an
eval_predictions-compatible CSV (file_idx,gt,pred,correct).

Runs inside a vLLM conda env. Typical invocation (from the main env):
    conda run --no-capture-output -n vllm \
        python tools/infer/predict_mllm.py \
        --mllm_model Qwen/Qwen3.5-27B-GPTQ-Int4 \
        --data_dir ~/data/STR/openocr/.../filter_train_medium_subset \
        --output_path weights/pretrained/mllm_<safe>/eval_predictions/<ds>__mllm_<safe>.csv

Forces direct-answer output via:
  * Short prompt: "Read the text in the image ath the word level."
  * max_tokens=32, temperature=0
"""
import argparse
import base64
import csv
import io
import os
import string
import sys

import lmdb as lmdb_lib
from PIL import Image

DEFAULT_PROMPT = 'Read the text in the image at the word level.'


def normalize_eval(text):
    """Match RecMetric (filter+space+lower, i.e. 36char)."""
    text = (text or '').replace(' ', '').lower()
    return ''.join(c for c in text if c in string.digits + string.ascii_lowercase)


def load_image_data_url(env, file_idx):
    with env.begin(buffers=True) as txn:
        buf = txn.get(f'image-{file_idx:09d}'.encode())
        if buf is None:
            return None
        img = Image.open(io.BytesIO(bytes(buf))).convert('RGB')
    out = io.BytesIO()
    img.save(out, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(out.getvalue()).decode()


def read_gt(env, file_idx):
    with env.begin() as txn:
        buf = txn.get(f'label-{file_idx:09d}'.encode())
    return buf.decode('utf-8') if buf else ''


def extract_text(completion):
    """No guided / no prefill anymore — just strip whitespace + surrounding
    quotes/braces and take the first line."""
    s = (completion or '').strip()
    s = s.split('\n', 1)[0].strip()
    return s.strip('{}').strip('"\'`').strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mllm_model', required=True)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--output_path', required=True)
    p.add_argument('--tp', type=int, default=1)
    p.add_argument('--max_model_len', type=int, default=1024)
    p.add_argument('--max_tokens', type=int, default=32)
    p.add_argument('--max_pixels', type=int, default=512 * 28 * 28)
    p.add_argument('--min_pixels', type=int, default=28 * 28)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--prompt', default=DEFAULT_PROMPT)
    p.add_argument('--gpu_mem', type=float, default=0.9)
    p.add_argument('--debug_n', type=int, default=5,
                   help='Print raw completion + parsed pred + gt for the '
                        'first N samples (0 to disable).')
    args = p.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    env = lmdb_lib.open(data_dir, readonly=True, lock=False,
                        readahead=False, meminit=False)
    with env.begin() as txn:
        num = int(txn.get(b'num-samples'))
    print(f'Dataset: {data_dir} ({num} samples)')

    from vllm import LLM, SamplingParams

    print(f'Loading MLLM: {args.mllm_model}')
    mllm = LLM(
        model=args.mllm_model,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        trust_remote_code=True,
        limit_mm_per_prompt={'image': 1},
        mm_processor_kwargs={'min_pixels': args.min_pixels,
                             'max_pixels': args.max_pixels},
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        disable_log_stats=True,
    )

    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    chat_kwargs = {}
    if 'Qwen3' in args.mllm_model:
        chat_kwargs['chat_template_kwargs'] = {'enable_thinking': False}

    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    batch_convs, batch_meta = [], []
    done = 0

    with open(args.output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file_idx', 'gt', 'pred', 'correct'])

        def flush():
            nonlocal batch_convs, batch_meta, done
            if not batch_convs:
                return
            outputs = mllm.chat(batch_convs, sampling_params=sampling_params,
                                use_tqdm=False, **chat_kwargs)
            for meta, out in zip(batch_meta, outputs):
                completion = out.outputs[0].text if out.outputs else ''
                pred = extract_text(completion)
                correct = int(normalize_eval(pred) == normalize_eval(meta['gt']))
                writer.writerow([meta['idx'], meta['gt'], pred, correct])
                done += 1
                if done <= args.debug_n:
                    mark = 'OK ' if correct else 'ERR'
                    print(f'  [debug {done:>3d}] {mark} idx={meta["idx"]:>6d} '
                          f'gt={meta["gt"]!r:<20} pred={pred!r:<20} '
                          f'raw={completion!r}',
                          flush=True)
            batch_convs, batch_meta = [], []
            print(f'  {done}/{num}', flush=True)

        for idx in range(1, num + 1):
            gt = read_gt(env, idx)
            data_url = load_image_data_url(env, idx)
            if data_url is None:
                continue
            messages = [
                {'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': data_url}},
                    {'type': 'text', 'text': args.prompt},
                ]},
            ]
            batch_convs.append(messages)
            batch_meta.append({'idx': idx, 'gt': gt})
            if len(batch_convs) >= args.batch_size:
                flush()
        flush()

    env.close()
    print(f'Saved {done} predictions to {args.output_path}')


if __name__ == '__main__':
    main()
