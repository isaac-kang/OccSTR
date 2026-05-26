"""Categorize SVTRv2 errors from WOST/HOST HTML reports.

Reads error_analysis/{WOST,HOST}_errors_svtrv2.html, extracts (pred_raw, gt_raw)
pairs, normalizes to 36-char form, and reports per-op statistics derived from
the same Levenshtein alignment used in eval_lmdb_html.render_pair.
"""

import html
import os
import re
import sys
from collections import Counter

from rapidfuzz.distance import Levenshtein

NORM_RE = re.compile(r'[^a-z0-9]')
VOWELS = set('aeiou')


def norm(s: str) -> str:
    return NORM_RE.sub('', s.lower())


def char_class(c):
    if c.isdigit(): return 'digit'
    if c in VOWELS: return 'vowel'
    if c.isalpha(): return 'consonant'
    return 'other'


def parse(path):
    with open(path) as f:
        s = f.read()
    rows = re.findall(
        r'<tr><td class="id">(.*?)</td>.*?'
        r'<td class="text raw">(.*?)</td><td class="text raw">(.*?)</td></tr>',
        s, re.S,
    )
    return [(sid,
             html.unescape(re.sub(r'<[^>]+>', '', p)),
             html.unescape(re.sub(r'<[^>]+>', '', g)))
            for sid, p, g in rows]


def categorize_edit(p, g):
    """(kind, n_sub, n_ins, n_del) from rapidfuzz opcodes."""
    n_sub = n_ins = n_del = 0
    for op in Levenshtein.opcodes(p, g):
        if op.tag == 'replace':
            n_sub += op.src_end - op.src_start
        elif op.tag == 'insert':
            n_ins += op.dest_end - op.dest_start
        elif op.tag == 'delete':
            n_del += op.src_end - op.src_start
    total = n_sub + n_ins + n_del
    if total == 0: return 'identical', 0, 0, 0
    if total == 1:
        if n_sub: return 'edit1_sub', n_sub, n_ins, n_del
        if n_ins: return 'edit1_ins', n_sub, n_ins, n_del
        return 'edit1_del', n_sub, n_ins, n_del
    if total == 2: return 'edit2', n_sub, n_ins, n_del
    if total <= 4: return 'edit3to4', n_sub, n_ins, n_del
    return 'edit5plus', n_sub, n_ins, n_del


def pos_bin(idx, total):
    if total <= 0: return 'end'
    if idx == 0: return 'start'
    if idx >= total - 1: return 'end'
    return 'mid'


def report(tag, rows):
    n = len(rows)
    cats = Counter()
    ed_dist = []
    gt_lens = []
    sub_total = ins_total = del_total = 0
    pos_sub = Counter(); pos_ins = Counter(); pos_del = Counter()
    confusion = Counter()      # (gt_char, pred_char) — substitutions only
    ins_chars = Counter()      # what pred dropped
    del_chars = Counter()      # what pred hallucinated
    cls_sub_gt = Counter()
    short_gt = 0

    for sid, p_raw, g_raw in rows:
        p, g = norm(p_raw), norm(g_raw)
        kind, n_sub, n_ins, n_del = categorize_edit(p, g)
        cats[kind] += 1
        sub_total += n_sub; ins_total += n_ins; del_total += n_del
        ed_dist.append(n_sub + n_ins + n_del)
        gt_lens.append(len(g))
        if len(g) <= 3: short_gt += 1

        for op in Levenshtein.opcodes(p, g):
            if op.tag == 'replace':
                k_len = max(op.src_end - op.src_start, op.dest_end - op.dest_start)
                for k in range(k_len):
                    si = op.src_start + k
                    di = op.dest_start + k
                    if si < op.src_end and di < op.dest_end:
                        cp, cg = p[si], g[di]
                        confusion[(cg, cp)] += 1
                        cls_sub_gt[char_class(cg)] += 1
                        pos_sub[pos_bin(di, len(g))] += 1
            elif op.tag == 'insert':
                for di in range(op.dest_start, op.dest_end):
                    ins_chars[g[di]] += 1
                    pos_ins[pos_bin(di, len(g))] += 1
            elif op.tag == 'delete':
                for si in range(op.src_start, op.src_end):
                    del_chars[p[si]] += 1
                    pos_del[pos_bin(si, len(p))] += 1

    print(f'\n=== {tag}  (n={n}) ===')
    print('Edit-kind breakdown:')
    for k in ['edit1_sub', 'edit1_ins', 'edit1_del', 'edit2', 'edit3to4', 'edit5plus']:
        v = cats.get(k, 0)
        bar = '#' * min(40, v // 5)
        print(f'  {k:10s} {v:4d}  ({v/n:5.1%}) {bar}')

    tot_ops = sub_total + ins_total + del_total or 1
    print(f'\nTotal char-level ops across all {n} errors (= {tot_ops}):')
    print(f'  substitutions : {sub_total:4d}  ({sub_total/tot_ops:5.1%})')
    print(f'  insertions    : {ins_total:4d}  ({ins_total/tot_ops:5.1%})   (gt has char pred dropped)')
    print(f'  deletions     : {del_total:4d}  ({del_total/tot_ops:5.1%})   (pred has extra hallucinated char)')

    print(f'\nEdit distance: mean={sum(ed_dist)/n:.2f}, median={sorted(ed_dist)[n//2]}, max={max(ed_dist)}')
    print(f'GT len: mean={sum(gt_lens)/n:.1f}, median={sorted(gt_lens)[n//2]}, '
          f'min={min(gt_lens)}, max={max(gt_lens)}   '
          f'short (≤3): {short_gt} ({short_gt/n:.1%})')

    def fmt(c):
        t = sum(c.values()) or 1
        return (f'start {c["start"]:3d} ({c["start"]/t:5.1%})  '
                f'mid {c["mid"]:3d} ({c["mid"]/t:5.1%})  '
                f'end {c["end"]:3d} ({c["end"]/t:5.1%})')
    print('\nOp position in gt/pred word:')
    print(f'  sub : {fmt(pos_sub)}')
    print(f'  ins : {fmt(pos_ins)}')
    print(f'  del : {fmt(pos_del)}')

    cls_t = sum(cls_sub_gt.values()) or 1
    print('\nGT char class for substitutions (what got mis-read):')
    for c in ('vowel', 'consonant', 'digit', 'other'):
        v = cls_sub_gt.get(c, 0)
        print(f'  {c:10s} {v:4d}  ({v/cls_t:5.1%})')

    print('\nTop ins chars (gt char that pred dropped):')
    for ch, k in ins_chars.most_common(8):
        print(f"  '{ch}' : {k}")
    print('Top del chars (extra char pred hallucinated):')
    for ch, k in del_chars.most_common(8):
        print(f"  '{ch}' : {k}")

    return confusion


def main():
    base = os.path.join(os.path.dirname(__file__), '..', 'error_analysis')
    files = [
        ('WOST', os.path.join(base, 'WOST_errors_svtrv2.html')),
        ('HOST', os.path.join(base, 'HOST_errors_svtrv2.html')),
    ]
    combined = Counter()
    for tag, path in files:
        rows = parse(path)
        if not rows:
            print(f'[warn] no rows parsed from {path}', file=sys.stderr)
            continue
        conf = report(tag, rows)
        combined.update(conf)

    print('\n=== Top char substitutions (gt → pred, WOST + HOST combined) ===')
    for (cg, cp), nn in combined.most_common(25):
        print(f"  '{cg}' → '{cp}' : {nn}")


if __name__ == '__main__':
    main()
