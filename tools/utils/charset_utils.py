"""Charset utilities for CCD (Confusion-aware Class Decomposition).

Provides:
  - build_pl_charset: Extend base charset with Unicode variant characters
  - PLCharsetAdapter: Map extended Unicode chars back to base chars during eval
  - generate_extended_dict: Create an extended dict file for training with CCD
"""

import json
import re
from pathlib import Path


def build_pl_charset(base_dict_path, unicode_mapping_path):
    """Extend charset with Unicode variant characters from unicode_mapping.json.

    Args:
        base_dict_path: Path to the base character dict file (e.g., EN_symbol_dict.txt)
        unicode_mapping_path: Path to unicode_mapping.json from confusion_and_pl.py

    Returns:
        ext_chars: list of extended Unicode characters (to append to charset)
        ext_to_base: dict mapping each extended Unicode char to its base char
    """
    with open(unicode_mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    ext_chars = [v['unicode'] for v in mapping.values()]
    ext_to_base = {v['unicode']: v['base_char'] for v in mapping.values()}
    return ext_chars, ext_to_base


def generate_extended_dict(base_dict_path, unicode_mapping_path, output_path=None):
    """Generate an extended dict file that includes Unicode variant chars.

    The extended dict is the base dict + one line per Unicode variant char.
    This can be used as `character_dict_path` in training configs for CCD mode.

    Args:
        base_dict_path: Path to the base dict file
        unicode_mapping_path: Path to unicode_mapping.json
        output_path: Where to write the extended dict. If None, writes next to unicode_mapping.

    Returns:
        output_path: Path to the generated extended dict file
    """
    base_dict_path = Path(base_dict_path)
    unicode_mapping_path = Path(unicode_mapping_path)

    if output_path is None:
        output_path = unicode_mapping_path.parent / 'extended_dict.txt'
    else:
        output_path = Path(output_path)

    # Read base dict
    with open(base_dict_path, 'r', encoding='utf-8') as f:
        base_lines = [line.rstrip('\n').rstrip('\r') for line in f.readlines()]

    # Get extended chars
    ext_chars, _ = build_pl_charset(base_dict_path, unicode_mapping_path)

    # Write extended dict
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in base_lines:
            f.write(line + '\n')
        for ch in ext_chars:
            f.write(ch + '\n')

    return str(output_path)


class PLCharsetAdapter:
    """Maps extended PL Unicode chars back to base chars, then filters unsupported chars.

    Used during evaluation so that predictions containing extended Unicode
    variant characters (from CCD training) are properly compared against
    standard GT labels.

    Usage in trainer:
        adapter = PLCharsetAdapter(target_charset, ext_to_base)
        pred_mapped = adapter(pred_text)  # 'hèllo' -> 'hello'
    """

    def __init__(self, target_charset, ext_to_base):
        """
        Args:
            target_charset: str or list of allowed characters for evaluation
            ext_to_base: dict mapping extended Unicode chars to base chars
                e.g., {'è': 'e', 'ç': 'c', ...}
        """
        self.ext_to_base = ext_to_base
        charset_str = ''.join(target_charset) if isinstance(target_charset, (list, tuple)) else target_charset
        self.lowercase_only = charset_str == charset_str.lower()
        self.uppercase_only = charset_str == charset_str.upper()
        self.unsupported = re.compile(f'[^{re.escape(charset_str)}]')

    def __call__(self, label):
        # First map extended chars to base chars
        label = ''.join(self.ext_to_base.get(c, c) for c in label)
        # Then apply case normalization
        if self.lowercase_only:
            label = label.lower()
        elif self.uppercase_only:
            label = label.upper()
        # Remove unsupported characters
        label = self.unsupported.sub('', label)
        return label
