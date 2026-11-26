"""Versioned Unicode rules shared by source preparation and model tokenization.

Python 3.11 and 3.14 otherwise use different Unicode databases. Keep NFC,
character names and categories identical across the CPU fleet and GPU runtime.
"""
import unicodedata as _stdlib

UNICODE_VERSION = "16.0.0"
REGEX_VERSION = "2026.7.19"

if _stdlib.unidata_version == UNICODE_VERSION:
    _unicode = _stdlib
else:
    try:
        import unicodedata2 as _unicode
    except ImportError as exc:
        raise RuntimeError("this tokenizer requires unicodedata2==16.0.0 on Python <3.14") from exc
    if _unicode.unidata_version != UNICODE_VERSION:
        raise RuntimeError("tokenizer Unicode database must be exactly 16.0.0")

unidata_version = _unicode.unidata_version
normalize = _unicode.normalize
category = _unicode.category
name = _unicode.name


def require_regex_version(regex_module):
    if regex_module.__version__ != REGEX_VERSION:
        raise RuntimeError(f"this tokenizer requires regex=={REGEX_VERSION}")
