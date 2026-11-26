"""Sarvam score-BPE and the script router.

English segments are not encoded here. The caller supplies Spark's tokenizer.
The merge rules match the frozen donor: NFC, a leading U+0120, highest score
first, equal scores merge the leftmost pair.
"""

from __future__ import annotations

import heapq
import re
import unicodedata
from bisect import bisect_left, insort

SPACE_MARK = "\u0120"
KEPT_INDIC_BLOCKS = (
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
)
_FORMAT_CHARS = {"\u200d", "\u200c"}
_WHITESPACE_RUN = re.compile(r"[ \t\n\r]+")
_LATIN_CATEGORIES = {"Lu", "Ll", "Lt", "Lm", "Lo"}
_BYTE_FALLBACK = re.compile(r"<0x[0-9A-F]{2}>")
INDIC_BASE = 544
ENGLISH_TAIL_BASE = 40347


def bytes_to_unicode():
    """GPT-2 byte alphabet, indexed by the original byte value."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codepoints = byte_values[:]
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    alphabet = [""] * 256
    for byte, point in zip(byte_values, codepoints):
        alphabet[byte] = chr(point)
    return alphabet


BYTE_ALPHABET = bytes_to_unicode()
BYTE_OF = {char: byte for byte, char in enumerate(BYTE_ALPHABET)}


def is_target_char(char):
    codepoint = ord(char)
    return any(lo <= codepoint <= hi for lo, hi in KEPT_INDIC_BLOCKS)


def has_target_script(text):
    return any(is_target_char(char) for char in text)


def _is_latin_letter(char):
    return (unicodedata.category(char) in _LATIN_CATEGORIES
            and unicodedata.name(char, "").startswith("LATIN "))


def _letter_classes(stretch):
    classes = []
    for index, char in enumerate(stretch):
        if _is_latin_letter(char):
            classes.append("L")
        elif is_target_char(char):
            classes.append("I")
        elif char in _FORMAT_CHARS:
            neighbor = None
            for cursor in range(index - 1, -1, -1):
                if stretch[cursor] not in _FORMAT_CHARS:
                    neighbor = stretch[cursor]
                    break
            if neighbor is None or not is_target_char(neighbor):
                for cursor in range(index + 1, len(stretch)):
                    if stretch[cursor] not in _FORMAT_CHARS:
                        neighbor = stretch[cursor]
                        break
            classes.append("I" if neighbor is not None and is_target_char(neighbor) else "O")
        else:
            classes.append("O")
    return classes


def _route_mixed_stretch(stretch, classes):
    units = []
    cursor = 0
    for word_index, word in enumerate(stretch.split(" ")):
        if word_index > 0:
            units.append(("space", " "))
        cursor += word_index > 0
        if word == "":
            continue
        word_classes = classes[cursor:cursor + len(word)]
        cursor += len(word)
        runs = []
        for char, cls in zip(word, word_classes):
            kind = "indic" if cls == "I" else ("english" if cls == "L" else None)
            if kind is None:
                if runs:
                    runs[-1][1] += char
                else:
                    runs.append(["pending", char])
            else:
                if runs and runs[-1][0] == "pending":
                    runs[-1][0] = kind
                if runs and runs[-1][0] == kind:
                    runs[-1][1] += char
                else:
                    runs.append([kind, char])
        if runs and runs[0][0] == "pending":
            runs[0][0] = "english"
        units.extend((kind, text) for kind, text in runs)
    return units


def route(text):
    """Yield ('indic'|'english', segment) pairs."""
    items = []
    pending = None
    cursor = 0
    for match in _WHITESPACE_RUN.finditer(text):
        if match.start() > cursor:
            pending = (pending or "") + text[cursor:match.start()]
        run = match.group(0)
        if run == " ":
            pending = (pending or "") + run
        else:
            if pending is not None:
                items.append(("stretch", pending))
                pending = None
            items.append(("english", run))
        cursor = match.end()
    if cursor < len(text):
        pending = (pending or "") + text[cursor:]
    if pending is not None:
        items.append(("stretch", pending))

    for item_kind, item in items:
        if item_kind == "english":
            yield ("english", item)
            continue
        classes = _letter_classes(item)
        has_indic = "I" in classes
        has_latin = "L" in classes
        if has_indic and not has_latin:
            yield ("indic", item)
        elif not has_indic:
            yield ("english", item)
        else:
            units = _route_mixed_stretch(item, classes)
            merged = []
            for index, (kind, value) in enumerate(units):
                if (kind == "space" and merged and merged[-1][0] == "indic"
                        and index + 1 < len(units) and units[index + 1][0] == "indic"):
                    merged[-1][1] += value
                elif kind == "indic" and merged and merged[-1][0] == "indic":
                    merged[-1][1] += value
                elif kind == "space":
                    merged.append(["english", " "])
                else:
                    merged.append([kind, value])
            for kind, value in merged:
                yield ("indic" if kind == "indic" else "english", value)


def decode_class(piece, indic_region):
    """'indic', 'english', or 'neutral', matching the donor look-back."""
    if _BYTE_FALLBACK.fullmatch(piece):
        return "indic"
    if piece in indic_region:
        return "indic"
    byte = BYTE_OF.get(piece)
    if byte is None:
        return "english"
    if byte >= 128 or byte in (9, 10, 13) or 65 <= byte <= 90 or 97 <= byte <= 122:
        return "english"
    return "neutral"


def _neutral(symbol):
    return len(symbol) == 1 and (symbol == SPACE_MARK or ord(symbol) < 128)


def _seed_symbols(transformed, known_pieces):
    symbols = []
    for char in transformed:
        if char == SPACE_MARK or char in known_pieces:
            symbols.append(char)
        elif ord(char) < 128:
            symbols.append(char)
        else:
            for byte in char.encode("utf-8"):
                symbols.append(f"<0x{byte:02X}>")
    return symbols


def indic_bpe(transformed, scored, *, restrict_space_merges):
    # `scored` contains every donor piece that has a score, including single
    # Indic letters. ASCII still falls through to the ord < 128 branch.
    texts = _seed_symbols(transformed, scored)
    n = len(texts)
    alive = [True] * n
    prev_of = [index - 1 for index in range(n)]
    next_of = [index + 1 for index in range(n)]
    non_neutral = []

    def restricted(position):
        if not restrict_space_merges or texts[position] != SPACE_MARK:
            return False
        return bisect_left(non_neutral, position) == 0

    heap = []

    def offer(left):
        right = next_of[left]
        if right >= n or not alive[left] or not alive[right]:
            return
        merged = texts[left] + texts[right]
        score = scored.get(merged)
        if score is None or restricted(left):
            return
        heapq.heappush(heap, (-score, left, right, merged))

    for index, symbol in enumerate(texts):
        if not _neutral(symbol):
            insort(non_neutral, index)
    for index in range(n - 1):
        offer(index)
    while heap:
        _, left, right, merged_text = heapq.heappop(heap)
        if not (alive[left] and alive[right]) or next_of[left] != right:
            continue
        if texts[left] + texts[right] != merged_text:
            continue
        if restricted(left):
            continue
        alive[right] = False
        texts[left] = merged_text
        nxt = next_of[right]
        next_of[left] = nxt
        if nxt < n:
            prev_of[nxt] = left
        if not _neutral(texts[left]):
            at = bisect_left(non_neutral, left)
            if at >= len(non_neutral) or non_neutral[at] != left:
                insort(non_neutral, left)
        cursor = bisect_left(non_neutral, right)
        if cursor < len(non_neutral) and non_neutral[cursor] == right:
            non_neutral.pop(cursor)
        offer(left)
        if prev_of[left] >= 0:
            offer(prev_of[left])
    return [texts[index] for index in range(n) if alive[index]]


def encode_indic_span(span, scored, *, with_dummy):
    body = unicodedata.normalize("NFC", span).replace(" ", SPACE_MARK)
    if not with_dummy:
        return indic_bpe(body, scored, restrict_space_merges=True)
    pieces = indic_bpe(SPACE_MARK + body, scored, restrict_space_merges=False)
    if len(pieces) > 1 and pieces[0] == SPACE_MARK:
        pieces = indic_bpe(body, scored, restrict_space_merges=True)
    return pieces


def encode_tagged(text, english_tokens, scored, indic_region):
    """Tag each piece 'english' or 'indic'. English pieces come from Spark."""
    tagged = []
    single_space_pending = False
    for path, segment in route(text):
        if path == "indic":
            if segment.startswith(" "):
                for token in english_tokens(" "):
                    tagged.append(("english", token))
                segment = segment[1:]
                if not segment:
                    single_space_pending = False
                    continue
            with_dummy = True
            if single_space_pending:
                cursor = len(tagged) - 2
                while cursor >= 0 and decode_class(tagged[cursor][1], indic_region) == "neutral":
                    cursor -= 1
                if cursor >= 0 and decode_class(tagged[cursor][1], indic_region) == "indic":
                    with_dummy = False
            for piece in encode_indic_span(segment, scored, with_dummy=with_dummy):
                tagged.append(("indic", piece))
            single_space_pending = False
        else:
            for token in english_tokens(segment):
                tagged.append(("english", token))
            single_space_pending = segment == " "
    return tagged
