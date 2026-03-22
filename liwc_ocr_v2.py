#!/usr/bin/env python3
"""LIWC-22 poster OCR — EasyOCR, block×strip tile approach.

Architecture (from visual inspection):
  - Poster is a single large raster image in the PDF.
  - 102 vertical color blocks separated by thick dark borders.
  - Each block is sliced into horizontal strips independently.
  - Each (block × strip) tile is OCR'd separately.
  - Words from a tile belong to that block's category — no x-coord guessing.
  - The first strip of each block contains the category name at the top,
    followed by words. Skip the header zone (--skip-top-px) to avoid
    the gradient title bar.

Category name matching:
  - OCR tokens from the header zone of strip 1 of each block are matched
    against VALID_LIWC_CATEGORIES.
  - Alternatively, supply --category-map (CSV: block_index,category_name)
    to skip header OCR entirely.

Usage:
    # Test on first 4 blocks, 3 strips
    python liwc_ocr_v1.py \\
        --pdf poster8985480095715875207.pdf \\
        --csv output/liwc_words.csv \\
        --test-blocks 4 --test-strips 3

    # Full run with category map
    nohup python liwc_ocr_v1.py \\
        --pdf poster8985480095715875207.pdf \\
        --csv output/liwc_words.csv \\
        --summary output/liwc_summary.csv \\
        --rollup output/liwc_rollup.csv \\
        --category-map output/blocks.csv &
"""

from __future__ import annotations

import base64

import argparse
import csv
import io
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

try:
    import fitz
except ImportError:
    fitz = None  # type: ignore

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    from PIL import Image, ImageEnhance, ImageFilter
    Image.MAX_IMAGE_PIXELS = None
except ImportError:
    Image = ImageEnhance = ImageFilter = None  # type: ignore

try:
    import easyocr
except ImportError:
    easyocr = None  # type: ignore

try:
    import requests as _requests
except ImportError:
    _requests = None  # type: ignore

VALID_LIWC_CATEGORIES: Set[str] = {
    c.lower() for c in {
        "Analytic","Clout","Authentic","Tone",
        "Linguistic","function","pronoun","ppron",
        "i","we","you","shehe","they","ipron",
        "det","article","number","prep","auxverb",
        "adverb","conj","negate","verb","adj","quantity",
        "Drives","affiliation","achieve","power",
        "Cognition","allnone","cogproc","insight","cause",
        "discrep","tentat","certitude","differ","memory",
        "Affect","tone_pos","tone_neg","emotion",
        "emo_pos","emo_neg","emo_anx","emo_anger","emo_sad",
        "swear",
        "Social","socbehav","prosocial","polite","conflict",
        "moral","comm","socrefs","family","friend",
        "female","male",
        "Culture","politic","ethnicity","tech",
        "Lifestyle","leisure","home","work","money","relig",
        "Physical","health","illness","wellness","mental",
        "substances","sexual","food","death",
        "need","want","acquire","lack","fulfill",
        "fatigue","reward","risk","curiosity","allure",
        "Perception","attention","motion","space",
        "visual","auditory","feeling",
        "time","focuspast","focuspresent","focusfuture",
        "Conversation","netspeak","assent","nonflu","filler",
        "AllPunc","Period","Comma","QMark","Exclam",
        "Apostro","OtherP","Emoji",
    }
}

# Noise filter: reject tokens that are clearly not LIWC entries.
# Kept deliberately permissive to preserve LIWC special formats:
#   achiev*          wildcard stems
#   brown boy*       multi-word phrases with wildcard
#   take* aback      mid-word wildcard
#   1st / 2nd        ordinal numbers
#   my child*        possessive + wildcard
#   full of sh*t     phrase with wildcard
# Rejected: purely numeric strings, empty tokens, non-ASCII characters,
# tokens longer than 60 chars (OCR garbage).
NOISE_RE = re.compile(r"^\d+$")          # purely numeric
MAX_TOKEN_LEN = 60


def is_valid_token(text: str) -> bool:
    """Return True if text looks like a legitimate LIWC dictionary entry.

    Accepts:
      - Plain words:          function, pronoun
      - Wildcard stems:       achiev*, hungr*
      - Multi-word phrases:   brown boy*, take* aback, full of sh*t
      - Ordinals/numbers:     1st, 2nd, 20th
      - Contractions:         don't, can't, they've
      - Hyphenated:           well-known, up-to-date

    Rejects:
      - Non-ASCII characters  (OCR noise from other languages)
      - Purely numeric        (123, 456)
      - Very long strings     (OCR garbage runs)
      - Empty strings
    """
    if not text:
        return False
    if len(text) > MAX_TOKEN_LEN:
        return False
    # Must be ASCII-only
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return False
    # Reject purely numeric
    if NOISE_RE.match(text):
        return False
    # Must contain at least one letter
    if not any(c.isalpha() for c in text):
        return False
    return True


# Keep is_valid_word as an alias for backward compatibility
is_valid_word = is_valid_token


def normalise(text: str) -> str:
    """Lowercase and collapse internal whitespace.

    NOTE: does NOT strip trailing * or other LIWC special characters —
    these are meaningful in the .dic format and must be preserved.
    """
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass
class WordEntry:
    word: str
    category: str
    block_index: int
    strip_index: int
    confidence: float

@dataclass
class CategorySummary:
    category: str
    word_count: int


def ensure_dependencies(use_paddle_api: bool = False) -> None:
    missing = []
    if fitz is None:  missing.append("pymupdf  (pip install pymupdf)")
    if np is None:    missing.append("numpy    (pip install numpy)")
    if Image is None: missing.append("pillow   (pip install pillow)")
    if use_paddle_api:
        if _requests is None:
            missing.append("requests (pip install requests)")
    else:
        if easyocr is None:
            missing.append("easyocr  (pip install easyocr)")
    if missing:
        raise RuntimeError("Missing packages:\n  " + "\n  ".join(missing))


def extract_poster_image(pdf_path: Path) -> "Image.Image":
    assert fitz is not None and Image is not None
    doc = fitz.open(str(pdf_path))
    images = doc[0].get_images()
    if not images:
        raise RuntimeError(f"No embedded images in {pdf_path}.")
    best_xref = max(images, key=lambda i: i[2] * i[3])[0]
    raw = doc.extract_image(best_xref)
    img = Image.open(io.BytesIO(raw["image"]))
    print(f"Full image: {img.width} x {img.height} px")
    return img


def detect_column_boundaries(
    strip: "Image.Image",
    darkness_threshold: int = 80,
    min_line_height_fraction: float = 0.5,
    min_line_width_px: int = 2,
    merge_gap_px: int = 8,
) -> List[Tuple[float, float]]:
    assert np is not None
    gray = np.array(strip.convert("L"))
    H, W = gray.shape
    min_dark = int(H * min_line_height_fraction)
    is_border = (gray < darkness_threshold).sum(axis=0) >= min_dark

    borders: List[Tuple[int, int]] = []
    in_b, start = False, 0
    for x in range(W):
        if is_border[x]:
            if not in_b:
                start, in_b = x, True
        else:
            if in_b:
                if borders and start - borders[-1][1] <= merge_gap_px:
                    borders[-1] = (borders[-1][0], x - 1)
                else:
                    borders.append((start, x - 1))
                in_b = False
    if in_b:
        borders.append((start, W - 1))

    borders = [(l, r) for l, r in borders if r - l + 1 >= min_line_width_px]
    if not borders:
        print("WARNING: No vertical borders detected.")
        return [(0.0, float(W))]

    sentinels = [(-1, -1)] + borders + [(W, W)]
    blocks = []
    for i in range(len(sentinels) - 1):
        xl = float(sentinels[i][1] + 1)
        xr = float(sentinels[i + 1][0] - 1)
        if xr - xl > 5:
            blocks.append((xl, xr))
    return blocks


def load_category_map(map_path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with map_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                try:
                    idx = int(parts[0].strip())
                    name = normalise(parts[1].strip())
                    if name and name not in ("block_index","name",
                                             "category_name","category"):
                        mapping[idx] = name
                except ValueError:
                    continue
    print(f"Loaded category map: {len(mapping)} entries from {map_path}")
    return mapping


def preprocess_tile(img: "Image.Image", upscale: int = 3) -> "Image.Image":
    """Preprocess a narrow block tile for OCR.

    Uses adaptive contrast stretching (percentile normalisation) instead of
    a fixed brightness threshold.  This handles both:
      - White background / dark text  (function, pronoun blocks)
      - Orange background / light-grey text  (Linguistic block)
    The p10-p90 stretch pulls the darkest pixels to 0 and the brightest to
    255 based on the tile's own content, so the binarisation threshold of
    128 works reliably regardless of background colour.
    """
    gray = np.array(img.convert("L"), dtype=np.float32)

    # Adaptive contrast stretch: map [p10, p90] -> [0, 255]
    p10 = float(np.percentile(gray, 10))
    p90 = float(np.percentile(gray, 90))
    if p90 > p10:
        gray = (gray - p10) / (p90 - p10) * 255.0
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    stretched = Image.fromarray(gray)

    sharp = stretched.filter(ImageFilter.SHARPEN)
    contrast = ImageEnhance.Contrast(sharp).enhance(2.5)
    big = contrast.resize(
        (contrast.width * upscale, contrast.height * upscale),
        Image.Resampling.LANCZOS,
    )
    # Binarise at midpoint (works well after adaptive stretch)
    return big.point(lambda px: 255 if px > 128 else 0)


def ocr_tile_easyocr(
    tile: "Image.Image",
    reader: "easyocr.Reader",
    upscale: int = 3,
    min_confidence: float = 0.3,
) -> List[Tuple[str, float, float, float]]:
    """OCR one tile using local EasyOCR."""
    assert np is not None
    processed = preprocess_tile(tile, upscale=upscale)
    try:
        results = reader.readtext(
            np.array(processed), detail=1, paragraph=False
        )
    except Exception as e:
        print(f"    WARNING: EasyOCR failed: {e}")
        return []

    tokens = []
    for bbox, text, conf in results:
        if conf < min_confidence:
            continue
        text = normalise(text)
        if not text:
            continue
        xs = [pt[0] / upscale for pt in bbox]
        ys = [pt[1] / upscale for pt in bbox]
        tokens.append((text, float(conf),
                        sum(xs)/len(xs), sum(ys)/len(ys)))
    return tokens


def ocr_tile_paddle_api(
    tile: "Image.Image",
    api_url: str,
    api_token: str,
    min_confidence: float = 0.3,
) -> List[Tuple[str, float, float, float]]:
    """OCR one tile using PaddleOCR online API.

    API returns ocrResults list. Each item has:
        prunedResult.boxes  : [[x1,y1,x2,y2,x3,y3,x4,y4], ...]
        prunedResult.rec_texts : [text, ...]
        prunedResult.rec_scores: [score, ...]
    Coordinates are in the original (non-upscaled) tile space.
    """
    assert _requests is not None

    # Save tile to bytes (PNG)
    import io as _io
    buf = _io.BytesIO()
    tile.save(buf, format="PNG")
    file_data = base64.b64encode(buf.getvalue()).decode("ascii")

    headers = {
        "Authorization": f"token {api_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "file": file_data,
        "fileType": 1,          # image
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useTextlineOrientation": False,
    }

    try:
        # Retry up to 3 times with increasing timeout
        resp = None
        for _attempt in range(3):
            try:
                resp = _requests.post(
                    api_url, json=payload, headers=headers,
                    timeout=120 + _attempt * 60   # 120s, 180s, 240s
                )
                break
            except Exception as _retry_exc:
                print(f"    WARNING: API attempt {_attempt+1}/3 failed: {_retry_exc}")
                if _attempt == 2:
                    return [], []
        if resp is None:
            return [], []
        resp.raise_for_status()
    except Exception as e:
        print(f"    WARNING: Paddle API call failed: {e}")
        return [], []

    tokens = []
    raw_texts = []
    try:
        result = resp.json()["result"]
        for ocr_res in result.get("ocrResults", []):
            pruned  = ocr_res.get("prunedResult", {})
            texts   = pruned.get("rec_texts",  [])
            scores  = pruned.get("rec_scores", [])
            boxes   = pruned.get("boxes", [])
            raw_texts.extend(texts)   # keep original strings for debug

            # One-time debug: print all available keys and first box entry
            if not raw_texts and texts:
                print(f"    [API DEBUG] pruned keys: {list(pruned.keys())}")
                if boxes:
                    print(f"    [API DEBUG] boxes[0] sample: {boxes[0]}")
                else:
                    print(f"    [API DEBUG] boxes is empty — no position data")

            for idx, raw_text in enumerate(texts):
                text = normalise(str(raw_text))
                if not text:
                    continue
                # No confidence filter here — done after wildcard merging
                conf = float(scores[idx]) if idx < len(scores) else 0.0

                if idx < len(boxes) and boxes[idx]:
                    pts = boxes[idx]
                    # Handle both formats:
                    # Nested: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    # Flat:   [x1,y1,x2,y2,x3,y3,x4,y4]
                    if isinstance(pts[0], (list, tuple)):
                        xs = [float(pt[0]) for pt in pts]
                        ys = [float(pt[1]) for pt in pts]
                    else:
                        xs = [float(pts[i])   for i in range(0, len(pts), 2)]
                        ys = [float(pts[i+1]) for i in range(0, len(pts)-1, 2)]
                    x_c = sum(xs) / len(xs) if xs else 0.0
                    y_c = sum(ys) / len(ys) if ys else 0.0
                else:
                    # No box info from API — use token order as y surrogate.
                    # Tokens are returned top-to-bottom, so idx * estimated_row_height
                    # gives a rough y position. We use a large value (9999) so these
                    # tokens are never mistakenly classified as header-zone tokens.
                    x_c = 0.0
                    y_c = 9999.0

                tokens.append((text, conf, x_c, y_c))
    except Exception as e:
        print(f"    WARNING: Paddle API response parse failed: {e}")

    return tokens, raw_texts


def _merge_wildcards(
    tokens: List[Tuple[str, float, float, float]]
) -> List[Tuple[str, float, float, float]]:
    """Re-merge wildcard stems split by OCR.

    PaddleOCR sometimes splits "achiev*" into two tokens: "achiev" and "*".
    This function detects a lone "*" token immediately after a word token
    and merges them back: "achiev" + "*" -> "achiev*".

    Also handles multi-word phrases with wildcards:
    "brown", "boy", "*" -> "brown boy*"
    by keeping adjacent tokens on the same approximate y-line together
    when the last one is "*".
    """
    if not tokens:
        return tokens

    merged: List[Tuple[str, float, float, float]] = []
    i = 0
    while i < len(tokens):
        text, conf, x_c, y_c = tokens[i]
        # Check if next token is a lone "*" on the same y-line (within 20px)
        if (i + 1 < len(tokens) and
                tokens[i + 1][0].strip() == "*" and
                abs(tokens[i + 1][3] - y_c) < 20):
            _, conf2, x_c2, y_c2 = tokens[i + 1]
            merged.append((
                text + "*",
                min(conf, conf2),   # use lower confidence (more conservative)
                (x_c + x_c2) / 2,
                (y_c + y_c2) / 2,
            ))
            i += 2
        else:
            merged.append((text, conf, x_c, y_c))
            i += 1
    return merged


def ocr_tile(
    tile: "Image.Image",
    reader,                     # easyocr.Reader OR None (when using API)
    upscale: int = 3,
    min_confidence: float = 0.3,
    api_url: str = "",
    api_token: str = "",
    debug_label: str = "",      # e.g. "b001_s01" for debug logging
) -> List[Tuple[str, float, float, float]]:
    """Dispatch to EasyOCR or PaddleOCR API, then apply wildcard merging."""
    if reader is not None:
        tokens = ocr_tile_easyocr(tile, reader, upscale, min_confidence)
        raw_texts = []
    else:
        tokens, raw_texts = ocr_tile_paddle_api(
            tile, api_url, api_token, min_confidence=0.0  # no conf filter yet
        )

    # Re-merge split wildcard tokens
    tokens = _merge_wildcards(tokens)

    # Debug logging for strip 1 (or any labelled tile)
    if debug_label:
        print(f"    [{debug_label}] raw API tokens: {len(raw_texts)} -> "
              f"after merge: {len(tokens)}")
        print(f"    [{debug_label}] raw sample: {raw_texts[:15]}")
        # Print first few tokens with y coords to debug header zone issue
        for _t, _c, _x, _y in tokens[:8]:
            print(f"      y={_y:.1f}  x={_x:.1f}  '{_t}'")

    # Apply confidence filter here (after merge)
    tokens = [(t, c, x, y) for t, c, x, y in tokens if c >= min_confidence]

    return tokens



def deduplicate(entries: List[WordEntry]) -> List[WordEntry]:
    best: Dict[Tuple[str, str], WordEntry] = {}
    for e in entries:
        key = (e.word, e.category)
        if key not in best or e.confidence > best[key].confidence:
            best[key] = e
    result = list(best.values())
    result.sort(key=lambda e: (e.category, e.word))
    return result


def build_summary(entries: List[WordEntry]) -> List[CategorySummary]:
    counts: Dict[str, int] = defaultdict(int)
    seen: Dict[str, set] = defaultdict(set)
    for e in entries:
        if e.word not in seen[e.category]:
            seen[e.category].add(e.word)
            counts[e.category] += 1
    return [CategorySummary(c, n) for c, n in sorted(counts.items())]


def rollup_categories(
    entries: List[WordEntry],
    hierarchy: Dict[str, List[str]],
) -> Dict[str, set]:
    cat_words: Dict[str, set] = defaultdict(set)
    for e in entries:
        cat_words[e.category].add(e.word)

    def _union(cat: str) -> set:
        result = set(cat_words.get(cat, set()))
        for child in hierarchy.get(cat, []):
            result |= _union(child)
        return result

    return {parent: _union(parent) for parent in hierarchy}


def write_words_csv(entries: List[WordEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "word","category","block_index","strip_index","confidence"])
        w.writeheader()
        for e in entries:
            w.writerow(asdict(e))
    print(f"Wrote {len(entries)} entries -> {path}")


def write_summary_csv(summaries: List[CategorySummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["category","word_count"])
        w.writeheader()
        for s in summaries:
            w.writerow(asdict(s))
    print(f"Wrote summary -> {path}")


def write_rollup_csv(rollup: Dict[str, set], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parent_category","word"])
        for cat, words in sorted(rollup.items()):
            for word in sorted(words):
                w.writerow([cat, word])
    print(f"Wrote rollup -> {path}")


def write_blocks_csv(
    blocks: List[Tuple[int, str, float, float]], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["block_index","category_name","x_left","x_right","width_px"])
        for idx, name, xl, xr in blocks:
            w.writerow([idx, name, f"{xl:.0f}", f"{xr:.0f}", f"{xr-xl:.0f}"])
    print(f"Wrote block listing -> {path}")


def parse_hierarchy(raw: Optional[str]) -> Dict[str, List[str]]:
    if not raw:
        return {"drives": ["affiliation","achieve","power"]}
    result: Dict[str, List[str]] = {}
    for part in raw.split(";"):
        if ":" not in part:
            continue
        parent, children_raw = part.split(":", 1)
        result[parent.strip()] = [
            c.strip() for c in children_raw.split(",") if c.strip()
        ]
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LIWC-22 poster OCR -- block x strip tile approach.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--csv", type=Path, default=Path("output/liwc_words.csv"))
    p.add_argument("--summary", type=Path, default=Path("output/liwc_summary.csv"))
    p.add_argument("--rollup", type=Path, default=Path("output/liwc_rollup.csv"))
    p.add_argument("--blocks-csv", type=Path, default=Path("output/blocks.csv"))
    p.add_argument("--category-map", type=Path, default=None,
                   help="CSV: block_index,category_name. Skips header OCR.")
    p.add_argument("--hierarchy",
                   help="e.g. 'drives:affiliation,achieve,power'")
    p.add_argument("--skip-top-px", type=int, default=378,
                   help="Skip px from top of poster (title bar height).")
    p.add_argument("--strip-height", type=int, default=2000)
    p.add_argument("--strip-overlap", type=int, default=200)
    p.add_argument("--darkness-threshold", type=int, default=80)
    p.add_argument("--min-line-height", type=float, default=0.5)
    p.add_argument("--min-line-width", type=int, default=2)
    p.add_argument("--merge-gap", type=int, default=8)
    p.add_argument("--upscale", type=int, default=3)
    p.add_argument("--header-y-fraction", type=float, default=0.08,
                   help="Top fraction of strip 1 used for category name detection.")
    p.add_argument("--languages", nargs="+", default=["en"])
    p.add_argument("--gpu", action="store_true", help="Use GPU for EasyOCR.")
    p.add_argument("--paddle-api-url", default="",
                   help="PaddleOCR API URL. If set, uses API instead of EasyOCR.")
    p.add_argument("--paddle-api-token", default="",
                   help="PaddleOCR API token.")
    p.add_argument("--min-confidence", type=float, default=0.3)
    p.add_argument("--test-blocks", type=int, default=0,
                   help="Test only first N blocks (0=all).")
    p.add_argument("--test-strips", type=int, default=0,
                   help="Test only first N strips per block (0=all).")
    p.add_argument("--save-tiles", type=Path, default=None,
                   help="Save tile images for inspection.")
    p.add_argument("--checkpoint-dir", type=Path,
                   default=Path("output/checkpoint"),
                   help="Directory for real-time per-tile API response logs "
                        "and incremental word CSV. Enables resume on failure.")
    p.add_argument("--resume", action="store_true",
                   help="Skip tiles already in checkpoint, resume from where "
                        "it left off.")
    return p


# ---------------------------------------------------------------------------
# Checkpoint helpers — real-time write so nothing is lost on failure
# ---------------------------------------------------------------------------

def checkpoint_path(checkpoint_dir: Path, block_idx: int, strip_idx: int) -> Path:
    """Return the path for a per-tile checkpoint file."""
    return checkpoint_dir / f"b{block_idx:03d}_s{strip_idx:02d}.done"


def write_checkpoint(
    checkpoint_dir: Path,
    block_idx: int,
    strip_idx: int,
    cat_name: str,
    entries: List[WordEntry],
    incremental_csv: Path,
) -> None:
    """Write a checkpoint file and append words to the incremental CSV.

    The checkpoint file is a small JSON with the tile's metadata.
    The incremental CSV is opened in append mode so every tile's words
    are flushed to disk immediately — if the process dies, we lose at
    most one tile's work.
    """
    import json
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Write per-tile JSON (raw API results summary)
    cp = checkpoint_path(checkpoint_dir, block_idx, strip_idx)
    cp.write_text(json.dumps({
        "block_index": block_idx,
        "strip_index": strip_idx,
        "category": cat_name,
        "word_count": len(entries),
    }))

    # Append words to incremental CSV (header written separately on first call)
    write_header = not incremental_csv.exists()
    incremental_csv.parent.mkdir(parents=True, exist_ok=True)
    with incremental_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["word","category","block_index","strip_index","confidence"]
        )
        if write_header:
            w.writeheader()
        for e in entries:
            w.writerow(asdict(e))


def is_tile_done(checkpoint_dir: Path, block_idx: int, strip_idx: int) -> bool:
    return checkpoint_path(checkpoint_dir, block_idx, strip_idx).exists()


def load_incremental_csv(path: Path) -> List[WordEntry]:
    """Load all word entries from the incremental checkpoint CSV."""
    entries = []
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(WordEntry(
                word=row["word"],
                category=row["category"],
                block_index=int(row["block_index"]),
                strip_index=int(row["strip_index"]),
                confidence=float(row["confidence"]),
            ))
    print(f"Resumed: loaded {len(entries)} entries from {path}")
    return entries


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    use_paddle_api = bool(args.paddle_api_url)
    ensure_dependencies(use_paddle_api=use_paddle_api)

    if use_paddle_api:
        print("\n[0] Using PaddleOCR API (no local model needed).")
        reader = None
        api_url   = args.paddle_api_url
        api_token = args.paddle_api_token
    else:
        print("\n[0] Initialising EasyOCR...")
        reader    = easyocr.Reader(args.languages, gpu=args.gpu)
        api_url   = ""
        api_token = ""

    print("\n[1] Extracting poster image...")
    poster = extract_poster_image(args.pdf)
    W, H = poster.size

    print("\n[2] Detecting column blocks...")
    boundary = poster.crop((0, 0, W, min(2000, H)))
    raw_blocks = detect_column_boundaries(
        strip=boundary,
        darkness_threshold=args.darkness_threshold,
        min_line_height_fraction=args.min_line_height,
        min_line_width_px=args.min_line_width,
        merge_gap_px=args.merge_gap,
    )
    print(f"Detected {len(raw_blocks)} column blocks.")

    if args.test_blocks > 0:
        raw_blocks = raw_blocks[:args.test_blocks]
        print(f"TEST MODE: first {len(raw_blocks)} blocks only.")

    category_map: Optional[Dict[int, str]] = None
    if args.category_map is not None:
        print(f"\n[3] Loading category map from {args.category_map}...")
        category_map = load_category_map(args.category_map)
    else:
        print("\n[3] No category map -- will detect from strip 1 OCR.")

    print(f"\n[4] Computing strips (skip_top={args.skip_top_px}px)...")
    strip_starts: List[int] = []
    y = args.skip_top_px
    while y < H:
        strip_starts.append(y)
        y += args.strip_height - args.strip_overlap

    if args.test_strips > 0:
        strip_starts = strip_starts[:args.test_strips]
        print(f"TEST MODE: first {len(strip_starts)} strips only.")

    print(f"Strips: {len(strip_starts)}")

    total = len(raw_blocks)
    print(f"\n[5] Processing {total} blocks x {len(strip_starts)} strips "
          f"= {total * len(strip_starts)} tiles...\n")

    checkpoint_dir  = args.checkpoint_dir
    incremental_csv = checkpoint_dir / "incremental_words.csv"

    # On resume, reload already-processed words
    all_entries: List[WordEntry] = []
    if args.resume:
        all_entries = load_incremental_csv(incremental_csv)

    block_listing: List[Tuple[int, str, float, float]] = []

    for b_idx, (xl, xr) in enumerate(raw_blocks, start=1):
        print(f"  Block {b_idx:3d}/{total}  x=[{xl:.0f}-{xr:.0f}]  "
              f"w={xr-xl:.0f}px")

        cat_name = (category_map.get(b_idx) if category_map else None) or f"block_{b_idx}"
        cat_detected = category_map is not None and b_idx in (category_map or {})
        block_entries: List[WordEntry] = []

        for s_idx, y_start in enumerate(strip_starts, start=1):

            # Skip if already checkpointed
            if args.resume and is_tile_done(checkpoint_dir, b_idx, s_idx):
                print(f"    strip {s_idx}  [SKIPPED — already done]")
                continue

            y_end = min(y_start + args.strip_height, H)
            xl_int = int(xl)
            xr_int = int(min(xr, W))
            tile = poster.crop((xl_int, y_start, xr_int, y_end))
            tile_h = tile.height

            if args.save_tiles is not None:
                args.save_tiles.mkdir(parents=True, exist_ok=True)
                tile.save(args.save_tiles / f"block{b_idx:03d}_strip{s_idx:02d}.png")

            debug_label = f"b{b_idx:03d}_s{s_idx:02d}" if s_idx == 1 else ""
            tokens = ocr_tile(tile, reader,
                              upscale=args.upscale,
                              min_confidence=args.min_confidence,
                              api_url=api_url,
                              api_token=api_token,
                              debug_label=debug_label)

            header_y_max = tile_h * args.header_y_fraction if s_idx == 1 else 0.0

            # Strip 1: first token is the category name — skip it,
            # collect everything else. All other strips: collect all tokens.
            tile_entries: List[WordEntry] = []
            for t_idx, (text, conf, x_c, y_c) in enumerate(tokens):
                if s_idx == 1 and t_idx == 0:
                    # First token of strip 1 = category name
                    norm_text = normalise(text)
                    if norm_text in VALID_LIWC_CATEGORIES:
                        cat_name = norm_text
                        cat_detected = True
                        print(f"    [b{b_idx:03d}] category: {cat_name!r}")
                    continue   # skip this token regardless
                if is_valid_word(text):
                    tile_entries.append(WordEntry(
                        word=text,
                        category=cat_name,
                        block_index=b_idx,
                        strip_index=s_idx,
                        confidence=conf,
                    ))

            # Only save checkpoint if we got actual words (or it's a
            # genuinely empty tile — detected by tokens being non-empty).
            # If tokens is empty it means the API failed; don't mark as done
            # so --resume will retry this tile.
            if tokens:
                write_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    block_idx=b_idx,
                    strip_idx=s_idx,
                    cat_name=cat_name,
                    entries=tile_entries,
                    incremental_csv=incremental_csv,
                )
                print(f"    strip {s_idx}  -> {len(tile_entries)} words  "
                      f"[checkpoint saved]")
            else:
                print(f"    strip {s_idx}  -> API failed, NOT checkpointed "
                      f"(will retry with --resume)")
            block_entries.extend(tile_entries)

        # Update category on all entries for this block (may have been
        # detected from strip 1 header after other strips were processed)
        for e in block_entries:
            e.category = cat_name

        print(f"  -> {cat_name}  ({len(block_entries)} tokens total)")
        all_entries.extend(block_entries)
        block_listing.append((b_idx, cat_name, xl, xr))

    deduped = deduplicate(all_entries)
    unmatched = sum(1 for e in deduped if e.category.startswith("block_"))
    print(f"\nDeduped: {len(deduped)} unique pairs. "
          f"Words in unmatched blocks: {unmatched}")

    print("\n[6] Writing outputs...")
    write_words_csv(deduped, args.csv)
    write_blocks_csv(block_listing, args.blocks_csv)

    if not (args.test_blocks or args.test_strips):
        write_summary_csv(build_summary(deduped), args.summary)
        hierarchy = parse_hierarchy(args.hierarchy)
        rollup = rollup_categories(deduped, hierarchy)
        write_rollup_csv(rollup, args.rollup)
        print("\nRollup summary:")
        for parent, words in sorted(rollup.items()):
            print(f"  {parent}: {len(words)} unique words (self + children)")
    else:
        print("\n(Summary/rollup skipped in test mode)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
