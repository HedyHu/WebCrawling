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

WORD_RE = re.compile(r"^[a-z][a-z'\-]{0,28}[a-z]$|^[a-z]$")

def is_valid_word(text: str) -> bool:
    cleaned = text.strip().lower()
    try:
        cleaned.encode("ascii")
    except UnicodeEncodeError:
        return False
    return bool(WORD_RE.match(cleaned))

def normalise(text: str) -> str:
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


def ensure_dependencies() -> None:
    missing = []
    if fitz is None: missing.append("pymupdf  (pip install pymupdf)")
    if np is None:   missing.append("numpy    (pip install numpy)")
    if Image is None: missing.append("pillow   (pip install pillow)")
    if easyocr is None: missing.append("easyocr  (pip install easyocr)")
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


def ocr_tile(
    tile: "Image.Image",
    reader: "easyocr.Reader",
    upscale: int = 3,
    min_confidence: float = 0.3,
) -> List[Tuple[str, float, float, float]]:
    assert np is not None
    processed = preprocess_tile(tile, upscale=upscale)
    try:
        results = reader.readtext(
            np.array(processed), detail=1, paragraph=False
        )
    except Exception as e:
        print(f"    WARNING: OCR failed: {e}")
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


def process_block(
    block_idx: int,
    x_left: float,
    x_right: float,
    poster: "Image.Image",
    strip_starts: List[int],
    strip_height: int,
    reader: "easyocr.Reader",
    category_map: Optional[Dict[int, str]],
    header_y_fraction: float,
    upscale: int,
    min_confidence: float,
    tiles_dir: Optional[Path],
) -> Tuple[str, List[WordEntry]]:
    W, H = poster.size
    xl = int(x_left)
    xr = int(min(x_right, W))

    cat_name = (category_map.get(block_idx) if category_map else None)
    cat_detected = cat_name is not None
    word_entries: List[WordEntry] = []

    for s_idx, y_start in enumerate(strip_starts, start=1):
        y_end = min(y_start + strip_height, H)
        tile = poster.crop((xl, y_start, xr, y_end))
        tile_h = tile.height

        if tiles_dir is not None:
            tiles_dir.mkdir(parents=True, exist_ok=True)
            tile.save(
                tiles_dir / f"block{block_idx:03d}_strip{s_idx:02d}.png"
            )

        tokens = ocr_tile(tile, reader, upscale=upscale,
                          min_confidence=min_confidence)

        # Header zone: top fraction of strip 1 only
        header_y_max = tile_h * header_y_fraction if s_idx == 1 else 0.0

        for text, conf, x_c, y_c in tokens:
            if s_idx == 1 and y_c < header_y_max:
                if not cat_detected and normalise(text) in VALID_LIWC_CATEGORIES:
                    cat_name = normalise(text)
                    cat_detected = True
                continue
            if is_valid_word(text):
                word_entries.append(WordEntry(
                    word=text,
                    category=cat_name or f"block_{block_idx}",
                    block_index=block_idx,
                    strip_index=s_idx,
                    confidence=conf,
                ))

    if cat_name is None:
        cat_name = f"block_{block_idx}"

    for e in word_entries:
        e.category = cat_name

    return cat_name, word_entries


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
    p.add_argument("--min-confidence", type=float, default=0.3)
    p.add_argument("--test-blocks", type=int, default=0,
                   help="Test only first N blocks (0=all).")
    p.add_argument("--test-strips", type=int, default=0,
                   help="Test only first N strips per block (0=all).")
    p.add_argument("--save-tiles", type=Path, default=None,
                   help="Save tile images for inspection.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_dependencies()

    print("\n[0] Initialising EasyOCR...")
    reader = easyocr.Reader(args.languages, gpu=args.gpu)

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

    all_entries: List[WordEntry] = []
    block_listing: List[Tuple[int, str, float, float]] = []

    for b_idx, (xl, xr) in enumerate(raw_blocks, start=1):
        print(f"  Block {b_idx:3d}/{total}  x=[{xl:.0f}-{xr:.0f}]  "
              f"w={xr-xl:.0f}px", end="  ")

        cat_name, entries = process_block(
            block_idx=b_idx,
            x_left=xl, x_right=xr,
            poster=poster,
            strip_starts=strip_starts,
            strip_height=args.strip_height,
            reader=reader,
            category_map=category_map,
            header_y_fraction=args.header_y_fraction,
            upscale=args.upscale,
            min_confidence=args.min_confidence,
            tiles_dir=args.save_tiles,
        )
        print(f"-> {cat_name}  ({len(entries)} tokens)")
        all_entries.extend(entries)
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
