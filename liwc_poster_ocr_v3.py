#!/usr/bin/env python3
"""LIWC-22 poster word extraction — headless / server-safe version.

Works directly from the PDF file (no screen, no display, no pyautogui).

The LIWC-22 poster is a single large raster image embedded in a PDF.
This script:
  1. Extracts that image directly from the PDF via PyMuPDF.
  2. Slices it into overlapping horizontal strips (tiles).
  3. Detects thick dark vertical borders in the first strip to find
     category block boundaries [x_left, x_right].
  4. OCRs each strip with EasyOCR and assigns every word to whichever
     block its x-centre falls inside.
  5. Exports: per-word CSV, per-category summary, rollup CSV.

Usage:
    python liwc_poster_ocr_v3.py \\
        --pdf poster.pdf \\
        --csv output/liwc_words.csv \\
        --summary output/liwc_summary.csv \\
        --rollup output/liwc_rollup.csv

    # Tune border detection if needed:
    python liwc_poster_ocr_v3.py \\
        --pdf poster.pdf \\
        --darkness-threshold 60 \\
        --min-line-height 0.4 \\
        --csv output/liwc_words.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    from PIL import Image, ImageEnhance, ImageFilter
    Image.MAX_IMAGE_PIXELS = None  # poster is large — disable bomb check
except ImportError:
    Image = ImageEnhance = ImageFilter = None  # type: ignore

try:
    import easyocr
except ImportError:
    easyocr = None  # type: ignore


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CategoryBlock:
    name: str
    x_left: float
    x_right: float

    @property
    def x_centre(self) -> float:
        return (self.x_left + self.x_right) / 2.0


@dataclass
class OCRItem:
    strip_index: int
    text: str
    x_centre: float   # in full-image pixel coordinates
    y_centre: float   # in full-image pixel coordinates
    confidence: float


@dataclass
class WordEntry:
    word: str
    category: str
    strip_index: int
    confidence: float


@dataclass
class CategorySummary:
    category: str
    word_count: int


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def ensure_dependencies() -> None:
    missing = []
    if fitz is None:
        missing.append("pymupdf  (pip install pymupdf)")
    if np is None:
        missing.append("numpy    (pip install numpy)")
    if Image is None:
        missing.append("pillow   (pip install pillow)")
    if easyocr is None:
        missing.append("easyocr  (pip install easyocr)")
    if missing:
        raise RuntimeError(
            "Missing required packages:\n  " + "\n  ".join(missing)
        )


# ---------------------------------------------------------------------------
# Step 1: Extract the poster image from PDF
# ---------------------------------------------------------------------------

def extract_poster_image(pdf_path: Path) -> "Image.Image":
    """Extract the single large raster image embedded in the LIWC-22 poster PDF."""
    assert fitz is not None and Image is not None

    doc = fitz.open(str(pdf_path))
    page = doc[0]
    images = page.get_images()

    if not images:
        raise RuntimeError(
            f"No embedded images found in {pdf_path}. "
            "The PDF may use vector graphics instead of a raster image."
        )

    # Pick the largest image (by pixel count)
    best_xref = None
    best_pixels = 0
    for img_info in images:
        xref = img_info[0]
        w, h = img_info[2], img_info[3]
        if w * h > best_pixels:
            best_pixels = w * h
            best_xref = xref

    print(f"Extracting embedded image (xref={best_xref}, "
          f"{int(math.sqrt(best_pixels))}² ~ {best_pixels:,} pixels)…")

    base_image = doc.extract_image(best_xref)
    img = Image.open(io.BytesIO(base_image["image"]))
    print(f"Full image size: {img.width} × {img.height} px")
    return img


# ---------------------------------------------------------------------------
# Step 2: Slice image into horizontal strips
# ---------------------------------------------------------------------------

def slice_into_strips(
    image: "Image.Image",
    strip_height: int = 2000,
    overlap_px: int = 200,
) -> List[Tuple[int, "Image.Image"]]:
    """Slice the full-height poster image into overlapping horizontal strips.

    Returns list of (y_offset, strip_image) where y_offset is the top of
    the strip in full-image coordinates.
    """
    W, H = image.size
    strips: List[Tuple[int, Image.Image]] = []
    y = 0
    while y < H:
        y_end = min(y + strip_height, H)
        strip = image.crop((0, y, W, y_end))
        strips.append((y, strip))
        if y_end >= H:
            break
        y += strip_height - overlap_px
    print(f"Sliced into {len(strips)} strips "
          f"(strip_height={strip_height}, overlap={overlap_px}px)")
    return strips


# ---------------------------------------------------------------------------
# Step 3: Detect vertical dark borders → block boundaries
# ---------------------------------------------------------------------------

def detect_column_boundaries(
    strip: "Image.Image",
    darkness_threshold: int = 80,
    min_line_height_fraction: float = 0.5,
    min_line_width_px: int = 2,
    merge_gap_px: int = 8,
) -> List[Tuple[float, float]]:
    """Find [x_left, x_right] ranges of category blocks by detecting vertical
    dark lines (thick black/dark borders between color blocks).

    Parameters
    ----------
    darkness_threshold:
        Pixels with brightness < this are "dark". Default 80 (0=black, 255=white).
        Decrease if borders are not being detected; increase if too many are found.
    min_line_height_fraction:
        A pixel column is a border if at least this fraction of its rows are dark.
    min_line_width_px:
        Minimum width (px) of a detected border to be counted.
    merge_gap_px:
        Gaps between dark columns <= this are merged into one border.
    """
    assert np is not None

    gray = np.array(strip.convert("L"))  # shape: (H, W)
    H, W = gray.shape
    min_dark_rows = int(H * min_line_height_fraction)

    # Count dark pixels per column
    dark_counts = (gray < darkness_threshold).sum(axis=0)  # shape: (W,)
    is_border = dark_counts >= min_dark_rows

    # Merge nearby border columns
    border_ranges: List[Tuple[int, int]] = []
    in_border = False
    start = 0
    for x in range(W):
        if is_border[x]:
            if not in_border:
                start = x
                in_border = True
        else:
            if in_border:
                if (border_ranges and
                        start - border_ranges[-1][1] <= merge_gap_px):
                    border_ranges[-1] = (border_ranges[-1][0], x - 1)
                else:
                    border_ranges.append((start, x - 1))
                in_border = False
    if in_border:
        border_ranges.append((start, W - 1))

    # Filter narrow noise
    border_ranges = [
        (l, r) for l, r in border_ranges if r - l + 1 >= min_line_width_px
    ]

    if not border_ranges:
        print("WARNING: No vertical borders detected. "
              "Try --darkness-threshold 60 or --min-line-height 0.3")
        return [(0.0, float(W))]

    # Build blocks from gaps between borders
    sentinels = [(-1, -1)] + border_ranges + [(W, W)]
    blocks: List[Tuple[float, float]] = []
    for i in range(len(sentinels) - 1):
        x_left = float(sentinels[i][1] + 1)
        x_right = float(sentinels[i + 1][0] - 1)
        if x_right - x_left > 5:
            blocks.append((x_left, x_right))

    return blocks


# ---------------------------------------------------------------------------
# Step 4: OCR each strip
# ---------------------------------------------------------------------------

def preprocess_subtile(subtile: "Image.Image") -> "Image.Image":
    """Preprocess a sub-tile for OCR.

    NOTE: We do NOT upscale 2× here because OpenCV inside EasyOCR has a
    hard limit of SHRT_MAX (32,767) on any image dimension.  The poster
    strips are already ~29,879 px wide; doubling would exceed the limit
    and crash with cv2.error.  Contrast enhancement alone is sufficient.
    """
    gray = subtile.convert("L")
    sharp = gray.filter(ImageFilter.SHARPEN)
    contrast = ImageEnhance.Contrast(sharp).enhance(2.0)
    # Threshold to pure black/white — helps OCR on dense small text
    return contrast.point(lambda px: 255 if px > 170 else 0)


def ocr_strips(
    strips: List[Tuple[int, "Image.Image"]],
    languages: Sequence[str],
    tiles_dir: Optional[Path] = None,
    subtile_width: int = 4000,
    subtile_overlap: int = 100,
) -> List[OCRItem]:
    """OCR each strip by further splitting it into narrow vertical sub-tiles.

    OpenCV (used internally by EasyOCR) crashes when any image dimension
    exceeds 32,767 px.  The poster is ~29,879 px wide, so each strip must
    be processed in sub-tiles narrower than that limit.

    Parameters
    ----------
    subtile_width:
        Width in pixels of each sub-tile.  Keep well below 32,767.
        Default 4,000 gives ~8 sub-tiles per strip.
    subtile_overlap:
        Horizontal overlap between sub-tiles to avoid missing words near
        sub-tile boundaries.
    """
    assert easyocr is not None and np is not None

    reader = easyocr.Reader(list(languages), gpu=False)
    items: List[OCRItem] = []

    for strip_idx, (y_offset, strip) in enumerate(strips, start=1):
        W = strip.width
        x_starts = list(range(0, W, subtile_width - subtile_overlap))
        print(f"  OCR strip {strip_idx}/{len(strips)} "
              f"(y_offset={y_offset}, {len(x_starts)} sub-tiles)…")

        if tiles_dir is not None:
            tiles_dir.mkdir(parents=True, exist_ok=True)
            strip.save(tiles_dir / f"strip_{strip_idx:04d}.png")

        for sub_idx, x_start in enumerate(x_starts):
            x_end = min(x_start + subtile_width, W)
            subtile = strip.crop((x_start, 0, x_end, strip.height))
            processed = preprocess_subtile(subtile)

            try:
                results = reader.readtext(
                    np.array(processed), detail=1, paragraph=False
                )
            except Exception as exc:
                print(f"    WARNING: sub-tile {sub_idx+1} failed: {exc}")
                continue

            for bbox, text, conf in results:
                text = text.strip()
                if not text:
                    continue
                # bbox coords are in sub-tile space (no upscale, so 1:1)
                xs = [pt[0] for pt in bbox]
                ys = [pt[1] for pt in bbox]
                # Convert back to full-image coordinates
                x_c = sum(xs) / len(xs) + x_start
                y_c = sum(ys) / len(ys) + y_offset

                items.append(OCRItem(
                    strip_index=strip_idx,
                    text=text,
                    x_centre=x_c,
                    y_centre=y_c,
                    confidence=float(conf),
                ))

    print(f"OCR complete: {len(items)} tokens across {len(strips)} strips.")
    return items


# ---------------------------------------------------------------------------
# Step 5: Match headers to blocks
# ---------------------------------------------------------------------------

def match_headers_to_blocks(
    items: List[OCRItem],
    blocks: List[Tuple[float, float]],
    header_y_max: float,
) -> List[CategoryBlock]:
    """Assign a category name to each block from header-row OCR items.

    header_y_max: full-image y coordinate below which items are headers.
    """
    header_items = [it for it in items if it.y_centre <= header_y_max]

    category_blocks: List[CategoryBlock] = []
    for i, (x_left, x_right) in enumerate(blocks):
        inside = [h for h in header_items if x_left <= h.x_centre <= x_right]
        if inside:
            inside.sort(key=lambda h: h.x_centre)
            name = " ".join(h.text.strip() for h in inside)
        else:
            name = f"BLOCK_{i + 1}"
        category_blocks.append(
            CategoryBlock(name=name, x_left=x_left, x_right=x_right)
        )

    return category_blocks


# ---------------------------------------------------------------------------
# Step 6: Assign words to blocks
# ---------------------------------------------------------------------------

def assign_words_to_blocks(
    items: List[OCRItem],
    blocks: List[CategoryBlock],
    header_y_max: float,
    min_confidence: float = 0.3,
) -> List[WordEntry]:
    entries: List[WordEntry] = []

    for item in items:
        if item.y_centre <= header_y_max:
            continue
        if item.confidence < min_confidence:
            continue
        text = item.text.strip()
        if len(text) < 2:
            continue

        category = "UNKNOWN"
        for block in blocks:
            if block.x_left <= item.x_centre <= block.x_right:
                category = block.name
                break

        entries.append(WordEntry(
            word=text.lower(),
            category=category,
            strip_index=item.strip_index,
            confidence=item.confidence,
        ))

    return entries


# ---------------------------------------------------------------------------
# Dedup, summary, rollup
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_csv(rows: list, path: Path, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(asdict(row) if hasattr(row, "__dataclass_fields__") else row)
    print(f"Wrote {len(rows)} rows → {path}")


def write_rollup_csv(rollup: Dict[str, set], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parent_category", "word"])
        for cat, words in sorted(rollup.items()):
            for word in sorted(words):
                w.writerow([cat, word])
    print(f"Wrote rollup → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_hierarchy(raw: Optional[str]) -> Dict[str, List[str]]:
    if not raw:
        return {"drives": ["affiliation", "achieve", "power"]}
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
        description="Extract LIWC-22 poster word lists — headless, no display needed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pdf", type=Path, required=True,
                   help="Path to the LIWC-22 poster PDF.")
    p.add_argument("--csv", type=Path, default=Path("output/liwc_words.csv"))
    p.add_argument("--summary", type=Path, default=Path("output/liwc_summary.csv"))
    p.add_argument("--rollup", type=Path, default=Path("output/liwc_rollup.csv"))
    p.add_argument("--hierarchy",
                   help="Rollup hierarchy e.g. 'drives:affiliation,achieve,power'.")
    p.add_argument("--strip-height", type=int, default=2000,
                   help="Height in pixels of each processing strip.")
    p.add_argument("--strip-overlap", type=int, default=200,
                   help="Overlap in pixels between strips.")
    p.add_argument("--languages", nargs="+", default=["en"])
    p.add_argument("--min-confidence", type=float, default=0.3)
    p.add_argument("--header-y-px", type=int, default=500,
                   help="Full-image y (px) below which text is treated as a header.")
    # Border detection
    p.add_argument("--darkness-threshold", type=int, default=80,
                   help="Pixel brightness below which a pixel counts as dark (0-255).")
    p.add_argument("--min-line-height", type=float, default=0.5,
                   help="Fraction of strip height a dark column must span.")
    p.add_argument("--min-line-width", type=int, default=2)
    p.add_argument("--merge-gap", type=int, default=8)
    # Debug
    p.add_argument("--save-strips", type=Path, default=None,
                   help="Optional directory to save strip images for inspection.")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_dependencies()

    # ── Step 1: Extract image from PDF ───────────────────────────────────
    print(f"\n[1] Extracting poster image from {args.pdf}…")
    poster = extract_poster_image(args.pdf)

    # ── Step 2: Detect column boundaries from top strip ───────────────────
    print("\n[2] Detecting column boundaries…")
    # Use a strip from the top of the image for border detection
    top_strip_h = min(args.strip_height, poster.height)
    top_strip = poster.crop((0, 0, poster.width, top_strip_h))

    raw_blocks = detect_column_boundaries(
        strip=top_strip,
        darkness_threshold=args.darkness_threshold,
        min_line_height_fraction=args.min_line_height,
        min_line_width_px=args.min_line_width,
        merge_gap_px=args.merge_gap,
    )
    print(f"Detected {len(raw_blocks)} column blocks:")
    for i, (xl, xr) in enumerate(raw_blocks):
        print(f"  Block {i+1:3d}: x=[{xl:6.0f} – {xr:6.0f}]  width={xr-xl:.0f}px")

    # ── Step 3: Slice poster into strips ──────────────────────────────────
    print("\n[3] Slicing poster into strips…")
    strips = slice_into_strips(
        image=poster,
        strip_height=args.strip_height,
        overlap_px=args.strip_overlap,
    )

    # ── Step 4: OCR all strips ────────────────────────────────────────────
    print("\n[4] Running EasyOCR on all strips…")
    all_items = ocr_strips(strips, args.languages, tiles_dir=args.save_strips)

    # ── Step 5: Match headers → blocks ────────────────────────────────────
    print("\n[5] Matching category headers to blocks…")
    category_blocks = match_headers_to_blocks(
        items=all_items,
        blocks=raw_blocks,
        header_y_max=float(args.header_y_px),
    )
    print("Category blocks:")
    for b in category_blocks:
        print(f"  [{b.x_left:6.0f} – {b.x_right:6.0f}]  →  {b.name!r}")

    # ── Step 6: Assign words ──────────────────────────────────────────────
    print("\n[6] Assigning words to blocks…")
    entries = assign_words_to_blocks(
        items=all_items,
        blocks=category_blocks,
        header_y_max=float(args.header_y_px),
        min_confidence=args.min_confidence,
    )
    deduped = deduplicate(entries)
    unknown = sum(1 for e in deduped if e.category == "UNKNOWN")
    print(f"Assigned: {len(deduped)} unique (word, category) pairs. "
          f"UNKNOWN: {unknown}")
    if unknown:
        print("  → To reduce UNKNOWN: try --darkness-threshold 60 "
              "or --min-line-height 0.3")

    # ── Step 7: Export ────────────────────────────────────────────────────
    print("\n[7] Writing outputs…")
    write_csv(deduped, args.csv,
              ["word", "category", "strip_index", "confidence"])
    write_csv(build_summary(deduped), args.summary,
              ["category", "word_count"])

    hierarchy = parse_hierarchy(args.hierarchy)
    rollup = rollup_categories(deduped, hierarchy)
    write_rollup_csv(rollup, args.rollup)

    print("\nRollup summary:")
    for parent, words in sorted(rollup.items()):
        print(f"  {parent}: {len(words)} unique words (self + all children)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
