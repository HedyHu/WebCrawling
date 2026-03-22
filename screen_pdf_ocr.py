#!/usr/bin/env python3
"""Screen-driven OCR pipeline for tiled PDF capture and LIWC-style row extraction.

This script is designed for situations where a PDF must be OCR'd from a desktop
viewer instead of being parsed directly as a PDF. It can:

1. Calibrate the PDF viewer region on screen.
2. Capture overlapping screenshot tiles while scrolling.
3. OCR each tile with EasyOCR.
4. Extract LIWC-like rows in the form:
      [Category Name] [ID] [Word Count]
5. Reconstruct hierarchy using OCR indentation/x-position hints.
6. Export raw OCR rows, deduplicated rows, and set-union anomaly flags.

Typical usage:
    python screen_pdf_ocr.py \
        --viewer-title Acrobat \
        --tiles-dir output/tiles \
        --csv output/liwc_rows.csv

Before starting, open the target PDF on screen and make sure the first page of
interest is visible.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - import error is handled at runtime
    np = None
    NUMPY_IMPORT_ERROR = exc
else:
    NUMPY_IMPORT_ERROR = None
try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageGrab
except Exception as exc:  # pragma: no cover - import error is handled at runtime
    Image = ImageEnhance = ImageFilter = ImageGrab = None
    PIL_IMPORT_ERROR = exc
else:
    PIL_IMPORT_ERROR = None

try:
    import pyautogui
except Exception as exc:  # pragma: no cover - import error is handled at runtime
    pyautogui = None
    PYAUTOGUI_IMPORT_ERROR = exc
else:
    PYAUTOGUI_IMPORT_ERROR = None

try:
    import easyocr
except Exception as exc:  # pragma: no cover - import error is handled at runtime
    easyocr = None
    EASYOCR_IMPORT_ERROR = exc
else:
    EASYOCR_IMPORT_ERROR = None

try:
    import pygetwindow as gw
except Exception:
    gw = None


ROW_PATTERN = re.compile(
    r"^(?P<category>.+?)\s+(?P<category_id>\d{1,6})\s+(?P<word_count>\d{1,12})$"
)


@dataclass
class OCRRow:
    tile_index: int
    tile_path: str
    text: str
    category: str
    category_id: int
    word_count: int
    x_left: float
    y_top: float
    level: int = 0
    parent_id: Optional[int] = None
    anomaly_sum_of_children_gt_parent: bool = False
    children_word_count_sum: int = 0


@dataclass
class ScreenRegion:
    left: int
    top: int
    width: int
    height: int

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.left + self.width, self.top + self.height)


def ensure_dependencies() -> None:
    missing = []
    if pyautogui is None:
        missing.append(f"pyautogui ({PYAUTOGUI_IMPORT_ERROR})")
    if easyocr is None:
        missing.append(f"easyocr ({EASYOCR_IMPORT_ERROR})")
    if np is None:
        missing.append(f"numpy ({NUMPY_IMPORT_ERROR})")
    if Image is None:
        missing.append(f"pillow ({PIL_IMPORT_ERROR})")
    if missing:
        raise RuntimeError(
            "Missing required dependency/dependencies: "
            + "; ".join(missing)
            + ". Install them before running this script."
        )


def identify_pdf_viewer_window(
    title_keywords: Sequence[str],
    manual_region: Optional[ScreenRegion] = None,
) -> ScreenRegion:
    """Identify the PDF viewer region.

    Detection strategy:
    1. Use a manually supplied region if present.
    2. Search desktop windows via pygetwindow for a title match.
    3. Fall back to an interactive corner capture workflow.
    """
    if manual_region is not None:
        return manual_region

    if gw is not None and title_keywords:
        normalized = [keyword.lower() for keyword in title_keywords]
        for window in gw.getAllWindows():
            title = (window.title or "").strip()
            if not title:
                continue
            lowered = title.lower()
            if any(keyword in lowered for keyword in normalized):
                if window.width > 100 and window.height > 100:
                    return ScreenRegion(
                        left=int(window.left),
                        top=int(window.top),
                        width=int(window.width),
                        height=int(window.height),
                    )

    return interactive_region_capture()


def interactive_region_capture() -> ScreenRegion:
    if pyautogui is None:
        raise RuntimeError(
            "pyautogui is required for interactive calibration, but it is unavailable."
        )

    print(
        "Could not auto-detect the PDF viewer window. "
        "Move the mouse to the TOP-LEFT corner of the viewer content area and keep it still."
    )
    top_left = wait_for_stable_mouse_position(3)
    print(f"Captured top-left corner at {top_left}. Now move to the BOTTOM-RIGHT corner.")
    bottom_right = wait_for_stable_mouse_position(3)

    left = min(top_left[0], bottom_right[0])
    top = min(top_left[1], bottom_right[1])
    right = max(top_left[0], bottom_right[0])
    bottom = max(top_left[1], bottom_right[1])
    return ScreenRegion(left=left, top=top, width=right - left, height=bottom - top)


def wait_for_stable_mouse_position(stable_seconds: int = 3) -> Tuple[int, int]:
    assert pyautogui is not None
    previous = None
    stable_since = None
    while True:
        current = pyautogui.position()
        if previous == current:
            stable_since = stable_since or time.time()
            elapsed = time.time() - stable_since
            print(f"  stable for {elapsed:.1f}/{stable_seconds}s at {current}", end="\r")
            if elapsed >= stable_seconds:
                print()
                return int(current.x), int(current.y)
        else:
            stable_since = None
            print(f"  waiting for stable cursor at {current}...", end="\r")
        previous = current
        time.sleep(0.2)


def compute_scroll_plan(region_height: int, overlap_ratio: float) -> Tuple[int, int]:
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must be within [0, 1).")
    effective_shift = int(region_height * (1 - overlap_ratio))
    effective_shift = max(effective_shift, 1)

    # Mouse wheel units vary by OS/app. A conservative estimate of ~120 screen
    # pixels per notch works well as a starting point and can be tuned via CLI.
    wheel_clicks = max(1, round(effective_shift / 120))
    return effective_shift, wheel_clicks


def capture_tiles(
    region: ScreenRegion,
    output_dir: Path,
    max_tiles: int,
    overlap_ratio: float = 0.10,
    settle_delay: float = 1.0,
    end_guard_identical_limit: int = 2,
    use_pagedown_fallback: bool = True,
) -> List[Path]:
    """Capture overlapping screenshot tiles while scrolling the viewer."""
    assert pyautogui is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_shift, wheel_clicks = compute_scroll_plan(region.height, overlap_ratio)
    print(
        f"Calibrated capture region={region} | shift≈{effective_shift}px | "
        f"scroll clicks={wheel_clicks} | overlap={overlap_ratio:.0%}"
    )

    previous_arrays: List[np.ndarray] = []
    tile_paths: List[Path] = []

    for tile_index in range(1, max_tiles + 1):
        image = ImageGrab.grab(bbox=region.bbox)
        tile_path = output_dir / f"tile_{tile_index:04d}.png"
        image.save(tile_path)
        tile_paths.append(tile_path)
        print(f"Saved {tile_path}")

        current_array = np.array(image)
        identical_to_previous = any(
            arrays_are_nearly_identical(current_array, earlier)
            for earlier in previous_arrays[-end_guard_identical_limit:]
        )
        previous_arrays.append(current_array)

        if identical_to_previous and tile_index > 1:
            print(
                "Detected repeated capture content; assuming the viewer reached the end "
                "of the document."
            )
            break

        pyautogui.moveTo(region.left + region.width // 2, region.top + region.height // 2)
        pyautogui.click()
        pyautogui.scroll(-wheel_clicks)
        if use_pagedown_fallback:
            pyautogui.press("pagedown")
        time.sleep(settle_delay)

    return tile_paths


def arrays_are_nearly_identical(
    a: np.ndarray,
    b: np.ndarray,
    tolerance: float = 1.0,
) -> bool:
    if a.shape != b.shape:
        return False
    mean_abs_error = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()
    return mean_abs_error <= tolerance


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    grayscale = image.convert("L")
    sharpened = grayscale.filter(ImageFilter.SHARPEN)
    contrasted = ImageEnhance.Contrast(sharpened).enhance(1.8)
    enlarged = contrasted.resize(
        (contrasted.width * 2, contrasted.height * 2), Image.Resampling.LANCZOS
    )
    thresholded = enlarged.point(lambda px: 255 if px > 175 else 0)
    return thresholded


def run_easyocr(tile_paths: Sequence[Path], languages: Sequence[str]) -> List[OCRRow]:
    assert easyocr is not None
    reader = easyocr.Reader(list(languages), gpu=False)
    extracted: List[OCRRow] = []

    for tile_index, tile_path in enumerate(tile_paths, start=1):
        image = Image.open(tile_path)
        processed = preprocess_for_ocr(image)
        processed_np = np.array(processed)
        results = reader.readtext(processed_np, detail=1, paragraph=False)
        extracted.extend(parse_easyocr_results(tile_index, tile_path, results))

    return extracted


def parse_easyocr_results(
    tile_index: int,
    tile_path: Path,
    results: Iterable[Sequence[object]],
) -> List[OCRRow]:
    rows: List[OCRRow] = []
    for item in results:
        if len(item) < 2:
            continue
        bbox = item[0]
        text = normalize_text(str(item[1]))
        match = ROW_PATTERN.match(text)
        if not match:
            continue
        x_left = min(point[0] for point in bbox)
        y_top = min(point[1] for point in bbox)
        rows.append(
            OCRRow(
                tile_index=tile_index,
                tile_path=str(tile_path),
                text=text,
                category=match.group("category"),
                category_id=int(match.group("category_id")),
                word_count=int(match.group("word_count")),
                x_left=float(x_left),
                y_top=float(y_top),
            )
        )
    return rows


def normalize_text(text: str) -> str:
    text = text.replace("|", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def deduplicate_rows(rows: Sequence[OCRRow]) -> List[OCRRow]:
    best_by_key: dict[Tuple[int, str], OCRRow] = {}
    for row in rows:
        key = (row.category_id, row.category.lower())
        previous = best_by_key.get(key)
        if previous is None:
            best_by_key[key] = row
            continue

        prev_score = previous.word_count + previous.tile_index * 0.001
        curr_score = row.word_count + row.tile_index * 0.001
        if curr_score >= prev_score:
            best_by_key[key] = row

    deduped = list(best_by_key.values())
    deduped.sort(key=lambda row: (row.tile_index, row.y_top, row.x_left, row.category_id))
    return deduped


def assign_hierarchy(rows: List[OCRRow], indent_tolerance: int = 25) -> List[OCRRow]:
    if not rows:
        return rows

    distinct_x = sorted({int(round(row.x_left)) for row in rows})
    levels: List[int] = []
    for x_value in distinct_x:
        if not levels or abs(x_value - levels[-1]) > indent_tolerance:
            levels.append(x_value)

    for row in rows:
        row.level = min(
            range(len(levels)), key=lambda index: abs(levels[index] - row.x_left)
        )

    stack: List[OCRRow] = []
    for row in rows:
        while stack and stack[-1].level >= row.level:
            stack.pop()
        row.parent_id = stack[-1].category_id if stack else None
        stack.append(row)

    children_by_parent: dict[int, List[OCRRow]] = {}
    for row in rows:
        if row.parent_id is not None:
            children_by_parent.setdefault(row.parent_id, []).append(row)

    for row in rows:
        children = children_by_parent.get(row.category_id, [])
        row.children_word_count_sum = sum(child.word_count for child in children)
        row.anomaly_sum_of_children_gt_parent = (
            row.children_word_count_sum > row.word_count
        )

    return rows


def write_csv(rows: Sequence[OCRRow], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(asdict(rows[0]).keys()) if rows else [
            "tile_index",
            "tile_path",
            "text",
            "category",
            "category_id",
            "word_count",
            "x_left",
            "y_top",
            "level",
            "parent_id",
            "anomaly_sum_of_children_gt_parent",
            "children_word_count_sum",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_manual_region(raw: Optional[str]) -> Optional[ScreenRegion]:
    if not raw:
        return None
    parts = [int(part.strip()) for part in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("manual region must be left,top,width,height")
    return ScreenRegion(*parts)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viewer-title",
        action="append",
        default=[],
        help="Keyword(s) used to locate the PDF viewer window title.",
    )
    parser.add_argument(
        "--manual-region",
        help="Optional fixed capture region: left,top,width,height",
    )
    parser.add_argument(
        "--tiles-dir",
        type=Path,
        default=Path("output/tiles"),
        help="Directory where screenshot tiles are written.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("output/liwc_rows.csv"),
        help="CSV path for the final structured output.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=200,
        help="Maximum number of tiles to capture before stopping.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.10,
        help="Fractional overlap between captures. Default: 0.10 (10%%).",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="EasyOCR language codes.",
    )
    parser.add_argument(
        "--indent-tolerance",
        type=int,
        default=25,
        help="Pixels used to group x positions into hierarchy levels.",
    )
    parser.add_argument(
        "--settle-delay",
        type=float,
        default=1.0,
        help="Seconds to wait after scrolling for the PDF viewer to sharpen text.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    ensure_dependencies()
    manual_region = parse_manual_region(args.manual_region)
    region = identify_pdf_viewer_window(args.viewer_title, manual_region)
    print(f"Using capture region: {region}")
    time.sleep(1)

    tile_paths = capture_tiles(
        region=region,
        output_dir=args.tiles_dir,
        max_tiles=args.max_tiles,
        overlap_ratio=args.overlap,
        settle_delay=args.settle_delay,
    )

    raw_rows = run_easyocr(tile_paths, args.languages)
    print(f"OCR extracted {len(raw_rows)} candidate LIWC-style rows")
    deduped_rows = deduplicate_rows(raw_rows)
    structured_rows = assign_hierarchy(deduped_rows, indent_tolerance=args.indent_tolerance)
    write_csv(structured_rows, args.csv)
    print(f"Wrote structured CSV to {args.csv}")

    flagged = [row for row in structured_rows if row.anomaly_sum_of_children_gt_parent]
    if flagged:
        print("Set-union anomaly flags:")
        for row in flagged:
            print(
                f"  category_id={row.category_id} category={row.category!r} "
                f"children_sum={row.children_word_count_sum} > parent_total={row.word_count}"
            )
    else:
        print("No set-union anomalies detected.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
