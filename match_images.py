#!/usr/bin/env python3
"""Command-line pair comparison and reproducible photo-manifest experiments."""

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import platform
import re
import statistics
import sys

import cv2
import numpy as np

from charts import write_charts
from matcher import MatchConfig, load_image, match_pair, save_visualizations


NUMERIC_FIELDS = ["keypoints1", "keypoints2", "knn_pairs", "good_matches", "inliers",
                  "inlier_ratio", "median_reprojection_error_px", "detect_describe_ms",
                  "match_filter_ms", "geometry_ms", "total_ms"]
FIELDS = ["scene", "pair_id", "variation", "level", "notes", "image1", "image2", "method",
          "status", "error", "width1", "height1", "width2", "height2"] + NUMERIC_FIELDS


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_manifest(path):
    """Resolve image paths relative to the CSV, never relative to the shell cwd."""
    path = Path(path).resolve()
    required = {"scene", "pair_id", "image1", "image2", "variation"}
    pairs, seen = [], set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Manifest needs columns: " + ", ".join(sorted(required)))
        for line, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Manifest line {line}: extra columns; quote notes containing commas")
            if not any((v or "").strip() for v in row.values()):
                continue
            pair = {key: (row.get(key) or "").strip() for key in
                    ("scene", "pair_id", "variation", "level", "notes", "image1", "image2")}
            if any(not pair[key] for key in required):
                raise ValueError(f"Manifest line {line}: required field is empty")
            identifier = (pair["scene"], pair["pair_id"])
            if identifier in seen:
                raise ValueError(f"Manifest line {line}: duplicate scene/pair_id {identifier}")
            seen.add(identifier)
            for key in ("image1", "image2"):
                image_path = (path.parent / pair[key]).resolve()
                if not image_path.is_file():
                    raise ValueError(f"Manifest line {line}: image does not exist: {image_path}")
                pair[key] = str(image_path)
            pairs.append(pair)
    if not pairs:
        raise ValueError("Manifest contains no image pairs")
    return pairs


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scene"], row["variation"], row["method"])].append(row)
    summary = []
    for (scene, variation, method), group in sorted(groups.items()):
        item = dict(scene=scene, variation=variation, method=method,
                    evaluations=len(group), models_estimated=sum(r["status"] == "ok" for r in group),
                    input_errors=sum(r["status"] == "input_error" for r in group))
        for field in NUMERIC_FIELDS:
            values = [r[field] for r in group if isinstance(r.get(field), (float, int))]
            item[f"mean_{field}"] = statistics.mean(values) if values else None
        summary.append(item)
    return summary


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")[:70] or "pair"


def execute(pairs, methods, output, config, max_size, max_draw):
    output = Path(output).resolve()
    # Refuse stale artifacts from earlier experiments, without deleting anything.
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    write_json(output / "run_config.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
        "methods": methods, "max_size": max_size, "max_draw": max_draw,
        "python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__,
        "platform": platform.platform(), "opencv_threads": cv2.getNumThreads(),
        "command": sys.argv, "pairs": pairs,
        "homography_coordinates": "processed/resized image1 -> processed/resized image2",
        "timing": "One warm-up; median of repeats. Includes detection+description in BOTH images, "
                  "matching+filtering and geometry. Excludes grayscale conversion, resize, IO, "
                  "detector construction and visualization. Total median need not equal sum of stage medians.",
    })
    (output / "opencv_build.txt").write_text(cv2.getBuildInformation(), encoding="utf-8")
    rows = []
    for index, pair in enumerate(pairs, start=1):
        try:
            image1 = load_image(pair["image1"], max_size)
            image2 = load_image(pair["image2"], max_size)
        except ValueError as exc:
            for method in methods:
                rows.append({**pair, "method": method, "status": "input_error", "error": str(exc)})
            print(f"[{index}/{len(pairs)}] {pair['scene']}/{pair['pair_id']}: {exc}", file=sys.stderr)
            continue
        for method in methods:
            result = match_pair(image1, image2, method, config)
            row = {**pair, **result.metrics, "error": ""}
            rows.append(row)
            folder = output / "figures" / f"{index:03d}_{safe_name(pair['scene'])}_{safe_name(pair['pair_id'])}" / method
            save_visualizations(result, image1, image2, folder, max_draw)
            write_json(folder / "details.json", {
                "metrics": row, "homography": result.homography.tolist() if result.homography is not None else None,
                "coordinate_system": "processed image pixels; image1 -> image2",
                "matches": [{"query_index": m.queryIdx, "train_index": m.trainIdx,
                             "point1": list(result.keypoints1[m.queryIdx].pt),
                             "point2": list(result.keypoints2[m.trainIdx].pt),
                             "descriptor_distance": float(m.distance), "inlier": bool(keep)}
                            for m, keep in zip(result.matches, result.inlier_mask)],
            })
            metrics = result.metrics
            ratio = metrics["inlier_ratio"]
            ratio_text = f"{ratio:.1%}" if ratio is not None else "N/A"
            print(f"[{index}/{len(pairs)}] {pair['scene']}/{pair['pair_id']} {method}: "
                  f"{metrics['status']}, {metrics['good_matches']} matches, "
                  f"{metrics['inliers']} inliers ({ratio_text}), {metrics['total_ms']:.1f} ms")
    write_csv(output / "results.csv", rows, FIELDS)
    summary = summarize(rows)
    write_csv(output / "summary.csv", summary, list(summary[0]))
    chart_dir = output / "charts"
    chart_dir.mkdir()
    write_charts(rows, chart_dir)
    failures = sum(row["status"] == "input_error" for row in rows)
    print(f"Saved {len(rows)} evaluations to {output}")
    return 1 if failures else 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pair = subparsers.add_parser("pair", help="Compare SIFT and ORB on two photographs")
    pair.add_argument("image1", type=Path)
    pair.add_argument("image2", type=Path)
    batch = subparsers.add_parser("batch", help="Run all pairs listed in a CSV manifest")
    batch.add_argument("manifest", type=Path)
    for sub in (pair, batch):
        sub.add_argument("--output", type=Path, help="New or empty folder; default: timestamped results folder")
        sub.add_argument("--methods", nargs="+", choices=["SIFT", "ORB"], default=["SIFT", "ORB"])
        sub.add_argument("--max-size", type=int, default=1600, help="Maximum image side in pixels (default: 1600)")
        sub.add_argument("--nfeatures", type=int, default=2000, help="Requested feature budget for each method")
        sub.add_argument("--ratio", type=float, default=0.75, help="Nearest-neighbor ratio threshold")
        sub.add_argument("--ransac-threshold", type=float, default=3.0, help="Reprojection threshold in processed pixels")
        sub.add_argument("--repeats", type=int, default=3, help="Timed repetitions after one warm-up")
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--max-draw", type=int, default=80, help="Maximum displayed matches; all matches are evaluated")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = MatchConfig(args.nfeatures, args.ratio, args.ransac_threshold, args.repeats, args.seed)
        if args.max_size < 1 or args.max_draw < 1:
            raise ValueError("max-size and max-draw must be positive")
        methods = list(dict.fromkeys(args.methods))
        if args.command == "batch":
            pairs = read_manifest(args.manifest)
        else:
            # Decode before creating output to catch an invalid single-pair request early.
            for path in (args.image1, args.image2):
                load_image(path, args.max_size)
            pairs = [dict(scene="single_pair", pair_id="pair", variation="unspecified", level="", notes="",
                          image1=str(args.image1.resolve()), image2=str(args.image2.resolve()))]
        output = args.output or Path("results") / f"{args.command}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        return execute(pairs, methods, output, config, args.max_size, args.max_draw)
    except (ValueError, OSError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
