"""SIFT/ORB matching and planar geometric verification (no GUI required)."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional
import math
import statistics

import cv2
import numpy as np


@dataclass(frozen=True)
class MatchConfig:
    nfeatures: int = 2000
    ratio: float = 0.75
    ransac_threshold: float = 3.0
    repeats: int = 3
    seed: int = 42

    def __post_init__(self):
        for name in ("nfeatures", "repeats"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.ratio) or not 0 < self.ratio < 1:
            raise ValueError("ratio must be between 0 and 1 (exclusive)")
        if not math.isfinite(self.ransac_threshold) or self.ransac_threshold <= 0:
            raise ValueError("ransac_threshold must be positive and finite")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer between 0 and 2147483647")


@dataclass
class MatchResult:
    metrics: Dict[str, Any]
    keypoints1: List[Any]
    keypoints2: List[Any]
    matches: List[Any]
    inlier_mask: np.ndarray
    homography: Optional[np.ndarray]


def load_image(path, max_size=1600):
    """Read a color image (OpenCV applies EXIF orientation); shrink longest side."""
    if not isinstance(max_size, int) or max_size < 1:
        raise ValueError("max_size must be a positive integer")
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Image does not exist: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}. Use JPEG or PNG.")
    height, width = image.shape[:2]
    scale = min(1.0, max_size / max(height, width))
    if scale < 1:
        image = cv2.resize(image, (max(1, round(width * scale)),
                                   max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
    return image


def _run(gray1, gray2, detector, norm, config):
    cv2.setRNGSeed(config.seed)
    start = perf_counter()
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)
    detected = perf_counter()
    status = "no_descriptors"
    matches, knn_pairs = [], 0
    if des1 is not None and des2 is not None and len(des1) and len(des2):
        status = "insufficient_neighbors"
        if len(des2) >= 2:
            neighbors = cv2.BFMatcher(norm).knnMatch(des1, des2, k=2)
            complete = [pair for pair in neighbors if len(pair) == 2]
            knn_pairs = len(complete)
            candidates = sorted(
                (m for m, n in complete if m.distance < config.ratio * n.distance),
                key=lambda m: (m.distance, m.queryIdx, m.trainIdx))
            # Avoid counting multiple query features at one target feature.
            used_targets = set()
            for match in candidates:
                if match.trainIdx not in used_targets:
                    matches.append(match)
                    used_targets.add(match.trainIdx)
            status = "insufficient_matches"
    matched = perf_counter()
    homography = None
    mask = np.zeros(len(matches), dtype=np.uint8)
    error = None
    if len(matches) >= 4:
        status = "homography_failed"
        src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        # SIFT can assign multiple orientations to the same spatial point.
        if len(np.unique(src.reshape(-1, 2), axis=0)) >= 4 and len(
                np.unique(dst.reshape(-1, 2), axis=0)) >= 4:
            try:
                candidate, candidate_mask = cv2.findHomography(
                    src, dst, cv2.RANSAC, config.ransac_threshold,
                    maxIters=2000, confidence=0.995)
            except cv2.error:
                candidate, candidate_mask = None, None
            if (candidate is not None and candidate_mask is not None
                    and np.isfinite(candidate).all()
                    and np.linalg.matrix_rank(candidate) == 3
                    and int(candidate_mask.sum()) >= 4):
                projected = cv2.perspectiveTransform(src, candidate)
                errors = np.linalg.norm(projected - dst, axis=2).ravel()
                if np.isfinite(errors).all():
                    homography = candidate
                    mask = candidate_mask.ravel().astype(np.uint8)
                    error = float(np.median(errors[mask.astype(bool)]))
                    status = "ok"
    ended = perf_counter()
    inliers = int(mask.sum())
    metrics = {
        "status": status, "keypoints1": len(kp1), "keypoints2": len(kp2),
        "knn_pairs": knn_pairs, "good_matches": len(matches), "inliers": inliers,
        "inlier_ratio": inliers / len(matches) if homography is not None else None,
        "median_reprojection_error_px": error,
        "detect_describe_ms": (detected - start) * 1000,
        "match_filter_ms": (matched - detected) * 1000,
        "geometry_ms": (ended - matched) * 1000,
        "total_ms": (ended - start) * 1000,
    }
    return MatchResult(metrics, list(kp1), list(kp2), matches, mask, homography)


def match_pair(image1, image2, method="SIFT", config=None):
    """Match BGR uint8 arrays. Timings are medians after one untimed warm-up.

    A returned homography maps image1 to image2 in their processed coordinates.
    'ok' means a model was estimated, not that scene identity was proven.
    """
    config = config or MatchConfig()
    method = method.upper()
    if method not in ("SIFT", "ORB"):
        raise ValueError("method must be SIFT or ORB")
    for image in (image1, image2):
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3 or not image.size):
            raise ValueError("Images must be nonempty BGR uint8 arrays")
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    if method == "SIFT":
        detector, norm = cv2.SIFT_create(nfeatures=config.nfeatures), cv2.NORM_L2
    else:
        detector, norm = cv2.ORB_create(nfeatures=config.nfeatures, WTA_K=2), cv2.NORM_HAMMING
    _run(gray1, gray2, detector, norm, config)
    runs = [_run(gray1, gray2, detector, norm, config) for _ in range(config.repeats)]
    result = runs[0]
    for field in ("detect_describe_ms", "match_filter_ms", "geometry_ms", "total_ms"):
        result.metrics[field] = statistics.median(run.metrics[field] for run in runs)
    result.metrics.update(method=method, width1=image1.shape[1], height1=image1.shape[0],
                          width2=image2.shape[1], height2=image2.shape[0])
    return result


def save_visualizations(result, image1, image2, output, max_draw=80):
    """Save keypoints, filtered matches, and RANSAC inliers; never open windows."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    def save(name, image, label):
        banner = np.full((58, image.shape[1], 3), 245, dtype=np.uint8)
        # Scale long labels to remain visible on smaller photographs.
        scale = min(0.6, max(0.1, (image.shape[1] - 16) / max(1, len(label) * 11)))
        cv2.putText(banner, label, (8, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (25, 25, 25), 1, cv2.LINE_AA)
        if not cv2.imwrite(str(output / name), np.vstack((banner, image))):
            raise OSError(f"Cannot write {output / name}")

    for index, (image, keypoints) in enumerate(
            ((image1, result.keypoints1), (image2, result.keypoints2)), start=1):
        drawn = cv2.drawKeypoints(image, keypoints, None, color=(60, 220, 60),
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        save(f"keypoints_{index}.jpg", drawn,
             f"{result.metrics['method']} | image {index} | {len(keypoints)} keypoints")
    inliers = [m for m, keep in zip(result.matches, result.inlier_mask) if keep]
    for filename, selected, label in (
        ("matches.jpg", result.matches, "ratio-filtered, unique-target matches"),
        ("inliers.jpg", inliers, "RANSAC inliers"),
    ):
        drawn = cv2.drawMatches(image1, result.keypoints1, image2, result.keypoints2,
                                selected[:max_draw], None, matchColor=(40, 210, 40),
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        save(filename, drawn, f"{result.metrics['method']} | {label}: {len(selected)} | "
             f"shown: {min(len(selected), max_draw)} | {result.metrics['status']}")
