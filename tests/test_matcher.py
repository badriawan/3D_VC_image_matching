"""Independent checks with synthetic images, not assignment photo evidence.

Run from the project root: python3 -m unittest discover -s tests -v
"""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from matcher import MatchConfig, load_image, match_pair


def textured_image():
    """Create a reproducible scene with unique corners at multiple scales."""
    rng = np.random.default_rng(2026)
    image = rng.integers(40, 215, (480, 640, 3), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    for index in range(100):
        x, y = rng.integers([25, 25], [590, 430])
        color = tuple(int(v) for v in rng.integers(0, 256, 3))
        radius = int(rng.integers(4, 18))
        if index % 2:
            cv2.circle(image, (int(x), int(y)), radius, color, 2)
        else:
            cv2.rectangle(image, (int(x), int(y)),
                          (int(x + radius), int(y + radius)), color, -1)
    for y, text in [(100, "OpenCV matching"), (240, "Independent check 2026"),
                    (380, "Texture / geometry")]:
        cv2.putText(image, text, (45, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (245, 245, 245), 2, cv2.LINE_AA)
    return image


class MatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)
        cls.reference = textured_image()
        cls.expected_h = np.array([
            [0.96, -0.08, 35.0],
            [0.07, 0.97, 8.0],
            [0.00004, -0.00005, 1.0],
        ], dtype=np.float64)
        cls.variant = cv2.warpPerspective(cls.reference, cls.expected_h, (640, 480))
        cls.config = MatchConfig(repeats=1)

    def test_transformed_scene_matches_known_geometry(self):
        anchors = np.float32([
            [120, 100], [500, 100], [500, 360], [120, 360], [320, 240]
        ]).reshape(-1, 1, 2)
        expected = cv2.perspectiveTransform(anchors, self.expected_h)
        for method in ("SIFT", "ORB"):
            with self.subTest(method=method):
                result = match_pair(self.reference, self.variant, method, self.config)
                metrics = result.metrics
                self.assertEqual(metrics["status"], "ok")
                self.assertIsNotNone(result.homography)
                self.assertGreater(metrics["inliers"], 30)
                self.assertGreater(metrics["inlier_ratio"], 0.5)
                self.assertLess(metrics["median_reprojection_error_px"], 3.0)
                actual = cv2.perspectiveTransform(anchors, result.homography)
                error = np.linalg.norm(actual - expected, axis=2)
                self.assertLess(float(error.max()), 3.0)
                self.assertEqual(len(result.matches), metrics["good_matches"])
                self.assertEqual(len(result.inlier_mask), len(result.matches))
                self.assertEqual(int(np.count_nonzero(result.inlier_mask)),
                                 metrics["inliers"])
                self.assertEqual(metrics["keypoints1"], len(result.keypoints1))
                self.assertEqual(metrics["keypoints2"], len(result.keypoints2))
                for field in ("detect_describe_ms", "match_filter_ms",
                              "geometry_ms", "total_ms"):
                    self.assertGreaterEqual(metrics[field], 0.0)
                    self.assertTrue(np.isfinite(metrics[field]))

    def test_featureless_images_return_recorded_failure(self):
        blank = np.full_like(self.reference, 160)
        for method in ("SIFT", "ORB"):
            for first in (blank, self.reference):
                with self.subTest(method=method, both_blank=first is blank):
                    result = match_pair(first, blank, method, self.config)
                    self.assertEqual(result.metrics["status"], "no_descriptors")
                    self.assertIsNone(result.homography)
                    self.assertIsNone(result.metrics["inlier_ratio"])
                    self.assertIsNone(result.metrics["median_reprojection_error_px"])
                    self.assertEqual(result.metrics["inliers"], 0)
                    self.assertEqual(len(result.matches), 0)
                    self.assertEqual(len(result.inlier_mask), 0)


class InputTests(unittest.TestCase):
    def test_load_image_downscales_without_upscaling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.png"
            source = np.full((400, 800, 3), 128, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(path), source))
            reduced = load_image(path, max_size=200)
            self.assertEqual(reduced.shape, (100, 200, 3))
            full = load_image(path, max_size=1600)
            self.assertEqual(full.shape, source.shape)

    def test_missing_or_unreadable_image_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                load_image(Path(directory) / "missing.jpg")
            invalid = Path(directory) / "invalid.jpg"
            invalid.write_text("This is not an image.", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_image(invalid)

    def test_invalid_configuration_is_rejected(self):
        for options in (
            {"nfeatures": 0}, {"ratio": 0.0}, {"ratio": 1.0},
            {"ransac_threshold": 0.0}, {"repeats": 0},
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    MatchConfig(**options)


if __name__ == "__main__":
    unittest.main()
