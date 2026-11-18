"""End-to-end CLI checks using temporary synthetic input files."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from match_images import main


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        rng = np.random.default_rng(90210)
        image = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        cv2.putText(image, "Scene 2026", (25, 130), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        shifted = cv2.warpAffine(image, np.float32([[1, 0, 8], [0, 1, 6]]),
                                 (image.shape[1], image.shape[0]))
        for name, content in (("reference.png", image), ("shifted.png", shifted),
                              ("blank.png", np.full_like(image, 128))):
            self.assertTrue(cv2.imwrite(str(self.root / name), content))
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def invoke(self, arguments):
        with redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            return main([str(arg) for arg in arguments])

    def read_results(self, output):
        with (output / "results.csv").open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def manifest(self, rows, name="pairs.csv"):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["scene", "pair_id", "image1", "image2", "variation"])
            writer.writerows(rows)
        return path

    def test_pair_exports_both_methods_and_reviewable_artifacts(self):
        output = self.root / "pair_results"
        code = self.invoke(["pair", self.root / "reference.png", self.root / "shifted.png",
                            "--output", output, "--repeats", 1])
        self.assertEqual(code, 0, self.stderr.getvalue())
        rows = self.read_results(output)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["method"] for row in rows}, {"SIFT", "ORB"})
        self.assertTrue(all(row["status"] == "ok" for row in rows))
        config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["config"]["repeats"], 1)
        self.assertEqual(set(config["methods"]), {"SIFT", "ORB"})
        self.assertGreater((output / "summary.csv").stat().st_size, 0)
        figures = list((output / "figures").glob("*/*"))
        self.assertEqual(len(figures), 2)
        for folder in figures:
            details = json.loads((folder / "details.json").read_text(encoding="utf-8"))
            self.assertEqual(details["metrics"]["method"], folder.name)
            self.assertTrue(details["matches"])
            self.assertEqual(np.asarray(details["homography"]).shape, (3, 3))
            for name in ("keypoints_1.jpg", "keypoints_2.jpg", "matches.jpg", "inliers.jpg"):
                image = cv2.imread(str(folder / name))
                self.assertIsNotNone(image, str(folder / name))
                self.assertGreater(image.size, 0)
        charts = list((output / "charts").glob("*.svg"))
        self.assertEqual(len(charts), 3)
        for chart in charts:
            self.assertEqual(ET.parse(chart).getroot().tag,
                             "{http://www.w3.org/2000/svg}svg")

    def test_batch_resolves_relative_paths_and_retains_geometry_failures(self):
        manifest = self.manifest([
            ["textured", "shift", "reference.png", "shifted.png", "position"],
            ["plain", "blank", "reference.png", "blank.png", "texture"],
        ])
        # The manifest lives outside the process cwd, as do its relative images.
        self.assertNotEqual(manifest.parent, Path.cwd())
        output = self.root / "batch_results"
        code = self.invoke(["batch", manifest, "--output", output, "--repeats", 1])
        self.assertEqual(code, 0, self.stderr.getvalue())
        rows = self.read_results(output)
        self.assertEqual(len(rows), 4)
        successful = [row for row in rows if row["scene"] == "textured"]
        failures = [row for row in rows if row["scene"] == "plain"]
        self.assertEqual(len(successful), 2)
        self.assertEqual(len(failures), 2)
        self.assertTrue(all(row["status"] == "ok" for row in successful))
        self.assertTrue(all(row["status"] == "no_descriptors" for row in failures))
        self.assertTrue(all(row["inlier_ratio"] == "" for row in failures))
        self.assertTrue(all(Path(row["image1"]).is_absolute() for row in rows))

    def test_batch_records_corrupt_input_and_continues(self):
        (self.root / "corrupt.jpg").write_text("not an image", encoding="utf-8")
        manifest = self.manifest([
            ["scene", "corrupt", "reference.png", "corrupt.jpg", "position"],
            ["scene", "valid", "reference.png", "shifted.png", "position"],
        ])
        output = self.root / "partial_results"
        code = self.invoke(["batch", manifest, "--output", output, "--repeats", 1])
        self.assertEqual(code, 1)
        rows = self.read_results(output)
        self.assertEqual(len(rows), 4)
        failures = [row for row in rows if row["pair_id"] == "corrupt"]
        valid = [row for row in rows if row["pair_id"] == "valid"]
        self.assertEqual(len(failures), 2)
        self.assertTrue(all(row["status"] == "input_error" and row["error"] for row in failures))
        self.assertEqual(len(valid), 2)
        self.assertTrue(all(row["status"] == "ok" for row in valid))

    def test_invalid_manifests_fail_before_creating_output(self):
        header = "scene,pair_id,image1,image2,variation\n"
        row = "scene,first,reference.png,shifted.png,position\n"
        cases = {
            "missing_columns": "scene,pair_id\nscene,first\n",
            "duplicate": header + row + row,
            "missing_image": header + "scene,first,reference.png,missing.png,position\n",
            "extra_columns": header + row.rstrip("\n") + ",unquoted extra\n",
        }
        for name, content in cases.items():
            with self.subTest(name=name):
                manifest = self.root / f"{name}.csv"
                manifest.write_text(content, encoding="utf-8")
                output = self.root / f"{name}_results"
                code = self.invoke(["batch", manifest, "--output", output, "--repeats", 1])
                self.assertEqual(code, 2)
                self.assertFalse(output.exists())

    def test_existing_nonempty_output_is_preserved(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_bytes(b"Original content\x00\n")
        code = self.invoke(["pair", self.root / "reference.png", self.root / "shifted.png",
                            "--output", output, "--repeats", 1])
        self.assertEqual(code, 2)
        self.assertEqual(sentinel.read_bytes(), b"Original content\x00\n")
        self.assertEqual(list(output.iterdir()), [sentinel])


if __name__ == "__main__":
    unittest.main()
