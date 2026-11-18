# OpenCV image matching assignment

Compare **SIFT** and **ORB** on your own photographs, individually or as a batch.
The program saves numerical results, keypoint/match figures, homographies, and
SVG comparison plots. It runs without a GUI, internet connection, or Matplotlib.

No assignment photographs or experimental findings are included yet. Images
generated inside the automated tests are synthetic validation fixtures only.

## Install

Python 3.9 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. Run the commands
below from this project folder. JPEG and PNG are recommended; export HEIC phone
photos to JPEG before using them.

## Match any two photographs

```bash
python match_images.py pair path/to/image1.jpg path/to/image2.jpg --output results/my_pair
```

Both methods run by default. To change settings:

```bash
python match_images.py pair path/to/image1.jpg path/to/image2.jpg \
  --methods SIFT ORB --nfeatures 2000 --max-size 1600 \
  --ratio 0.75 --ransac-threshold 3 --repeats 5 --output results/my_pair_v2
```

The output folder must be new or empty to avoid mixing results from different
runs. Omit `--output` for an automatically timestamped folder under `results/`.
Use `python match_images.py pair --help` or `batch --help` for options.

## Photograph the three scenes

Use your own camera/phone and take **seven separate photographs per scene**.
Choose an indoor textured scene (poster/noticeboard), an outdoor textured scene
(wall/rock surface), and a low-texture scene (plain door/wall with some edges).
Prefer roughly planar subjects for this homography-based experiment. Bark or
other natural subjects can be used, but curved surfaces/depth introduce model
limitations that should be discussed.

Place these seven files in **each** of `data/indoor_textured/`,
`data/outdoor_textured/`, and `data/low_texture/`:

- `reference.jpg`: front-facing baseline image.
- `position_left.jpg`: move the camera left, keeping the subject in view.
- `position_right.jpg`: move the camera right, keeping the subject in view.
- `angle_roll.jpg`: rotate the camera around its viewing axis (for example 25°).
- `angle_oblique.jpg`: photograph from an oblique viewpoint (for example 35°).
- `zoom_1.jpg`: a second zoom setting (for example 1.5× the baseline).
- `zoom_2.jpg`: a larger zoom setting (for example 2× the baseline).

The suggested angles/zoom factors are targets, not measured capture metadata.
Keep the subject static, lighting and focus similar, and substantial overlap
between views. Change one factor as closely as practical. Record the actual
displacements, angles, lens/zoom settings and any unavoidable coupled changes
in `data/pairs.csv`. Keep the camera position fixed for zoom captures when
possible; record whether zoom is optical or digital. Do not substitute generated
images or crops of the reference for your separately captured photographs.

`data/pairs.csv` already lists the **18 reference-to-variant pairs**. Edit its
paths if using different filenames or formats. Paths are resolved relative to
the CSV file, so `indoor_textured/reference.jpg` refers to a file under `data/`.
Required columns are `scene,pair_id,image1,image2,variation`; `level,notes` are
optional. Scene/pair ID combinations must be unique. Quote fields containing
commas using normal CSV quoting.

## Run all experiments

After adding the 21 photographs:

```bash
python match_images.py batch data/pairs.csv --output results/experiment_01
```

This produces **36 evaluations** (18 pairs × 2 methods). All manifest paths
are checked before output is created. Missing photos produce a clear error.
A file that exists but cannot be decoded is recorded as `input_error`, and
the other pairs are still processed.

## Saved outputs

- `results.csv`: one row per pair and method, including failures and capture notes.
- `summary.csv`: mean metrics grouped by scene, variation, and method; evaluation,
  estimated-model and input-error counts are included.
- `charts/inliers.svg`: mean inlier counts by scene and method.
- `charts/inlier_ratio.svg`: mean inlier ratio **only for estimated models**.
- `charts/total_ms.svg`: mean of per-pair median processing times.
- `figures/<pair>/<method>/keypoints_1.jpg` and `keypoints_2.jpg`: detected keypoints.
- `figures/<pair>/<method>/matches.jpg`: ratio-filtered matches.
- `figures/<pair>/<method>/inliers.jpg`: matches consistent with the estimated model.
- `figures/<pair>/<method>/details.json`: metrics, full match coordinates,
  descriptor distances, inlier flags, and the homography (or `null`).
- `run_config.json` and `opencv_build.txt`: parameters, resolved photo paths,
  software/platform versions, thread setting, and timing definition.

Open JPEGs and SVGs in an image viewer/browser. `--max-draw` limits displayed
match lines only; **all retained matches** contribute to metrics and JSON.

## Method and interpretation

1. Decode color images with OpenCV, respecting EXIF orientation, and resize each
   so its longest side is at most 1600 pixels. Smaller images are not enlarged.
   Convert to grayscale for feature extraction.
2. Detect and describe SIFT/ORB features with the same requested feature budget
   (2000 by default). Actual detected counts can differ; SIFT can exceed its
   requested count at tied responses.
3. Use brute-force two-neighbor matching: L2 for SIFT and Hamming for ORB
   (`WTA_K=2`). Retain a candidate when `d_best < ratio * d_second`. Where several
   candidates point to one target feature, keep only the best-distance match.
   Matching is directional from image1 to image2; cross-check is not enabled.
4. With at least four distinct point positions, estimate a homography using
   RANSAC (3 processed-image pixels, confidence 0.995, max 2000 iterations).
   Missing, singular or invalid models are recorded as failures.

OpenCV references: [descriptor matching](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
and [feature matching with homography](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html).

The homography maps the **resized first image to the resized second image**.
Match coordinates and reprojection errors use these processed pixel coordinates,
not the original camera resolution. Do not apply the saved matrix directly to
full-resolution images without accounting for their resizing scales.

Metrics include keypoint counts for both images, the number of complete
two-neighbor query pairs (`knn_pairs`), filtered unique-target matches
(`good_matches`), RANSAC inliers, `inliers / good_matches`, and median forward
reprojection error over inliers. A failed geometry fit has inlier count zero
and undefined ratio/error (blank in CSV, `null` in JSON). An unreadable image
has undefined numerical metrics; summaries exclude these missing values.

**Inlier ratio measures geometric consistency, not ground-truth matching
accuracy.** An `ok` status means a model was estimated; it does not prove that
the images show the same scene. Inspect the match figures, especially for
repeated patterns and small/clustered inlier sets. A high ratio with four
matches is weaker evidence than many spatially distributed inliers.
Homography assumes a planar scene (or an appropriate camera motion); depth,
parallax, moving objects, and lens distortion can reduce consistency even when
descriptor matches are correct.

Timings use one untimed warm-up and the median of three timed runs per method
and pair. `detect_describe_ms` covers **both images**; `match_filter_ms` includes
matching/ratio filtering/target deduplication; `geometry_ms` covers model
estimation and validation. `total_ms` covers the complete measured core pipeline.
Loading, resizing, grayscale conversion, detector construction, and saving are
excluded. The median total need not equal the sum of the stage medians. OpenCV
uses one thread and a fixed RANSAC seed; timing still depends on hardware/load.

Keep all settings fixed for the main comparison. Equal ratio thresholds are a
controlled baseline, not necessarily equally optimal operating points for the
two descriptor types. An optional sensitivity experiment can rerun the same
pairs at ratios 0.65, 0.75, and 0.85 in separate output folders. Compare both
inlier counts and ratios, and keep the failed-fit counts visible.

Statuses are `ok`, `no_descriptors`, `insufficient_neighbors`,
`insufficient_matches`, `homography_failed`, and batch-only `input_error`.
Exit codes: 0 for a completed run (including expected geometry failures),
1 for a completed batch with input decoding errors, 2 for invalid requests or
an execution error. Geometry failures are useful experimental observations.

## Verify the implementation

```bash
python -m unittest discover -s tests -v
```

Tests use temporary synthetic images to check known geometric transformations,
low-feature failure handling, input validation, and CLI artifacts. They do not
replace the required real-photo experiments. After taking your photos, inspect
the outputs and write conclusions from the observed results.
