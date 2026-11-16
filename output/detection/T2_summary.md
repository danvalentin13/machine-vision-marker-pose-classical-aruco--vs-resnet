# Task 2 - marker detection with the laptop camera: summary

Sources (all in `output/detection/`): `guided/guided_report.md` and its files (live demo, 2026-11-15), `detection_matrix.md` / `.csv`
(detection limits, 2026-11-10). Code: `code/t2_detect_live.py`, `t2_guided.py`, `t2_matrix.py`; detector `MarkerDetector` in `code/mvlib.py`.

## Set-up

- Dictionary `DICT_6X6_250` (6x6 bits, 8 cells across with the border); F = id 0 (fixed), M = id 23 (moving).
- Printed black square **77 mm** (measured with a calliper; intended 80 mm, the printer scaled to about 96 %).
- Laptop camera, **1280x720**, calibrated (`data/calibration/camera.npz`, fx = 846.6 px).
- `cv2.aruco.ArucoDetector.detectMarkers` with **sub-pixel corner refinement** (`CORNER_REFINE_SUBPIX`).
- Frame rate of detection + drawing: **about 30-31 fps** (= the camera's frame rate; median 31 fps in the demo, 29 fps in the limit tests).

## Live demo (task check: both ids read correctly)

| step | file | result |
|---|---|---|
| 1. both markers, still | `guided/step1_both.png` | both ids read correctly (M ~58 cm) |
| 2. M moving, F still, 33.8 s | `guided/step2_moving.avi` | F detected in 94 %, M in 100 % of 1014 frames; M moved between 31 and 66 cm; no foreign ids |
| 3. M tilted ~45 deg | `guided/step3_tilted.png` | both ids read correctly (M tilt measured ~45 deg) |
| 4. M far | `guided/step4_far.png` | both ids read correctly (M ~95 cm) |

Both ids were read correctly in every step; no wrong or foreign id was ever reported.

## Detection limits (task check: largest distance and tilt with detection in >= 9 of 10 frames)

Marker M, distance from the camera lens (tape measure), tilt about the marker's vertical axis (protractor); 10 frames per cell.
The set-up allows at most 1 m.

| distance | px per cell at 0 deg | 0-75 deg | 80 deg | 85 deg | largest tilt with PASS |
|---|---|---|---|---|---|
| 30 cm | 29.7 | 10/10 | 10/10 | 0/10 | **80 deg** |
| 50 cm | 17.8 | 10/10 | 10/10 | 0/10 | **80 deg** |
| 75 cm | 11.0 | 10/10 | 10/10 | 0/10 | **80 deg** |
| 100 cm | 8.4 | 10/10 (75 deg: 10/10) | 5/10 | not tested | **75 deg** |

- **Largest distance:** the marker was detected in 10/10 frames at every distance up to 1 m (the end of the set-up): the distance limit
  was **not reached**. The marker size in pixels follows 1/distance (px per cell x distance = 828-892 px*cm at the four distances,
  close to fx L / 8 = 815), so the lecture rule (detection fails near 2 px per cell) predicts a limit of about **4.2 m** for this marker.
- **Largest tilt:** **80 deg** up to 75 cm, **75 deg** at 1 m.
- The tilt limit is the same pixel rule in disguise: a tilted marker is squeezed by cos(tilt) in one direction. At the limits that
  leaves 1.5-3 px per cell across the marker: 100 cm / 80 deg (1.5 px) fails, 75 cm / 80 deg (1.9 px) just passes, 50 cm / 85 deg (1.6 px) fails.
  Exception: 30 cm / 85 deg fails with 2.6 px per cell, so at very steep angles perspective, the thin quiet zone and the corner
  refinement limit detection before the pixel count does.
- Cross-check: the distance and tilt estimated from the marker pose agreed with the tape measure / protractor within 2.9 cm and 4 deg
  (mean differences +0.3 cm and -0.6 deg), which also supports the calibration scale.

## Notes / limitations

- Only marker M was tested for the limits (F is identical in size and dictionary).
- 100 cm / 80 deg is borderline (5/10); 85 deg was not tested at 100 cm after that clear fail.
- Constant indoor lighting, matte paper prints; glare and motion blur were not part of T2 (they are E4 in T6).
