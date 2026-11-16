# Machine Vision - Assignment 2: Marker Pose Tracking (classical vs. learned)

Every step of the project, in the order it is done. Each step has **one command on one line**, what it does, and what it writes.

**How to use this file**
- Use ONE PowerShell terminal (VS Code: Ctrl+`). Do steps [0.1](#step-01---open-the-project-folder), [0.5](#step-05---activate-the-venv) and [0.6](#step-06---go-to-the-code-folder) in every new terminal; all later commands run in the `code` folder with the venv active.
- Scripts that ask questions: press **Enter** to accept the `[default]`; the answers that differ from the default are listed in the step.
- Windows with a camera: click the window before pressing a key. Only one camera window can be open at a time.
- Longer explanations of the windows, keys and options: [docs/README_details.md](docs/README_details.md). History and results by date: [PROJECT_LOG.md](PROJECT_LOG.md). What the assignment asks and where it is: [ASSIGNMENT_CHECKLIST.md](ASSIGNMENT_CHECKLIST.md).

**What this repository contains - and what it leaves out**
- In: all the code, the calibration result, the truth files and notes of every recording, and the results (tables, reports, logs, plots).
- Out, for privacy: every camera recording, frame, screenshot and camera video (they show a person and a room), so `data\recordings\*.avi`, `data\calibration\images`, `training\real_v*\*.avi` and the camera animations are not here. They are handed in separately.
- Out, for size or rights: the trained networks and data sets (`training\models`, `training\synthetic`), the course PDFs and the lecturer's reference scripts.
- Animations without any camera picture (3D view and curves only) are in `output\public\`; `python make_public_media.py` makes them.
- Steps that need a recording or a network therefore run only after you record or train your own (steps 3-5).

## Contents

| Part | Steps |
|---|---|
| Setup | [0.1](#step-01---open-the-project-folder) [0.2](#step-02---create-the-venv-once) [0.3](#step-03---install-the-packages-once) [0.4](#step-04---install-pytorch-with-cuda-once) [0.5](#step-05---activate-the-venv) [0.6](#step-06---go-to-the-code-folder) [0.7](#step-07---check-the-folders) [0.8](#step-08---self-test-without-hardware) [0.9](#step-09---self-test-of-the-network-code) |
| T1 markers | [1.1](#step-11---generate-the-two-markers) [1.2](#step-12---make-the-chessboard) [1.3](#step-13---make-the-print-sheet) [1.4](#step-14---print-measure-and-write-the-measured-sizes) |
| T2 detection | [2.1](#step-21---live-detection) [2.2](#step-22---guided-live-demo) [2.3](#step-23---detection-limits-distance-x-tilt) [2.4](#step-24---rebuild-the-demo-report) |
| T3 calibration and poses | [3.1](#step-31---capture-the-calibration-views) [3.2](#step-32---calibrate-the-camera) [3.3](#step-33---record-the-5-measured-positions-check-a) [3.4](#step-34---record-the-camera-invariance-sequence-check-b) [3.5](#step-35---poses-of-a-recording) [3.6](#step-36---camera-invariance-report-check-b) |
| T4 animation | [4.1](#step-41---guided-recording-of-m-moving) [4.2](#step-42---poses-of-that-recording) [4.3](#step-43---export-the-animation) [4.4](#step-44---live-animation) |
| T5 network | [5.1](#step-51---synthetic-data-set) [5.2](#step-52---train-the-whole-image-network-stage-1) [5.3](#step-53---crop-data-set) [5.4](#step-54---train-the-crop-network-stage-2) [5.5](#step-55---record-the-real-training-frames) [5.6](#step-56---fine-tune-both-networks-on-real--synthetic-frames) [5.7](#step-57---test-the-networks-on-real-recordings) [5.8](#step-58---test-on-degraded-frames) |
| T6 comparison | [6.1](#step-61---live-check-of-a-placement) [6.2](#step-62---record-the-static-test-recordings-e1-e2-e4) [6.3](#step-63---record-the-known-motion-e3) [6.4](#step-64---run-the-comparison-e1-e6) [6.5](#step-65---both-methods-side-by-side-video) [6.6](#step-66---t4-animation-with-the-networks-poses) |
| Bonus | [7.1](#step-71---live-check-of-the-occlusion-set-up) [7.2](#step-72---record-passes-without-the-obstacle) [7.3](#step-73---record-passes-behind-the-obstacle) [7.4](#step-74---measure-the-passes-of-a-recording) [7.5](#step-75---predictor-on-the-uncovered-passes-hidden-in-software) [7.6](#step-76---predictor-behind-the-real-obstacle) [7.7](#step-77---animation-with-the-predicted-pose) [7.8](#step-78---live-classical-pipeline--predictor) |
| Reference | [Folders](#folders) - [Results: where each number comes from](#results-where-each-number-comes-from) - [Rules](#rules-from-the-assignment) |

---

## Setup

### Step 0.1 - Open the project folder
Every new terminal starts here.
```
cd "C:\Study\TUIASI\MLRC\an2\sem1\Machine Vision\Lab 2"
```

### Step 0.2 - Create the venv (once)
Only if the folder `.venv` does not exist yet.
```
python -m venv .venv
```

### Step 0.3 - Install the packages (once)
OpenCV (with ArUco), NumPy, Matplotlib, Pillow.
```
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Step 0.4 - Install PyTorch with CUDA (once)
The CUDA 12.8 build for the RTX 5080 (training and the network).
```
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Step 0.5 - Activate the venv
Every new terminal; the prompt then starts with `(.venv)`. If PowerShell blocks it: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then repeat.
```
.venv\Scripts\Activate.ps1
```

### Step 0.6 - Go to the code folder
All commands below run here.
```
cd code
```

### Step 0.7 - Check the folders
Prints `DATA_DIR`, `OUTPUT_DIR`, `TRAINING_DIR`; they must end in `Lab 2\data`, `Lab 2\output`, `Lab 2\training`. All paths are defined in this one file.
```
python config.py
```

### Step 0.8 - Self-test without hardware
Runs the whole classical pipeline on simulated data in a temporary folder (about 1 minute); prints PASS / FAIL per check.
```
python selftest.py
```

### Step 0.9 - Self-test of the network code
Checks the synthetic renderer, the rotation output, the loss and a throw-away training run.
```
python selftest_learned.py
```

---

## T1 - Markers

### Step 1.1 - Generate the two markers
Asks: folders, dictionary (`DICT_6X6_250`), ids (F = 0, M = 23), side (80 mm). Writes `output\markers\marker_F_id0.png`, `marker_M_id23.png`, `output\dictionary\dictionary_sheet.pdf`, `dictionary.csv`, and `data\markers.json`.
```
python t1_generate_markers.py
```

### Step 1.2 - Make the chessboard
The calibration target: 9 x 6 inner corners, 25 mm squares. Writes `output\calibration\chessboard_9x6_25mm.png` and `data\calibration\board.json`.
```
python t3_make_board.py
```

### Step 1.3 - Make the print sheet
One PDF with marker F, marker M and the chessboard on A4: `output\print\print_sheet.pdf`.
```
python print_sheet.py
```

### Step 1.4 - Print, measure and write the measured sizes
Print at 100 % ("actual size"). Measure the black square of each marker and one chessboard square. Write them as `measured_side_mm` (this project: 77.0) and, in `data\calibration\board.json`, `measured_square_mm` (24.0). Leave at least 1 cm of white around every black square.
```
notepad ..\data\markers.json
```

---

## T2 - Detection

### Step 2.1 - Live detection
Camera with the marker outlines and ids, the frame rate; keys in the window (`H` = help, `Q` = quit).
```
python t2_detect_live.py --mode live
```

### Step 2.2 - Guided live demo
Four steps on screen (both markers / M moving / M tilted / M far); `S` screenshot, `R` recording. Writes `output\detection\guided\` with `guided_report.md`.
```
python t2_detect_live.py --mode guided
```

### Step 2.3 - Detection limits (distance x tilt)
Guided matrix: 30, 50, 75, 100 cm x 0-85 deg, 10 frames per cell, pass = 9 of 10. Writes `output\detection\detection_matrix.csv` / `.md`.
```
python t2_detect_live.py --mode matrix
```

### Step 2.4 - Rebuild the demo report
Rewrites `guided_report.md` from the saved files of step 2.2.
```
python t2_guided.py --report
```

---

## T3 - Calibration and poses

### Step 3.1 - Capture the calibration views
Guided: 20 views of the chessboard (distance, image region and tilt asked per view; `SPACE` saves a view). Writes `data\calibration\images\`.
```
python t3_calibrate.py capture
```

### Step 3.2 - Calibrate the camera
Model `k1k2`. Writes `data\calibration\camera.npz` (used by every later step) and `output\calibration\report.json`, `per_view.csv`, `calibration_report.png` (RMS, per-view errors, held-out check). This project: RMS 1.18 px.
```
python t3_calibrate.py calibrate
```

### Step 3.3 - Record the 5 measured positions (check A)
One command per position: M to the left of F at a measured gap (truth x = -(gap + 77) mm), 100 frames. The five recordings of this project: gaps 200, 150, 100, 50, 250 mm (`E1_p1` ... `E1_p5`; `E1_p4` has M 3 mm higher). Example for position 1:
```
python t6_record.py --name E1_p1 --tag E1_static --truth -277 0 0 0 0 0
```

### Step 3.4 - Record the camera-invariance sequence (check B)
F and M stand still, the laptop is moved slowly. Answer: tag `other`, name `invariance_1`; `R` starts and stops.
```
python t3_record.py
```

### Step 3.5 - Poses of a recording
Asks: recording, camera file, camera still (y; **n** for `invariance_1`), pose of M from `classical` or `network`. Writes `output\pose\<name>_poses.csv` and `<name>_annotated.avi`.
```
python t3_pose.py
```

### Step 3.6 - Camera-invariance report (check B)
Compares how far the camera moved with how much the relative pose changed. Writes `output\pose\invariance_1_invariance.md` / `.png`.
```
python t3_check_invariance.py
```

---

## T4 - Animation

### Step 4.1 - Guided recording of M moving
30 s with the moves shown on screen; `R` starts, it stops by itself. Writes `data\recordings\T4_motion_1.avi` / `.csv` / `.json`.
```
python t4_record_guided.py --name T4_motion_1 --seconds 30
```

### Step 4.2 - Poses of that recording
Choose `T4_motion_1`, camera still `y`, `classical`. Writes `output\pose\T4_motion_1_poses.csv`.
```
python t3_pose.py
```

### Step 4.3 - Export the animation
Camera video + 3D view in the frame of F + x, y, z and roll, pitch, yaw curves, synchronised. Writes `output\animation\T4_motion_1_animation.mp4` / `.gif`.
```
python t4_animate.py export
```

### Step 4.4 - Live animation
The same display from the camera (last 10 s); `S` screenshot, `Q` quit.
```
python t4_animate.py live
```

---

## T5 - Network

### Step 5.1 - Synthetic data set
10 000 / 1 000 / 1 000 rendered images with the calibrated camera (seed 0). Answer: name `synth_v1`. Writes `training\synthetic\synth_v1\` and `output\network\synth_v1\examples.png`.
```
python t5_make_data.py
```

### Step 5.2 - Train the whole-image network (stage 1)
ResNet-18, 6-D rotation + (x/Z, y/Z, log Z), chordal + L1 loss. Answers: data set `synth_v1`, name `resnet18_v2`, epochs `40`, batch 64, learning rate 5e-4, weights 1 / 1, seed 0, live window `y`. Writes `training\models\resnet18_v2.pt` and `output\network\resnet18_v2\`. (The loss-weight study: the same with name `resnet18_v3` and rotation weight `2`.)
```
python t5_train.py
```

### Step 5.3 - Crop data set
30 000 / 2 000 / 2 000 crops of 224 px around M, placed by stage 1. Answers: stage-1 network `resnet18_v2`, name `crop_v1`. Writes `training\synthetic\crop_v1\`.
```
python t5_make_crops.py
```

### Step 5.4 - Train the crop network (stage 2)
Answers: data set `crop_v1`, name `crop_resnet18_v1`, epochs `80`, the rest as in step 5.2. Writes `training\models\crop_resnet18_v1.pt`.
```
python t5_train.py
```

### Step 5.5 - Record the real training frames
Guided: `R` starts, `SPACE` pauses, it stops by itself; the recordings are numbered automatically. Run it 5 times with `--set real_v2 --seconds 36`, 5 times with `--set real_v3 --seconds 42`, 7 times as below (camera pitched / turned; plans in `training\real_v3\INSTRUCTIONS.md` and `training\real_v4\INSTRUCTIONS.md`).
```
python t4_record_guided.py --plan train --set real_v4 --seconds 42
```

### Step 5.6 - Fine-tune both networks on real + synthetic frames
Labels the 17 recordings with the classical pipeline, builds the sets, fine-tunes stage 1 (25 epochs) and stage 2 (50 epochs): half real / half synthetic per batch, learning rate 1e-4, batch 128, seed 0, 4 whole recordings for validation, best validation epoch kept. Writes `training\models\resnet18_v2_ft3.pt` and `crop_resnet18_v1_ft3.pt` (the final network) and `output\network\<name>_ft3\`. Interrupted: run it again, it continues.
```
python t5_finetune.py --sets real_v2 real_v3 real_v4 --suffix _ft3 --val real_v2_05 real_v3_05 real_v4_03 real_v4_07 --yes
```

### Step 5.7 - Test the networks on real recordings
Classical against the networks on `E1_p1..p5` and `T4_motion_1`. Writes `output\network\real_E1_comparison_ft3.json`.
```
python t5_eval_real.py
```

### Step 5.8 - Test on degraded frames
E1 frames with part of M covered, motion blur, low resolution, dim light (made in software). Writes `output\network\hard_cases_E1.json` and `output\network\figures\`.
```
python t5_test_hard_cases.py
```

---

## T6 - Comparison

Truth convention for every test recording: the pose of M in the directions of F, measured by ruler / protractor: `x` + = M right of F, `y` + = M higher, `z` + = M nearer the camera [mm]; `roll pitch yaw` [deg], pitch = turn about the vertical axis. F and M mounted the same way up, at least 1 cm of white around each black square.

### Step 6.1 - Live check of a placement
Shows what the camera measures now (pose of M in F, how each marker faces the camera, steadiness); records nothing.
```
python t6_record.py --check
```

### Step 6.2 - Record the static test recordings (E1, E2, E4)
One command per recording: the table in the window compares the truth you give with the camera; `R` records 100 frames and stops. The recordings of this project (run each line once, after setting up what its name says):

| Recording | Command |
|---|---|
| E1, M turned 20 deg | `python t6_record.py --name E1_turn20 --tag E1_static --truth -115 -20 90 0 20 0 --distance 61 --tilt 20` |
| E1, M turned 40 deg | `python t6_record.py --name E1_turn40 --tag E1_static --truth -115 -20 90 0 40 0 --distance 61 --tilt 40` |
| E2, camera 40 cm, 0 deg | `python t6_record.py --name E2_d40_t0 --tag E2_dist_tilt --truth -115 -20 90 0 0 0 --distance 40 --tilt 0` |
| E2, camera 60 cm, 0 deg | `python t6_record.py --name E2_d60_t0 --tag E2_dist_tilt --truth -115 -20 90 0 0 0 --distance 60 --tilt 0` |
| E2, camera 75 cm, 0 deg | `python t6_record.py --name E2_d75_t0 --tag E2_dist_tilt --truth -115 -20 90 0 0 0 --distance 75 --tilt 0` |
| E2, camera 40 cm, 40 deg | `python t6_record.py --name E2_d40_t40 --tag E2_dist_tilt --truth -115 -20 90 0 40 0 --distance 40 --tilt 40` |
| E2, camera 60 cm, 40 deg | `python t6_record.py --name E2_d60_t40 --tag E2_dist_tilt --truth -115 -20 90 0 40 0 --distance 61 --tilt 40` |
| E2, camera 75 cm, 40 deg | `python t6_record.py --name E2_d75_t40 --tag E2_dist_tilt --truth -115 -20 90 0 40 0 --distance 75 --tilt 40` |
| E4, dim light | `python t6_record.py --name E4_dim --tag E4_robust --truth -135 3.5 0 0 0 0 --distance 61 --tilt 0` |
| E4, left edge of M covered | `python t6_record.py --name E4_cover --tag E4_robust --truth -135 3.5 0 0 0 0 --distance 61 --tilt 0` |
| E4, clutter around the markers | `python t6_record.py --name E4_clutter --tag E4_robust --truth -135 3.5 0 0 0 0 --distance 61 --tilt 0` |
| E4, table shaken (mild) | `python t6_record.py --name E4_blur --tag E4_robust --frames 150 --truth -135 3.5 0 0 0 0 --distance 61 --tilt 0` |
| E4, camera wobbled (blur) | `python t6_record.py --name E4_blur_2 --tag E4_robust --frames 150 --truth -135 3.5 0 0 0 0 --distance 61 --tilt 0` |

Each writes `data\recordings\<name>.avi` / `.csv` / `.json` / `.truth.csv`. How each was set up: [output/evaluation/T6_notes.md](output/evaluation/T6_notes.md).

### Step 6.3 - Record the known motion (E3)
M slides 60 mm along a ruler and back, following an outline on the screen that moves at 1 cm/s; the truth file is written from that schedule. Run 3 times with names `E3_motion_01`, `_02`, `_03`.
```
python t4_record_guided.py --plan motion --name E3_motion_01 --seconds 20 --pass-s 6 --truth-start -140 -20 90 --truth-end -200 -20 90
```

### Step 6.4 - Run the comparison (E1-E6)
Both pipelines on every recording with a truth file, the same frames and the same truth, plus the held-out synthetic set (E5) and the cost (E6). Writes `output\evaluation\comparison_table.md` (the assignment's table), `report.md`, `summary.json`, `worst_frames.csv`, `worst\`, `E3_*.png`. The fps values change a little from run to run.
```
python t6_evaluate.py --network crop_resnet18_v1_ft3 --yes
```

### Step 6.5 - Both methods side by side (video)
The same frames twice: classical on the left, network on the right, with the axes of F and M, the pose and the error against the truth; cropped to the markers. Writes `output\evaluation\side_by_side_<name>.mp4`.
```
python t6_side_by_side.py E3_motion_01 E4_cover E1_turn40
```

### Step 6.6 - T4 animation with the network's poses
The T4 display driven by the network (F classical, M network). Writes `output\animation\T4_motion_1_network_animation.mp4`.
```
python t6_side_by_side.py T4_motion_1 --animation
```

---

## Bonus - prediction behind an obstacle

Set-up: F on the right, a ruler along the table, M slides along it toward F between two stops; an obstacle stands in front of the middle of the path. Details, layouts and all numbers: [output/bonus/E7_summary.md](output/bonus/E7_summary.md) (use its corrected sections).

### Step 7.1 - Live check of the occlusion set-up
Shows the path of M, where M ends and where the obstacle must stand; the numbers are the start distance between the centres, the travel and the ruler readings the obstacle hides [mm].
```
python t7_setup_guided.py --centre 325 --travel 228 --obstacle 180 263
```

### Step 7.2 - Record passes without the obstacle
Guided cycle: rest - GO - rest - back, with beeps; several passes per recording. This project: `E7_nocover_slow_2` and `E7_nocover_fast`.
```
python t4_record_guided.py --plan bonus --name E7_nocover_slow_2 --seconds 60
```

### Step 7.3 - Record passes behind the obstacle
The same with the obstacle in place: a countdown, then an outline that moves at the wanted speed (`--pass-s` = seconds per pass). This project: `E7_box_slow` (2.5 s), `E7_box_fast` (1.7 s), `E7_box_5cms` and `E7_box_5cms_2` (4.6 s, `--seconds 95`), and with the first layout `E7_cover_slow_3`, `E7_cover_slow_4`, `E7_cover_fast`.
```
python t4_record_guided.py --plan bonus --name E7_box_slow --seconds 75 --pass-s 2.5 --travel 228 --centre 325
```

### Step 7.4 - Measure the passes of a recording
Finds each pass (rest - slide - rest): start and end distance to F, travel, speed, straightness, hidden time. Writes `output\bonus\<name>_passes.csv` / `.png`.
```
python t7_passes.py E7_box_slow
```

### Step 7.5 - Predictor on the uncovered passes, hidden in software
M is hidden in software where an obstacle would hide it; the prediction is compared with the measured pose in every hidden frame, and with 'hold the last pose'. Writes `output\bonus\virtual_occlusion.md` / `.csv` / `.png` (errors against time hidden, horizon, jump).
```
python t7_virtual_occlusion.py E7_nocover_slow_2 E7_nocover_fast
```

### Step 7.6 - Predictor behind the real obstacle
Error at the moment M is seen again (predictor against 'hold the last pose'), for every pass. Writes `output\bonus\real_occlusion.md` / `.csv` / `.png`.
```
python t7_real_occlusion.py E7_box_5cms_2 E7_box_5cms E7_box_slow E7_box_fast E7_cover_slow_4 E7_cover_slow_3 E7_cover_fast --min-runup 50
```

### Step 7.7 - Animation with the predicted pose
T4 display with the predicted segment dashed, the growing uncertainty, a red box where M is predicted; everything outside the path is cut out. Writes `output\bonus\E7_box_5cms_2_occlusion.mp4` / `.gif`.
```
python t7_animate.py E7_box_5cms_2 --start 37.5 --seconds 50 --gif
```

### Step 7.8 - Live: classical pipeline + predictor
Free motion on the camera: green = measured, red box = predicted while M is hidden, plots of position and angles with the predicted stretches shaded.
```
python t7_live.py
```

---

## Folders

```
Lab 2\
  README.md  PROJECT_LOG.md  ASSIGNMENT_CHECKLIST.md  requirements.txt
  docs\        the assignment and lectures (PDF), README_details.md (long explanations)
  code\        config.py (the one place for the paths), mvlib.py + synth.py (lecture toolkit), mvtools.py (shared helpers),
               t1_ .. t7_ scripts, synthdata.py, netpose.py, selftest.py, selftest_learned.py
  data\        INPUTS: markers.json, calibration\ (board.json, images\, camera.npz), recordings\ (test recordings + truth)
  training\    FOR THE NETWORK: synthetic\ (data sets), real_v2\ real_v3\ real_v4\ (real training recordings), models\ (networks)
  output\      GENERATED: markers, dictionary, print, detection, calibration, pose, animation, network, evaluation, bonus
  .venv\       virtual environment (local to each machine)
```

Other locations for `data`, `output`, `training`: edit the `*_OVERRIDE` lines in `code\config.py`, or set `MV_DATA_DIR`, `MV_OUTPUT_DIR`, `MV_TRAINING_DIR`.

## Results: where each number comes from

| Result | Step | File |
|---|---|---|
| Calibration RMS, per-view errors | [3.2](#step-32---calibrate-the-camera) | `output\calibration\report.json`, `per_view.csv` |
| Detection limits, frame rate | [2.3](#step-23---detection-limits-distance-x-tilt) | `output\detection\detection_matrix.md`, `T2_summary.md` |
| Check A (5 positions) | [6.4](#step-64---run-the-comparison-e1-e6) | `output\evaluation\report.md`, section E1 (`output\pose\E1_checkA.md` is an earlier hand-made summary) |
| Check B (camera invariance) | [3.6](#step-36---camera-invariance-report-check-b) | `output\pose\invariance_1_invariance.md` |
| Network training, loss weights | [5.2](#step-52---train-the-whole-image-network-stage-1), [5.4](#step-54---train-the-crop-network-stage-2), [5.6](#step-56---fine-tune-both-networks-on-real--synthetic-frames) | `output\network\<name>\train_log.json`, `output\network\T5_runs.md` |
| Comparison table, E1-E6, worst frames | [6.4](#step-64---run-the-comparison-e1-e6) | `output\evaluation\comparison_table.md`, `report.md`, `worst_frames.csv` |
| Bonus | [7.5](#step-75---predictor-on-the-uncovered-passes-hidden-in-software), [7.6](#step-76---predictor-behind-the-real-obstacle) | `output\bonus\virtual_occlusion.md`, `real_occlusion.md`, `E7_summary.md` |

Notes on reproducing:
- Training uses seed 0; GPU training is still not bit-for-bit repeatable, so retrained networks give slightly different numbers. The networks used for the reported numbers are kept in `training\models\`.
- `_ft` and `_ft2` in `training\models\` are the two earlier fine-tuning rounds (fewer recordings, older settings); their logs are in `output\network\logs\`. The reported network is `_ft3` (step 5.6).
- Speed values (fps, ms per frame) depend on what else the laptop is doing.

## Rules from the assignment

State every library, pretrained backbone and AI coding assistant used in the report. No test frame is used for training or tuning (see the honesty points in [ASSIGNMENT_CHECKLIST.md](ASSIGNMENT_CHECKLIST.md)). Every number must be reproducible from this file.
