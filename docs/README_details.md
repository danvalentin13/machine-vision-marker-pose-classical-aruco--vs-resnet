# Machine Vision - Assignment 2: Marker Pose Tracking (classical vs. learned)

Project folder: `C:\Study\TUIASI\MLRC\an2\sem1\Machine Vision\Lab 2`

Use ONE PowerShell terminal (VS Code: Ctrl+`) for all steps below. **Do not close it and do not run `deactivate`** - you stay in the venv the whole time.
Every script asks its questions in the terminal; press **Enter** to accept the `[default]`. Copy only the command lines, never the line with the word `powershell`.

**Contents:** [Setup](#setup-do-once) - [T1 markers](#t1---generate-the-markers) - [T2 detection](#t2---detect-the-markers-live) - [T3 calibration and poses](#t3---calibration-recordings-and-poses) - [T4 animation](#t4---animation) - [T5 network](#t5---the-neural-network) - [T6 comparison](#t6---comparison) - [Bonus](#bonus---prediction-through-an-obstacle) - [Test without hardware](#test-everything-without-hardware) - [Folders](#folders)

---

## Setup (do once)

**Step 1 - Create the venv** (only if the folder `.venv` does not exist yet). PyTorch needs the CUDA build for the RTX 5080:

```
cd "C:\Study\TUIASI\MLRC\an2\sem1\Machine Vision\Lab 2"
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Step 2 - Activate the venv** (every new terminal):

```
cd "C:\Study\TUIASI\MLRC\an2\sem1\Machine Vision\Lab 2"
.venv\Scripts\Activate.ps1
```

The prompt starts with `(.venv)`. If PowerShell blocks the script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then repeat.

**Step 3 - Go to the code folder and check the folders** (all following commands run here):

```
cd code
python config.py
```

Expected: `DATA_DIR`, `OUTPUT_DIR` and `TRAINING_DIR` end in `Lab 2\data`, `Lab 2\output` and `Lab 2\training`.

---

## T1 - Generate the markers

```
python t1_generate_markers.py
```

| Question | Default |
|---|---|
| 1) Folder for the marker images | `output\markers` |
| 2) Folder for the dictionary table and sheet | `output\dictionary` |
| 3) Dictionary (`?` lists all OpenCV dictionaries) | `DICT_6X6_250` |
| 4) Id of F and id of M | `0` and `23` |
| 5) Side of the black square [mm] | `80` |

Last two lines must say `OK` (OpenCV reads the ids back). Results: `marker_F_id0.png`, `marker_M_id23.png` (print them),
`dictionary_sheet.pdf` (find any marker by its id; F and M framed in orange), `dictionary.csv`, and `data\markers.json`.

**Print at 100 % scale**, measure the black square with a calliper (target 80 mm, tolerance 1 mm) and write it in `data\markers.json` as `"measured_side_mm"`.
Glue F and M on rigid boards. Report: ids, dictionary name, measured L.

## T2 - Detect the markers live

```
python t2_detect_live.py
```

Questions: output folder (`output\detection`), mode, camera index (`0`), resolution (`1280x720` - keep it for ALL later tasks). Then Enter starts the camera.

| Mode | Call | What it is |
|---|---|---|
| `live` | `python t2_detect_live.py --mode live` | live view with outlines and ids; keys for single tests, screenshots |
| `guided` | `python t2_detect_live.py --mode guided` | **the guided live demo** (task check: both ids read correctly) |
| `matrix` | `python t2_detect_live.py --mode matrix` | the detection-limit experiments (distance x tilt) |

`python t2_detect_live.py` alone asks for the mode.

**Markers from earlier sessions:** T2 does not need T1 to be re-run. If `data\markers.json` is missing or does not match your images, T2 searches `output\markers` (subfolders too) for `marker_F_id*.png` / `marker_M_id*.png`,
recognises the dictionary from the images themselves (and the printed size from the PNG), lists the sets it found and asks which one to use; the choice is written back to `data\markers.json`.
Markers stored elsewhere: `python t2_detect_live.py --markers-folder "D:\old_markers"`. The other scripts (`t3_record`, `t3_pose`, ...) also adopt the markers found in `output\markers` when `data\markers.json` is missing.

**Guided live demo** - the T2 check "both ids read correctly in a live demo":

```
python t2_detect_live.py --mode guided
```

(or answer `guided` to the mode question, or press `G` in the live view). The camera runs and detects all the time; **nothing is saved automatically**.
A step list on the right of the window (and the terminal) tells you what to show and when to save it:

| Step | What to show | You save it with |
|---|---|---|
| 1. Both markers | F and M in view together, 40-60 cm, still | `S` screenshot |
| 2. M moving | F still, M moved slowly (closer, farther, sideways), about 10 s | `R` start recording ... `R` stop |
| 3. M tilted | M turned about 45 deg, F in view | `S` screenshot |
| 4. M far | M at about 1 m, F in view | `S` screenshot |

The panel shows live whether F and M are detected, M's distance and tilt, the fps and `REC` while recording.
**Checkpoints:** after every screenshot / recording the step is checked at once - a large green `STEP n OK` on the video and a `CHECKPOINT ... DONE - OK` line in the terminal,
or a red `NOT OK` with the reason (a marker missing, M tilted less than 30 deg in step 3, M closer than 80 cm in step 4, a recording shorter than 5 s or with a marker detected in less than 90 % of the frames).
Then press `S` / `R` again (it replaces the file) or `N` to keep it. When all four are OK: `ALL STEPS OK - press Q`.
**Safeguards against moving the wrong marker:** F's position at step 1 is remembered and F must stay there (at most 60 px away, else `NOT OK: F is the FIXED marker, put it back`).
In the recording F must stay still (at most 40 px) and M must really move (10 cm closer/farther or 80 px sideways); if F moved and M did not: `you moved F instead of M`.
In steps 3 and 4: if M is not turned / far enough but F is: `it looks like F was turned / moved away, not M`. While recording, the panel shows live `F: still / MOVING` and how far M has moved.
`R` always records the "M moving" step if it is not done yet, whatever step is shown. Steps saved in an earlier session are kept and the demo continues at the first missing one;
to start over: answer `fresh` when asked, or `python t2_detect_live.py --mode guided --fresh`. Rebuild the report from the saved files: `python t2_guided.py --report`. Keys: `S` screenshot - `R` start/stop recording - `N` next step - `B` previous step - `Q` finish.
Results in `output\detection\guided\`: `step1_both.png`, `step2_moving.avi` (the video with outlines and ids), `step3_tilted.png`, `step4_far.png`
and `guided_report.md` (ids in every screenshot, detection rate while recording, fps, "both ids read correctly in every step: yes/no").
F and M must be on separate boards: M has to move and tilt on its own.

Keys (click the video window first): `SPACE` pause - `C` camera off/on - `S` screenshot - `T` 10-frame test - `L` detection-limit experiments - `G` guided demo - `H` help - `Q`/`Esc` quit.

Detection runs automatically on every frame (green outline, `F id 0` / `M id 23`, fps shown). **`T` = detection-limit test:** type distance [cm] (ruler, camera lens to marker)
and tilt [deg] (protractor, 0 = facing the camera), hold still, Enter. 10 frames are captured; PASS = detected in at least 9. Each test adds a row to `output\detection\detection_tests.csv`.
**Detection limits - guided matrix (recommended):** its own call:

```
python t2_detect_live.py --mode matrix
```

(the same as answering `matrix` to question 2 of `python t2_detect_live.py`), or press **`L` in the live view**: the experiments start at once with the default plan (no questions in the terminal, so the keys stay in the window), and you are back in the live view when they are done. For other distances or tilts use `--mode matrix`. The test is a matrix of distances x tilts, default
4 distances `30,50,75,100` cm (a 1 m set-up) x 8 tilts `0,30,45,60,70,75,80,85` deg = 32 cells; both lists can be changed.
For each cell the terminal says where to put the marker (e.g. `Test 11/32: put marker M (id 23) at 50 cm from the camera lens, turned 45 deg about its vertical axis`).
The panel on the right shows the marker's distance and tilt **measured live from its pose** (calibrated camera, measured L) next to the target,
a FIT bar that turns **red** when both match, and the matrix so far (P pass, F fail). Set the marker with the ruler / protractor, use the panel to fine-tune, press `SPACE`:
10 frames are tested (PASS = at least 9 detected). After a clear FAIL (5/10 or less) the larger tilts at that distance are skipped (optional).
Keys: `SPACE` test the cell - `N` skip it - `U` back one cell - `Q` stop. Results: `output\detection\detection_matrix.csv` (every cell, incl. px per cell and the
measured distance/tilt) and `detection_matrix.md` (the matrix as a table, the distance limit, the largest tilt per distance, fps).
If every distance passes at tilt 0, the summary says the distance limit was not reached and extrapolates it with the 2-px-per-cell rule.
Without printed markers: `python t2_detect_live.py --image ..\output\markers\marker_F_id0.png`, or show the PNG on a phone.

---

## T3 - Calibration, recordings and poses

**3a. Chessboard for the calibration** (print on A4 at 100 %, glue on rigid flat board):

```
python t3_make_board.py
```

Questions: inner corners 9 x 6, square 25 mm, folder (`output\calibration`). Measure a printed square and write `measured_square_mm` in `data\calibration\board.json`.

**Print everything at once (A4):** after T1 and 3a run `python print_sheet.py`. It writes `output\print\print_sheet.pdf` with 3 A4 pages: marker F, marker M (portrait) and the chessboard (landscape).
Print at 100 % / "actual size" (not "fit to page"), then measure the black squares and one chessboard square with a calliper.

**3b. Calibrate the camera** (at least 10 views, 20-40 is better):

```
python t3_calibrate.py capture
```

**Guided:** it asks the number of views (default 20, at least 10), then gives one instruction per view in the terminal and at the bottom of the window, e.g.
`View 9/20: MEDIUM distance (the board covers about half of the image width, roughly 35 cm), in the TOP-LEFT corner of the image, turned so its RIGHT edge is farther from the camera, about 30 deg. Hold it still and press SPACE.`
The orange frame in the window shows the image region asked for. A **panel on the right** shows the target next to the measured values: distance [cm], board size in the image, region,
left/right edge away [deg], top/bottom edge away [deg], total tilt and in-plane rotation (OK / -- per item). The **FIT bar** grows as the view gets closer to the instruction and turns **red** when distance, region and tilt all fit: then hold still and press SPACE.
Before the first calibration the angles and distances come from an approximate camera (field of view about 70 deg): good to a few degrees and about 10-15 % in distance; after a calibration `camera.npz` is used. After each view it says `OK` or gives a hint (closer, farther, tilt more, other region); the view is kept either way.
No ruler or protractor needed: distances are judged by how big the board looks, and the program estimates the tilt itself.
Keys: `SPACE` save the view (only when the board is found) - `U` remove the last view and redo its instruction - `N` skip an instruction - `Q` finish.
Views left in the folder from an earlier session are moved to a subfolder `old_<date>` (not deleted) unless you choose to keep them. At the end answer `y` to calibrate, or run later:

```
python t3_calibrate.py calibrate
```

Distortion model: `k1k2` (simplest, recommended start) or `full`. Output: RMS, per-view errors, a held-out check, warnings (few views, too little tilt),
`data\calibration\camera.npz` (used by everything after) and `output\calibration\` (report.json, per_view.csv, calibration_report.png).
A low RMS alone does not prove a good calibration: look at tilt, coverage and the held-out error. Do not change zoom, focus or resolution afterwards.

**3c. Record** (calibration views are separate; this is for the tests, the training frames and the animation):

```
python t3_record.py
```

Questions: experiment tag (`E1_static`, `E2_dist_tilt`, `E3_motion`, `E4_robust`, `E7_occlusion`, `train`, `demo`), folder, name, notes, distance/tilt (optional), camera, resolution.
Folder default: `data\recordings` for all test recordings, `training\real` for tag `train` (frames for fine-tuning the network, kept apart on purpose).
Keys: `R` start/stop recording - `D` detection on/off - `S` screenshot - `Q` quit. Writes `<name>.avi`, `<name>.csv` (time stamps), `<name>.json`.
Every recording is one unit: never use frames of the same recording for training and for testing.

**3d. Poses of F, M and the relative pose `F_T_M`** for a recording:

```
python t3_pose.py
```

Questions: recording, camera file, camera still? (a missing F is then replaced by the last F), pose of M from `classical` or `network`, output folder, annotated video.
Uses the measured side L, `solvePnP` (IPPE_SQUARE + LM), and handles the planar ambiguity (takes the candidate closest to the previous frame when both are similar).
Output `output\pose\<name>_poses.csv` (every frame: flags, reprojection errors, `FM_*` = M in F as x,y,z [mm] + rotation vector [rad], roll/pitch/yaw [deg]) and `_annotated.avi`.
**Check B - camera invariance:** F and M stand still, you move the laptop slowly (record with `t3_record.py`), run `t3_pose.py` on it
(answer **n** to "camera still?"), then:

```
python t3_check_invariance.py
```

It compares how much the camera moved (position of F seen from the camera) with how much `F_T_M` changed (RMS jitter, max deviation, rotation),
checks the distance between the marker centres against the one you measured (black edge-to-edge distance + one side L), and writes
`output\pose\<name>_invariance.png` and `_invariance.md`.

## T4 - Animation

**Guided recording of M moving (for the export):** `python t4_record_guided.py` opens the camera with the instructions on the window
(press `R`, then follow the moves shown - left/right, closer/farther, turn, lift, circle - with a 30 s countdown; it saves by itself as
`data
ecordings\T4_motion_1`; `--name` / `--seconds` to change). Then `python t3_pose.py` on it and the export below.

```
python t4_animate.py export
python t4_animate.py live
```

`export`: reads `*_poses.csv`, writes `output\animation\<name>_animation.mp4` / `.gif`: annotated camera video, 3D view (F and M frames x red, y green, z blue, trajectory of M), x,y,z and roll,pitch,yaw curves with a synchronised cursor.
Optional Kalman / alpha-beta smoothing (frames without a measurement are shown as predicted, for at most 1 s). `live`: the same display from the camera, last 10 s (`S` screenshot - the window shows SCREENSHOT SAVED - , `Q` quit); nothing is recorded.

## T5 - The neural network

Everything for the network lives in `Lab 2\training\`: `synthetic\` (rendered data sets), `real\` (real training frames, recordings tagged `train`), `models\` (trained networks). The real TEST recordings stay in `data\recordings\`, so test and training frames cannot be mixed by accident.

**5a. Synthetic data set** (use the calibrated camera: make the final one AFTER the calibration; without `camera.npz` a provisional camera is used and the set is for tests only):

```
python t5_make_data.py
```

Questions: camera file, set name (`synth_v1`), folder (`training\synthetic`), number of train/val/test images (10000 / 1000 / 1000), network input width (640), distance range (0.3-1.2 m), max tilt (60 deg), seed.
About 2 min for 10000 images. Open `output\network\<name>\examples.png`: the green outline must sit on the markers.

**5b. Train** (ResNet-18, grey image, 6-D rotation + x/Z, y/Z, log Z; loss = chordal rotation + L1 translation, with weights):

```
python t5_train.py
```

Questions: data set, model name (`resnet18_v1`), epochs (15), batch (64), learning rate, loss weights (the assignment wants the weights and how you chose them: train with several), seed, pretrained.
Prints validation errors every epoch. A checkpoint is saved after every epoch (`training\models\<model>_checkpoint.pt`): if the run is
interrupted (update, crash, closed terminal), start `t5_train.py` again with the same model name and settings and it continues where it stopped. Question 9 opens a **live training window** (`t5_dashboard.py`), refreshed about once a second:
status (epoch, step, ETA, images/s, learning rate, GPU memory), the training loss per step with its two weighted parts (rotation, translation),
the validation rotation [deg] and translation [mm] error per epoch (median, 95th percentile), the learning-rate schedule, and 4 validation images
with the true (green) and predicted (orange) marker outline. Keys: `Q` stop early (the model is saved as it is), `S` screenshot; the last view is
saved as `output
etwork\<model>\dashboard.png`. Output `training\models\<name>.pt` and `output\network\<name>\` (train_log.json, training_curves.png).
A network is valid only for the camera K and marker size it was trained with.

**5b+. Two-stage network (better rotation):** stage 1 (a full-image network, e.g. `resnet18_v2.pt`) finds M; stage 2 sees only a crop
around M cut from the full-resolution camera image, so the marker has about 1.7x more pixels (3.4x when far):

```
python t5_make_crops.py
python t5_train.py
```

`t5_make_crops.py` asks the stage-1 network, the name (`crop_v1`), 30000 / 2000 / 2000 crops of 224 px, distances, tilt and the stage-1
error to simulate (0.12 of the marker side). `t5_train.py` recognises the crop data set and trains the 3-channel network (grey + viewing
direction of every pixel) with stronger augmentation and gradient clipping; the checkpoint remembers its stage 1. `t3_pose.py` and
`t6_evaluate.py` use a two-stage network like any other network file.

**5c. Use it in the pipeline:** run `t3_pose.py` and choose `network` (undistorts and resizes each frame, then predicts `C_T_M`; F stays classical).
Real training frames labelled by the classical pipeline may be used for fine-tuning; real test recordings never.

**5d. Fine-tune on real frames:** record training frames (guided, completion bar, stops by itself), then fine-tune both networks
(half real, half synthetic in every batch, live window):

```
cd code
python t4_record_guided.py --plan train --set real_v2 --seconds 36
python t5_finetune.py
```

Round 1 = `training/real_v2` -> `resnet18_v2_ft.pt`, `crop_resnet18_v1_ft.pt`. Round 2 = `training/real_v3` -> `*_ft2.pt`; its step-by-step
instructions are in `training/real_v3/INSTRUCTIONS.md`:

```
cd code
python t4_record_guided.py --plan train --set real_v3 --seconds 42
python t5_finetune.py --sets real_v2 real_v3 --suffix _ft2
```

## T6 - Comparison

Test recordings need a truth file `data\recordings\<name>.truth.csv` (hand-measured, NEVER taken from the classical pipeline):

```
t_s,x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg
0,150,200,0,0,0,0
```

M in the frame of F. One row = static pose; several rows (with time) = known motion, interpolated. Place F and M with the same orientation (marker "up" the same way) so that yaw = rotation of M about the vertical axis.

```
python t6_evaluate.py
```

Runs both pipelines on all recordings with a truth file (same frames, same truth), plus the held-out synthetic set (E5), and writes to `output\evaluation\`:
`comparison_table.md` (the assignment table: median and 95th percentile), `report.md` (E1 bias and jitter, E2 distance and tilt, E3 RMSE/median/max, E4 output rate, E5, E6 fps/training time/model size/lines of code),
`worst_frames.csv` and `worst\*.png` (show the worst frames, not only averages), `E3_*.png` (error against time).
Tags decide the experiment (the bonus has its own script, see below): `E1_static` (M at 5 measured positions, 3 angles, 100 frames each), `E2_dist_tilt`, `E3_motion` (known path, 3 x 20 s), `E4_robust` (blur, dim light, occlusion, clutter: write it in the notes).

---

## Bonus - Prediction through an obstacle

Record M passing behind a fixed obstacle (F stays visible), at least 5 passes at two speeds, hidden 0.5-2 s each, **on a known path at constant speed** (turntable / ruler / rail at a measured speed): tag `E7_occlusion`, and write the path in `data\recordings\<name>.truth.csv` with several rows (same format as T6). Then:

```
python t3_pose.py
python t7_bonus_occlusion.py
```

`t3_pose.py` gives the measured poses (frames without a pose of M are the hidden ones). `t7_bonus_occlusion.py` finds the hidden episodes (0.3-4 s) and runs the predictor:
constant-velocity Kalman filter (position) + alpha-beta filter on SO(3) (rotation). While M is hidden only the prediction runs and the uncertainty (sigma) grows; the pose is declared **lost** when 3 sigma exceeds a threshold (question 4, default 2 L).
When M reappears, the first measurements are **gated** against the prediction (chi-square on the position, angle gate against planar-ambiguity flips), so the output does not jump; after a lost pose it restarts from the measurement.

Questions: which recordings, truth folder, lost threshold in marker sides L (2), process noise (0.15 m/s^2), error threshold for the horizon (20 mm), output folder (`output\occlusion`).
Output: `summary.md` (error against time hidden at 0.25/0.5/1/2 s, predictor vs "hold the last pose", per speed; the horizon where the error passes the threshold; lost time; the jump at re-acquisition),
`episodes.csv`, `occlusion_curves.png`, and `<name>_predicted_poses.csv`. Show it in the animation (predicted segment dashed, 3 sigma circles, "LOST" text):

```
python t4_animate.py export
```

and give the `output\occlusion\<name>_predicted_poses.csv` file as the poses file. The optional learned predictor (small MLP/GRU) of the assignment is not implemented.

---

## Test everything without hardware

```
python selftest.py
```

Runs the whole pipeline on synthetic data in a temporary folder (about 35 s, nothing of yours is touched): markers, chessboard calibration of simulated views,
poses against exact truth, animation, the bonus (prediction through an obstacle), synthetic data set and label check, the network code (a throwaway net of 2 epochs, NOT a trained model), and the evaluation. It prints `PASS`/`FAIL` per check. Use `--quick` to skip the slow steps.

---

## Folders

```
Lab 2\
  README.md  requirements.txt
  docs\                   assignment and lectures (PDF)
  code\
    config.py             the ONE place that defines data\ and output\ (see below)
    mvlib.py  synth.py    lecture toolkit (transforms, PnP, Kalman filters) and synthetic scene
    mvtools.py            shared helpers of t1..t6 (questions, pose estimator, truth, metrics)
    t1_generate_markers.py  t2_detect_live.py  t3_make_board.py  t3_calibrate.py  t3_record.py  t3_pose.py
    t4_animate.py  t5_make_data.py  t5_train.py  t6_evaluate.py  t7_bonus_occlusion.py  print_sheet.py
    synthdata.py  netpose.py  selftest.py  selftest_learned.py
    lecture_reference\    the lecturer's example scripts (not needed for the assignment)
  data\                   INPUTS: markers.json, calibration\ (board.json, images\, camera.npz), recordings\ (test recordings + truth)
  training\               FOR THE NETWORK: synthetic\ (data sets), real\ (training frames), models\ (trained networks)
  output\                 GENERATED: markers, dictionary, print, detection, calibration, pose, animation, network, evaluation, occlusion
  .venv\                  virtual environment (local to each machine)
```

**Changing the folders.** All scripts take `data\`, `output\` and `training\` from `code\config.py`. Defaults are next to `code\`, so copying the whole project to another machine needs no change.
Other places: edit `DATA_DIR_OVERRIDE` / `OUTPUT_DIR_OVERRIDE` / `TRAINING_DIR_OVERRIDE` in `config.py`, or set `$env:MV_DATA_DIR = "D:\mv_data"` (and `MV_OUTPUT_DIR`, `MV_TRAINING_DIR`) for the session.
On another machine: copy the project, redo Step 1 (`.venv` is not portable), use the new path in the commands.

**Rules from the assignment:** state every library and AI assistant used in the report; no test frame is used for training or tuning; every number must be reproducible from this README.
