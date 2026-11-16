# Project log - Assignment 2: Marker Pose Tracking (classical vs. learned)

Last update: 2026-11-15. How to run everything: [README.md](README.md). Assignment: [docs/MV_Assignment2_Marker_Pose.pdf](docs/MV_Assignment2_Marker_Pose.pdf).

## 1. Where we are

All code for T1-T6 and the bonus is written. It is tested on **synthetic data** (`python selftest.py`, 32/32 checks pass, about 35 s).
Nothing has been run on **real hardware** except T1 and T2. The network has **not been trained** yet.

| Task | Code | Tested | Still to do |
|---|---|---|---|
| T1 markers | done | ids read back OK; **printed and measured: 77 mm** (intended 80) | state the print deviation in the report (see 2) |
| T2 live detection | done (`t2_detect_live`, guided demo `t2_guided`, limits `t2_matrix`) | **COMPLETE 2026-11-15**: guided demo all 4 steps OK (31 fps); limits: 80 deg up to 75 cm, 75 deg at 1 m, distance limit not reached in 1 m (~4.2 m predicted) | summary: `output/detection/T2_summary.md` |
| T3 calibration | done (`t3_make_board`, `t3_calibrate`, guided capture) | **real calibration done 2026-11-10** (20 views, RMS 1.18 px, accepted, see 2) | optional re-capture with a flat board |
| T3 recording and poses | done (`t3_record` with truth helper, `t3_pose`, `t3_check_invariance`) | **COMPLETE 2026-11-15**: check B camera invariance (RMS 2.7 mm while the camera moved ~140 mm); check A 5 positions E1_p1-p5: translation error median 8.8 mm (3.8-13.5 per position), rotation 4.0 deg, jitter 0.2-0.5 mm | reports: `output/pose/invariance_1_invariance.md`, `output/pose/E1_checkA.md` |
| T4 animation | done (`t4_animate` export + live, guided recording `t4_record_guided`) | **COMPLETE 2026-11-15**: export `output/animation/T4_motion_1_animation.mp4` (30 s, synchronised with the camera video); live mode works (F and M frames in 3D, curves; screenshots `output/animation/live_*.png`) | - |
| T5 network | done (`t5_make_data`, `t5_train` + live dashboard `t5_dashboard`, `netpose`) | **synthetic data set made 2026-11-15**: `training/synthetic/synth_v1` (10000/1000/1000, 640x360, calibrated camera, L 77 mm, 0.25-1.0 m, tilt <= 60 deg; label check: classical PnP 3.8 mm / 0.33 deg) | training run(s), real fine-tuning, T5 check on unseen recordings |
| T6 comparison | done (`t6_evaluate`) | machinery tested with a throwaway network | real test recordings with measured truth |
| Bonus (+10) | done (`t7_bonus_occlusion`) | synthetic: 18.7 mm at 0.5 s vs 89 mm for hold; lost rule and re-acquisition OK | recordings with an obstacle and a known path |

**Not live-tested** (only the code behind them): `t3_calibrate capture`, `t3_record`, `t4_animate live`, the key controls of those windows.

## 2. Decisions made

- Dictionary `DICT_6X6_250`, F = id 0, M = id 23, black square 80 mm intended, 2-cell white quiet zone.
- **Printed sizes accepted as measured (2026-11-10):** both markers 77 mm, chessboard square 24 mm. The printer scaled every page to about 96 %,
  so the T1 check (printed side within 1 mm of the intended 80 mm) is NOT met. We keep these prints and use the measured values
  (`data/markers.json` measured_side_mm = 77.0, `data/calibration/board.json` measured_square_mm = 24.0); record in
  `output/calibration/measurements.csv`. State this in the report. Measured to the whole millimetre only.
- **Calibration accepted as is (2026-11-10):** 20 views, model k1, k2; RMS 1.18 px (all views), held-out 0.58 px; fx 846.6, fy 845.2,
  cx 641.1, cy 361.8 px (horizontal FOV 74 deg); k1 +0.092, k2 -0.135; coverage good, tilts 8-38 deg (mostly 20-32).
  Known limitation, to state in the report: the printed board was on a clipboard and bent by the fingers; the worst views
  (view_019, view_014: 2.44 px) show systematic residuals; leave-one-out fx spread 833-860 px (3 %), i.e. an expected scale error of
  about 1-3 % in distances. Without the 4 worst views: RMS 0.82 px, fx 839.1 (not used). Files: `data/calibration/camera.npz`,
  `output/calibration/` (report.json, per_view.csv, calibration_report.png). A re-capture with the print glued flat would improve it.
- Camera: built-in laptop camera, index 0, **1280x720** for everything (the calibration is valid only for this resolution, zoom and focus).
- Calibration: chessboard 9x6 inner corners, 25 mm squares; simplest distortion model (k1, k2) first; held-out check.
- Classical pose: `solvePnP` IPPE_SQUARE + LM, planar ambiguity handled by temporal continuity; F held only if the camera is still.
- Network: ResNet-18 (ImageNet-pretrained), grey input 640x360, output 6-D rotation + (x/Z, y/Z, log Z); loss = chordal rotation + L1 translation (weights 1 and 1 to start; to be chosen by comparing runs).
- Data separation: training in `training\` (synthetic, real training frames, models); test recordings with measured truth only in `data\recordings\`.
- Folders: `data\`, `output\`, `training\`, all defined in `code\config.py`; defaults next to `code\`, so the project is movable.
- Environment: Python 3.12, `.venv`; PyTorch 2.11 (CUDA 12.8 build) works on the RTX 5080 laptop GPU (16 GB).
- Backup of the project before the reorganisation: `Machine Vision\Lab 2 - backup before reorganisation`.

## 3. Next steps (in order)

**A. Tomorrow - needs the printer**
1. ~~Print and measure~~ done 2026-11-10 (markers 77 mm, square 24 mm, values written to the config files). Glue on rigid, flat boards if not done yet.
2. Keep these prints for the whole project: a reprint changes L (re-measure), and a new chessboard needs a new calibration.
3. ~~Calibrate~~ done 2026-11-10 (20 views, RMS 1.18 px, accepted). Do not change resolution, zoom or focus from now on.
4. ~~T2~~ done 2026-11-15 (limits matrix 2026-11-10, guided live demo 2026-11-15; summary `output/detection/T2_summary.md`).
5. T3 poses: short recording (`t3_record.py`), `t3_pose.py`, camera-invariance check (move the laptop; the relative pose must stay constant).

**B. With the calibration done (do together)**
6. Real synthetic data set: `python t5_make_data.py` (10000 / 1000 / 1000, about 2 min, 2.8 GB).
7. Start the training: `python t5_train.py`; compare a few runs (loss weights, epochs) for the report.

**C. Recordings**
8. Training frames: `t3_record.py` with tag `train` (M close, sharp, lit; several recordings) -> `training\real\`; label with `t3_pose.py`. No ruler needed.
9. Test recordings with **measured truth** (`<name>.truth.csv`): E1 (5 positions x 3 angles x 100 frames), E2 (3 distances, tilts 0-60), E3 (known path, 3 x 20 s), E4 (blur, dim light, occlusion, clutter).
10. Bonus recordings: M behind an obstacle, at least 5 passes at two speeds, known constant-speed path.

**D. Finish**
11. `t6_evaluate.py` -> comparison table, worst frames, E1-E6; `t7_bonus_occlusion.py` -> E7.
12. T4 export for the final video (both pipelines side by side), report (<= 12 pages, contribution statement per student), 2-3 min video, presentation 10 + 5 min.

## 4. Open items and known gaps

- **Fine-tuning on real frames is not implemented**: `t5_train.py` trains on synthetic data only. Add it once there are `training\real` recordings (load them, label with the classical poses, continue training at a lower learning rate, split by recording).
- The optional learned occlusion predictor (MLP/GRU) of the bonus is not implemented; the Kalman + alpha-beta predictor is.
- The synthetic renderer has no motion blur, lens distortion, real backgrounds or glare: expect a synthetic-real gap in E5 (to be reported).
- A network is valid only for the camera K and the marker size L it was trained with: always train after calibration and after measuring L.
- Distinguish in the report: jitter (averages out) vs bias (does not); show the worst frames; every number must come from the code and the README.
- Cleanup left to do by hand: a leftover test folder `mv_selftest_...` in the Windows temp folder (harmless), and the backup folder next to the project once it is no longer needed.
- Report rules: state every library and AI assistant used; no test frame in training or tuning; one track per student (A: classical and calibration, B: data and network, C: truth, evaluation, animation).

## 5. Checkpoints (assignment)

| When | Show |
|---|---|
| End of week 1 | live demo T1-T3, calibration file with RMS, first classical E1 numbers |
| End of week 2 | T4 animation live, synthetic data set, first network with E5 results |
| Final (week 3) | code + README, report, video, data, presentation |

## 6. Change log

- 2026-11-15: two-stage `crop_resnet18_v1` trained (80 epochs, 27 min). End to end on 1000 new full-resolution synthetic images:
  rotation 7.6 -> 1.7 deg, translation 14.3 -> 9.7 mm, usable 50 -> 83 % against resnet18_v2. **Chosen network: crop_resnet18_v1.**
  Details in `output/network/T5_runs.md`.

- 2026-11-15: two-stage network (best-gain package): stage 1 = resnet18_v2 finds M; stage 2 = ResNet-18 on a 224 px crop around M cut from
  the full-resolution image (marker ~112 px instead of ~65 px), input grey + 2 viewing-direction channels; 30000 crops rendered at full
  resolution with simulated stage-1 errors; stronger augmentation (noise, blur); gradient clipping 5. Code: `t5_make_crops.py`, `netpose.py`,
  `t5_train.py`. Bug fixed: the E5 synthetic evaluation undistorted distortion-free synthetic images. Run `crop_resnet18_v1` (80 epochs) started.

- 2026-11-15: T5 run 3 `resnet18_v3` (40 epochs, w_rot 2): rotation 8.9 deg, translation 16.7 mm - worse than run 2. Chosen network:
  **resnet18_v2** (w_rot = w_trans = 1, 40 epochs). Comparison table for the report: `output/network/T5_runs.md`.

- 2026-11-15: T5 run 2 `resnet18_v2` (40 epochs, w_rot 1, 20.7 min): validation rotation 7.3 deg (p95 33), translation 13.5 mm (p95 75)
  - training longer was the main gain against run 1. Run 3 (w_rot 2) was interrupted by a system update at epoch 9 (no checkpoint yet) and restarted.
  Training now saves a checkpoint after every epoch and resumes from it (tested with a simulated crash).

- 2026-11-15: T5 run 1 `resnet18_v1` (synth_v1, 15 epochs, batch 64, lr 5e-4, w_rot 1, w_trans 1, pretrained; 7.5 min on the RTX 5080):
  validation translation 26 mm median (p95 134), rotation 23 deg (p95 61). Rotation error is the TILT (normal 22 deg; in-plane only 4.3 deg),
  systematically underestimated (predicted 22 vs true 35 deg median); mirror (planar-ambiguity) choice only 16 %; rotation loss still falling
  -> run 2 planned: 40 epochs, w_rot 2 (also the loss-weight comparison); option: crop around M.

- 2026-11-15: live training dashboard (`t5_dashboard.py`); final synthetic data set `synth_v1` generated with the calibrated camera (45 s, 2.8 GB).

- 2026-11-15: T4 live mode tested on the camera (3D box now always includes F, on-screen screenshot confirmation, instruction bar). T4 complete.

- 2026-11-15: T4 - guided recording window for M moving (instructions + countdown on the camera window); animation exported;
  3D view oriented like the camera (F's y up, x right, z toward the viewer), solid trajectory, predicted segments dashed.

- 2026-11-15: T3 check A - M at 5 ruler-measured positions (gaps 50-250 mm, boards in one plane, `data/recordings/E1_p1..p5` + truth files);
  error mostly a depth bias along F's normal (+3 to +9 mm) and a 3.4 % short spacing at the widest position. T3 complete.

- 2026-11-15: T3 check B (camera invariance) on the real camera: recording `data/recordings/invariance_1` (50 s), report
  `output/pose/invariance_1_invariance.md`. Error mostly along F's normal (z, RMS 2.6 mm) - markers nearly frontal (planar ambiguity flagged in 89 % of the frames).

- 2026-11-15: T2 complete - guided live demo (step list, checkpoints, safeguards against moving the wrong marker) passed all 4 steps;
  limit experiments done 2026-11-10 with the guided matrix. T2 summary written for the report.

- 2026-11-10: markers and chessboard printed (scale about 96 %); the measured 77 mm / 24 mm accepted and written to the config files.
  T2 now recognises markers generated in earlier sessions (`--markers-folder`).
- 2026-11-10: guided calibration capture (instructions, angle panel, fit bar); real camera calibrated (20 views, RMS 1.18 px) and accepted
  with its known limitation (board not flat).
- 2026-11-08: read the assignment and lectures; set up venv and `config.py`; T1 (markers, dictionary sheet) and T2 (live detection) written, T2 confirmed working on the camera.
- 2026-11-08: reorganised folders (task-prefixed scripts, `lecture_reference\`, `docs\`); wrote T3-T6, bonus, `selftest.py`, `print_sheet.py`; installed PyTorch with CUDA; moved training into `training\`; chessboard and print sheet generated.

### 2026-11-15 - real training frames + fine-tuning (T5)
- Recorded 5 real training recordings in `training/real_v2` (`t4_record_guided.py --plan train --seconds 36`, completion bar instead of countdown); F at 41 cm (01) and 31 cm (02-05).
- `t5_finetune.py`: 3164 train / 589 val classical-labelled frames; fine-tuned `resnet18_v2_ft` (4.5 min) and `crop_resnet18_v1_ft` (1.0 min), half real / half synthetic per batch.
- Dashboard now also saves the example photos of every epoch (`output/network/<model>/examples/epoch_NNN.png`).
- Real test (E1, T4, never trained on): two-stage fine-tuned 10.0 mm / 5.7 deg on E1 (was 26.5 mm / 9.5 deg), T4 difference to classical 2.7 deg (was 142 deg). Classical still best: 8.8 mm / 4.0 deg, jitter 0.4 mm. Details: `output/network/T5_runs.md`.
- Open: E1_p4 outliers (stage 1 jumps on that view).
- Prepared round 2 (tomorrow): empty `training/real_v3` with `INSTRUCTIONS.md`. New options: `t4_record_guided.py --set real_v3` (7th train move: M right next to F, aimed at E1_p4) and `t5_finetune.py --sets real_v2 real_v3 --suffix _ft2` (keeps the `_ft` networks, asks before overwriting).

### 2026-11-16 - round 2: real_v3 recordings + fine-tuning `_ft2` (T5)
- Recorded `real_v3_01..05` (42 s each, pause/resume with SPACE, S = stop and save; F at 46.4 / 60.1 / 67 / 40 / 55 cm, camera cross-check in each .json; 05 = laptop tilted ~6.6 deg, validation).
- First attempt (no augmentation) stopped at epoch 7: stage 1 put M ~82 px too low on the tilted recording (it used the room as reference).
- Fixes: camera-rotation augmentation for stage 1 (`t5_train.rotate_camera`, `rot_aug`), wider crop offsets for stage 2 (`CROP_JITTER` 0.25), two-pass crop at inference (`NetPose(passes=2)`), best validation epoch kept (`keep_best`), batch 128 + cudnn benchmark.
- `t5_finetune.py` cleaned: checks before work, labels reused per recording (`--relabel`), `--yes`, preparation window (steps, bar, one picture per 100 frames); dashboard shows the validation numbers as text and new preview frames every epoch.
- Result (E1 + T4, never trained on): two-stage `_ft2`, 2 pass = 9.2 mm / 3.9 deg (p95 17.8 mm / 6.6 deg), jitter 2.1 mm, E1_p4 2.8 mm, T4 difference to classical 4.3 mm / 2.1 deg. Classical: 8.8 mm / 4.0 deg, jitter 0.4 mm. Details: `output/network/T5_runs.md`.
- Use `crop_resnet18_v1_ft2.pt` (with `resnet18_v2_ft2.pt`) as the network for T6.
- Open: optional extra tilted recordings (training + a tilted test recording); validation on several recordings would need a small change in `t5_finetune.py`.

### 2026-11-16 - round 3: real_v4 recordings (tilted / turned camera) + fine-tuning `_ft3` (T5)
- Recorded `real_v4_01..07` (camera pitched inward / outward and yawed left / right at 44-72 cm; 03 and 07 = validation). Turn directions checked against the picture (heater visible = turned right: 02, 03, 07).
- New: hard-case augmentation (`t5_train.hard_cases`, 40 %: cover, motion blur, low resolution, dim light); validation on several whole recordings (`--val`); separate stage-2 epochs (50); data sets cached (`build_info.json`), `--rebuild`, `--restart`; images kept in RAM when the GPU would overflow (the first attempt froze at 15.9 / 16.3 GB).
- Result (E1, never trained on): two-stage `_ft3`, 2 pass = 6.7 mm / 3.56 deg (p95 12.5 mm / 6.3 deg), jitter 1.26 mm; classical 8.8 mm / 3.96 deg, jitter 0.38 mm; `_ft2` 9.2 mm / 3.88 deg. T4 difference to classical 4.3 mm / 2.2 deg.
- Usability (E1 frames degraded in software): 30 % of M covered -> classical 0 %, `_ft3` 91 % usable; motion blur 31 px -> classical 0 %, `_ft3` 100 %; very dark -> both 0 %. Details: `output/network/T5_runs.md`.
- Use `crop_resnet18_v1_ft3.pt` (with `resnet18_v2_ft3.pt`) for T6. Open: real hard-case test recordings with truth (covered, shaken, dim, tilted camera) for T6.
- Saved for the paper (2026-11-16): evaluation scripts moved into the project (`code/t5_eval_real.py`, `code/t5_test_hard_cases.py`), figures in `output/network/figures/` (augmentation examples, real_v4 first frames, degraded test frames), console logs of the three fine-tuning rounds in `output/network/logs/`. Per network: `output/network/<name>/` = train_log.json, training_curves.png, dashboard.png, examples/epoch_NNN.png.

### 2026-11-16 - bonus E7 (occlusion): recordings and first results
- Setup: F right, ruler along the table, M slides from the start stop (left corner at ruler 0, 377 mm centre to centre) to the end stop (~273 mm); tin box in front hides M from ~90 to ~265 mm.
- New tools: `t7_setup_guided.py` (live setup check with path / obstacle overlay), `t4_record_guided.py --plan bonus --pass-s` (pacing bar + beeps), `t7_passes.py` (finds and measures passes), `t7_virtual_occlusion.py` (uncovered passes hidden in software), `t7_real_occlusion.py` (real obstacle: error at reappearance).
- Predictor change: the turning speed fades out while hidden (`rot_tau` 0.1 s) - rotation error 42 -> ~11 deg on the fast uncovered passes.
- Recordings: E7_nocover_slow_2 / _fast (4 + 7 passes), E7_cover_slow_3 / _slow_4 / _fast (19 covered passes, 10 clean; hidden 0.9-1.9 s, two speed groups ~10 and ~15 cm/s). E7_cover_slow / _slow_2 are too slow (hidden 3-6 s).
- Result: error at reappearance 35-41 mm (predictor) against 176 mm (hold the last pose); rotation no gain (measurement noise). Summary: `output/bonus/E7_summary.md`.
- Open: animation of a covered pass with the predicted segment (t4_animate), optional: network poses during partial cover as extra measurements.
- Bonus animation: `code/t7_animate.py` -> `output/bonus/E7_cover_slow_4_occlusion.mp4` / `.gif` (predicted segment dashed, 3 sigma circles, LOST caption) and `_predicted_poses.csv`.
- Bonus round 2 (evening): layout 2 (start moved, bulb box 7 cm), recorder with a moving 'keep M here' outline and save-on-close, animation with red predicted box and pixelated surroundings (face hidden). Recordings E7_box_slow / E7_box_fast: error at reappearance ~32 mm (10 cm/s) and ~50 mm (13 cm/s) against ~165 mm for holding - the same level as round 1; the limit is the hand's changing speed. Details in `output/bonus/E7_summary.md`.
- CORRECTION to the bonus numbers: `t7_real_occlusion.py` measured the hidden stretch from the first to the last missing frame of a pass, so single dropped frames shifted it (passes wrongly called 'lost at the start', some reappearance errors too small). Now the longest run of missing frames is used. Corrected: predictor / hold at reappearance = 50 / 174 mm (~10 cm/s, 1.6 s hidden, 14 passes), 96 / 182 mm (~14 cm/s, 1.2 s, 18 passes), 111 / 158 mm (~4.5 cm/s, 3.5 s, 7 passes: E7_box_5cms recorded with countdown + outline). The earlier '35-41 mm, a fifth of the baseline' is superseded. See the last section of `output/bonus/E7_summary.md`.
- Bonus: second slow set `E7_box_5cms_2` (7 passes, ~5.5 cm/s, hidden ~2.9 s): predictor / hold at reappearance 55 / 157 mm (first slow set 111 / 158 mm); 5.9 / 27.7 mm at 0.5 s, 15.2 / 53.5 mm at 1 s. Recorder: countdown 3-2-1 before GO. Tested a straight-path option for the predictor (`line_s`, `--line`): mixed results, off by default.
- Bonus animation of the slow set: `output/bonus/E7_box_5cms_2_occlusion.mp4` / `.gif` (passes 4-7); `t7_animate.py` now cuts the person out by default (`--pixelate`, `--show-all`) and has `--start`.

### 2026-11-16 (night) - Task 6: test recordings and comparison
- New tools: `t6_record.py` (one command per test recording, table truth / camera / difference, `--check` live mode), `t4_record_guided.py --plan motion` (E3: outline goes there and back, writes the truth file from its schedule), `t7_live.py` (live classical + predictor view).
- Recorded with ruler / protractor truth: E1_turn20, E1_turn40; E2_d40/d60/d75 x t0/t40; E3_motion_01..03; E4_dim, E4_cover, E4_clutter, E4_blur, E4_blur_2.
- `t6_evaluate.py` with `crop_resnet18_v1_ft3.pt` -> `output/evaluation/` (table, report, worst frames). Classical / network: E1 7.8 / 8.1 mm, 3.84 / 4.04 deg, jitter 0.55 / 1.29 mm; E3 6.6 / 6.3 mm, 5.82 / 5.21 deg; E4 output 80 / 100 % (partial cover: 0 / 100 %); 170 / 30 fps.
- Finding: a marker sheet cut to ~1 mm of white margin gives a false 18 deg lean on a dark background (fixed with a white backing); affects the bonus rotation readings. Details: `output/evaluation/T6_notes.md`.
- Open: report, video, presentation; backup of the project.
- Small finishes before the paper (2026-11-16, night): `ASSIGNMENT_CHECKLIST.md` (every requirement of the PDF with its status); E1 numbers for the untouched recordings (T6_notes.md); bonus horizon (virtual_occlusion.md); `t6_side_by_side.py` (side-by-side videos + T4 animation with the network's poses); `t6_evaluate.py --network ... --yes` (one command); README rewritten as 49 numbered one-command steps (the old text is in `docs/README_details.md`); `training/README.md` describes the training data.
- GitHub repository danvalentin13/machine-vision-marker-pose-classical-aruco--vs-resnet (2026-11-16): code, text results and computer-generated images only. Privacy rule in .gitignore: no camera recording, frame or screenshot is published; camera-free animations are in output/public (code/make_public_media.py). Commits use the GitHub noreply address. This line and later changes are not pushed yet.
