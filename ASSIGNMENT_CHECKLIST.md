# Assignment checklist (against docs/MV_Assignment2_Marker_Pose.pdf) - 2026-11-16

Legend: DONE = result exists in the project; PARTIAL = exists but not exactly as asked; MISSING = not there yet.

## Tasks and checks

| Slide | Requirement | Status | Where / note |
|---|---|---|---|
| 4 | two markers, one dictionary, different IDs, L 50-100 mm | DONE | DICT_6X6_250, F id 0, M id 23, 77 mm |
| 4 | white quiet zone >= 1 cell | PARTIAL | M's sheet was cut to ~1 mm at the sides; fixed with a white backing for T6 (see output/evaluation/T6_notes.md); the bonus recordings were made before the fix |
| 4 | glued on a rigid board, measured with a calliper | PARTIAL | paper clipped / taped to a board, measured with a ruler (1 mm divisions) |
| 4 | calibration: >= 10 views, RMS and per-view errors | DONE | output/calibration/report.json, per_view.csv; RMS 1.18 px (accepted as is) |
| 4 | explain every reused starter function | MISSING | goes into the report |
| 6 | T1 check: printed side within 1 mm of the intended side | NOT MET | intended 80 mm, printed 77 mm; the measured 77 mm is used everywhere - state it in the report |
| 6 | T1: IDs and dictionary in the report | MISSING | report |
| 6 | T2: live detection, outlines + IDs, sub-pixel refinement, frame rate | DONE | t2_detect_live.py; 31 fps |
| 6 | T2 check: both IDs read in a live demo | DONE | output/detection/guided |
| 6 | T2 check: largest distance and tilt with >= 9 of 10 frames | PARTIAL | tilt limit found (80 deg to 75 cm, 75 deg at 1 m); distance limit not reached within the 1 m ruler (predicted ~4.2 m) |
| 7 | T3: calibration before pose work; IPPE_SQUARE + refine LM; relative pose; mm and roll-pitch-yaw with the convention stated | DONE (convention: report) | t3_pose.py, mvtools.py |
| 7 | T3: planar ambiguity, frames with one marker (hold F if the camera is still) | DONE | MarkerPoseEstimator |
| 7 | T3 check A: 5 measured positions, error per position | DONE | output/pose/E1_checkA.md |
| 7 | T3 check B: camera invariance | DONE | output/pose/invariance_1_invariance.md |
| 8 | T4: 3D view with both frames + trajectory, x y z and roll pitch yaw against time, live mode, export >= 20 s | DONE | t4_animate.py; output/animation/T4_motion_1_animation.mp4 (30 s) |
| 8 | T4 check: animation next to the camera video, synchronised, in the presentation | DONE (file) | to be shown in the presentation |
| 9 | T5: rotation not Euler (6-D), chordal loss; translation = log Z + normalised image coordinates | DONE | netpose.py |
| 9 | T5: loss weights reported and how they were chosen | DONE | output/network/T5_runs.md (w 1/1 against w_rot 2) |
| 9 | T5 check: the network runs in the T3 / T4 pipeline on test sequences never seen in training | DONE | t3_pose.py / t6_evaluate.py; output/animation/T4_motion_1_network_animation.mp4 (README step 6.6) |
| 10 | synthetic set >= 10 000 images, own K, backgrounds, blur, noise, brightness | DONE | training/synthetic/synth_v1 (10 000) + crop_v1 (30 000) |
| 10 | real training set labelled with the T3 poses | DONE | training/real_v2, real_v3, real_v4 (17 recordings) |
| 10 | real test set with independent truth | DONE | data/recordings E1-E4 (.truth.csv) |
| 10 | split by recording; no test frame for training OR for tuning | SEE 'Honesty points' | no test frame was trained on; some choices were made looking at E1_p1..p5 and T4_motion_1 |
| 11 | Bonus: >= 5 passes at two speeds, hidden 0.5-2 s | DONE | E7_cover_slow_3/_4/_fast, E7_box_slow/_fast (hidden 0.9-1.9 s); E7_box_5cms/_2 are slower (2.7-4.3 s) |
| 11 | Bonus: CV Kalman + alpha-beta SO(3); covariance grows; lost past a threshold (2L) | DONE | t7_bonus_occlusion.OcclusionPredictor |
| 11 | Bonus: gated re-acquisition, no jump | PARTIAL | gated; the jump at re-acquisition is measured (it equals the prediction error, 50-100 mm), it is not zero |
| 11 | Bonus: truth while hidden (known path at constant speed) | PARTIAL | ruler path, hand speed guided by an outline; exact truth only at reappearance + software-hidden uncovered passes |
| 11 | Bonus check: e_t, e_R against time hidden for both speeds; horizon where the error passes the threshold; jump | DONE | output/bonus/virtual_occlusion.md: errors against time hidden, horizon for 20 / 40 mm, jump (README step 7.5) |
| 11 | Bonus: predicted segment dashed with uncertainty in the T4 animation | DONE | output/bonus/*_occlusion.mp4 |
| 12 | E1: >= 5 positions and >= 3 angles, 100 frames each; bias and jitter per position | DONE | E1_p1..p5, E1_turn20, E1_turn40; output/evaluation/report.md |
| 12 | E2: 3 distances, tilts 0-60 deg; where each fails | PARTIAL | 41 / 61 / 77 cm x 0 / 40 deg; no 60 deg recording; neither method failed in this range |
| 12 | E3: known path, >= 3 recordings of 20 s; RMSE, median, max; plots | DONE | E3_motion_01..03; E3_*.png |
| 12 | E4: fast motion (blur), dim light, partial occlusion, clutter; output rate and error per condition | DONE | blur made by wobbling the camera, not by moving M fast |
| 12 | E5: held-out rendered images; synthetic-real gap | DONE | report.md |
| 12 | E6: fps, training time, model size, lines of code | DONE | report.md |
| 13 | comparison table (median, 95th percentile) | DONE | output/evaluation/comparison_table.md |
| 13 | worst frames of each method | DONE | output/evaluation/worst_frames.csv, worst/ |
| 13 | conclusion paragraph: which pipeline, why, where each fails | MISSING | report |

## Deliverables (slide 15)

| Deliverable | Status | Note |
|---|---|---|
| code in a repository | MISSING | the project is not a git repository; the grading also looks at the repository history |
| README that reproduces every number, fixed seeds | DONE | rewritten 2026-11-16: 49 numbered steps, one command each, with a table 'where each number comes from'; seed 0. Limits stated there: GPU training is not bit-for-bit repeatable, _ft / _ft2 are earlier rounds |
| report <= 12 pages: set-up, calibration, classical method, network and data, E1-E6, comparison table, failure cases, conclusion, contribution statement per student | MISSING | |
| video 2-3 min: markers and axes on the camera image, the T4 animation, both methods side by side | PARTIAL | the clips exist (output/evaluation/side_by_side_*.mp4, output/animation/*.mp4, output/bonus/*_occlusion.mp4); the 2-3 min video still has to be cut together |
| data: test recordings with measured truth; description of the training data | DONE | data/recordings (+ .truth.csv); training/README.md describes all 17 training recordings and the synthetic sets |
| presentation 10 + 5 min | MISSING | |

## Rules (slide 17) and honesty points for the report

1. State every use of libraries, pretrained backbones (ResNet-18, ImageNet) and AI coding assistants. Each student must be able to explain every line.
2. 'No test frame is used for training or hyperparameter tuning':
   - no test frame was ever trained on;
   - BUT the three fine-tuning rounds (_ft, _ft2, _ft3), the two-pass crop and the hard-case augmentation were compared on E1_p1..p5
     and T4_motion_1, and the software hard-case test used E1 frames: those recordings influenced design choices;
   - E1_turn20/40, all of E2, E3 and E4 were recorded AFTER the network was final: they are untouched.
   Say this in the report. The E1 numbers for the untouched recordings alone are in output/evaluation/T6_notes.md (last section).
3. Bonus: the fade-out of the turning speed (rot_tau) was chosen on the uncovered E7 recordings; the straight-path option was tested on
   the E7 recordings and not adopted. Say so.
4. The first real-obstacle numbers of the bonus were wrong (hidden stretch mis-detected) and are superseded: use only the corrected
   sections of output/bonus/E7_summary.md.
5. E3 truth = the schedule of an on-screen outline followed by hand (within 0-0.5 s, 2-5 mm).
6. Numbers must be reproducible from the README: the README has to be completed before hand-in.
