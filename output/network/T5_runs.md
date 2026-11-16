# Task 5 - training runs (synthetic data set synth_v1: 10000 train / 1000 validation images, 640x360)

ResNet-18 (ImageNet-pretrained, grey input), output 6-D rotation + (x/Z, y/Z, log Z); loss = w_rot * chordal ||R_hat - R||^2 + w_trans * L1;
AdamW, one-cycle learning rate (peak 5e-4), batch 64, seed 0, RTX 5080 laptop GPU. Errors on the 1000 synthetic VALIDATION images (no real test frame used).

| run | epochs | w_rot | w_trans | minutes | val rotation median (p95) [deg] | val translation median (p95) [mm] | final train loss (rot / trans) |
|---|---|---|---|---|---|---|---|
| resnet18_v1 | 15 | 1 | 1 | 7.5 | 23.5 (61) | 26.4 (134) | 0.4094 / 0.1087 |
| resnet18_v2 | 40 | 1 | 1 | 20.7 | 7.3 (33) | 13.5 (75) | 0.0145 / 0.0487 |
| resnet18_v3 | 40 | 2 | 1 | 20.3 | 8.9 (36) | 16.7 (95) | 0.0160 / 0.0628 |

- Run 1 -> run 2 (15 -> 40 epochs, same weights): rotation 23.5 -> 7.3 deg, translation 26.4 -> 13.5 mm. The longer schedule is the main gain;
  run 1 was under-trained (its rotation loss was still falling, the tilt was systematically underestimated).
- Run 2 -> run 3 (w_rot 1 -> 2, same 40 epochs): rotation 7.3 -> 8.9 deg, translation 13.5 -> 16.7 mm: doubling the rotation weight did NOT
  help, it made both errors slightly worse (one seed per run: differences of ~1-2 deg / 3 mm may partly be run-to-run variation).
- Choice: **resnet18_v2 (w_rot = w_trans = 1, 40 epochs)** - lowest validation error in rotation and translation.
- Gap: the final training rotation loss (~0.015, i.e. about 3.5 deg) is well below the validation error (7-9 deg): the network fits the
  training images better than new ones (generalisation gap); the 95th percentiles (33-36 deg) are the hard cases (small, far, steep or occluded markers).

## Two-stage network (crop around M) - `crop_resnet18_v1`

Stage 1 = `resnet18_v2` (whole 640x360 image) finds M; stage 2 = ResNet-18 on a 224x224 crop around M cut from the full-resolution
1280x720 image (window = 2 x the marker's apparent side, so the marker has ~112 px instead of ~65 px), input grey + the viewing direction
(x/Z, y/Z) of every crop pixel. Trained on 30000 synthetic crops (`crop_v1`, rendered at full resolution, centre and scale perturbed by
0.12 of the marker side like stage-1 errors), strong augmentation (gain, offset, noise, blur), gradient clipping 5, 80 epochs,
w_rot = w_trans = 1, batch 64, lr 5e-4; 27 min. Crop validation: rotation 1.95 deg (p95 6.3), translation 8.8 mm (p95 31).

**End to end** on 1000 NEW full-resolution synthetic test images (`synth_full_test`, 1280x720, never used for training or tuning):

| network | rotation median (p95) [deg] | tilt median [deg] | translation median (p95) [mm] | usable (e_t < 20 mm and e_R < 10 deg) |
|---|---|---|---|---|
| resnet18_v2 (whole image) | 7.6 (27.4) | 6.7 | 14.3 (87) | 50.5 % |
| **crop_resnet18_v1 (two-stage)** | **1.7 (6.9)** | **1.4** | **9.7 (41)** | **83.2 %** |

By distance (translation [mm] / rotation [deg], medians): 0.25-0.4 m 7.8/5.4 -> 5.1/1.5; 0.4-0.6 m 12.8/6.9 -> 9.3/1.6; 0.6-1.0 m 27.1/10.8 -> 16.7/2.0.
The crop removes most of the tilt error (the weak point of the whole-image network) and helps most for far markers.
Translation still comes partly from stage 1's centre estimate and grows with distance (depth from apparent size).

## Fine-tuning on real frames (real_v2, 2026-11-15)

Real training set: 5 recordings in `training/real_v2` (36 s each, F at 41 cm in real_v2_01 and 31 cm in the others, M moved by hand:
close, far, edges, strong tilt, in-plane rotation, free). Labels = classical poses (^C T_M) with reprojection error <= 1 px:
3164 training frames (real_v2_01..04) and 589 validation frames (real_v2_05). Fine-tuning: 15 epochs, lr 1e-4 (OneCycle), every batch
half real and half synthetic (synth_v1 / crop_v1), starting from resnet18_v2 and crop_resnet18_v1. Time: 4.5 min + 1.0 min (RTX 5080).

Validation (real_v2_05, last epoch): whole-image 8.2 deg / 14.3 mm; crop (centred on the label) 8.1 deg / 6.8 mm.

Test on the real recordings never used in training (`output/network/real_E1_comparison_ft.json`):

| method | E1 e_t median (p95) | E1 e_R median (p95) | jitter | T4 difference to classical (median / p95) | ms/frame |
|---|---|---|---|---|---|
| classical (ArUco + IPPE + LM) | 8.8 (13.6) mm | 3.96 (6.1) deg | 0.38 mm | - | 6.2 |
| resnet18_v2, synthetic only | 59.8 (190.9) mm | 11.88 (32.2) deg | 21.3 mm | 157 mm / 25.0 deg (230 / 53.6) | 19.3 |
| two-stage, synthetic only | 26.5 (106.5) mm | 9.46 (37.6) deg | 26.5 mm | 196 mm / 142.5 deg (281 / 174.2) | 33.7 |
| **resnet18_v2_ft** | 13.7 (135.8) mm | 5.83 (8.4) deg | 11.0 mm | 12.3 mm / 7.1 deg (33 / 12.6) | 21.1 |
| **two-stage, fine-tuned** | **10.0** (124.3) mm | **5.69** (36.7) deg | 16.3 mm | **7.2 mm / 2.7 deg** (13 / 5.3) | 33.7 |

Reading: the real frames closed most of the synthetic-to-real gap (two-stage E1 error 26.5 -> 10.0 mm, T4 difference 142 -> 2.7 deg).
The remaining tail comes from E1_p4 (e_t 34-103 mm, jitter 45-74 mm): stage 1 still jumps between positions on some frames of that view.
Classical remains more precise (jitter 0.4 mm against 11-16 mm) and faster.

## Fine-tuning round 2 (real_v2 + real_v3, 2026-11-16) -> `*_ft2.pt`

Real data: 10 recordings (real_v2_01..05, real_v3_01..04 for training = 7358 labelled frames; real_v3_05 = validation, 930 frames,
the only recording with the laptop screen tilted, ~6.6 deg). real_v3: F at 46 / 60 / 67 / 40 / 55 cm, 7 moves incl. "M right next to F".
Settings: 25 epochs, lr 1e-4, batch 128 (64 real + 64 synthetic), start from resnet18_v2 / crop_resnet18_v1, best validation epoch kept.
New in this round
- stage 1: camera-rotation augmentation (image warped by K Ra K^-1, labels turned with it; pitch/yaw up to 8 deg, roll 10 deg).
  Without it stage 1 placed M ~82 px too low on the tilted recording (it used the room as a reference): validation 13.9 deg / 92 mm.
  With it: 6.6 deg / 29 mm (epoch 8 kept).  28.8 min.
- stage 2: crop-window offsets 25 % of the marker size in the real set (was 12 %); epoch 21 kept: 3.7 deg / ~7 mm.  4.1 min.
- inference: two-pass crop (the window is re-centred once on the crop network's own answer), `NetPose(..., passes=2)` (default).

Test on the real recordings never used in training (`output/network/real_E1_comparison_ft2.json`):

| method | E1 e_t median (p95) | E1 e_R median (p95) | jitter | E1_p4 e_t | T4 difference to classical (median / p95) | ms/frame |
|---|---|---|---|---|---|---|
| classical | 8.8 (13.6) mm | 3.96 (6.1) deg | 0.38 mm | 3.8 mm | - | 5.4 |
| two-stage _ft, 1 pass (round 1 as reported) | 10.0 (124.3) mm | 5.69 (36.7) deg | 16.3 mm | 34.0 mm | 7.2 mm / 2.7 deg (13 / 5.3) | 28.0 |
| two-stage _ft, 2 pass | 7.3 (81.0) mm | 4.43 (25.5) deg | 18.6 mm | 7.0 mm (jitter 88) | 7.6 mm / 2.6 deg (11 / 4.7) | 27.4 |
| whole-image _ft2 alone | 15.8 (132.4) mm | 8.45 (14.5) deg | 10.1 mm | 99.4 mm | 28.8 mm / 5.2 deg (49 / 9.0) | 17.7 |
| two-stage _ft2, 1 pass | 12.0 (19.5) mm | 4.87 (7.5) deg | 8.4 mm | 8.1 mm | 4.3 mm / 2.1 deg (9 / 4.1) | 31.0 |
| **two-stage _ft2, 2 pass** | **9.2 (17.8) mm** | **3.88 (6.6) deg** | **2.1 mm** | **2.8 mm** | **4.3 mm / 2.1 deg (9 / 4.4)** | 27.1 |

Tilted-laptop recording real_v3_05, end to end against the classical labels (310 frames; it is the validation recording, so slightly
optimistic for _ft2): two-stage _ft 79.4 mm / 16.2 deg -> _ft2 1 pass 5.2 mm (p95 42) / 2.5 deg -> _ft2 2 pass 5.2 mm (p95 16) / 2.6 deg.

Reading: the outlier tail is gone (E1 p95 124 -> 18 mm, 37 -> 6.6 deg) and E1_p4 is solved (34 -> 2.8 mm). The learned pipeline now
matches classical accuracy on E1 (9.2 mm / 3.9 deg against 8.8 mm / 4.0 deg; the classical depth bias is inherited through the labels).
Classical is still ~5x steadier (jitter 0.4 against 2.1 mm) and ~5x faster. Stage 1 alone still fails on E1_p4 (99 mm): the crop
stage and its second pass recover it.

## Fine-tuning round 3 (real_v2 + real_v3 + real_v4, 2026-11-16) -> `*_ft3.pt`

Real data: 17 recordings; 13 for training (11562 labelled frames), 4 whole recordings for validation (3725 frames): real_v2_05 (upright),
real_v3_05 (screen pitched inward), real_v4_03 (pitch + yaw), real_v4_07 (yaw right, 63 cm). real_v4 = camera pitched / yawed at
44-72 cm (see training/real_v4/INSTRUCTIONS.md and the .json notes: measured distance and angles + the camera's own estimate).
Settings: lr 1e-4, batch 128 (half real / half synthetic), stage 1 25 epochs (35 min, epoch 22 kept: 5.8 deg / 15.4 mm),
stage 2 50 epochs (13.6 min, epoch 48 kept: 3.66 deg / 6.4 mm), camera-rotation augmentation (stage 1), crop offsets 25 %, two-pass crop.
New: hard-case augmentation in 40 % of the training images (`t5_train.hard_cases`): partial cover of the marker, motion blur,
low resolution, dim low-contrast light - the label stays valid, so the network learns poses where the detector gives none.
(Images are kept in RAM when they would fill the 16 GB card: the first attempt froze in epoch 3 with 15.9 GB used.)

Test on the real recordings never used in training (`output/network/real_E1_comparison_ft3.json`):

| method | E1 e_t median (p95) | E1 e_R median (p95) | jitter | E1_p4 e_t | T4 difference to classical (median / p95) | ms/frame |
|---|---|---|---|---|---|---|
| classical | 8.8 (13.6) mm | 3.96 (6.1) deg | 0.38 mm | 3.8 mm | - | 5.4 |
| two-stage _ft2, 2 pass | 9.2 (17.8) mm | 3.88 (6.6) deg | 2.12 mm | 2.8 mm | 4.3 mm / 2.1 deg (9 / 4.4) | 27.8 |
| whole-image _ft3 alone | 12.6 (88.0) mm | 6.90 (9.9) deg | 6.97 mm | 70.8 mm | 7.0 mm / 7.0 deg (18 / 9.8) | 18.4 |
| two-stage _ft3, 1 pass | 8.2 (12.8) mm | 3.74 (6.1) deg | 1.98 mm | 3.5 mm | 3.7 mm / 2.6 deg (8 / 4.8) | 32.8 |
| **two-stage _ft3, 2 pass** | **6.7 (12.5) mm** | **3.56 (6.3) deg** | **1.26 mm** | **3.1 mm** | 4.3 mm / 2.2 deg (8 / 4.8) | 28.1 |

Usability on degraded test frames (`output/network/hard_cases_E1.json`): 250 frames of E1_p1..p5, degraded in software; reference =
the classical pose of the clean frame; usable = within 20 mm and 10 deg of it.

| case | classical: output | classical: usable | _ft2 usable (median) | _ft3 usable (median) |
|---|---|---|---|---|
| clean | 100 % | 100 % | 100 % (3.7 mm / 1.1 deg) | 100 % (4.7 mm / 1.2 deg) |
| cover 15 % of M | 29 % | 28 % | 99.6 % (5.5 / 3.1) | 100 % (6.9 / 2.3) |
| cover 30 % of M | 0 % | 0 % | 51 % (10.9 / 9.0) | 91 % (5.9 / 5.1) |
| motion blur 15 px | 100 % | 100 % | 100 % (3.2 / 1.5) | 100 % (4.7 / 1.3) |
| motion blur 31 px | 0 % | 0 % | 14 % (20.0 / 11.5) | 100 % (8.9 / 4.1) |
| resolution / 4 | 100 % | 100 % | 99 % (3.6 / 1.5) | 100 % (3.9 / 1.0) |
| resolution / 6 | 100 % | 100 % | 100 % (4.4 / 1.7) | 100 % (4.2 / 0.9) |
| light x 0.25 | 97 % | 97 % | 100 % (6.7 / 4.3) | 100 % (5.0 / 1.9) |
| light x 0.10 | 0 % | 0 % | 0 % (209 / 19.8) | 0 % (124 / 31.2) |

Reading: on E1 the learned pipeline is now more accurate than classical against the measured truth (6.7 mm / 3.6 deg against
8.8 mm / 4.0 deg) - it averages out part of the classical depth bias it was trained on (z bias +2..+6 mm against +3..+9 mm).
Classical is still ~3x steadier (0.38 against 1.26 mm jitter) and ~5x faster. Usability: where the detector cannot decode the
marker (30 % covered, strong motion blur) classical gives nothing and _ft3 still gives a usable pose in 91-100 % of the frames;
the hard-case augmentation is what makes the difference (_ft2: 51 % / 14 %). Very dark frames (light x 0.10) defeat both.
Caveats: the degradations are simulated and of the same kind as the training augmentation; the reference there is the classical
pose, not the ruler truth. Real hard-case test recordings (covered, shaken, dim) are still to be made for T6.
