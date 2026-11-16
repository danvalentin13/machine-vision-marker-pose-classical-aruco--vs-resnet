# training - everything for training the pose network (Task 5)

    synthetic\   rendered data sets (t5_make_data.py): one folder per set, e.g. synth_v1\ with X_/R_/t_ train, val, test + meta.json
    real\        real frames for fine-tuning: recordings with tag "train" (t3_record.py), labelled later by the classical pipeline (t3_pose.py)
    models\      trained networks (t5_train.py): <name>.pt = weights + the camera and marker they are valid for

Three sets, three roles (assignment):
  synthetic  exact labels                     -> training, validation, and E5 (its own test split)
  real       labels = classical T3 poses      -> training / fine-tuning ONLY (the labels carry classical errors)
  real TEST  recordings with hand-measured truth stay in data\recordings\ (NOT here) -> used only by t6_evaluate.py

Keeping the training frames here and the test recordings in data\recordings\ makes it impossible to mix them by accident.
Split by recording, never by frame: neighbouring video frames are almost identical.

## The real training recordings (description of the training data)

17 recordings made with `t4_record_guided.py --plan train` (laptop camera, 1280x720, 30 fps): F stands still, M is moved by hand
(close, far, edges of the picture, strong tilts, in-plane rotation, right next to F, free). Labels = the classical T3 pose of M
(ArUco + IPPE + LM), kept only where M is detected with a reprojection error of at most 1 px (`t5_finetune.py`, saved in
`<set>\labels`). 20450 frames recorded, 15287 labelled: 11562 for training (13 recordings), 3725 for validation
(4 whole recordings, never trained on). None of these recordings is a test recording; the test recordings are in `data\recordings\`.

| recording | frames | labelled | use | set-up (from the recording's notes) |
|---|---|---|---|---|
| real_v2_01 | 1075 | 916 | training | F fixed at 41 cm from the lens |
| real_v2_02 | 1079 | 774 | training | F fixed at 31 cm from the lens |
| real_v2_03 | 1074 | 738 | training | F fixed at 31 cm from the lens |
| real_v2_04 | 1074 | 736 | training | F fixed at 31 cm from the lens |
| real_v2_05 | 1074 | 589 | validation | F fixed at 31 cm from the lens |
| real_v3_01 | 1256 | 882 | training | start setup: marker planes 46.4 cm from the lens (corrected; 65 cm was a mistake), parallel to it (0 deg tilt); F left, M right; 10.8 cm between the b |
| real_v3_02 | 1258 | 1114 | training | F fixed at 60.1 cm from the lens (measured; camera estimate 61.0 cm), on the left of the picture |
| real_v3_03 | 1257 | 879 | training | F fixed at 67 cm from the lens (measured; camera estimate 69.2 cm), on the right of the picture |
| real_v3_04 | 1255 | 730 | training | F fixed at 40 cm from the lens (measured; camera estimate 41.1 cm), in the middle of the picture; room light on (other light than 01-03) |
| real_v3_05 | 1258 | 930 | validation | F fixed at 55 cm from the lens (measured; camera estimate 53.2 cm), laptop screen closed to about 80 deg (camera looks ~10 deg down); validation recor |
| real_v4_01 | 1252 | 773 | training | start setup: marker plane 58 cm from the lens (measured), screen hinge 75 deg (15 deg inward from vertical), laptop not turned; camera estimate: F at  |
| real_v4_02 | 1254 | 1111 | training | start setup: marker plane 71 cm from the lens (measured), laptop turned 15 deg to the RIGHT (corrected; in the picture F sits left of centre), screen  |
| real_v4_03 | 1254 | 1130 | validation | start setup: marker plane 53 cm from the lens (measured), laptop turned 12 deg to the RIGHT (corrected; in the picture F sits left of centre) and scre |
| real_v4_04 | 1256 | 749 | training | start setup: marker plane 44 cm from the lens (measured), laptop turned 8 deg to the LEFT, screen upright; camera estimate: F at 44.0 cm, seen pitched |
| real_v4_05 | 1260 | 938 | training | start setup: marker plane 50.5 cm from the lens (measured), screen opened 15 deg outward and laptop turned 15 deg to the LEFT; camera estimate: F at 4 |
| real_v4_06 | 1260 | 1222 | training | start setup: marker plane 70 cm from the lens (measured), screen opened 15 deg outward, laptop not turned; camera estimate: F at 68.4 cm, seen pitched |
| real_v4_07 | 1254 | 1076 | validation | start setup: marker plane 63 cm from the lens (measured), laptop turned 15 deg to the RIGHT (corrected), screen upright; VALIDATION recording; camera  |

real_v2 (2026-11-15): camera upright, F at 31-41 cm.  real_v3 (2026-11-16): F at 40-67 cm, left / middle / right, other light,
real_v3_05 with the screen pitched ~7 deg.  real_v4 (2026-11-16): camera pitched inward / outward and the laptop turned left /
right, 44-72 cm.  Distances and angles were measured by hand and cross-checked with the camera (both are in each .json).

Synthetic sets: `synthetic\synth_v1` (10 000 / 1 000 / 1 000 whole images 640x360, seed 0) and `synthetic\crop_v1`
(30 000 / 2 000 / 2 000 crops of 224 px), rendered with the calibrated camera: random pose, background, blur, noise, brightness.
`synthetic\real_v2_real_v3_real_v4_full` / `_crop` are the real frames above in the same file format (built by `t5_finetune.py`).
In fine-tuning every batch is half real, half synthetic; augmentation: camera rotation up to 8 deg (stage 1), crop offsets
(stage 2), and in 40 % of the images a hard case (part of the marker covered, motion blur, low resolution, dim light).
