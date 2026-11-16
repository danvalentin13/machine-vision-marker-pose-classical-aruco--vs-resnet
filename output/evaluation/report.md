# Task 6 - comparison

Real test set; median with the 95th percentile in brackets.

| Method | E1 e_t [mm] | E1 e_R [deg] | E1 jitter [mm] | E3 e_t [mm] | E3 e_R [deg] | E4 output [%] | E6 [fps] |
|---|---|---|---|---|---|---|---|
| Classical | 7.8 (13.6) | 3.84 (6.04) | 0.55 | 6.6 (12.4) | 5.82 (9.54) | 80 | 147 |
| Network | 8.1 (13.0) | 4.04 (6.48) | 1.29 | 6.3 (13.7) | 5.21 (7.11) | 100 | 25 |

## E1 static accuracy (bias and jitter per recording)

| recording | method | bias [mm] | jitter [mm] | e_t median [mm] | e_R median [deg] |
|---|---|---|---|---|---|
| E1_p1 | classical | 8.77 | 0.35 | 8.77 | 5.25 |
| E1_p1 | network | 6.81 | 1.08 | 6.78 | 5.00 |
| E1_p2 | classical | 6.09 | 0.46 | 6.03 | 6.02 |
| E1_p2 | network | 3.24 | 1.11 | 3.13 | 6.06 |
| E1_p3 | classical | 10.67 | 0.41 | 10.68 | 3.96 |
| E1_p3 | network | 7.88 | 1.61 | 8.07 | 3.48 |
| E1_p4 | classical | 3.75 | 0.18 | 3.76 | 1.95 |
| E1_p4 | network | 3.10 | 1.43 | 3.09 | 1.87 |
| E1_p5 | classical | 13.44 | 0.49 | 13.48 | 3.40 |
| E1_p5 | network | 12.21 | 1.08 | 12.22 | 3.25 |
| E1_turn20 | classical | 3.93 | 0.97 | 4.06 | 2.07 |
| E1_turn20 | network | 10.98 | 1.29 | 11.13 | 3.97 |
| E1_turn40 | classical | 7.83 | 1.00 | 7.85 | 3.90 |
| E1_turn40 | network | 12.29 | 1.42 | 12.22 | 6.28 |

## E2 distance and tilt

| recording | distance [cm] | tilt [deg] | method | output [%] | e_t median [mm] | e_R median [deg] |
|---|---|---|---|---|---|---|
| E2_d40_t0 | 40.0 | 0.0 | classical | 100 | 8.1 | 4.44 |
| E2_d40_t0 | 40.0 | 0.0 | network | 100 | 7.9 | 3.40 |
| E2_d40_t40 | 40.0 | 40.0 | classical | 100 | 8.0 | 3.61 |
| E2_d40_t40 | 40.0 | 40.0 | network | 100 | 11.2 | 5.18 |
| E2_d60_t0 | 60.0 | 0.0 | classical | 100 | 8.4 | 4.15 |
| E2_d60_t0 | 60.0 | 0.0 | network | 100 | 6.4 | 2.94 |
| E2_d60_t40 | 61.0 | 40.0 | classical | 100 | 7.5 | 3.85 |
| E2_d60_t40 | 61.0 | 40.0 | network | 100 | 11.8 | 6.50 |
| E2_d75_t0 | 75.0 | 0.0 | classical | 100 | 8.7 | 4.38 |
| E2_d75_t0 | 75.0 | 0.0 | network | 100 | 12.4 | 2.48 |
| E2_d75_t40 | 75.0 | 40.0 | classical | 100 | 7.7 | 4.31 |
| E2_d75_t40 | 75.0 | 40.0 | network | 100 | 25.1 | 10.09 |

## E3 motion (known path)

| recording | method | RMSE [mm] | median [mm] | max [mm] | e_R median [deg] | e_R max [deg] |
|---|---|---|---|---|---|---|
| E3_motion_01 | classical | 9.0 | 6.5 | 58.8 | 6.02 | 16.0 |
| E3_motion_01 | network | 9.1 | 6.9 | 20.4 | 5.39 | 10.3 |
| E3_motion_02 | classical | 9.0 | 7.4 | 58.7 | 5.68 | 13.6 |
| E3_motion_02 | network | 7.9 | 6.7 | 16.4 | 5.20 | 7.7 |
| E3_motion_03 | classical | 8.4 | 5.6 | 59.4 | 5.72 | 12.8 |
| E3_motion_03 | network | 6.5 | 5.5 | 14.4 | 5.11 | 7.8 |

## E4 robustness

| recording | condition (notes) | method | output [%] | e_t median [mm] | e_R median [deg] |
|---|---|---|---|---|---|
| E4_blur | mild shake (too gentle to blur the picture: the classical detector kept 100 %): the table / laptop shaken gently during the recording (the markers do not move relative to each other), normal light, the clutter objects still in place; M left of F: centre to centre 135 mm (ruler), M 3.5 mm higher, camera ~61 cm | classical | 100 | 3.1 | 3.22 |
| E4_blur | mild shake (too gentle to blur the picture: the classical detector kept 100 %): the table / laptop shaken gently during the recording (the markers do not move relative to each other), normal light, the clutter objects still in place; M left of F: centre to centre 135 mm (ruler), M 3.5 mm higher, camera ~61 cm | network | 100 | 12.2 | 3.43 |
| E4_blur_2 | motion blur, stronger: the laptop screen (camera) wobbled quickly by hand during the recording (the markers do not move relative to each other), normal light; M left of F: centre to centre 135 mm (ruler), M 3.5 mm higher, camera ~61 cm | classical | 100 | 4.3 | 5.49 |
| E4_blur_2 | motion blur, stronger: the laptop screen (camera) wobbled quickly by hand during the recording (the markers do not move relative to each other), normal light; M left of F: centre to centre 135 mm (ruler), M 3.5 mm higher, camera ~61 cm | network | 100 | 12.9 | 5.04 |
| E4_clutter | clutter: a tape measure, a roll of tape, a highlighter, a glue stick, a jar and a paper star in front of and between the markers (nothing covers them), normal light; M left of F: centre to centre 135 mm (ruler), M 3.5 mm higher, same plane, camera ~61 cm | classical | 100 | 3.0 | 4.25 |
| E4_clutter | clutter: a tape measure, a roll of tape, a highlighter, a glue stick, a jar and a paper star in front of and between the markers (nothing covers them), normal light; M left of F: centre to centre 135 mm (ruler), M 3.5 mm higher, same plane, camera ~61 cm | network | 100 | 16.5 | 2.87 |
| E4_cover | partial occlusion: the left vertical edge of M covered by a paper strip (about 20 % of the black square), normal light; M left of F: centre to centre 135 mm (ruler), M 3.5 mm higher, same plane, camera ~61 cm | classical | 0 | nan | nan |
| E4_cover | partial occlusion: the left vertical edge of M covered by a paper strip (about 20 % of the black square), normal light; M left of F: centre to centre 135 mm (ruler), M 3.5 mm higher, same plane, camera ~61 cm | network | 100 | 32.6 | 2.70 |
| E4_dim | dim light; M left of F: gap between the black squares 58 mm (centre to centre 135 mm, ruler), M centre 3.5 mm higher, same plane, both facing the camera at ~61 cm; M on a white backing sheet | classical | 100 | 3.7 | 3.66 |
| E4_dim | dim light; M left of F: gap between the black squares 58 mm (centre to centre 135 mm, ruler), M centre 3.5 mm higher, same plane, both facing the camera at ~61 cm; M on a white backing sheet | network | 100 | 12.9 | 4.33 |

## E5 synthetic held-out set (exact truth, camera frame) and the synthetic-real gap

| method | output [%] | e_t median (p95) [mm] | e_R median (p95) [deg] |
|---|---|---|---|
| classical | 87 | 3.8 (25.2) | 0.33 (3.91) |
| network | 100 | 11.8 (81.2) | 1.96 (6.80) |

## E6 cost (same laptop)

- classical_ms_per_frame: 6.82
- loc_classical: 266
- loc_network: 770
- classical_fps: 146.68
- network_gpu_ms: 32.19
- network_cpu_ms: 44.30
- network_fps_gpu: 31.06
- network_fps_cpu: 22.57
- pipeline_network_ms_per_frame: 39.83
- model_params_M: 11.18
- model_file_MB: 44.80
- train_minutes: 13.58
- train_device: cuda
- n_train_images: 11562
