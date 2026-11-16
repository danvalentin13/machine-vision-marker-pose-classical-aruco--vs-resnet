# Task 3 - check A: M at 5 hand-measured positions (E1, classical pipeline)

Both boards upright against one straight edge (same plane), the edge turned about 20-30 deg from facing the camera; markers 34-37 cm from the lens
(measured by hand: P4 34 cm, P5 37 cm; the camera puts the marker centres at 35-43 cm); camera still; 100 frames per position. Truth = ruler: gap between the black squares (1 mm divisions) + side L = 77 mm;
height difference of the centres measured at P4 only (+3 mm), 0 elsewhere; printed surfaces at the same depth (checked).

| position | gap [mm] | truth x [mm] | camera - truth: x / y / z [mm] | translation error median (max) [mm] | rotation error median [deg] | jitter [mm] |
|---|---|---|---|---|---|---|
| P1 | 200 | -277 | -3.5 / +1.7 / +7.9 | 8.8 (9.4) | 5.25 | 0.35 |
| P2 | 150 | -227 | +0.5 / +1.2 / +5.9 | 6.0 (7.5) | 6.02 | 0.46 |
| P3 | 100 | -177 | -2.0 / +5.4 / +9.0 | 10.7 (11.7) | 3.96 | 0.41 |
| P4 | 50 | -127 | -1.7 / +1.5 / +3.0 | 3.8 (4.1) | 1.95 | 0.18 |
| P5 | 250 | -327 | +11.1 / +1.9 / +7.3 | 13.5 (14.1) | 3.40 | 0.49 |

- All 500 frames: translation error median **8.8 mm** (95 %: 13.6, max 14.1); rotation error median **3.96 deg** (max 6.3).
- Along the line of the markers (x, the direction measured best by hand): bias +0.9 mm on average (from -3.5 to +11.1); jitter 0.2-0.5 mm: the error is **bias, not noise** (averaging frames does not remove it).
- The largest component is the depth along F's normal (z): +3.0 to +9.0 mm although the printed surfaces are at the same depth:
  the depth of a near-frontal marker is the weakest direction of a planar pose (cf. check B); the boards' residual angle (2-4 deg, by hand) adds to it.
- P5 (largest spacing, markers near the image edges; 37 cm by hand, 42-43 cm to the marker centres by the camera): x +11 mm, i.e. the camera measures the spacing 3.4 % too short - the largest scale error, where lens distortion and the
  calibration error (RMS 1.18 px, fx spread 3 %) matter most.
