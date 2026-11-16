# Task 6 - how the test recordings were made, and what to keep in mind (2026-11-16)

Results: `comparison_table.md`, `report.md`, `summary.json`, `worst_frames.csv`, `worst/`, `E3_*.png` (from `python t6_evaluate.py`,
network = `training/models/crop_resnet18_v1_ft3.pt`, two-stage, 2 passes). No test recording was used for training.

## Recordings (data/recordings, each with .truth.csv measured by ruler / protractor)
- E1_static: E1_p1..p5 (2026-11-15: 5 gaps 50-250 mm, camera ~34 cm, 0 deg) + E1_turn20, E1_turn40 (M on its own support 90 mm in
  front of F, 115 mm to the side, 20 mm lower, turned 20 / 40 deg about the vertical axis, camera ~61 cm).
- E2_dist_tilt: E2_d40 / d60 / d75 x t0 / t40: the same M placement as E1_turn, only the laptop moved (camera 40.9 / 61.4 / 76.7 cm
  from F by the camera's own estimate; 75 cm is the farthest the room allows, 90 cm was planned).
- E3_motion: E3_motion_01..03 (20 s each): M on its box slides 60 mm along a ruler parallel to F's board and back (s 140 -> 200 mm,
  n 90 mm, 20 mm lower, facing the camera), 1 cm/s, rests of 2 s. Truth = the schedule of the on-screen outline the hand follows
  (`t4_record_guided.py --plan motion`) between the two measured stops. The hand followed it within 0.0-0.5 s / 2-5 mm along the path
  (checked with the classical track), so ~3-5 mm of the E3 error is the truth itself. Two failed takes are in `_discarded/`
  (too fast to follow; start position shifted by 26 mm).
- E4_robust (M 135 mm left of F, 3.5 mm higher, same plane, camera ~61 cm): E4_dim (dim light), E4_cover (left vertical edge of M
  covered by a paper strip, ~20 %), E4_clutter (objects around the markers), E4_blur (mild shake: no visible blur), E4_blur_2 (laptop
  screen wobbled by hand: ~30 % of the frames clearly blurred; the camera moved, the markers did not).
- Tools: `t6_record.py` (table: truth / camera now / difference, live check with `--check`), `t4_record_guided.py --plan motion`.

## Findings while recording
- White margin: M's sheet had been cut to ~1 mm of white at its left and right edges. On the black board the detector's corners
  slipped outward: the square was measured 9 % too wide, which the pose turned into a false 18 deg lean and 2 cm 'nearer'.
  A white backing sheet behind M fixed it (width/height 1.09 -> 1.01, lean 18 -> 2 deg, jitter 3-4 -> 0.7 mm). The same effect was
  present in the bonus recordings (M in front of the black board): part of the noisy rotation there.
- F had been re-clipped a quarter turn rotated after the E1 recordings of 2026-11-15; it was turned back for T6 (the truth is
  expressed in F's own directions).

## Reading of the table
- Fully visible marker: classical is steadier (jitter 0.55 against 1.29 mm) and ~5x faster (170 against 30 fps); position error
  E1 7.8 against 8.1 mm, E3 6.6 against 6.3 mm, angle about equal. A common ~8 mm in E2 for classical at every distance points to
  the ruler truth (and the +3..+9 mm depth bias seen in check A) rather than to the method.
- Network: equal to classical when M faces the camera up to ~60 cm; worse when M is far AND turned (E2_d75_t40: 25 mm / 10 deg);
  12-17 mm in the E4 recordings at 61 cm with M next to F.
- Partial cover (E4_cover): classical 0 % output, network 100 % (32.6 mm / 2.7 deg). E4 output rate over the 5 recordings: 80 % / 100 %.
- Dim light, clutter and hand-made blur did not break either method at ~61 cm.
- E5 (held-out synthetic): classical 87 % output, 3.8 mm / 0.33 deg; network 100 % output, 11.8 mm / 1.96 deg.

## E1 with and without the recordings that informed design choices

E1_p1..p5 (and T4_motion_1) were looked at when choosing between the fine-tuning rounds, the two-pass crop and the hard-case augmentation.
E1_turn20/40 and all of E2, E3, E4 were recorded after the network was final. Median (95th percentile); jitter = mean over the recordings.

| recordings | method | e_t [mm] | e_R [deg] | jitter [mm] |
|---|---|---|---|---|
| E1 as in the table (E1_p1..p5 + E1_turn20/40) | classical | 7.8 (13.6) | 3.84 (6.04) | 0.55 |
| E1 as in the table (E1_p1..p5 + E1_turn20/40) | network | 8.1 (13.0) | 4.04 (6.48) | 1.29 |
| E1 untouched only (E1_turn20/40) | classical | 6.2 (8.4) | 2.80 (4.20) | 0.99 |
| E1 untouched only (E1_turn20/40) | network | 11.8 (13.8) | 5.46 (6.70) | 1.36 |
| all untouched static (E1_turn + E2, 8 recordings) | classical | 8.0 (9.0) | 3.97 (4.69) | 0.96 |
| all untouched static (E1_turn + E2, 8 recordings) | network | 11.5 (25.6) | 4.47 (10.37) | 1.90 |

> CORRECTION (2026-11-16, paper audit): in E3 the hand is behind the guide outline by 0.6-1.0 s (median) and up to 2.0 s (20 mm) on the way back, not '0.0-0.5 s / 2-5 mm' as written above (`python paper/extra_checks.py lag`). E5 as evaluated here uses the 640x360 test split; at full resolution the classical pipeline has 1.6 mm / 0.11 deg and the network 6.8 mm / 1.85 deg (`paper/data/e5_full.json`).
