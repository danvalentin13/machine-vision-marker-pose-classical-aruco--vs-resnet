# Bonus (E7) - predicting M behind a fixed obstacle: summary (2026-11-16)

> **CORRECTION (2026-11-16, late evening).** The real-obstacle numbers in the sections 'Results' and 'Second round' below are
> SUPERSEDED: the script took the hidden stretch as 'first missing frame to last missing frame' of a pass, so a single dropped
> frame (detector flicker) before or after the obstacle moved its ends. That made some passes look 'lost at the start' (they
> were not) and measured some 'reappearance' errors after the filter had already been updated (too small).
> `t7_real_occlusion.py` now takes the LONGEST run of missing frames as the obstacle. Corrected results: the last section.
> The software-hidden results (virtual_occlusion) are not affected.


## Setup
- F fixed on the right; a ruler along the table; M slides along the ruler toward F, upright, facing the camera.
- Ruler: M's left corner at 0 at the start stop, F's left edge at 377 mm (centre to centre 377 mm: camera 377.0-377.3 mm);
  end stop at ~273 mm (centre to centre ~104 mm). Camera ~59 cm from F. F is mounted a quarter turn rotated and ~7 deg yawed against M:
  the path is therefore taken from the two rest positions the camera measures, not from F's axes.
- Obstacle: a tin box standing in front of the ruler (picture: setup_obstacle_sugus_box.jpg); it hides M from ruler ~90 mm to ~265 mm
  (measured from where the detection drops out and returns), i.e. ~175 mm of travel.
- Recording: `t4_record_guided.py --plan bonus --pass-s <s>` (rest 2 s - GO with a pacing bar and beeps - rest 2 s - back).

## Recordings (data/recordings)
| recording | obstacle | passes | speed | hidden | use |
|---|---|---|---|---|---|
| E7_nocover_slow | no | 1 | 1.5 cm/s | - | first try, too slow |
| E7_nocover_slow_2 | no | 4 | 5.5-7.5 cm/s | - | software-hidden test, slow |
| E7_nocover_fast | no | 7 | 9.5-19 cm/s | - | software-hidden test, fast |
| E7_cover_slow, E7_cover_slow_2 | yes | 3 + 4 | 3-5 cm/s | 3.3-6.2 s | too slow (hidden longer than 2 s): examples of a lost pose |
| E7_cover_slow_3 | yes | 6 (2 clean) | 13-17 cm/s | 1.2-2.0 s | faster group |
| E7_cover_slow_4 | yes | 6 (5 clean) | 9-11.5 cm/s | 1.5-1.9 s | slower group |
| E7_cover_fast | yes | 7 (3 clean) | 14-25 cm/s | 0.9-1.6 s | faster group |
Clean = M was seen moving for at least 60 mm before it disappeared (otherwise the detector lost it at the start: blur / fingers).

## Results
Predictor: constant-velocity Kalman (position) + alpha-beta on SO(3) with the turning speed faded out while hidden (rot_tau 0.1 s);
lost when 3 sigma > 2 L; gated re-acquisition. Baseline: hold the last pose.

Real obstacle, clean passes, error at the moment M is seen again (exact truth) - real_occlusion.md / .csv / .png:
| speed group | passes | speed | hidden | e_t predictor / hold | e_R predictor / hold | declared lost before reappearing |
|---|---|---|---|---|---|---|
| slower | 5 | 10 cm/s | 1.7 s (1.5-1.9) | 35 / 176 mm | 15.6 / 15.9 deg | 5 of 5 (after ~1.4 s) |
| faster | 5 | 15 cm/s | 1.2 s (0.9-1.2) | 41 / 177 mm | 19.0 / 17.6 deg | 2 of 5 |

Uncovered passes hidden in software (truth = the measured pose in every hidden frame) - virtual_occlusion.md / .csv / .png:
| speed | e_t predictor / hold at 0.25 s | at 0.5 s | at 1 s |
|---|---|---|---|
| ~7 cm/s (4 passes) | 7.6 / 22 mm | 15 / 42 mm | 25 / 82 mm |
| ~17 cm/s (7 passes) | 23 / 52 mm | 56 / 91 mm | 94 / 165 mm |

## Reading
- Position: the predictor cuts the error at reappearance to about a fifth of 'hold the last pose' (35-41 mm against 176 mm after 1.2-1.7 s).
- The remaining error comes from the hand: the speed changes by 40-50 % within a pass, the model assumes it stays constant.
- Rotation: no gain over holding the last orientation; ~10-19 deg is the noise of the single-frame rotation of a marker that
  faces the camera (and of the first frame after the obstacle). Extrapolating the turning speed made it worse (42 deg): hence the fade-out.
- Uncertainty: with sigma_a = 0.15 m/s^2 the error stays inside 3 sigma in 87 % of the software-hidden frames and in all clean
  real passes at reappearance; smaller values are over-confident (0.05-0.10: 70-82 %). With it the pose is declared lost after ~1.4 s:
  for a hand-moved marker that is the honest horizon, although the prediction is often still good later.
- Limits: truth while hidden behind the real obstacle exists only at reappearance; hand speed is not constant; 9 of 19 covered passes
  were not clean.

## Reproduce
python t7_passes.py <recording>;  python t7_virtual_occlusion.py E7_nocover_slow_2 E7_nocover_fast;
python t7_real_occlusion.py E7_cover_slow_4 E7_cover_slow_3 E7_cover_fast;  python t7_setup_guided.py (live setup check)

## Animation
`python t7_animate.py E7_cover_slow_4 --seconds 32 --gif` -> E7_cover_slow_4_occlusion.mp4 / .gif (first 32 s: 3 passes and their returns),
E7_cover_slow_4_predicted_poses.csv (measured / predicted / lost / sigma per frame), E7_cover_slow_4_occlusion_frame.jpg (one frame
while M is hidden: caption 'predicted (M not measured) +-99 mm (3 sigma)', uncertainty circles around the predicted M).
Note: F is mounted a quarter turn rotated, so in F's frame the ruler runs along F's y axis: in the 3D view M moves 'vertically'.
The roll curve jumps by +-20 deg at times: the planar ambiguity of a marker that faces the camera.

## Second round (layout 2, 2026-11-16 evening): guided speed, red box, face hidden
- Layout 2: an edge blocked M at the old start, so the start moved: M's right black edge 248 mm from F's left black edge
  (centre to centre 325 mm), end stop ~20 mm between the black squares (travel ~228 mm). Obstacle: a light-bulb box (7 cm wide),
  ~6 cm of visible run-up, M not readable over ~16 cm. Pictures: setup_layout2_bulb_box.jpg.
- Recorder: a moving outline 'keep M here' travels at the target speed (`--pass-s`, `--travel`, `--centre`); closing the window
  while recording now saves what was recorded.
- Recordings: E7_box_slow (7 passes, guide 2.5 s per pass), E7_box_fast (9 passes, guide 1.7 s per pass).
- Results (`t7_real_occlusion.py E7_box_slow E7_box_fast --min-runup 50`), error at reappearance, predictor / hold:
  E7_box_slow: passes 1, 2, 5, 6, 7 good (hidden 1.4-1.9 s, 9-12 cm/s): 27, 56, 45, 32, 13 mm against 160-175 mm;
  pass 3 lost M early (137 mm); pass 4 stalled behind the box for 4.3 s (254 mm against 164 mm: the one case worse than holding).
  E7_box_fast: passes 1, 4, 6 clean (hidden 1.2-1.3 s, 12-14 cm/s): 18, 47, 57 mm against 159-166 mm; in the other six M was lost
  3-5 cm into the run (blur at the faster start): 10-143 mm.
  Medians: ~10 cm/s 32 mm / 164 mm; ~13 cm/s 47-52 mm / 165 mm.
- Reading: the same level as round 1 (35-41 mm): the outline makes the speed steadier (9-12 cm/s) but not constant, and the
  predictor tends to lag (the hand is still accelerating when M disappears, the speed it has learned is too low).
- Animation: `t7_animate.py E7_box_slow --seconds 31 --gif` -> E7_box_slow_occlusion.mp4 / .gif, frames in E7_box_slow_occlusion_frames.jpg:
  green outline = measured, red box 'predicted' drawn over the obstacle, everything outside the path pixelated (face hidden).

## CORRECTED results - real obstacle, all usable passes (supersedes the real-obstacle tables above)
`python t7_real_occlusion.py E7_box_5cms E7_box_slow E7_box_fast E7_cover_slow_4 E7_cover_slow_3 E7_cover_fast --min-runup 50`
-> real_occlusion.csv / .md / .png.  42 passes, every one with 5-10 cm of visible run-up.  Medians per speed group;
error at reappearance = predicted pose against the first measured pose after the obstacle (exact truth); the errors at 0.5 s and 1 s
are against a straight line at constant speed between the last and the next measured pose (assumption).

| speed while hidden | passes | hidden | e_t at reappearance: predictor / hold | predictor better in | e_t at 0.5 s | e_t at 1 s | e_R at reappearance |
|---|---|---|---|---|---|---|---|
| ~4.5 cm/s (E7_box_5cms, countdown + outline) | 7 | 3.5 s (2.8-4.3) | 111 / 158 mm | 6 of 7 | 16 / 22 mm | 28 / 44 mm | 32 / 22 deg |
| ~10 cm/s | 14 | 1.6 s | 50 / 174 mm | 14 of 14 | 18 / 60 mm | 32 / 113 mm | 23 / 20 deg |
| ~14 cm/s | 18 | 1.2 s | 96 / 182 mm | 16 of 18 | 44 / 82 mm | 76 / 161 mm | 21 / 18 deg |
Left out: 3 passes (one stalled behind the box for 4.3 s, two faster than 18 cm/s): 254 / 171 mm.

Reading (corrected)
- Position: the predictor is better than holding the last pose in 36 of 39 passes, by a factor of about 3.5 at ~10 cm/s
  (50 against 174 mm after 1.6 s) and about 2 at ~14 cm/s (96 against 182 mm after 1.2 s) - not the factor 5 reported first.
- Slower is not more accurate at reappearance: at ~4.5 cm/s M is hidden for 3.5 s and the error has grown to 111 mm (hold 158 mm).
  At equal hidden time the slow passes are the most accurate in absolute terms (16 mm at 0.5 s, 28 mm at 1 s), but there the
  baseline is also good (22 / 44 mm) because M has not moved far.
- The pose is declared lost after ~1.4 s (0.7-0.9 s when the run-up was short): in 27 of the 39 passes before M reappears.
- Rotation: no gain (20-30 deg for both: the first frames after the obstacle are noisy and a frontal marker is ambiguous).
- Limit: the hand. Speed within a pass changes by tens of percent and M is still accelerating when it disappears.

## Third slow set and a tested option (2026-11-16, night)
- E7_box_5cms_2 (second set at ~5 cm/s with countdown + outline): 7 passes, 4.8-6.0 cm/s, hidden 2.7-3.3 s.
  Error at reappearance, predictor / hold: 182, 7, 116, 55, 49, 51, 57 mm against 155-159 mm -> median 55 / 157 mm
  (the first slow set: 111 / 158 mm). At 0.5 s and 1 s hidden: 5.9 / 27.7 mm and 15.2 / 53.5 mm - the most accurate of all sets.
  Along the path the last six passes are within -26..+17 mm after ~3 s (the speed learned before hiding was within ~7 mm/s of the
  mean speed while hidden); pass 1 started at 12 cm/s and slowed to 5 cm/s behind the box (+179 mm ahead). The rest of the error
  is across the path (a hand-held marker also moves a little in depth and height).
- All real-obstacle results now (`real_occlusion.csv`, 49 passes): ~5 cm/s set 2: 55 / 157 mm (2.9 s); ~5 cm/s set 1: 111 / 158 mm
  (3.5 s); ~10 cm/s: 50 / 174 mm (1.6 s); ~14 cm/s: 96 / 182 mm (1.2 s).
- Tested and NOT adopted: a straight-path option (`OcclusionPredictor(line_s=0.6)`, `--line`): when M disappears the velocity is kept
  only along the line M was moving on. Slow sets: 55 -> 40 mm and 111 -> 87 mm; ~10 cm/s: 50 -> 57 mm; ~14 cm/s: 96 -> 106 mm;
  software-hidden: mixed. No clear gain, so it stays off by default (and the numbers above are without it).
- Animation of the slow set: `python t7_animate.py E7_box_5cms_2 --start 37.5 --seconds 50 --gif` -> E7_box_5cms_2_occlusion.mp4 / .gif
  (passes 4-7), frames in E7_box_5cms_2_occlusion_frames.jpg. New options: `--start` (begin later in the recording; the predictor
  still runs from the beginning) and the default privacy mode now CUTS OUT everything outside the path and the small box around M
  (plain dark grey; `--pixelate` gives the earlier mosaic, `--show-all` the whole picture).
