# real_v4 - round 3 of real training frames: tilted / turned camera at three distances

Goal: make the network robust to the camera angle (screen pitch, laptop yaw) at near / middle / far distance.
7 recordings of 42 s: 5 for training, 2 for validation (whole recordings, never trained on).

| recording | F distance (target) | camera | use |
|---|---|---|---|
| real_v4_01 | ~58 cm | screen closed inward ~15 deg (hinge ~75 deg) | training |
| real_v4_02 | ~73 cm | laptop turned right ~15 deg | training |
| real_v4_03 | ~58 cm | screen inward ~12 deg + laptop turned right ~12 deg | **validation** |
| real_v4_04 | ~43 cm | laptop turned left ~8 deg | training |
| real_v4_05 | ~40 cm | screen outward ~15 deg (hinge ~105 deg) + laptop turned left ~15 deg | training |
| real_v4_06 | ~70 cm | screen opened outward ~15 deg (hinge ~105 deg) | training |
| real_v4_07 | ~63 cm | laptop turned right ~15 deg | **validation** |

The numbers are targets: anything close is fine, what counts is the measured value (it goes into the recording's .json notes).

## Each recording

```
cd code
python t4_record_guided.py --plan train --set real_v4 --seconds 42
```

1. Set the camera angle first, then place F at the distance so both markers are visible; M next to F, both still.
2. Measure: lens -> marker plane distance, screen (hinge) angle, laptop turn angle.
3. Press `R`; keep both markers still for 2-3 s (known-distance frames).
4. `SPACE` = pause (detach M, no hurry), `SPACE` = resume; follow the 7 moves. `S` = stop and save early. `Q` = close.

## Fine-tuning (round 3)

```
cd code
python t5_finetune.py --sets real_v2 real_v3 real_v4 --suffix _ft3 --val real_v2_05 real_v3_05 real_v4_03 real_v4_07 --yes
```

Validation = one whole recording per case: upright (real_v2_05), pitch (real_v3_05), pitch + yaw (real_v4_03), yaw far (real_v4_07).
The `_ft` and `_ft2` networks are kept.
