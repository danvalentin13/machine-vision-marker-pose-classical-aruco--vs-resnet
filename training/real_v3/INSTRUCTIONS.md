# real_v3 - second round of real training frames + fine-tuning

This folder is empty on purpose: tomorrow's recordings go here (`real_v3_01.avi`, `real_v3_02.avi`, ...).
The first round (`training/real_v2`, networks `*_ft.pt`) stays untouched.

## Why a second round

After the first fine-tuning (see `output/network/T5_runs.md`) the two-stage network is close to classical
(E1: 10.0 mm / 5.7 deg against 8.8 mm / 4.0 deg), but **E1_p4 still fails** (34-103 mm).
E1_p4 is the view where **M lies right next to F** (50 mm gap, ~13 cm between centres) - the real_v2 recordings
almost never had M that close to F. real_v3 adds exactly that, plus F at new distances.

## What you need

- F (id 0) and M (id 23) printed, the laptop camera, a ruler for F's distance.
- Close every other camera window first.

## Step 1 - record 5 recordings (about 1 minute each, ~10 minutes in total)

Open the terminal, activate the venv, then:

```
cd code
python t4_record_guided.py --plan train --set real_v3 --seconds 42
```

For each recording:
1. Put F somewhere new and **measure F's distance to the lens** (write it down).
2. Press `R`. Follow the instruction on the screen; it changes by itself. The bar shows how much is done; it stops by itself.
3. The 7 moves: close - far - edges - strong tilt - steering wheel - **M right next to F (5-15 cm gap, flat, slide around F)** - free.
4. Press `Q` to close. Run the same command again for the next recording (the number goes up by itself).

Vary the setup between recordings:

| recording | F distance to lens | change something |
|---|---|---|
| real_v3_01 | ~35 cm | F in the middle |
| real_v3_02 | ~50 cm | F to the left |
| real_v3_03 | ~65 cm | F to the right |
| real_v3_04 | ~40 cm | other light (lamp on/off) |
| real_v3_05 | ~55 cm | laptop screen tilted a bit - this one is kept for **validation** |

Tell Claude the distances: they go into each recording's `.json` notes.

## Step 2 - fine-tune (about 10 minutes, with the live window)

```
cd code
python t5_finetune.py --sets real_v2 real_v3 --suffix _ft2
```

Press Enter at every question (defaults: `resnet18_v2`, `crop_resnet18_v1`, validation = the last recording `real_v3_05`,
15 epochs, learning rate 1e-4, live window on). It uses the recordings of **both** folders and writes new networks
`resnet18_v2_ft2.pt` and `crop_resnet18_v1_ft2.pt` - the `_ft` networks of today are kept.
If a network with that name already exists, it asks before overwriting (answer `n` to stop).

In the live window: `S` screenshot, `Q` stop and keep the network as it is.
The example photos of every epoch are saved in `output/network/<network>_ft2/examples/`.

## Step 3 - test (Claude runs it)

Compare classical, `_ft` and `_ft2` on the test recordings (E1_p1..p5, T4_motion_1 - never trained on).
Goal: E1_p4 below ~20 mm, the rest at least as good as `_ft`.
