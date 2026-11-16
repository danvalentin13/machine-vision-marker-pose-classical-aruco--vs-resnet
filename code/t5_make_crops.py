"""t5_make_crops.py -- Task 5: synthetic CROP data set for the two-stage network (stage 2 sees only a window around M).

  python t5_make_crops.py

Asks (Enter = [default]): stage-1 network, data-set name, number of train / validation / test crops, crop size, distance range,
maximum tilt, stage-1 error to simulate, seed.
Every sample: M at a random pose in the FULL-resolution calibrated camera (1280x720); a square window around it with side
2.0 x the marker's apparent side, its centre and size perturbed like stage-1 errors, rendered directly at full resolution and
resized to crop_size x crop_size.  Labels: ^C T_M (R, t) and the window (u0, v0, side) in camera pixels.
Writes  training/synthetic/<name>/X_*.npy, R_*.npy, t_*.npy, box_*.npy, meta.json (kind "crop")
        output/network/<name>/examples.png   crops with the label drawn (green)
Then train it with t5_train.py (it recognises a crop data set and builds the 3-channel network; stage 1 is stored with it).
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from config import OUTPUT_DIR, training_path
from mvtools import ask, ask_folder, ask_number, default_camera_path, load_camera, load_markers_cfg
from synthdata import generate_crop_set, load_boxes, load_set


def make_crop_dataset(out_dir, name, cfg, K, size, stage1, n_train, n_val, n_test, S, z_range, tilt, jitter, seed, workers=None):
    rcfg = dict(dict_id=cfg["dict_id"], id_m=cfg["id_moving"], id_f=cfg["id_fixed"], L=cfg["L"], K=np.asarray(K).tolist(), size=list(size))
    par = dict(z_range=list(z_range), tilt_max_deg=tilt, p_distractor=0.7, p_occlude=0.15, jitter=jitter)
    d = os.path.join(out_dir, name)
    t0 = time.time()
    for split, n, s in (("train", n_train, seed), ("val", n_val, seed + 1), ("test", n_test, seed + 2)):
        if n > 0:
            generate_crop_set(d, split, n, s, rcfg, par, S, workers)
    meta = dict(name=name, kind="crop", dictionary=cfg["dictionary"], id_moving=cfg["id_moving"], id_fixed=cfg["id_fixed"], L_m=cfg["L"],
                K_cam=np.asarray(K).tolist(), size_cam=list(size), K_net=np.asarray(K).tolist(), net_size=[S, S], crop_size=S, stage1=stage1,
                n_train=n_train, n_val=n_val, n_test=n_test, seed=seed, provisional_camera=False, params=par,
                generation_seconds=round(time.time() - t0, 1))
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    return d, meta


def examples_figure(d, meta, path, n=8):
    from mvlib import marker_object_points
    split = "test" if meta["n_test"] else "train"
    X, R, t = load_set(d, split, mmap=True)
    B = load_boxes(d, split)
    K, S, obj = np.array(meta["K_cam"]), meta["crop_size"], marker_object_points(meta["L_m"]).astype(float)
    tiles = []
    for i in range(min(n, len(X))):
        u0, v0, side = B[i]
        k = S / side
        Kc = np.array([[K[0, 0] * k, 0, (K[0, 2] - u0) * k], [0, K[1, 1] * k, (K[1, 2] - v0) * k], [0, 0, 1]])
        img = cv2.cvtColor(np.ascontiguousarray(X[i]), cv2.COLOR_GRAY2BGR)
        uv = (Kc @ ((R[i].astype(float) @ obj.T).T + t[i]).T).T
        cv2.polylines(img, [(uv[:, :2] / uv[:, 2:3]).astype(np.int32)], True, (60, 200, 60), 1, cv2.LINE_AA)
        cv2.putText(img, f"{t[i][2]:.2f} m", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (26, 115, 217), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(img, (2 * S, 2 * S), interpolation=cv2.INTER_NEAREST))
    rows = [np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.vstack(rows))


def main():
    argparse.ArgumentParser(description="Task 5: synthetic crop data set (two-stage network).").parse_args()
    cfg = load_markers_cfg()
    print("Task 5 - crop data set for the two-stage network.  Press Enter to accept the [default].\n")
    K, _, size = load_camera(ask_folder("1) Camera file (the calibrated camera)", default_camera_path()))
    stage1 = ask("2) Stage-1 network (finds M in the whole image; file in training/models)", "resnet18_v2.pt")
    if not os.path.exists(training_path("models", stage1)):
        raise SystemExit(f"{training_path('models', stage1)} not found: train a full-image network first")
    name = ask("3) Data-set name", "crop_v1")
    out_dir = ask_folder("4) Folder for the data sets", training_path("synthetic"))
    n_train = ask_number("5a) Training crops", 30000, int, 0)
    n_val = ask_number("5b) Validation crops", 2000, int, 0)
    n_test = ask_number("5c) Test crops", 2000, int, 0)
    S = ask_number("6) Crop size [px] (224 = the ResNet input size)", 224, int, 32)
    z_lo = ask_number("7a) Nearest marker distance [m]", 0.25, float, 0.05)
    z_hi = ask_number("7b) Farthest marker distance [m]", 1.0, float, z_lo)
    tilt = ask_number("8) Maximum marker tilt [deg]", 60.0, float, 0)
    jitter = ask_number("9) Stage-1 error to simulate (centre and scale, fraction of the marker side)", 0.12, float, 0)
    seed = ask_number("10) Random seed", 0, int, 0)
    print(f"\n   {n_train + n_val + n_test} crops, about {(n_train + n_val + n_test) * S * S / 1e9:.1f} GB\n")
    d, meta = make_crop_dataset(out_dir, name, cfg, K, size, stage1, n_train, n_val, n_test, S, (z_lo, z_hi), tilt, jitter, seed)
    ex = os.path.join(OUTPUT_DIR, "network", name, "examples.png")
    examples_figure(d, meta, ex)
    print(f"\ndata set -> {d}   ({meta['generation_seconds']:.0f} s)\nexamples (green = label) -> {ex}")


if __name__ == "__main__":
    main()
