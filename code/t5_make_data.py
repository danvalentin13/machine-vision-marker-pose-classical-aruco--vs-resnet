"""t5_make_data.py -- Task 5, step 1: the SYNTHETIC data set (exact labels) for the pose network.

  python t5_make_data.py

Asks (Enter = [default]): camera file, data-set name and folder, number of train / validation / test images, network input
width, distance range, maximum tilt, seed.   The camera is the calibrated camera (data/calibration/camera.npz); if it does not
exist yet a PROVISIONAL camera (fx = 900 px at 1280x720, like the lecture) is used: such a data set is good for tests only,
a network is valid only for the camera it was trained for - make the final data set after the calibration.
Writes  <folder>/<name>/X_train.npy, R_train.npy, t_train.npy (and _val, _test)   images (N,H,W) uint8 and labels ^C T_M
        <folder>/<name>/meta.json                                                 camera, marker, parameters
        output/network/<name>/examples.png                                       examples with the label drawn as a green outline
Three sets, three roles (assignment): synthetic (this), real training (frames labelled by T3), real test (measured truth, T6 only).
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from config import data_path, OUTPUT_DIR, training_path
from mvtools import ask, ask_folder, ask_number, default_camera_path, load_camera, load_markers_cfg
from synthdata import SyntheticRenderer, generate_set, load_set, scaled_camera

PROVISIONAL = dict(K=np.array([[900.0, 0, 640], [0, 900, 360], [0, 0, 1]]), size=(1280, 720))


def make_dataset(out_dir, name, cfg, K_cam, size_cam, n_train, n_val, n_test, net_w, z_range, tilt_max, seed, provisional, workers=None,
                 p_distractor=0.7, p_occlude=0.15):
    net_h = int(round(net_w * size_cam[1] / size_cam[0]))
    K_net = scaled_camera(K_cam, size_cam, (net_w, net_h))
    rcfg = dict(dict_id=cfg["dict_id"], id_m=cfg["id_moving"], id_f=cfg["id_fixed"], L=cfg["L"], K=K_net.tolist(), size=[net_w, net_h])
    par = dict(z_range=list(z_range), tilt_max_deg=tilt_max, p_distractor=p_distractor, p_occlude=p_occlude)
    d = os.path.join(out_dir, name)
    t0 = time.time()
    for split, n, s in (("train", n_train, seed), ("val", n_val, seed + 1), ("test", n_test, seed + 2)):
        if n > 0:
            generate_set(d, split, n, s, rcfg, par, workers)
    meta = dict(name=name, dictionary=cfg["dictionary"], id_moving=cfg["id_moving"], id_fixed=cfg["id_fixed"], L_m=cfg["L"],
                K_cam=np.asarray(K_cam).tolist(), size_cam=list(size_cam), K_net=K_net.tolist(), net_size=[net_w, net_h],
                n_train=n_train, n_val=n_val, n_test=n_test, seed=seed, provisional_camera=provisional, params=par,
                generation_seconds=round(time.time() - t0, 1))
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    return d, meta


def examples_figure(d, meta, path, n=8):
    """First n test (or train) images with the label drawn as a green outline: a visual check that labels and pixels agree."""
    split = "test" if meta["n_test"] else "train"
    X, R, t = load_set(d, split, mmap=True)
    K = np.array(meta["K_net"])
    from mvlib import marker_object_points
    obj = marker_object_points(meta["L_m"]).astype(float)
    tiles = []
    for i in range(min(n, len(X))):
        img = cv2.cvtColor(np.ascontiguousarray(X[i]), cv2.COLOR_GRAY2BGR)
        P = (R[i].astype(float) @ obj.T).T + t[i]
        uv = (K @ P.T).T
        uv = (uv[:, :2] / uv[:, 2:3]).astype(np.int32)
        cv2.polylines(img, [uv], True, (60, 200, 60), 2, cv2.LINE_AA)
        cv2.putText(img, f"Z={t[i][2]:.2f} m", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (26, 115, 217), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA))
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.vstack(rows))


def main():
    argparse.ArgumentParser(description="Task 5: synthetic data set.").parse_args()
    cfg = load_markers_cfg()
    print("Task 5 - synthetic data set.  Press Enter to accept the [default].\n")
    cam_path = ask_folder("1) Camera file (the calibrated camera)", default_camera_path())
    if os.path.exists(cam_path):
        K, _, size = load_camera(cam_path)
        provisional = False
        print(f"   camera: fx {K[0, 0]:.1f} px at {size[0]}x{size[1]}")
    else:
        K, size, provisional = PROVISIONAL["K"], PROVISIONAL["size"], True
        print("   NOTE: no calibrated camera found - using the PROVISIONAL lecture camera (fx 900 px at 1280x720). Test data only!")
    name = ask("2) Data-set name", "synth_v1")
    out_dir = ask_folder("3) Folder for the data sets", training_path("synthetic"))
    n_train = ask_number("4a) Training images (assignment: >= 10000)", 10000, int, 0)
    n_val = ask_number("4b) Validation images", 1000, int, 0)
    n_test = ask_number("4c) Test images (E5, synthetic)", 1000, int, 0)
    net_w = ask_number("5) Network input width [px] (height keeps the aspect ratio)", 640, int, 64)
    z_lo = ask_number("6a) Nearest marker distance [m]", 0.3, float, 0.05)
    z_hi = ask_number("6b) Farthest marker distance [m]", 1.2, float, z_lo)
    tilt = ask_number("7) Maximum marker tilt [deg]", 60.0, float, 0)
    seed = ask_number("8) Random seed", 0, int, 0)
    size_gb = (n_train + n_val + n_test) * net_w * int(round(net_w * size[1] / size[0])) / 1e9
    print(f"\n   {n_train + n_val + n_test} images, about {size_gb:.1f} GB on disk\n")
    d, meta = make_dataset(out_dir, name, cfg, K, size, n_train, n_val, n_test, net_w, (z_lo, z_hi), tilt, seed, provisional)
    ex = os.path.join(OUTPUT_DIR, "network", name, "examples.png")
    examples_figure(d, meta, ex)
    print(f"\ndata set -> {d}   ({meta['generation_seconds']:.0f} s)")
    print(f"examples with the labels drawn on them (check that the green outline sits on the markers) -> {ex}")


if __name__ == "__main__":
    main()
