"""Read-only inspection of the project's real files (formats for Code_Documentation.docx).  Writes nothing except stdout."""
import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
J = lambda *p: os.path.join(ROOT, *p)


def size(p):
    if os.path.isdir(p):
        s = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)
    else:
        s = os.path.getsize(p)
    for u in ("B", "KB", "MB", "GB"):
        if s < 1024:
            return f"{s:.1f} {u}"
        s /= 1024
    return f"{s:.1f} TB"


def head_csv(p, n=2):
    with open(p, newline="") as f:
        r = list(csv.reader(f))
    print(f"CSV {os.path.relpath(p, ROOT)} ({size(p)}, {len(r) - 1} rows)\n  header: {r[0]}")
    for row in r[1:1 + n]:
        print("  row:", row)


def show_json(p, full=False):
    d = json.load(open(p, encoding="utf-8"))
    print(f"JSON {os.path.relpath(p, ROOT)} ({size(p)})")
    if full:
        print("  ", json.dumps(d)[:1500])
    else:
        if isinstance(d, dict):
            for k, v in d.items():
                s = json.dumps(v)
                print(f"   {k}: {s[:160]}")
        else:
            print("   list of", len(d), "first:", json.dumps(d[0])[:300])


def npy(p):
    a = np.load(p, mmap_mode="r")
    print(f"NPY {os.path.relpath(p, ROOT)}: shape {a.shape} dtype {a.dtype} ({size(p)})")


def video(p):
    import cv2
    c = cv2.VideoCapture(p)
    w, h, n, fps = c.get(3), c.get(4), c.get(7), c.get(5)
    fourcc = int(c.get(cv2.CAP_PROP_FOURCC))
    fc = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
    c.release()
    print(f"VIDEO {os.path.relpath(p, ROOT)}: {int(w)}x{int(h)}, {int(n)} frames, {fps:.2f} fps, fourcc {fc!r} ({size(p)})")


sec = sys.argv[1] if len(sys.argv) > 1 else "all"

if sec in ("all", "data"):
    show_json(J("data", "markers.json"), True)
    show_json(J("data", "calibration", "board.json"), True)
    show_json(J("data", "calibration", "camera.json"), True)
    z = np.load(J("data", "calibration", "camera.npz"))
    print("NPZ camera.npz:", {k: (z[k].shape, str(z[k].dtype), z[k].tolist() if z[k].size < 10 else "...") for k in z.files}, size(J("data", "calibration", "camera.npz")))
    imgs = glob.glob(J("data", "calibration", "images", "*.png"))
    print("calibration images:", len(imgs), size(J("data", "calibration", "images")))
    for n in ("E1_p1", "E3_motion_01", "T4_motion_1", "invariance_1", "E7_box_slow", "E4_blur"):
        b = J("data", "recordings", n)
        video(b + ".avi")
        head_csv(b + ".csv")
        show_json(b + ".json", True)
        if os.path.exists(b + ".truth.csv"):
            head_csv(b + ".truth.csv", 3)
    show_json(J("output", "markers", "markers_info.json"), True)
    show_json(J("output", "dictionary", "dictionary.json"), True)
    head_csv(J("output", "dictionary", "dictionary.csv"), 1)

if sec in ("all", "out"):
    for p in ("pose/E1_p1_poses.csv", "pose/T4_motion_1_poses.csv", "pose/invariance_1_poses.csv"):
        head_csv(J("output", p), 1)
    video(J("output", "pose", "invariance_1_annotated.avi"))
    show_json(J("output", "calibration", "report.json"))
    head_csv(J("output", "calibration", "per_view.csv"), 2)
    head_csv(J("output", "detection", "detection_matrix.csv"), 1)
    for p in sorted(glob.glob(J("output", "detection", "guided", "*"))):
        print("guided:", os.path.basename(p), size(p))
    for p in ("animation/T4_motion_1_animation.mp4", "animation/T4_motion_1_network_animation.mp4", "evaluation/side_by_side_E4_cover.mp4",
              "bonus/E7_box_5cms_2_occlusion.mp4"):
        video(J("output", p))
    for p in ("animation/T4_motion_1_animation.gif", "bonus/E7_box_5cms_2_occlusion.gif"):
        print("GIF", p, size(J("output", p)))
    head_csv(J("output", "animation", "T4_motion_1_animation_filtered_poses.csv"), 1)
    head_csv(J("output", "evaluation", "comparison_table.csv"), 2)
    head_csv(J("output", "evaluation", "worst_frames.csv"), 1)
    show_json(J("output", "evaluation", "summary.json"))
    print("worst/:", len(os.listdir(J("output", "evaluation", "worst"))), "files", size(J("output", "evaluation", "worst")))
    show_json(J("output", "evaluation", "t6_live.json"), True)
    show_json(J("output", "network", "real_E1_comparison_ft3.json"))
    d = json.load(open(J("output", "network", "real_E1_comparison_ft3.json")))
    k = list(d)[0]
    print("   per method keys:", list(d[k]), "per rec keys:", list(d[k][list(d[k])[0]]))
    show_json(J("output", "network", "hard_cases_E1.json"))
    print("figures:", os.listdir(J("output", "network", "figures")))
    for p in ("E7_box_slow_passes.csv", "E7_nocover_fast_track.csv", "virtual_occlusion.csv", "real_occlusion.csv", "E7_box_5cms_2_predicted_poses.csv"):
        pp = J("output", "bonus", p)
        if os.path.exists(pp):
            head_csv(pp, 1)
        else:
            print("MISSING", p)
    show_json(J("output", "bonus", "setup_live.json"), True)
    for p in sorted(os.listdir(J("output", "bonus"))):
        print("bonus:", p, size(J("output", "bonus", p)))
    for p in sorted(os.listdir(J("output", "pose"))):
        print("pose:", p, size(J("output", "pose", p)))
    for p in sorted(os.listdir(J("output", "evaluation"))):
        print("evaluation:", p, size(J("output", "evaluation", p)))
    for p in sorted(os.listdir(J("output", "print"))) + sorted(os.listdir(J("output", "calibration"))):
        print("print/calib:", p)

if sec in ("all", "train"):
    for d in sorted(glob.glob(J("training", "synthetic", "*"))):
        print("\nSET", os.path.basename(d), size(d), sorted(os.listdir(d)))
        for f in sorted(glob.glob(os.path.join(d, "*.npy"))):
            npy(f)
        if os.path.exists(os.path.join(d, "meta.json")):
            show_json(os.path.join(d, "meta.json"))
        if os.path.exists(os.path.join(d, "build_info.json")):
            show_json(os.path.join(d, "build_info.json"))
    for s in ("real_v2", "real_v3", "real_v4"):
        d = J("training", s)
        fs = sorted(os.listdir(d))
        print("\n", s, size(d), fs)
        vids = sorted(glob.glob(os.path.join(d, "*.avi")))
        if vids:
            video(vids[0])
            show_json(vids[0][:-4] + ".json", True)
        labs = sorted(glob.glob(os.path.join(d, "labels", "*.npz")))
        if labs:
            z = np.load(labs[0])
            print("  labels", os.path.basename(labs[0]), {k: (z[k].shape, str(z[k].dtype)) for k in z.files}, size(labs[0]))

if sec in ("all", "models"):
    import torch
    for p in sorted(glob.glob(J("training", "models", "*.pt"))):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {})
        print(f"\nPT {os.path.basename(p)} ({size(p)}): keys {list(ck)}; state_dict tensors {len(ck['state_dict'])}; "
              f"conv1 {tuple(ck['state_dict']['conv1.weight'].shape)}, fc {tuple(ck['state_dict']['fc.weight'].shape)}")
        print("  config keys:", list(cfg))
        print("  ", {k: cfg.get(k) for k in ("kind", "data_set", "epochs", "epochs_done", "best_epoch", "batch", "lr", "w_rot", "w_trans", "net_size", "crop_size",
                                             "stage1", "n_train", "n_params", "train_minutes", "device", "fine_tuned_from", "mixed_with", "hard_aug", "rot_aug_deg",
                                             "validation_recording", "L_m", "size_cam")})
        print("   val:", cfg.get("val"))
    for p in sorted(glob.glob(J("output", "network", "*", "train_log.json"))):
        d = json.load(open(p))
        print(f"\nLOG {os.path.relpath(p, ROOT)} ({size(p)}): keys {list(d)}; {len(d['log'])} epochs; epoch keys {list(d['log'][0])}")
        print("   files:", sorted(os.listdir(os.path.dirname(p))))
    print("logs dir:", os.listdir(J("output", "network", "logs")))
    for d in ("synth_v1", "crop_v1"):
        print(d, os.listdir(J("output", "network", d)))
