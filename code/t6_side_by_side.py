"""t6_side_by_side.py -- Task 6 / video: both methods side by side on the same frames, and the T4 animation of the NETWORK poses.

  python t6_side_by_side.py E3_motion_01 E4_cover E1_turn40          side-by-side videos
  python t6_side_by_side.py T4_motion_1 --animation                  T4 animation driven by the network's poses

Side by side (output/evaluation/side_by_side_<name>.mp4): the same camera frames twice,
  left   CLASSICAL  ArUco + IPPE + LM: the marker outlines and the axes of F and M (x red, y green, z blue)
  right  NETWORK    F classical, M from the two-stage network: the axes and the projected outline of M (orange)
under each picture the pose of M in the frame of F and, when the recording has a truth file, the errors e_t and e_R.
The picture is cropped to the area around the markers, so the person moving M is not in the video.
--animation: the T4 display (camera | 3D view | curves) with the network's poses; only the two markers stay visible in the camera part.
"""
import argparse
import os

import cv2
import numpy as np

from config import OUTPUT_DIR, data_path, training_path
from mvlib import marker_object_points
from mvtools import MarkerPoseEstimator, cols_to_pose, load_camera, load_markers_cfg, load_truth, pose_errors, truth_pose
from netpose import NetPose

GREEN, ORANGE, WHITE, GREY, RED = (60, 200, 60), (26, 150, 255), (236, 236, 236), (160, 160, 160), (60, 60, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def outline(T, obj, K, dist):
    return cv2.projectPoints(obj, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, dist)[0].reshape(-1, 2)


def read_all(video):
    cap, frames = cv2.VideoCapture(video), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def side_by_side(name, cfg, K, dist, net, out_dir, step=2):
    video = data_path("recordings", name + ".avi")
    frames = read_all(video)
    t = np.loadtxt(data_path("recordings", name + ".csv"), delimiter=",", skiprows=1)[:, 1]
    tp = data_path("recordings", name + ".truth.csv")
    truth = load_truth(tp) if os.path.exists(tp) else None
    obj = marker_object_points(cfg["L"]).astype(np.float64)
    est = MarkerPoseEstimator(cfg, K, dist, camera_still=True)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    res = [est.process(g) for g in grays]
    Rn, tn = [], []
    for s in range(0, len(grays), 32):
        a, b = net.predict_batch(grays[s:s + 32])
        Rn.append(a); tn.append(b)
    Rn, tn = np.concatenate(Rn), np.concatenate(tn)
    T_net = []
    for i in range(len(frames)):
        T = np.eye(4); T[:3, :3], T[:3, 3] = Rn[i], tn[i]
        T_net.append(T)
    last_CF, CF = None, []
    for r in res:
        last_CF = r["T_CF"] if r["T_CF"] is not None else last_CF
        CF.append(last_CF)
    pts = [c.reshape(-1, 2) for r in res for c in r["corners"].values()] + [outline(T, obj, K, dist) for T in T_net[::5]]
    pts = np.vstack(pts)
    H, W = frames[0].shape[:2]
    x0, x1 = int(max(0, np.percentile(pts[:, 0], 1) - 70)), int(min(W, np.percentile(pts[:, 0], 99) + 70))
    y0, y1 = int(max(0, np.percentile(pts[:, 1], 1) - 60)), int(min(H, np.percentile(pts[:, 1], 99) + 60))
    scale = 560 / (x1 - x0)                                           # every panel 560 px wide
    ph, bar = int((y1 - y0) * scale), 96
    out = os.path.join(out_dir, f"side_by_side_{name}.mp4")
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30.0 / step, (2 * 560 + 12, ph + bar + 44))
    n_c = n_n = 0
    for i in range(0, len(frames), step):
        Tt = truth_pose(truth, t[i]) if truth is not None else None
        panels = []
        for kind in ("classical", "network"):
            f = frames[i].copy()
            Tcm = res[i]["T_CM"] if kind == "classical" else (T_net[i] if CF[i] is not None else None)
            if CF[i] is not None:
                cv2.drawFrameAxes(f, K, dist, cv2.Rodrigues(CF[i][:3, :3])[0], CF[i][:3, 3], 0.6 * cfg["L"], 3)
            for mid, c in res[i]["corners"].items():
                if mid == cfg["id_fixed"] or kind == "classical":
                    cv2.polylines(f, [c.astype(np.int32)], True, GREEN, 2, cv2.LINE_AA)
            if Tcm is not None:
                if kind == "network":
                    cv2.polylines(f, [outline(Tcm, obj, K, dist).astype(np.int32)], True, ORANGE, 2, cv2.LINE_AA)
                cv2.drawFrameAxes(f, K, dist, cv2.Rodrigues(Tcm[:3, :3])[0], Tcm[:3, 3], 0.6 * cfg["L"], 3)
            crop = cv2.resize(f[y0:y1, x0:x1], (560, ph), interpolation=cv2.INTER_AREA)
            p = np.full((ph + bar + 44, 560, 3), 30, np.uint8)
            p[44:44 + ph] = crop
            cv2.putText(p, "CLASSICAL  (ArUco + IPPE + LM)" if kind == "classical" else "NETWORK  (ResNet-18, two-stage)", (10, 30), FONT, 0.75,
                        GREEN if kind == "classical" else ORANGE, 2, cv2.LINE_AA)
            yb = 44 + ph + 28
            if Tcm is None or CF[i] is None:
                cv2.putText(p, "no pose of M", (10, yb), FONT, 0.75, RED, 2, cv2.LINE_AA)
            else:
                Tfm = np.linalg.inv(CF[i]) @ Tcm
                x, y, z = 1000 * Tfm[:3, 3]
                cv2.putText(p, f"M in F:  x {x:+.0f}   y {y:+.0f}   z {z:+.0f} mm", (10, yb), FONT, 0.65, WHITE, 2, cv2.LINE_AA)
                if Tt is not None:
                    et, er = pose_errors(Tfm, Tt)
                    cv2.putText(p, f"error against the ruler truth:  {et:.0f} mm   {er:.1f} deg", (10, yb + 32), FONT, 0.65, WHITE, 2, cv2.LINE_AA)
                n_c += kind == "classical"; n_n += kind == "network"
            panels.append(p)
        wr.write(np.hstack([panels[0], np.full((ph + bar + 44, 12, 3), 12, np.uint8), panels[1]]))
    wr.release()
    n = len(range(0, len(frames), step))
    print(f"{name}: {n} frames, pose of M: classical {100 * n_c / n:.0f} %, network {100 * n_n / n:.0f} % -> {out}")


def network_animation(name, cfg, K, dist, net, out_dir):
    """The T4 display with the network's poses; only the two markers stay visible in the camera picture."""
    import t3_pose
    from t4_animate import export
    video = data_path("recordings", name + ".avi")
    rows, _ = t3_pose.process_video(video, cfg, K, dist, True, net, verbose=False)
    t = np.loadtxt(data_path("recordings", name + ".csv"), delimiter=",", skiprows=1)[:len(rows), 1]
    rows = rows[:len(t)]
    obj = marker_object_points(cfg["L"]).astype(np.float64)
    CF, CM = [cols_to_pose(r, "CF") for r in rows], [cols_to_pose(r, "CM") for r in rows]

    def draw(k, frame):
        out = np.full_like(frame, 38)
        for T, col in ((CF[k], GREEN), (CM[k], ORANGE)):
            if T is None:
                continue
            uv = outline(T, obj, K, dist)
            a, b = np.maximum(uv.min(0).astype(int) - 28, 0), uv.max(0).astype(int) + 28
            out[a[1]:b[1], a[0]:b[0]] = frame[a[1]:b[1], a[0]:b[0]]
            cv2.polylines(out, [uv.astype(np.int32)], True, col, 3, cv2.LINE_AA)
            cv2.drawFrameAxes(out, K, dist, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], 0.6 * cfg["L"], 3)
        cv2.putText(out, "F: classical      M: network", (16, 40), FONT, 1.0, WHITE, 2, cv2.LINE_AA)
        return out
    files = export(rows, t, [cols_to_pose(r, "FM") for r in rows], video, os.path.join(out_dir, name + "_network_animation"), formats=("mp4",),
                   with_filter=False, frame_fn=draw, verbose=False)
    print(f"{name}: network animation -> {', '.join(files)}")


def main():
    ap = argparse.ArgumentParser(description="Both methods side by side on the same frames / T4 animation with the network's poses.")
    ap.add_argument("names", nargs="+", help="recordings in data/recordings (without extension)")
    ap.add_argument("--network", default="crop_resnet18_v1_ft3", help="network in training/models")
    ap.add_argument("--animation", action="store_true", help="write the T4 animation driven by the network instead of the side-by-side video")
    a = ap.parse_args()
    cfg = load_markers_cfg()
    K, dist, _ = load_camera()
    net = NetPose(training_path("models", a.network + ".pt"), K=K, dist=dist)
    out_dir = os.path.join(OUTPUT_DIR, "animation" if a.animation else "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    for name in a.names:
        (network_animation if a.animation else side_by_side)(name, cfg, K, dist, net, out_dir)


if __name__ == "__main__":
    main()
