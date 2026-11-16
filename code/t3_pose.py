"""t3_pose.py -- Task 3, step 3: poses of F and M and the relative pose ^F T_M for every frame of a recording.

  python t3_pose.py

Asks (Enter = [default]): recording, camera file, camera still?, source of the pose of M (classical | network), output folder.
Needs  data/markers.json (T1, with measured_side_mm)  and  data/calibration/camera.npz (calibration).
Per frame:  ^C T_F and ^C T_M  with cv2.solvePnP (SOLVEPNP_IPPE_SQUARE, then solvePnPRefineLM), using the measured side L,
            ^F T_M = (^C T_F)^-1 ^C T_M, planar ambiguity handled (see mvtools.MarkerPoseEstimator).
Writes  <output>/<recording>_poses.csv   one row per frame: found / held / ambiguous flags, reprojection errors, and the poses
                                         FM_*, CM_*, CF_*  as x,y,z [mm] and rotation vector rx,ry,rz [rad]; roll/pitch/yaw columns for reading
        <output>/<recording>_annotated.avi  (optional) the video with outlines and the relative pose written on it
Convention: ^F T_M maps coordinates of M into F; roll-pitch-yaw is ZYX (R = Rz(yaw) Ry(pitch) Rx(roll)).
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np

from config import data_path, OUTPUT_DIR, training_path
from mvtools import (MarkerPoseEstimator, ask, ask_choice, ask_folder, load_camera, load_markers_cfg, open_video, recording_times,
                     result_to_row, rpy_deg, write_csv, default_camera_path, cols_to_pose, ask_yes)


def newest_recording():
    files = sorted([f for d in (data_path("recordings"), training_path("real")) for e in ("*.avi", "*.mp4")
                    for f in glob.glob(os.path.join(d, e))], key=os.path.getmtime)
    return files[-1] if files else data_path("recordings", "name.avi")


def draw_result(frame, r, K, dist, L):
    img = frame.copy()
    for mid, c in r["corners"].items():
        cv2.polylines(img, [c.astype(np.int32)], True, (60, 200, 60), 2, cv2.LINE_AA)
    for key in ("T_CF", "T_CM"):
        T = r[key]
        if T is not None and np.all(np.isfinite(T)):
            cv2.drawFrameAxes(img, K, dist, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3].reshape(3, 1), L * 0.7, 3)
    T = r["T_FM"]
    text = "no relative pose" if T is None else \
        "M in F: x %.0f y %.0f z %.0f mm  rpy %.0f %.0f %.0f deg" % (*(1000 * T[:3, 3]), *rpy_deg(T[:3, :3]))
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (30, 30, 30), -1)
    cv2.putText(img, text + ("   [F held]" if r["F_held"] else ""), (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def process_video(video_path, cfg, K, dist, camera_still=True, net=None, annotated_path=None, max_frames=None, verbose=True):
    """Runs the estimator over a video; returns (rows, summary)."""
    cap = open_video(video_path)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    times = recording_times(video_path, n_total)
    est = MarkerPoseEstimator(cfg, K, dist, camera_still, net)
    rows, writer, t_cpu, i = [], None, 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        t0 = time.perf_counter()
        r = est.process(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        t_cpu += time.perf_counter() - t0
        t = times[i] if i < len(times) else i / 30.0
        row = result_to_row(i, t, r)
        if r["T_FM"] is not None:
            row.update(dict(zip(("roll_deg", "pitch_deg", "yaw_deg"), np.round(rpy_deg(r["T_FM"][:3, :3]), 3))))
        rows.append(row)
        if annotated_path:
            if writer is None:
                writer = cv2.VideoWriter(annotated_path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (frame.shape[1], frame.shape[0]))
            writer.write(draw_result(frame, r, K, dist, cfg["L"]))
        i += 1
        if verbose and i % 100 == 0:
            print(f"  {i}/{n_total} frames")
    cap.release()
    if writer is not None:
        writer.release()
    n = max(1, len(rows))
    summary = dict(frames=len(rows), F_found_pct=100 * sum(r["F_found"] for r in rows) / n,
                   M_found_pct=100 * sum(r["M_found"] for r in rows) / n,
                   relative_pose_pct=100 * sum(1 for r in rows if np.isfinite(r["FM_x_mm"])) / n,
                   F_held_frames=sum(r["F_held"] for r in rows), ambiguous_frames=sum(r["amb_F"] or r["amb_M"] for r in rows),
                   ms_per_frame=1000 * t_cpu / n)
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description="Task 3: poses of F, M and the relative pose for a recording.")
    ap.add_argument("--max-frames", type=int, help=argparse.SUPPRESS)
    a = ap.parse_args()
    cfg = load_markers_cfg()
    print(f"Task 3 - poses.  Dictionary {cfg['dictionary']}, F = id {cfg['id_fixed']}, M = id {cfg['id_moving']}, "
          f"side L = {cfg['L'] * 1000:g} mm ({'measured' if cfg['side_is_measured'] else 'NOMINAL - measure the print and set measured_side_mm'}).")
    print("Press Enter to accept the [default].\n")
    video = ask_folder("1) Recording (video file)", newest_recording())
    cam_file = ask_folder("2) Camera file", default_camera_path())
    K, dist, size = load_camera(cam_file)
    still = ask_yes("3) Was the camera still during the recording? (then a missing F is replaced by the last F)", "y")
    source = ask_choice("4) Pose of M from", ["classical", "network"], "classical")
    net = None
    if source == "network":
        from netpose import NetPose
        model_path = ask_folder("   network file", training_path("models", "resnet18_v1.pt"))
        net = NetPose(model_path, K=K, dist=dist)
    out_dir = ask_folder("5) Output folder", os.path.join(OUTPUT_DIR, "pose"))
    save_video = ask_yes("6) Also write the annotated video?", "y")
    name = os.path.splitext(os.path.basename(video))[0] + ("" if source == "classical" else "_net")
    os.makedirs(out_dir, exist_ok=True)

    print()
    rows, s = process_video(video, cfg, K, dist, still, net, os.path.join(out_dir, name + "_annotated.avi") if save_video else None,
                            a.max_frames)
    csv_path = os.path.join(out_dir, name + "_poses.csv")
    cols = list(rows[0])
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    write_csv(csv_path, [{c: r.get(c, "") for c in cols} for r in rows], cols)
    print(f"{s['frames']} frames: F found {s['F_found_pct']:.0f} %, M found {s['M_found_pct']:.0f} %, relative pose in {s['relative_pose_pct']:.0f} %, "
          f"{s['ambiguous_frames']} ambiguous, {s['F_held_frames']} with F held, {s['ms_per_frame']:.1f} ms/frame")
    print(f"poses -> {csv_path}")


if __name__ == "__main__":
    main()
