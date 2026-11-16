"""t7_animate.py -- Bonus: animation of passes behind the obstacle with the predicted pose.

  python t7_animate.py E7_cover_slow_4 [--seconds 32] [--gif] [--show-all]

Runs the classical pipeline and the occlusion predictor (t7_bonus_occlusion.OcclusionPredictor) over the recording and writes
the T4-style animation: the camera video next to the 3D view in the frame of F and the x, y, z / roll, pitch, yaw curves.
On the camera video
  GREEN outline  M as measured
  RED box        where M is PREDICTED while it is hidden (drawn over the obstacle), with the 3 sigma uncertainty in the caption;
                 no box once the pose is declared LOST
  privacy        (default: cut out; --pixelate: pixelated instead) only the path (from just right of M's start position to F, with the obstacle) and a small box around M
                 are shown; the rest of the picture is removed, so the person moving M is not in the video
                 (--show-all switches this off).
Writes output/bonus/<name>_predicted_poses.csv and output/bonus/<name>_occlusion.mp4 (and .gif with --gif).
"""
import argparse
import os

import cv2
import numpy as np

from config import OUTPUT_DIR, data_path
from mvlib import marker_object_points
from mvtools import MarkerPoseEstimator, load_camera, load_markers_cfg, pose_to_cols, write_csv
from t4_animate import export
from t7_bonus_occlusion import OcclusionPredictor

GREEN, RED = (60, 200, 60), (40, 40, 235)


def measure(video, cfg, K, dist, seconds=0.0):
    """Per frame: t, ^F T_M (or None), ^C T_F (the last known one) and the image corners of F and M."""
    t = np.loadtxt(os.path.splitext(video)[0] + ".csv", delimiter=",", skiprows=1)[:, 1]
    est = MarkerPoseEstimator(cfg, K, dist, camera_still=True)
    cap = cv2.VideoCapture(video)
    T_FM, T_CF, corners, last_CF = [], [], [], None
    while True:
        ok, f = cap.read()
        if not ok or (seconds > 0 and len(T_FM) < len(t) and t[len(T_FM)] > seconds):
            break
        r = est.process(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        last_CF = r["T_CF"].copy() if r["T_CF"] is not None else last_CF
        T_FM.append(None if r["T_FM"] is None else r["T_FM"].copy())
        T_CF.append(last_CF)
        corners.append({i: c.reshape(-1, 2).copy() for i, c in r["corners"].items()})
    cap.release()
    n = min(len(t), len(T_FM))
    return t[:n], T_FM[:n], T_CF[:n], corners[:n]


def work_area(corners, shape, id_moving, side=40, top=25, bottom=60):
    """The fixed part of the picture that stays sharp: from just right of M's START position to F (the path and the obstacle).
    The start position itself is shown only while M is there (see the moving box in draw): the person moving M sits behind it."""
    pts = np.vstack([c for d in corners for c in d.values()])
    m = [d[id_moving] for d in corners if id_moving in d]
    H, W = shape[:2]
    left = pts[:, 0].min() - side
    if m:                                                             # M's leftmost resting place: its right edge
        cx = np.array([c[:, 0].mean() for c in m])
        start = [c for c, x in zip(m, cx) if x <= np.percentile(cx, 10)]
        left = float(np.median([c[:, 0].max() for c in start])) + 12
    return (int(max(0, left)), int(max(0, pts[:, 1].min() - top)), int(min(W, pts[:, 0].max() + side)), int(min(H, pts[:, 1].max() + bottom)))


def pixelate_outside(frame, boxes, block=40, cut=True):
    """Everything outside the boxes (x0, y0, x1, y1) is removed (cut: plain dark grey) or pixelated."""
    H, W = frame.shape[:2]
    if cut:
        out = np.full_like(frame, 38)
    else:
        small = cv2.resize(frame, (max(1, W // block), max(1, H // block)), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
    for x0, y0, x1, y1 in boxes:
        x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
    return out


def main():
    ap = argparse.ArgumentParser(description="Bonus: animation with the predicted pose behind the obstacle.")
    ap.add_argument("name", help="recording in data/recordings (without extension)")
    ap.add_argument("--seconds", type=float, default=32.0, help="animate the first ... seconds (0 = the whole recording)")
    ap.add_argument("--gif", action="store_true", help="also write a GIF (half size)")
    ap.add_argument("--show-all", action="store_true", help="show the whole picture (no privacy masking)")
    ap.add_argument("--pixelate", action="store_true", help="pixelate the picture outside the work area instead of cutting it out")
    ap.add_argument("--start", type=float, default=0.0, help="start the animation at this time of the recording [s]")
    a = ap.parse_args()
    cfg = load_markers_cfg()
    K, dist, _ = load_camera()
    video = data_path("recordings", a.name + ".avi")
    t, T, T_CF, corners = measure(video, cfg, K, dist, a.start + a.seconds if a.seconds > 0 else 0.0)
    pr = OcclusionPredictor(float(np.median(np.diff(t))), cfg["L"])
    outs = [pr.step(x) for x in T]                                 # the predictor runs from the beginning of the recording
    i0 = int(np.searchsorted(t, a.start))                          # ... the animation shows the part from --start on
    t, T, T_CF, corners, outs = t[i0:], T[i0:], T_CF[i0:], corners[i0:], outs[i0:]
    rows = [dict(frame=str(k + i0), t_s=f"{t[k]:.4f}", measured=str(int(T[k] is not None)), predicted=str(int(o["predicted"] and not o["lost"])),
                 lost=str(int(o["lost"])), sigma_mm=(f"{o['sigma_mm']:.2f}" if np.isfinite(o["sigma_mm"]) else ""), **pose_to_cols("FM", o["T"]))
            for k, o in enumerate(outs)]
    out = os.path.join(OUTPUT_DIR, "bonus")
    os.makedirs(out, exist_ok=True)
    write_csv(os.path.join(out, a.name + "_predicted_poses.csv"), rows)
    print(f"{a.name}: {len(rows)} frames ({t[-1]:.1f} s): measured {sum(r['measured'] == '1' for r in rows)}, "
          f"predicted {sum(r['predicted'] == '1' for r in rows)}, lost {sum(r['lost'] == '1' for r in rows)}")
    obj = marker_object_points(cfg["L"]).astype(np.float64)
    cap = cv2.VideoCapture(video)
    box = None if a.show_all else work_area(corners, (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))), cfg["id_moving"])
    cap.release()

    own = cv2.VideoCapture(video) if i0 else None                  # export reads frame k; with --start the frame k + i0 is needed

    def draw(k, frame):
        if own is not None:
            own.set(cv2.CAP_PROP_POS_FRAMES, k + i0)
            ok, f2 = own.read()
            frame = f2 if ok else frame
        o = outs[k]
        measured = cfg["id_moving"] in corners[k] and T[k] is not None
        uv = None
        if not measured and o["T"] is not None and o["predicted"] and not o["lost"] and T_CF[k] is not None:
            Tc = T_CF[k] @ o["T"]
            uv = cv2.projectPoints(obj, cv2.Rodrigues(Tc[:3, :3])[0], Tc[:3, 3], K, dist)[0].reshape(-1, 2).astype(np.int32)
        if box is not None:                                               # sharp: the path + a small box around M where it is now
            cur = corners[k][cfg["id_moving"]] if measured else uv
            boxes = [box] + ([] if cur is None else [(cur[:, 0].min() - 16, max(box[1], cur[:, 1].min() - 16), cur[:, 0].max() + 16, min(box[3], cur[:, 1].max() + 16))])
            frame = pixelate_outside(frame, boxes, cut=not a.pixelate)
        if measured:
            cv2.polylines(frame, [corners[k][cfg["id_moving"]].astype(np.int32)], True, GREEN, 3, cv2.LINE_AA)
        elif uv is not None:                                              # predicted: the red box
            cv2.polylines(frame, [uv], True, RED, 4, cv2.LINE_AA)
            cv2.putText(frame, "predicted", (int(uv[:, 0].min()), int(uv[:, 1].min()) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2, cv2.LINE_AA)
        return frame
    files = export(rows, t, [o["T"] for o in outs], video, os.path.join(out, a.name + "_occlusion"), formats=("mp4", "gif") if a.gif else ("mp4",),
                   frame_fn=draw)
    print("->", ", ".join(files))


if __name__ == "__main__":
    main()
