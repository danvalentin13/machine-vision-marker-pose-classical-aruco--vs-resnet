"""t7_live.py -- Bonus, live: the classical pipeline with the occlusion predictor on the camera, free motion.

  python t7_live.py

No timer, no countdown: F stays still, move M as you like, also behind an obstacle.
Left   the camera: GREEN outline = M measured by the classical pipeline (ArUco + IPPE + LM);
       RED box 'predicted' = where the predictor puts M while it is not measured (behind the obstacle), with the 3 sigma
       uncertainty in the caption; 'pose LOST' once the uncertainty is too large; then the re-acquisition.
Right  the pose of M in the frame of F over the last seconds: 3D view (measured dotted, predicted dashed orange with the
       uncertainty circles), x, y, z [mm] and roll, pitch, yaw [deg]; the predicted stretches are shaded orange.
Keys (click the window first):  S screenshot (output/bonus)    C clear the plots    Q or Esc stop
"""
import argparse
import os
import time

import cv2
import numpy as np

from config import OUTPUT_DIR
from mvlib import marker_object_points
from mvtools import MarkerPoseEstimator, load_camera, load_markers_cfg
from t2_detect_live import Camera
from t4_animate import PosePlot, hstack_to_height
from t7_bonus_occlusion import OcclusionPredictor

GREEN, RED, ORANGE = (60, 200, 60), (40, 40, 235), (26, 150, 255)


def set_dt(pr, dt):
    """The live frame time varies: give the filters the time since the last frame."""
    kf = pr.kf
    I3, Z3 = np.eye(3), np.zeros((3, 3))
    kf.dt = dt
    kf.F = np.block([[I3, dt * I3], [Z3, I3]])
    kf.Q = kf.sigma_a ** 2 * np.block([[dt ** 4 / 4 * I3, dt ** 3 / 2 * I3], [dt ** 3 / 2 * I3, dt ** 2 * I3]])
    pr.rf.dt = pr.dt = dt


def main():
    ap = argparse.ArgumentParser(description="Bonus, live: classical pipeline + occlusion predictor, free motion.")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--window-s", type=float, default=20.0, help="how many seconds the plots show")
    a = ap.parse_args()
    cfg = load_markers_cfg()
    K, dist, _ = load_camera()
    obj = marker_object_points(cfg["L"]).astype(np.float64)
    cam = Camera(a.camera, 1280, 720)
    if not cam.start():
        raise SystemExit("cannot open the camera (close other camera windows; check the camera privacy key)")
    est = MarkerPoseEstimator(cfg, K, dist, camera_still=True)
    pr = OcclusionPredictor(1 / 30, cfg["L"])
    plot = PosePlot(live=True)
    win = "Bonus live - classical pipeline + predictor  (S screenshot, C clear, Q stop)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    out_dir = os.path.join(OUTPUT_DIR, "bonus")
    os.makedirs(out_dir, exist_ok=True)
    print("Window open: F stays still, move M freely (also behind an obstacle).  S screenshot, C clear, Q stop.")
    t, poses, pred, sigma = [], [], [], []
    t0 = t_last = time.time()
    last_CF, plot_img, n_plot, shot_until = None, None, 0, 0.0
    while True:
        frame = cam.read()
        if frame is None:
            continue
        now = time.time()
        set_dt(pr, float(np.clip(now - t_last, 0.005, 0.2)))
        t_last = now
        r = est.process(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        last_CF = r["T_CF"] if r["T_CF"] is not None else last_CF
        o = pr.step(r["T_FM"])
        is_pred = bool(o["predicted"] and not o["lost"] and o["T"] is not None)
        t.append(now - t0); poses.append(o["T"]); pred.append(is_pred); sigma.append(o["sigma_mm"] if is_pred else np.nan)
        keep = int(np.searchsorted(t, t[-1] - a.window_s))
        if keep > 200:                                                # forget what is older than the plot window
            t, poses, pred, sigma = t[keep:], poses[keep:], pred[keep:], sigma[keep:]
        disp = frame.copy()
        for mid, c in r["corners"].items():
            cv2.polylines(disp, [c.astype(np.int32)], True, GREEN, 3, cv2.LINE_AA)
            cv2.putText(disp, "F" if mid == cfg["id_fixed"] else "M" if mid == cfg["id_moving"] else "?", tuple(c.reshape(-1, 2)[0].astype(int) - [0, 12]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2, cv2.LINE_AA)
        if r["T_FM"] is None and is_pred and last_CF is not None:     # the red box: where M is predicted
            Tc = last_CF @ o["T"]
            uv = cv2.projectPoints(obj, cv2.Rodrigues(Tc[:3, :3])[0], Tc[:3, 3], K, dist)[0].reshape(-1, 2).astype(np.int32)
            cv2.polylines(disp, [uv], True, RED, 4, cv2.LINE_AA)
            cv2.putText(disp, "predicted", (int(uv[:, 0].min()), int(uv[:, 1].min()) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2, cv2.LINE_AA)
        if plot_img is None or len(t) - n_plot >= 3 or n_plot > len(t):
            plot.set_data(np.array(t), poses, pred, sigma)
            plot_img = plot.render(len(poses) - 1)
            n_plot = len(t)
        view = hstack_to_height(disp, plot_img, 480)
        if last_CF is None:
            msg, col = "F not seen yet - show F to the camera (F stays still)", RED
        elif r["T_FM"] is not None:
            msg, col = "M MEASURED (classical).  Move M freely - also behind an obstacle", GREEN
        elif o["lost"]:
            msg, col = "pose LOST (uncertainty too large) - show M again", RED
        elif is_pred:
            msg, col = f"M not measured: PREDICTED  +-{3 * o['sigma_mm']:.0f} mm (3 sigma)", ORANGE
        else:
            msg, col = "M not seen yet - show M to the camera", ORANGE
        cv2.rectangle(view, (0, view.shape[0] - 32), (view.shape[1], view.shape[0]), (30, 30, 30), -1)
        cv2.putText(view, msg + "     S screenshot  C clear  Q stop", (10, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 2, cv2.LINE_AA)
        if now < shot_until:
            cv2.putText(view, "SCREENSHOT SAVED", (view.shape[1] // 2 - 170, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, GREEN, 3, cv2.LINE_AA)
        cv2.imshow(win, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord("s"), ord("S")):
            p = os.path.join(out_dir, f"live_{time.strftime('%H%M%S')}.png")
            cv2.imwrite(p, view)
            print("screenshot ->", p)
            shot_until = now + 1.5
        if key in (ord("c"), ord("C")):
            t, poses, pred, sigma, n_plot = [], [], [], [], 0
            t0 = now
    cam.stop()
    cv2.destroyAllWindows()
    plot.close()


if __name__ == "__main__":
    main()
