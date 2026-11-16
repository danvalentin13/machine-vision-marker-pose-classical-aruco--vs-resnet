"""t4_animate.py -- Task 4: animate the relative position and attitude of M in the frame of F.

  python t4_animate.py                 asks:  export | live
  export  reads <recording>_poses.csv (from t3_pose.py) and writes an MP4 and/or GIF:
          left the camera video with the detections, then a 3D view in the F frame (x red, y green, z blue, trajectory of M)
          and the x, y, z [mm] and roll, pitch, yaw [deg] curves against time, with a cursor synchronised to the video
  live    the same display from the camera, frame by frame (needs camera.npz)
Optional smoothing: constant-velocity Kalman filter (position) and alpha-beta filter on SO(3) (orientation), gated, as in the lecture.
Writes  output/animation/<name>.mp4 / .gif   (and <name>_filtered_poses.csv when the filter is on)
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np

from config import data_path, OUTPUT_DIR
from mvlib import AlphaBetaSO3, ConstantVelocityKalman
from mvtools import (MarkerPoseEstimator, ask, ask_choice, ask_folder, ask_number, ask_yes, cols_to_pose, default_camera_path,
                     load_camera, load_markers_cfg, pose_to_cols, read_csv, rpy_deg, write_csv, make_T)

AXIS_COLORS = ("#D62728", "#2CA02C", "#1F77B4")          # x red, y green, z blue


# ----------------------------------------------------------------------------------------------
# smoothing
# ----------------------------------------------------------------------------------------------
def filter_poses(t, poses, sigma_a=0.15, sigma_m=0.004, max_coast_s=1.0):
    """Gated constant-velocity Kalman + alpha-beta on SO(3).  Returns (poses, predicted_flags); a pose is None
    until the first measurement and after `max_coast_s` without an accepted measurement."""
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    kf, rf = ConstantVelocityKalman(dt, sigma_a, sigma_m), AlphaBetaSO3(dt)
    out, pred, since, rej = [], [], 0.0, 0
    for T in poses:
        if kf.x is None:
            if T is not None:
                kf.init(T[:3, 3]); rf.init(T[:3, :3]); since = 0.0
                out.append(T); pred.append(False)
            else:
                out.append(None); pred.append(False)
            continue
        kf.predict(); rf.predict()
        accepted = False
        if T is not None:
            if kf.update(T[:3, 3]):
                rf.update(T[:3, :3]); accepted, rej = True, 0
            else:
                rej += 1
                if rej >= 5:
                    kf.init(T[:3, 3]); rf.init(T[:3, :3]); accepted, rej = True, 0
        since = 0.0 if accepted else since + dt
        if since > max_coast_s:
            out.append(None); pred.append(True)
        else:
            out.append(make_T(rf.R, kf.position)); pred.append(not accepted)
    return out, pred


# ----------------------------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------------------------
class PosePlot:
    """3D view + x,y,z and roll,pitch,yaw curves; render(k) returns a BGR image."""

    def __init__(self, size=(720, 480), live=False, lims=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.live, self.size = live, size
        self.fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.15, 1], left=0.0, right=0.98, top=0.97, bottom=0.09, wspace=0.18, hspace=0.12)
        self.ax3 = self.fig.add_subplot(gs[:, 0], projection="3d")
        self.axp = self.fig.add_subplot(gs[0, 1])
        self.axr = self.fig.add_subplot(gs[1, 1])
        self.lims, self.t, self.poses = lims, np.zeros(0), []
        self.p = np.zeros((0, 3)); self.e = np.zeros((0, 3))
        self.cur = [self.axp.axvline(0, color="k", lw=0.8), self.axr.axvline(0, color="k", lw=0.8)]
        self.curves = []

    def set_data(self, t, poses, pred=None, sigma=None):
        """pred: per-frame flag 'M not measured, pose predicted'; sigma: per-frame position uncertainty 1 sigma [mm] (or None)."""
        self.t, self.poses = np.asarray(t, float), poses
        self.pred = np.zeros(len(poses), bool) if pred is None else np.asarray(pred, bool)
        self.sigma = None if sigma is None else np.asarray(sigma, float)
        self.p = np.array([1000 * T[:3, 3] if T is not None else [np.nan] * 3 for T in poses]).reshape(-1, 3)
        e = np.array([rpy_deg(T[:3, :3]) if T is not None else [np.nan] * 3 for T in poses]).reshape(-1, 3)
        for k in range(3):                                 # unwrap the angles for plotting only (never average Euler angles)
            ok = np.isfinite(e[:, k])
            if ok.sum() > 1:
                e[ok, k] = np.degrees(np.unwrap(np.radians(e[ok, k])))
        self.e = e
        if (self.lims is None or self.live) and np.isfinite(self.p).any():
            lo = np.minimum(np.nanmin(self.p, 0) / 1000, 0.0)            # F (the origin) is always inside the box
            hi = np.maximum(np.nanmax(self.p, 0) / 1000, 0.0)
            c, span = (lo + hi) / 2, max(float((hi - lo).max()), 0.1) * 0.6 + 0.12
            self.lims = (c - span, c + span)
        for ln in self.curves:
            ln.remove()
        self.curves = []
        for k, n in enumerate("xyz"):
            self.curves += self.axp.plot(self.t, self.p[:, k], color=AXIS_COLORS[k], lw=1, label=n)
        for k, n in enumerate(("roll", "pitch", "yaw")):
            self.curves += self.axr.plot(self.t, self.e[:, k], color=AXIS_COLORS[k], lw=1, label=n)
        if self.pred.any():                                # predicted stretches (M not measured): shaded on both curve plots
            edges = np.flatnonzero(np.diff(np.concatenate([[0], self.pred.astype(int), [0]])))
            for i0, i1 in zip(edges[::2], edges[1::2]):
                for ax in (self.axp, self.axr):
                    self.curves.append(ax.axvspan(self.t[i0], self.t[min(i1, len(self.t) - 1)], color="#D9731A", alpha=0.18, lw=0))
        for ax, yl in ((self.axp, "[mm]"), (self.axr, "[deg]")):
            ax.set_ylabel(yl, fontsize=7); ax.tick_params(labelsize=6); ax.grid(alpha=0.3, lw=0.4)
            ax.legend(fontsize=6, frameon=False, ncol=3, loc="upper right")
            if len(self.t) > 1:
                ax.set_xlim(self.t[0], max(self.t[-1], self.t[0] + 1))
            ax.relim(); ax.autoscale_view(scalex=False)
        self.axp.set_xticklabels([]); self.axr.set_xlabel("t [s]", fontsize=7)

    def render(self, k):
        k = min(k, len(self.poses) - 1)
        for c in self.cur:
            c.set_xdata([self.t[k], self.t[k]])
        ax = self.ax3
        ax.cla()
        lo, hi = self.lims if self.lims is not None else (np.array([-.4, -.1, -.1]), np.array([.4, .6, .3]))
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]))
        ax.tick_params(labelsize=5, pad=-3)
        ax.set_xlabel("x [m]", fontsize=6, labelpad=-8); ax.set_ylabel("y [m]", fontsize=6, labelpad=-8); ax.set_zlabel("z [m]", fontsize=6, labelpad=-8)
        ax.view_init(elev=20, azim=-30, vertical_axis="y")           # like the camera: F's y up, x right, z toward the viewer
        s = 0.18 * float(np.max(hi - lo))
        for T, sc, lw in ((np.eye(4), s, 2.5), (self.poses[k], 0.6 * s, 2.0)):          # F, M
            if T is None:
                continue
            for i in range(3):
                ax.plot(*zip(T[:3, 3], T[:3, 3] + sc * T[:3, i]), color=AXIS_COLORS[i], lw=lw)
        ax.text(0, -0.06 * float(np.max(hi - lo)), 0, "F", fontsize=9, color="#1F3A5F", weight="bold")
        tr = self.p[:k + 1] / 1000
        if np.isfinite(tr).any():
            pr = self.pred[:k + 1]
            start = 0
            for i in range(1, len(tr) + 1):                       # measured parts dotted black, predicted parts dashed orange
                if i == len(tr) or pr[i] != pr[start]:
                    seg = tr[max(start - 1, 0):i]
                    ax.plot(*seg.T, "--" if pr[start] else "-", color="#D9731A" if pr[start] else "#333333", lw=1.6 if pr[start] else 1.1)
                    start = i
        if self.poses[k] is not None:
            ax.text(*(self.poses[k][:3, 3] + [0, 0.06 * float(np.max(hi - lo)), 0]), "M", fontsize=9, color="#1E8C85", weight="bold")
            if self.pred[k] and self.sigma is not None and np.isfinite(self.sigma[k]):        # 3 sigma uncertainty as three circles
                c, r = self.poses[k][:3, 3], 3 * self.sigma[k] / 1000
                a = np.linspace(0, 2 * np.pi, 60)
                for u, v in ((0, 1), (0, 2), (1, 2)):
                    pts = np.tile(c, (len(a), 1))
                    pts[:, u] += r * np.cos(a); pts[:, v] += r * np.sin(a)
                    ax.plot(*pts.T, color="#D9731A", lw=0.7, alpha=0.8)
        self.fig.canvas.draw()
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        return cv2.cvtColor(np.ascontiguousarray(rgba[..., :3]), cv2.COLOR_RGB2BGR)

    def close(self):
        import matplotlib.pyplot as plt
        plt.close(self.fig)


def hstack_to_height(left, right, h):
    def fit(im):
        return cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h), interpolation=cv2.INTER_AREA)
    return np.hstack([fit(left), fit(right)])


# ----------------------------------------------------------------------------------------------
# export
# ----------------------------------------------------------------------------------------------
def load_poses_csv(path):
    rows = read_csv(path)
    t = np.array([float(r["t_s"]) for r in rows])
    return rows, t, [cols_to_pose(r, "FM") for r in rows]


def export(rows, t, poses, video_path, out_base, formats=("mp4",), target_fps=15, with_filter=True, verbose=True, frame_fn=None):
    """Writes out_base.mp4 / .gif.  Returns the list of files.
    frame_fn(k, frame) -> frame: optional drawing on the camera frame k before it is shown (overlays, hiding parts of the picture)."""
    pred, sigma, lost = [False] * len(poses), None, [False] * len(poses)
    if "predicted" in rows[0]:                       # file from t7_bonus_occlusion.py: predicted / lost / sigma are already there
        pred = [r["predicted"] == "1" for r in rows]
        lost = [r.get("lost") == "1" for r in rows]
        sigma = [float(r["sigma_mm"]) if r.get("sigma_mm") not in (None, "") else np.nan for r in rows]
        with_filter = False
    elif with_filter:
        poses, pred = filter_poses(t, poses)
        write_csv(out_base + "_filtered_poses.csv",
                  [dict(frame=r["frame"], t_s=r["t_s"], predicted=int(p), **pose_to_cols("FM", T)) for r, T, p in zip(rows, poses, pred)])
    src_fps = 1.0 / float(np.median(np.diff(t))) if len(t) > 1 else 30.0
    step = max(1, int(round(src_fps / target_fps)))
    out_fps = src_fps / step
    plot = PosePlot()
    plot.set_data(t, poses, pred, sigma)
    cap = cv2.VideoCapture(video_path) if video_path and os.path.exists(video_path) else None
    frames = []
    writer, files = None, []
    for k in range(0, len(poses), step):
        right = plot.render(k)
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, vf = cap.read()
            if ok and frame_fn is not None:
                vf = frame_fn(k, vf)
        else:
            ok, vf = False, None
        img = hstack_to_height(vf, right, right.shape[0]) if ok else right
        if lost[k]:
            cv2.putText(img, "pose LOST (uncertainty too large)", (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 230), 2, cv2.LINE_AA)
        elif pred[k]:
            extra = "" if sigma is None or not np.isfinite(sigma[k]) else f"  +-{3 * sigma[k]:.0f} mm (3 sigma)"
            cv2.putText(img, "predicted (M not measured)" + extra, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (26, 115, 217), 2, cv2.LINE_AA)
        if "mp4" in formats:
            if writer is None:
                writer = cv2.VideoWriter(out_base + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (img.shape[1], img.shape[0]))
            writer.write(img)
        if "gif" in formats:
            frames.append(cv2.cvtColor(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        if verbose and (k // step) % 50 == 0:
            print(f"  frame {k}/{len(poses)}")
    if writer is not None:
        writer.release(); files.append(out_base + ".mp4")
    if frames:
        from PIL import Image
        ims = [Image.fromarray(f) for f in frames]
        ims[0].save(out_base + ".gif", save_all=True, append_images=ims[1:], duration=int(1000 / out_fps), loop=0)
        files.append(out_base + ".gif")
    if cap is not None:
        cap.release()
    plot.close()
    return files


# ----------------------------------------------------------------------------------------------
# live
# ----------------------------------------------------------------------------------------------
def live(index, size, cfg, K, dist, camera_still, with_filter, out_dir):
    from t2_detect_live import Camera
    from t3_pose import draw_result
    cam = Camera(index, *size)
    print("Keys (click the window first):  Q or Esc = stop,  S = screenshot,  SPACE = pause")
    input("Press Enter to start the camera... ")
    if not cam.start():
        raise SystemExit("cannot open the camera")
    est = MarkerPoseEstimator(cfg, K, dist, camera_still)
    plot = PosePlot(live=True)
    win = "Task 4 - live relative pose"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    t, poses, t0, last_plot, plot_img, paused = [], [], time.time(), 0, None, False
    shot_until = 0.0
    while True:
        frame = cam.read()
        if frame is None:
            continue
        r = est.process(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        t.append(time.time() - t0); poses.append(r["T_FM"])
        shown_poses = poses
        if with_filter and len(poses) > 5:
            shown_poses, _ = filter_poses(np.array(t[-300:]), poses[-300:])
        if plot_img is None or len(poses) - last_plot >= 3:
            plot.set_data(np.array(t[-300:]), shown_poses[-300:])          # last ~10 s
            plot_img = plot.render(len(shown_poses[-300:]) - 1)
            last_plot = len(poses)
        view = hstack_to_height(draw_result(frame, r, K, dist, cfg["L"]), plot_img, 480)
        both = r["F_found"] and r["M_found"]
        if time.time() < shot_until:
            cv2.rectangle(view, (view.shape[1] // 2 - 190, 40), (view.shape[1] // 2 + 190, 95), (20, 20, 20), -1)
            cv2.putText(view, "SCREENSHOT SAVED", (view.shape[1] // 2 - 170, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (60, 200, 60), 3, cv2.LINE_AA)
        msg = ("LIVE: F stays still, move M slowly (left/right, closer/farther, turn, lift)   S screenshot   Q stop" if both else
               ("F not seen - F must stay in the picture   Q stop" if not r["F_found"] else "M not seen - bring M into the picture   Q stop"))
        cv2.rectangle(view, (0, view.shape[0] - 30), (view.shape[1], view.shape[0]), (30, 30, 30), -1)
        cv2.putText(view, msg, (10, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (26, 180, 255) if both else (60, 60, 235), 2, cv2.LINE_AA)
        cv2.imshow(win, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord("s"), ord("S")):
            os.makedirs(out_dir, exist_ok=True)
            p = os.path.join(out_dir, f"live_{time.strftime('%H%M%S')}.png")
            cv2.imwrite(p, hstack_to_height(draw_result(frame, r, K, dist, cfg["L"]), plot_img, 480)); print("screenshot ->", p)
            shot_until = time.time() + 1.5
    cam.stop(); cv2.destroyAllWindows(); plot.close()


def main():
    ap = argparse.ArgumentParser(description="Task 4: animate the relative pose.")
    ap.add_argument("mode", nargs="?", choices=["export", "live"])
    a = ap.parse_args()
    print("Task 4 - animation.  Press Enter to accept the [default].\n")
    mode = a.mode or ask_choice("Mode", ["export", "live"], "export")
    out_dir = ask_folder("Output folder", os.path.join(OUTPUT_DIR, "animation"))
    os.makedirs(out_dir, exist_ok=True)
    with_filter = ask_yes("Smooth with the Kalman / alpha-beta filter?", "y")
    if mode == "export":
        newest = sorted(glob.glob(os.path.join(OUTPUT_DIR, "pose", "*_poses.csv")), key=os.path.getmtime)
        csv_path = ask_folder("1) Poses file (from t3_pose.py)", newest[-1] if newest else os.path.join(OUTPUT_DIR, "pose", "name_poses.csv"))
        name = os.path.basename(csv_path).replace("_predicted_poses.csv", "").replace("_poses.csv", "")
        annotated = os.path.join(OUTPUT_DIR, "pose", name + "_annotated.avi")        # markers and axes drawn by t3_pose.py
        guess = annotated if os.path.exists(annotated) else data_path("recordings", name.replace("_net", "") + ".avi")
        video = ask_folder("2) Video to show next to the animation (the annotated one if it exists; - for none)", guess)
        fmt = ask_choice("3) Format", ["mp4", "gif", "both"], "mp4")
        rows, t, poses = load_poses_csv(csv_path)
        print(f"\n{len(poses)} frames, {t[-1] - t[0]:.1f} s, M measured in {sum(p is not None for p in poses)} frames")
        files = export(rows, t, poses, None if video == os.path.abspath("-") else video, os.path.join(out_dir, name + "_animation"),
                       ("mp4", "gif") if fmt == "both" else (fmt,), with_filter=with_filter)
        print("written:", *files, sep="\n  ")
    else:
        cfg = load_markers_cfg()
        K, dist, size = load_camera(ask_folder("Camera file", default_camera_path()))
        still = ask_yes("Is the camera still? (a missing F is replaced by the last F)", "y")
        live(ask_number("Camera index", 0, int, 0), size, cfg, K, dist, still, with_filter, out_dir)


if __name__ == "__main__":
    main()
