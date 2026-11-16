"""t8_live_all.py -- everything at once, live: classical pipeline, network and occlusion predictor on the same picture.

  python t8_live_all.py                        the laptop camera     (easiest: python live.py in the project folder)
  python t8_live_all.py --video E4_cover       the same display on a recording of data/recordings (no camera needed)

Display (1920 x 1080, scalable window, anti-aliased TrueType text)
  top left     the camera picture at full resolution:
               GREEN  F and M as the classical pipeline measures them (ArUco + IPPE + LM), with the axes of F
               RED    M as the two-stage network sees it (it answers in every frame, also when M is covered - or absent)
               BLUE   M where the occlusion predictor puts it while the classical pipeline has no measurement of M
  bottom left  six live plots of the pose of M in the frame of F over the last seconds: x, y, z [mm], roll, pitch, yaw [deg];
               green = classical, red = network, blue = predictor (drawn only while it predicts)
  right        the numbers: x, y, z, roll, pitch, yaw, distance F-M, distance camera-M for each method, the difference
               network - classical, the predictor's state and its 3 sigma uncertainty, frame rate
F is always measured classically (if F is not seen, its last pose is held: keep the camera still).
Keys (click the window first):  S screenshot (output/live)   N network on / off   C restart the predictor   P pause plots   Q or Esc quit
Options: --network NAME (in training/models, default crop_resnet18_v1_ft3), --no-network, --camera N, --camera-file PATH,
         --plot-seconds S (default 15), --video NAME [--frames N] [--no-window] [--save-frame PATH].
"""
import argparse
import functools
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import OUTPUT_DIR, data_path, training_path
from mvlib import inv_T, marker_object_points, rot_error_deg
from mvtools import MarkerPoseEstimator, load_camera, load_markers_cfg, rpy_deg
from t2_detect_live import Camera
from t7_bonus_occlusion import OcclusionPredictor

# colours (BGR for OpenCV)
GREEN, RED, BLUE = (70, 200, 70), (60, 60, 235), (240, 150, 50)              # classical, network, predictor
WHITE, LIGHT, GREY, DIM = (240, 240, 240), (205, 205, 205), (150, 150, 150), (90, 90, 90)
BG, BG2, GRID = (28, 28, 28), (38, 38, 38), (62, 62, 62)
W, H = 1920, 1080                                                           # the whole display
CAM_W, CAM_H = 1280, 720                                                    # camera picture inside it
PANEL_X = CAM_W                                                             # numbers panel: x 1280-1920, full height
PLOT_Y = CAM_H                                                              # plots: under the picture, 1280 x 360
COMP = ("x", "y", "z", "roll", "pitch", "yaw")
UNIT = ("mm", "mm", "mm", "deg", "deg", "deg")


# ----------------------------------------------------------------------------------------------------------- text
def _font_file(mono=False, bold=False):
    """A good TrueType font on this computer: Segoe UI / Consolas on Windows, else DejaVu (always shipped with matplotlib)."""
    win = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    names = (["consolab.ttf", "consola.ttf"] if mono else ["segoeuib.ttf", "segoeui.ttf"] if bold else ["segoeui.ttf"])
    for n in names:
        if os.path.exists(os.path.join(win, n)):
            return os.path.join(win, n)
    try:
        from matplotlib import font_manager
        return font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans Mono" if mono else "DejaVu Sans", weight="bold" if bold else "normal"))
    except Exception:
        return None


@functools.lru_cache(maxsize=64)
def font(size, mono=False, bold=False):
    f = _font_file(mono, bold)
    return ImageFont.truetype(f, size) if f else ImageFont.load_default()


class Text:
    """Collects text while the frame is drawn, then renders all of it in one pass with PIL (anti-aliased)."""

    def __init__(self):
        self.items = []

    def put(self, x, y, s, col=WHITE, size=22, mono=False, bold=False, anchor="la"):
        self.items.append((x, y, s, col, size, mono, bold, anchor))

    def render(self, img):
        im = Image.fromarray(img[:, :, ::-1])
        d = ImageDraw.Draw(im)
        for x, y, s, col, size, mono, bold, anchor in self.items:
            d.text((x, y), s, fill=(col[2], col[1], col[0]), font=font(size, mono, bold), anchor=anchor)
        return np.ascontiguousarray(np.asarray(im)[:, :, ::-1])


# ----------------------------------------------------------------------------------------------------------- helpers
def set_dt(pr, dt):
    """The live frame time varies: give the filters the time since the last frame."""
    kf = pr.kf
    I3, Z3 = np.eye(3), np.zeros((3, 3))
    kf.dt = dt
    kf.F = np.block([[I3, dt * I3], [Z3, I3]])
    kf.Q = kf.sigma_a ** 2 * np.block([[dt ** 4 / 4 * I3, dt ** 3 / 2 * I3], [dt ** 3 / 2 * I3, dt ** 2 * I3]])
    pr.rf.dt = pr.dt = dt


def outline(T, obj, K, dist):
    return cv2.projectPoints(obj, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, dist)[0].reshape(-1, 2)


def pose6(T):
    """x, y, z [mm] and roll, pitch, yaw [deg] of a 4x4 pose."""
    return np.r_[1000 * T[:3, 3], rpy_deg(T[:3, :3])]


def nice_range(v):
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return -1.0, 1.0
    lo, hi = float(v.min()), float(v.max())
    span = max(hi - lo, 4.0)
    mid = (hi + lo) / 2
    return mid - 0.6 * span, mid + 0.6 * span


def fmt(v, d=1, w=7):
    return f"{v:+{w}.{d}f}".replace("-", "−") if np.isfinite(v) else "—".rjust(w)


# ----------------------------------------------------------------------------------------------------------- drawing
def draw_camera(canvas, tx, disp, labels):
    h, w = disp.shape[:2]
    if (w, h) != (CAM_W, CAM_H):
        disp = cv2.resize(disp, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
    canvas[:CAM_H, :CAM_W] = disp
    sx, sy = CAM_W / w, CAM_H / h
    for x, y, s, col in labels:                                             # labels next to the outlines, on a dark ground
        x, y = int(x * sx), int(y * sy)
        f = font(24, bold=True)
        tw = f.getlength(s)
        cv2.rectangle(canvas, (x - 4, y - 28), (int(x + tw + 6), y + 4), (20, 20, 20), -1)
        tx.put(x, y, s, col, 24, bold=True, anchor="ls")


def draw_plots(canvas, tx, hist, now, span):
    """Six small plots under the camera picture: x y z (top row), roll pitch yaw (bottom row)."""
    canvas[PLOT_Y:, :CAM_W] = BG
    cw, ch = CAM_W // 3, (H - PLOT_Y) // 2
    t = np.array([h[0] for h in hist]) if hist else np.zeros(0)
    keep = t >= now - span
    t = t[keep] - now
    data = {k: np.array([h[1][k] for h in hist])[keep] if hist else np.zeros((0, 6)) for k in ("c", "n", "p")}
    for i in range(6):
        cx, cy = (i % 3) * cw, PLOT_Y + (i // 3) * ch
        x0, x1, y0, y1 = cx + 78, cx + cw - 16, cy + 48, cy + ch - 30
        cv2.rectangle(canvas, (x0, y0), (x1, y1), BG2, -1)
        allv = np.concatenate([data[k][:, i] for k in ("c", "n", "p")]) if len(t) else np.zeros(0)
        lo, hi = nice_range(allv)
        for f in (0.0, 0.5, 1.0):                                           # three grid lines with their values
            yy = int(y1 - f * (y1 - y0))
            cv2.line(canvas, (x0, yy), (x1, yy), GRID, 1, cv2.LINE_AA)
            tx.put(x0 - 8, yy, f"{lo + f * (hi - lo):.0f}".replace("-", "−"), GREY, 16, anchor="rm")
        for s in range(0, int(span) + 1, 5):                                 # time grid every 5 s
            xx = int(x1 - s / span * (x1 - x0))
            cv2.line(canvas, (xx, y0), (xx, y1), GRID, 1, cv2.LINE_AA)
            if i >= 3:
                tx.put(xx, y1 + 6, f"−{s} s" if s else "now", GREY, 15, anchor="ma")
        tx.put(cx + 14, cy + 8, COMP[i], WHITE, 21, bold=True)
        tx.put(cx + 14 + font(21, bold=True).getlength(COMP[i]) + 8, cy + 11, f"[{UNIT[i]}]", GREY, 17)
        for k, col, th in (("n", RED, 2), ("c", GREEN, 2), ("p", BLUE, 3)):  # classical drawn over network, predictor on top
            v = data[k][:, i] if len(t) else np.zeros(0)
            if not len(v):
                continue
            xs = x1 - (-t) / span * (x1 - x0)
            ys = y1 - (v - lo) / (hi - lo) * (y1 - y0)
            ok = np.isfinite(ys)
            # draw the finite runs as separate polylines (gaps where the method has no value)
            idx = np.where(ok)[0]
            if len(idx) < 2:
                continue
            breaks = np.where(np.diff(idx) > 1)[0]
            for seg in np.split(idx, breaks + 1):
                if len(seg) >= 2:
                    pts = np.stack([xs[seg], ys[seg]], 1)
                    cv2.polylines(canvas, [np.round(pts * 4).astype(np.int32)], False, col, th, cv2.LINE_AA, shift=2)
    cv2.line(canvas, (0, PLOT_Y), (CAM_W, PLOT_Y), GRID, 1)


def draw_panel(canvas, tx, info):
    canvas[:, PANEL_X:] = BG
    x, xr = PANEL_X + 32, W - 32
    y = 30
    tx.put(x, y, "Pose of M in the frame of F", WHITE, 30, bold=True)
    y += 50
    tx.put(x, y, info["status"], GREY, 19)
    y += 34
    cols = (x + 250, x + 400, x + 550)                                       # right edges of the three value columns

    def block(title, col, pose, extra):
        nonlocal y
        cv2.rectangle(canvas, (x, y + 6), (x + 12, y + 26), col, -1)
        tx.put(x + 24, y, title, col, 25, bold=True)
        y += 42
        if pose is None:                                                    # no pose: one grey line, same height as a full block
            tx.put(x + 24, y, extra[0] if extra else "—", GREY, 20)
            y += 34 * 2 + 30 * 2 + 18
            return
        for names, vals, unit in ((("x", "y", "z"), pose[:3], "mm"), (("roll", "pitch", "yaw"), pose[3:6], "deg")):
            for c, n_, v in zip(cols, names, vals):                     # name, value, unit on one baseline
                tx.put(c - 108, y + 24, n_, GREY, 18, anchor="rs")
                tx.put(c, y + 24, fmt(v, 1, 7), WHITE, 23, mono=True, anchor="rs")
            tx.put(cols[2] + 12, y + 24, unit, GREY, 18, anchor="ls")
            y += 34
        for e in extra:
            tx.put(x + 24, y, e[0], e[1], 19)
            y += 30
        y += 18

    block("CLASSICAL  ·  ArUco + IPPE + LM", GREEN, info["c"], info["c_extra"])
    block("NETWORK  ·  ResNet-18, two stages", RED, info["n"], info["n_extra"])
    block("PREDICTOR  ·  Kalman + α-β on SO(3)", BLUE, info["p"], info["p_extra"])
    yk = H - 92
    cv2.line(canvas, (x, yk - 16), (xr, yk - 16), GRID, 1)
    tx.put(x, yk, "S screenshot    N network on/off    C restart predictor", GREY, 18)
    tx.put(x, yk + 30, "P pause plots    Q quit", GREY, 18)


# ----------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Live: classical (green), network (red) and occlusion predictor (blue) on one picture.")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--network", default="crop_resnet18_v1_ft3", help="network in training/models (without .pt)")
    ap.add_argument("--video", default=None, help="recording in data/recordings (without extension) instead of the camera")
    ap.add_argument("--frames", type=int, default=0, help="with --video: stop after this many frames (0 = all)")
    ap.add_argument("--no-window", action="store_true", help="with --video: no window, only the summary (for testing)")
    ap.add_argument("--save-frame", default=None, help="with --video: write the display of the last frame to this file")
    ap.add_argument("--camera-file", default=None, help="camera calibration file (default: data/calibration/camera.npz)")
    ap.add_argument("--no-network", action="store_true", help="run without the network (green and blue only)")
    ap.add_argument("--plot-seconds", type=float, default=15.0, help="how many seconds the plots show")
    a = ap.parse_args()
    cfg = load_markers_cfg()
    K, dist, cal_size = load_camera(a.camera_file)
    L = cfg["L"]
    obj = marker_object_points(L).astype(np.float64)
    net, net_on = None, True
    path = training_path("models", a.network + ".pt")
    if a.no_network:
        print("running without the network")
    elif os.path.exists(path):
        from netpose import NetPose
        net = NetPose(path, K=K, dist=dist)
    else:
        print(f"network {path} not found: running without the network (train it first, README steps 5.1 - 5.6)")
    est = MarkerPoseEstimator(cfg, K, dist, camera_still=True)
    pr = OcclusionPredictor(1 / 30, L)
    if a.video:
        cap = cv2.VideoCapture(data_path("recordings", a.video + ".avi"))
        tp = data_path("recordings", a.video + ".csv")
        stamps = np.loadtxt(tp, delimiter=",", skiprows=1)[:, 1] if os.path.exists(tp) else None
        if not cap.isOpened():
            raise SystemExit(f"cannot open the recording {a.video}")
        read = lambda: (lambda ok_f: ok_f[1] if ok_f[0] else None)(cap.read())
    else:
        cam = Camera(a.camera, cal_size[0], cal_size[1])                   # ask the camera for the calibrated size
        if not cam.start():
            raise SystemExit("cannot open the camera (close other camera windows; check the camera privacy key)")
        read = cam.read
    win = "Live: classical (green) | network (red) | predictor (blue)"
    show = not a.no_window
    if show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        try:                                                                # open at a size that fits the screen
            import ctypes
            sw, sh = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
            f = min(1.0, 0.92 * sw / W, 0.88 * sh / H)
            cv2.resizeWindow(win, int(W * f), int(H * f))
        except Exception:
            cv2.resizeWindow(win, 1600, 900)
    out_dir = os.path.join(OUTPUT_DIR, "live")
    os.makedirs(out_dir, exist_ok=True)
    print("GREEN classical   RED network   BLUE occlusion predictor.   S screenshot, N network on/off, C restart predictor, P pause plots, Q quit.")
    last_CF, t_last, fps, n = None, None, 0.0, 0
    stat = dict(frames=0, classical=0, predicted=0, lost=0, diff_mm=[], diff_deg=[])
    view, hidden_s, hist, paused, t_start = None, 0.0, [], False, None
    first, t_wait = True, time.time()
    canvas = np.zeros((H, W, 3), np.uint8)
    while True:
        frame = read()
        if frame is None:
            if a.video:
                break
            if time.time() - t_wait > 8:
                raise SystemExit("the camera sends no picture (another program may be using it, or the camera privacy switch / Windows "
                                 "camera permission blocks it); try --camera 1 for a second camera")
            continue
        t_wait = time.time()
        if first:                                                           # a camera with another resolution: scale the calibration
            first = False
            h, w = frame.shape[:2]
            if (w, h) != tuple(cal_size):
                sx, sy = w / cal_size[0], h / cal_size[1]
                K = K.copy()
                K[0] *= sx
                K[1] *= sy
                print(f"note: the camera gives {w} x {h} px, the calibration is for {cal_size[0]} x {cal_size[1]} px: calibration scaled, "
                      "distances are approximate (recalibrate this camera: README steps 3.1 - 3.2)")
                est = MarkerPoseEstimator(cfg, K, dist, camera_still=True)
                if net is not None:
                    from netpose import NetPose
                    net = NetPose(path, K=K, dist=dist)
        now = float(stamps[n]) if (a.video and stamps is not None and n < len(stamps)) else time.time()
        t_start = now if t_start is None else t_start
        dt = 1 / 30 if t_last is None else float(np.clip(now - t_last, 0.005, 0.2))
        t_last = now
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        r = est.process(gray)                                               # classical: F and M
        last_CF = r["T_CF"] if r["T_CF"] is not None else last_CF
        T_CM_net = net.predict_one(gray) if (net is not None and net_on) else None
        set_dt(pr, dt)
        o = pr.step(r["T_FM"])                                              # predictor on the classical relative pose
        predicted = bool(r["T_FM"] is None and o["predicted"] and not o["lost"] and o["T"] is not None)
        hidden_s = 0.0 if r["T_FM"] is not None else hidden_s + dt
        el = max(1e-3, time.perf_counter() - t0)
        fps = 0.9 * fps + 0.1 / el if fps else 1.0 / el
        # ---- the camera picture with the outlines
        disp, labels = frame.copy(), []
        for mid, c in r["corners"].items():
            if mid in (cfg["id_fixed"], cfg["id_moving"]):
                cv2.polylines(disp, [np.round(c * 4).astype(np.int32)], True, GREEN, 3, cv2.LINE_AA, shift=2)
                labels.append((c[:, 0].min(), c[:, 1].min() - 10, "F" if mid == cfg["id_fixed"] else "M  classical", GREEN))
        if last_CF is not None:
            cv2.drawFrameAxes(disp, K, dist, cv2.Rodrigues(last_CF[:3, :3])[0], last_CF[:3, 3], 0.6 * L, 3)
        T_FM_net = None
        if T_CM_net is not None:
            uvn = outline(T_CM_net, obj, K, dist)
            cv2.polylines(disp, [np.round(uvn * 4).astype(np.int32)], True, RED, 2, cv2.LINE_AA, shift=2)
            labels.append((uvn[:, 0].min(), uvn[:, 1].max() + 34, "M  network", RED))
            if last_CF is not None:
                T_FM_net = inv_T(last_CF) @ T_CM_net
        if predicted and last_CF is not None:
            uvp = outline(last_CF @ o["T"], obj, K, dist)
            cv2.polylines(disp, [np.round(uvp * 4).astype(np.int32)], True, BLUE, 4, cv2.LINE_AA, shift=2)
            labels.append((uvp[:, 0].min(), uvp[:, 1].min() - 10, "M  predicted", BLUE))
        # ---- history for the plots
        p_c = pose6(r["T_FM"]) if r["T_FM"] is not None else np.full(6, np.nan)
        p_n = pose6(T_FM_net) if T_FM_net is not None else np.full(6, np.nan)
        p_p = pose6(o["T"]) if predicted else np.full(6, np.nan)
        if not paused:
            hist.append((now, dict(c=p_c, n=p_n, p=p_p)))
            while hist and hist[0][0] < now - a.plot_seconds - 1:
                hist.pop(0)
        # ---- the numbers
        def dist_text(T_FM, T_CM):
            s = f"F to M  {np.linalg.norm(1000 * T_FM[:3, 3]):.1f} mm"
            return s + ("" if T_CM is None else f"      camera to M  {100 * np.linalg.norm(T_CM[:3, 3]):.1f} cm")
        f_state = "F measured" if r["F_found"] else "F held (not seen)" if last_CF is not None else "F not seen yet"
        cam_f = "" if last_CF is None else f"   ·   camera to F {100 * np.linalg.norm(last_CF[:3, 3]):.1f} cm"
        info = dict(status=f"{fps:.0f} frames/s   ·   {f_state}{cam_f}" + ("   ·   plots paused" if paused else ""))
        if r["T_FM"] is not None:
            info["c"] = p_c
            info["c_extra"] = [(dist_text(r["T_FM"], r["T_CM"]), LIGHT),
                               (f"reprojection {r['rms_M']:.2f} px" + ("   ·   planar pose ambiguous" if r["amb_M"] else ""), GREY)]
        else:
            info["c"], info["c_extra"] = None, ["no pose of M (not detected)" if last_CF is not None else "show F to the camera"]
        if T_FM_net is not None:
            info["n"] = p_n
            ex = [(dist_text(T_FM_net, T_CM_net), LIGHT)]
            if r["T_FM"] is not None:
                dmm = 1000 * float(np.linalg.norm(T_FM_net[:3, 3] - r["T_FM"][:3, 3]))
                ddeg = rot_error_deg(r["T_FM"][:3, :3], T_FM_net[:3, :3])
                ex.append((f"difference to classical  {dmm:.1f} mm   {ddeg:.1f}°", GREY))
                stat["diff_mm"].append(dmm); stat["diff_deg"].append(ddeg)
            else:
                ex.append(("classical has no pose: the network answers alone", GREY))
            info["n_extra"] = ex
        else:
            info["n"], info["n_extra"] = None, ["network off (N)" if net is not None and not net_on else "no network file" if net is None else "show F to the camera"]
        if o["T"] is not None and not o["lost"]:
            info["p"] = pose6(o["T"])
            info["p_extra"] = [(dist_text(o["T"], None if last_CF is None else last_CF @ o["T"]), LIGHT),
                               ((f"PREDICTING for {hidden_s:.1f} s" if predicted else "following the measurement") + f"   ·   3σ {3 * o['sigma_mm']:.0f} mm",
                                BLUE if predicted else GREY)]
        elif o["lost"]:
            info["p"], info["p_extra"] = None, ["pose LOST (3σ larger than 2 marker sides): show M again"]
        else:
            info["p"], info["p_extra"] = None, ["waiting for the first measurement of M"]
        # ---- compose
        tx = Text()
        draw_camera(canvas, tx, disp, labels)
        draw_plots(canvas, tx, hist, hist[-1][0] if hist else now, a.plot_seconds)
        draw_panel(canvas, tx, info)
        view = tx.render(canvas)
        stat["frames"] += 1
        stat["classical"] += r["T_FM"] is not None
        stat["predicted"] += predicted
        stat["lost"] += bool(o["lost"])
        n += 1
        if a.video and a.frames and n >= a.frames:
            break
        if show:
            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("s"), ord("S")):
                fn = os.path.join(out_dir, f"live_all_{time.strftime('%H%M%S')}.png")
                cv2.imwrite(fn, view)
                print("screenshot ->", fn)
            if key in (ord("n"), ord("N")) and net is not None:
                net_on = not net_on
            if key in (ord("c"), ord("C")):
                pr = OcclusionPredictor(1 / 30, L)
            if key in (ord("p"), ord("P")):
                paused = not paused
    if a.video:
        cap.release()
    else:
        cam.stop()
    if show:
        cv2.destroyAllWindows()
    if a.save_frame and view is not None:
        cv2.imwrite(a.save_frame, view)
    m = stat["frames"]
    if m:
        d = stat["diff_mm"]
        print(f"{m} frames, {fps:.0f} frames/s at the end; classical pose of M in {100 * stat['classical'] / m:.0f} %, predicted in {stat['predicted']} frames, "
              f"lost in {stat['lost']} frames" + (f"; network - classical: median {np.median(d):.1f} mm, {np.median(stat['diff_deg']):.1f} deg" if d else ""))


if __name__ == "__main__":
    main()
