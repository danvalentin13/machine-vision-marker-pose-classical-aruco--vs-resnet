"""t3_calibrate.py -- Task 3, step 2: calibrate the camera (K and lens distortion) with the chessboard.

  python t3_calibrate.py            asks what to do:  capture | calibrate
  capture    guided: asks the number of views (default 20), then one instruction per view (distance, image region,
             tilt) in the terminal and in the window; SPACE saves the view, U redoes the last, N skips, Q finishes
  calibrate  reads the saved views, runs cv2.calibrateCamera and reports RMS, per-view errors and a held-out check

Writes  data/calibration/images/view_NNN.png     the saved views (full frames)
        data/calibration/camera.npz / .json      K, distortion, image size, RMS  (used by t3_pose.py and later tasks)
        output/calibration/                      per_view.csv, report.json, calibration_report.png
The calibration is valid only for the resolution and the lens settings used (do not change zoom/focus afterwards).
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

from config import data_path, OUTPUT_DIR
from mvtools import ask, ask_choice, ask_folder, ask_number, save_camera, write_csv

MODELS = {"k1k2": cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,        # k1, k2 only: simplest model that usually fits
          "full": 0}                                                      # k1, k2, p1, p2, k3
HELD_OUT_EVERY = 5
SB_FLAGS = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ACCURACY


# ----------------------------------------------------------------------------------------------
# board and corners
# ----------------------------------------------------------------------------------------------
def load_board(path=None):
    path = path or data_path("calibration", "board.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run t3_make_board.py first")
    b = json.load(open(path))
    b["square_used_mm"] = b.get("measured_square_mm") or b["square_mm"]
    return b


def board_points(board):
    """3D corner coordinates [m] on the board plane (Z = 0), in the order cv2 returns them."""
    cols, rows, s = board["cols"], board["rows"], board["square_used_mm"] / 1000.0
    return np.array([[c * s, r * s, 0.0] for r in range(rows) for c in range(cols)], np.float32)


def find_corners(gray, board):
    """Corner positions (N,1,2) float32 (sub-pixel) or None."""
    ok, c = cv2.findChessboardCornersSB(gray, (board["cols"], board["rows"]), flags=SB_FLAGS)
    return c.astype(np.float32) if ok else None


def tilt_estimate_deg(corners, board):
    """Board tilt from the foreshortening of the corner grid: arccos(s_min/s_max) of the affine map grid -> image."""
    cols, rows = board["cols"], board["rows"]
    grid = np.array([[c, r] for r in range(rows) for c in range(cols)], float)
    P = corners.reshape(-1, 2).astype(float)
    A = np.hstack([grid, np.ones((len(grid), 1))])
    M = np.linalg.lstsq(A, P, rcond=None)[0][:2].T                 # 2x2 linear part
    s = np.linalg.svd(M, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s[1] / s[0], 0, 1))))


# ----------------------------------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------------------------------
def calibrate(objp, corners_list, size, model="k1k2"):
    objs = [objp] * len(corners_list)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objs, corners_list, size, None, None, flags=MODELS[model])
    errs = []
    for o, c, rv, tv in zip(objs, corners_list, rvecs, tvecs):
        p, _ = cv2.projectPoints(o, rv, tv, K, dist)
        errs.append(float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - c.reshape(-1, 2)) ** 2, axis=1)))))
    tilts = [float(np.degrees(np.arccos(abs(cv2.Rodrigues(rv)[0][2, 2])))) for rv in rvecs]
    return dict(rms=float(rms), K=K, dist=dist.ravel(), per_view_rms=errs, tilt_deg=tilts, rvecs=rvecs, tvecs=tvecs)


def held_out_check(objp, corners_list, size, model="k1k2", every=HELD_OUT_EVERY):
    """Calibrate with 4 of 5 views, measure the reprojection error on the remaining views (poses re-estimated)."""
    n = len(corners_list)
    held = [i for i in range(n) if i % every == every - 1]
    if n < 10 or not held:
        return None
    train = [corners_list[i] for i in range(n) if i not in held]
    cal = calibrate(objp, train, size, model)
    errs = []
    for i in held:
        ok, rv, tv = cv2.solvePnP(objp, corners_list[i], cal["K"], cal["dist"])
        p, _ = cv2.projectPoints(objp, rv, tv, cal["K"], cal["dist"])
        errs.append(float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - corners_list[i].reshape(-1, 2)) ** 2, axis=1)))))
    return dict(views=held, rms_per_view=errs, rms_mean=float(np.mean(errs)), train_rms=cal["rms"],
                fx_train=float(cal["K"][0, 0]))


def run_calibration(images_dir, board, model, out_dir, camera_path):
    paths = sorted(glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg")))
    if not paths:
        raise SystemExit(f"no images in {images_dir}: capture views first")
    objp, corners, names, size = board_points(board), [], [], None
    for p in paths:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if size is None:
            size = (g.shape[1], g.shape[0])
        elif (g.shape[1], g.shape[0]) != size:
            print(f"  skipped {os.path.basename(p)}: different size {g.shape[1]}x{g.shape[0]}")
            continue
        c = find_corners(g, board)
        if c is None:
            print(f"  skipped {os.path.basename(p)}: board not found")
            continue
        corners.append(c)
        names.append(os.path.basename(p))
    print(f"{len(corners)} usable views of {len(paths)} images, image size {size[0]}x{size[1]}")
    if len(corners) < 3:
        raise SystemExit("need at least 3 views (assignment: >= 10)")

    cal = calibrate(objp, corners, size, model)
    held = held_out_check(objp, corners, size, model)
    K, dist = cal["K"], cal["dist"]
    fov = np.degrees(2 * np.arctan(size[0] / (2 * K[0, 0])))
    warnings = []
    if len(corners) < 10:
        warnings.append(f"only {len(corners)} views: the assignment asks for at least 10")
    if sum(t > 15 for t in cal["tilt_deg"]) < 3 or max(cal["tilt_deg"]) < 20:
        warnings.append("too little tilt: tilt the board by 20-45 degrees in several views (frontal views are degenerate)")
    if not board.get("measured_square_mm"):
        warnings.append("board.json has no measured_square_mm: all translations in metres rest on the nominal square size - measure the printed square")
    worst = int(np.argmax(cal["per_view_rms"]))
    rep = dict(model=model, n_views=len(corners), image_size=list(size), rms_px=cal["rms"],
               fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]), hfov_deg=float(fov),
               dist=[float(x) for x in dist], per_view_rms_max=cal["per_view_rms"][worst], worst_view=names[worst],
               held_out=held, square_used_mm=board["square_used_mm"], warnings=warnings)

    save_camera(camera_path, K, dist, size, cal["rms"], extra=dict(model=model, n_views=len(corners), square_mm=board["square_used_mm"]))
    os.makedirs(out_dir, exist_ok=True)
    write_csv(os.path.join(out_dir, "per_view.csv"),
              [dict(view=n, rms_px=round(e, 3), tilt_deg=round(t, 1), held_out=int(i % HELD_OUT_EVERY == HELD_OUT_EVERY - 1 and len(corners) >= 10))
               for i, (n, e, t) in enumerate(zip(names, cal["per_view_rms"], cal["tilt_deg"]))])
    json.dump(rep, open(os.path.join(out_dir, "report.json"), "w"), indent=1)
    report_figure(os.path.join(out_dir, "calibration_report.png"), corners, cal, size, names)

    print(f"\nRMS reprojection error : {cal['rms']:.3f} px  (all views)   worst view {names[worst]}: {cal['per_view_rms'][worst]:.3f} px")
    print(f"K: fx {K[0, 0]:.1f}  fy {K[1, 1]:.1f}  cx {K[0, 2]:.1f}  cy {K[1, 2]:.1f}   horizontal FOV {fov:.1f} deg")
    print(f"distortion (k1 k2 p1 p2 k3): {np.array2string(dist, precision=4, suppress_small=True)}")
    if held:
        print(f"held-out views ({len(held['views'])}): mean RMS {held['rms_mean']:.3f} px   (calibrated without them: {held['train_rms']:.3f} px)")
    print("(a low RMS alone does not prove a good calibration: check tilt, coverage and the held-out error)")
    for w in warnings:
        print("WARNING:", w)
    print(f"\ncamera -> {camera_path}\nreport -> {out_dir}")
    return rep


def report_figure(path, corners, cal, size, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    cm = plt.get_cmap("viridis")
    for i, c in enumerate(corners):
        p = c.reshape(-1, 2)
        ax[0].plot(p[:, 0], p[:, 1], ".", ms=2, color=cm(i / max(1, len(corners) - 1)))
    ax[0].set_xlim(0, size[0]); ax[0].set_ylim(size[1], 0); ax[0].set_aspect("equal")
    ax[0].set_title("detected corners of all views (coverage)", fontsize=9)
    ax[1].bar(range(len(names)), cal["per_view_rms"], color="#1F3A5F")
    ax[1].axhline(cal["rms"], color="#D9731A", lw=1); ax[1].set_title(f"per-view RMS [px] (line = overall {cal['rms']:.2f})", fontsize=9)
    ax[1].set_xlabel("view")
    ax[2].bar(range(len(names)), cal["tilt_deg"], color="#1E8C85")
    ax[2].set_title("board tilt per view [deg]", fontsize=9); ax[2].set_xlabel("view")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# live capture
# ----------------------------------------------------------------------------------------------
HELP = """
Keys (click the video window first):
  SPACE  save this view (only when the board is found)     U  remove the last saved view (back one instruction)
  N      skip this instruction                              H  this help                 Q or Esc  finish
Follow the instruction shown in the terminal and in the window (orange frame = where the board should be).
The panel on the right shows the measured distance and angles next to the target; the FIT bar grows and turns RED
when distance, region and tilt all fit. Angles are approximate before the first calibration (assumed field of view).
Keep the board still when you press SPACE; same resolution and focus as for the later tasks.
"""

# ---- guided capture: one instruction per view -------------------------------------------------------------
DIST = {"close": ("CLOSE", "the board fills about 2/3 of the image width, roughly 25 cm", 0.48, 1.01),        # ranges are loose:
        "medium": ("MEDIUM", "the board covers about half of the image width, roughly 35 cm", 0.26, 0.62),   # a tilted board looks
        "far": ("FAR", "the board covers about 1/3 of the image width, roughly 50 cm", 0.0, 0.42)}            # narrower
POS = {"centre": ((1, 1), "in the CENTRE of the image"), "top-left": ((0, 0), "in the TOP-LEFT corner of the image"),
       "top-right": ((2, 0), "in the TOP-RIGHT corner of the image"), "bottom-left": ((0, 2), "in the BOTTOM-LEFT corner of the image"),
       "bottom-right": ((2, 2), "in the BOTTOM-RIGHT corner of the image"), "left": ((0, 1), "at the LEFT edge of the image (middle)"),
       "right": ((2, 1), "at the RIGHT edge of the image (middle)"), "top": ((1, 0), "at the TOP edge of the image (middle)"),
       "bottom": ((1, 2), "at the BOTTOM edge of the image (middle)")}
TILT = {"none": "facing the camera straight on", "left": "turned so its LEFT edge is farther from the camera",
        "right": "turned so its RIGHT edge is farther from the camera", "top": "tilted so its TOP edge is farther from the camera",
        "bottom": "tilted so its BOTTOM edge is farther from the camera", "diag": "tilted so its TOP-LEFT corner is farther from the camera"}
PLAN = [("medium", "centre", "none", 0), ("medium", "centre", "left", 30), ("medium", "centre", "right", 30),
        ("medium", "centre", "top", 30), ("medium", "centre", "bottom", 30), ("close", "centre", "left", 30),
        ("close", "centre", "top", 30), ("close", "centre", "diag", 35), ("medium", "top-left", "right", 30),
        ("medium", "top-right", "left", 30), ("medium", "bottom-left", "top", 30), ("medium", "bottom-right", "diag", 30),
        ("far", "top-left", "bottom", 30), ("far", "top-right", "bottom", 30), ("far", "bottom-left", "top", 30),
        ("far", "bottom-right", "top", 30), ("far", "left", "right", 35), ("far", "right", "left", 35),
        ("medium", "top", "bottom", 30), ("medium", "bottom", "top", 30), ("close", "centre", "none", 10),
        ("medium", "centre", "diag", 45), ("far", "centre", "left", 45), ("close", "centre", "bottom", 45)]
DEFAULT_VIEWS = 20


def instruction(i, n):
    d, p, t, deg = PLAN[i % len(PLAN)]
    again = "  (again, a little different from before)" if i >= len(PLAN) else ""
    tilt = TILT[t] + (f", about {deg} deg" if deg and t != "none" else "")
    text = (f"View {i + 1}/{n}: {DIST[d][0]} distance ({DIST[d][1]}), {POS[p][1]}, {tilt}.{again}\n"
            f"           Hold it still and press SPACE.")
    short = f"View {i + 1}/{n}: {DIST[d][0]}, {p.upper()}, " + ("straight on" if t == "none" else f"{t.upper()} away ~{deg} deg")
    return text, short, (d, p, t, deg)


def guide_camera(size):
    """Camera used to estimate the board pose while capturing: camera.npz if it exists (same size), else an approximate
    laptop camera (horizontal field of view ~70 deg).  Only for the guide display - the calibration does not use it."""
    path = data_path("calibration", "camera.npz")
    if os.path.exists(path):
        d = np.load(path)
        if (int(d["width"]), int(d["height"])) == tuple(size):
            return d["K"].astype(float), d["dist"].astype(float).ravel(), "camera.npz"
    f = size[0] / (2 * np.tan(np.radians(35)))
    return np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]]), np.zeros(5), "approximate camera (before calibration)"


def board_angles(corners, board, K, dist):
    """Board pose from its corners: distance [cm], 'left edge away' and 'top edge away' [deg] (negative = the other edge),
    total tilt to the image plane [deg], in-plane rotation [deg]."""
    obj = board_points(board)
    ok, rv, tv = cv2.solvePnP(obj, corners, K, dist)
    if not ok:
        return None
    R = cv2.Rodrigues(rv)[0]
    n = R[:, 2]                                                    # board normal in camera coordinates
    P = (R @ obj.T).T + tv.ravel()
    ctr = P.mean(axis=0)
    left_away = float(np.degrees(np.arctan2(n[0], n[2]))) * (1 if n[2] > 0 else -1)   # dZ/dX < 0: left edge farther
    top_away = float(np.degrees(np.arctan2(n[1], n[2]))) * (1 if n[2] > 0 else -1)    # dZ/dY < 0: top edge farther
    left_away = left_away if abs(left_away) <= 90 else np.sign(left_away) * (180 - abs(left_away))
    top_away = top_away if abs(top_away) <= 90 else np.sign(top_away) * (180 - abs(top_away))
    return dict(dist_cm=100 * float(np.linalg.norm(ctr)), left_away=left_away, top_away=top_away,
                tilt=float(np.degrees(np.arccos(min(1.0, abs(n[2]))))),
                inplane=float((np.degrees(np.arctan2(R[1, 0], R[0, 0])) + 90) % 180 - 90))   # the board has no "up": fold to -90..90


def edge_text(a, pos_name, neg_name):
    """'LEFT edge away 28 deg' / 'RIGHT edge away 12 deg' / 'level' for a signed 'edge away' angle."""
    if abs(a) < 2:
        return "level (0 deg)"
    return f"{pos_name if a > 0 else neg_name} edge away {abs(a):.0f} deg"


def board_extent(corners, size):
    P = corners.reshape(-1, 2)
    frac = (P[:, 0].max() - P[:, 0].min()) / size[0] * 10 / 8          # inner corners span 8 of the 10 squares of a row
    cell = (min(2, int(P[:, 0].mean() / size[0] * 3)), min(2, int(P[:, 1].mean() / size[1] * 3)))
    return frac, cell


CELL_NAMES = {v[0]: k for k, v in POS.items()}


def match(target, corners, ang, size):
    """How well the current view fits the instruction: score 0..1 and per-item flags (distance, region, tilt)."""
    d, p, t, deg = target
    frac, cell = board_extent(corners, size)
    lo, hi = DIST[d][2], DIST[d][3]
    s_dist = 1.0 if lo <= frac <= hi else max(0.0, 1 - min(abs(frac - lo), abs(frac - hi)) / 0.2)
    s_reg = 1.0 if cell == POS[p][0] else 0.0
    la, ta = ang["left_away"], ang["top_away"]
    if t == "none":
        lim = max(15.0, deg + 10.0)
        s_tilt = 1.0 if ang["tilt"] <= lim else max(0.0, 1 - (ang["tilt"] - lim) / 30)
    else:
        v = {"left": la, "right": -la, "top": ta, "bottom": -ta, "diag": min(la, ta)}[t]
        need = (deg / np.sqrt(2) - 8) if t == "diag" else (deg - 10)
        s_tilt = 1.0 if v >= need else max(0.0, v / need)
    flags = dict(distance=s_dist == 1.0, region=s_reg == 1.0, tilt=s_tilt == 1.0)
    return (s_dist + s_reg + s_tilt) / 3, flags, frac, cell


def side_panel(h, ang, fit, target, cam_note, short, width=400):
    """Panel on the right of the video: all measured angles and distance next to the target, and the fit bar."""
    pan = np.full((h, width, 3), 32, np.uint8)
    font, white, grey, orange, green = cv2.FONT_HERSHEY_SIMPLEX, (235, 235, 235), (150, 150, 150), (26, 160, 255), (60, 200, 60)

    def put(y, text, col=white, scale=0.55, th=1):
        cv2.putText(pan, text, (14, y), font, scale, col, th, cv2.LINE_AA)
    put(30, "CALIBRATION GUIDE", white, 0.7, 2)
    put(54, cam_note, grey, 0.42)
    put(88, "TARGET", orange, 0.6, 2)
    d, p, t, deg = target
    put(112, f"distance  {d.upper()}  ({DIST[d][1].split(', ')[-1]})")
    put(134, f"region    {p.upper()}")
    put(156, "tilt      " + ("straight on" if t == "none" else f"{t.upper()} edge away ~{deg} deg" if t != "diag" else f"TOP-LEFT corner away ~{deg} deg"))
    put(196, "MEASURED", orange, 0.6, 2)
    if ang is None:
        put(222, "board not found", (80, 80, 255))
        return pan
    score, flags, frac, cell = fit

    def mark(ok):
        return ("OK", green) if ok else ("--", (80, 80, 255))
    rows = [(f"distance  ~{ang['dist_cm']:.0f} cm   board {100 * frac:.0f} % of width", flags["distance"]),
            (f"region    {CELL_NAMES.get(cell, '?').upper()}", flags["region"]),
            ("left/right " + edge_text(ang["left_away"], "LEFT", "RIGHT"), None),
            ("up/down    " + edge_text(ang["top_away"], "TOP", "BOTTOM"), None),
            (f"total tilt {ang['tilt']:.0f} deg", flags["tilt"]),
            (f"in-plane rotation {ang['inplane']:.0f} deg", None)]
    y = 222
    for text, ok in rows:
        put(y, text)
        if ok is not None:
            m, c = mark(ok)
            cv2.putText(pan, m, (width - 46, y), font, 0.55, c, 2, cv2.LINE_AA)
        y += 24
    # fit bar: grows with the score, turns RED when distance, region and tilt all fit
    y0, x0, x1 = y + 24, 14, width - 14
    put(y0 - 8, "FIT", orange, 0.6, 2)
    cv2.rectangle(pan, (x0, y0), (x1, y0 + 28), (90, 90, 90), 1)
    full = all(flags.values())
    col = (40, 40, 230) if full else (int(120 - 60 * score), int(120 + 60 * score), 200)
    cv2.rectangle(pan, (x0 + 1, y0 + 1), (x0 + 1 + int((x1 - x0 - 2) * score), y0 + 27), col, -1)
    put(y0 + 56, "FITS - hold still, press SPACE" if full else f"{100 * score:.0f} % - adjust the board", (40, 40, 230) if full else white, 0.58, 2)
    return pan


def view_feedback(corners, size, ang, target):
    """Text after saving a view: measured values and hints (the view is kept either way)."""
    if ang is None:
        return "pose not estimated"
    score, flags, frac, cell = match(target, corners, ang, size)
    hints = []
    if not flags["distance"]:
        hints.append("closer next time" if frac < DIST[target[0]][2] else "farther next time")
    if not flags["region"]:
        hints.append("not in the asked region")
    if not flags["tilt"]:
        hints.append("tilt differently (see the target)")
    lr, tb = edge_text(ang["left_away"], "LEFT", "RIGHT"), edge_text(ang["top_away"], "TOP", "BOTTOM")
    return (f"~{ang['dist_cm']:.0f} cm, {lr}, {tb}, tilt {ang['tilt']:.0f} deg" +
            ("  -> " + ", ".join(hints) + " (kept anyway; U = redo it)" if hints else "  OK"))


def capture_views(index, size, images_dir, board, n_views=DEFAULT_VIEWS):
    from t2_detect_live import Camera
    import time
    from datetime import datetime
    os.makedirs(images_dir, exist_ok=True)
    old = sorted(glob.glob(os.path.join(images_dir, "view_*.png")))
    if old:
        keep = ask(f"The folder already has {len(old)} views from an earlier session. Keep them and add (k), "
                   "or move them to a subfolder and start fresh (f)?", "f").lower().startswith("k")
        if not keep:
            arch = os.path.join(images_dir, f"old_{datetime.now():%Y%m%d_%H%M%S}")
            os.makedirs(arch)
            for f in old:
                os.replace(f, os.path.join(arch, os.path.basename(f)))
            print(f"   {len(old)} old views moved to {arch} (not deleted)")
    cam = Camera(index, *size)
    print(HELP)
    print(f"Plan: {n_views} views at 3 distances (judge the distance by how big the board looks), all image regions, tilts 20-45 deg.")
    print("No ruler or protractor is needed: the program estimates the tilt and position of every view itself.\n")
    input("Press Enter to start the camera... ")
    if not cam.start():
        raise SystemExit("cannot open the camera")
    win = "Task 3 - calibration views"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    saved = sorted(glob.glob(os.path.join(images_dir, "view_*.png")))
    step, msg, msg_t = 0, "", 0.0
    K_g, dist_g, cam_note, ang, fit = None, None, "", None, None
    text, short, target = instruction(step, n_views)
    print("\n" + text)
    while True:
        frame = cam.read()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = find_corners(gray, board)
        disp = frame.copy()
        h, w = disp.shape[:2]
        tilt = None
        for k in range(1, 3):                                         # 3x3 regions
            cv2.line(disp, (w * k // 3, 0), (w * k // 3, h), (90, 90, 90), 1)
            cv2.line(disp, (0, h * k // 3), (w, h * k // 3), (90, 90, 90), 1)
        if step < n_views:                                            # the region asked for
            cx, cy = POS[target[1]][0]
            cv2.rectangle(disp, (w * cx // 3 + 3, h * cy // 3 + 3), (w * (cx + 1) // 3 - 3, h * (cy + 1) // 3 - 3), (26, 115, 217), 3)
        if K_g is None:
            K_g, dist_g, cam_note = guide_camera((w, h))
            cam_note = "angles from " + cam_note
        ang = fit = None
        if corners is not None:
            cv2.drawChessboardCorners(disp, (board["cols"], board["rows"]), corners, True)
            tilt = tilt_estimate_deg(corners, board)
            ang = board_angles(corners, board, K_g, dist_g)
            if ang is not None:
                tilt = ang["tilt"]
                if step < n_views:
                    fit = match(target, corners, ang, (w, h))
        status = f"saved {len(saved)}   board {'FOUND, tilt ~%d deg' % tilt if tilt is not None else 'not found'}"
        cv2.rectangle(disp, (0, 0), (w, 30), (30, 30, 30), -1)
        cv2.putText(disp, status, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(disp, (0, h - 36), (w, h), (30, 30, 30), -1)
        bottom = short if step < n_views else f"All {n_views} views done - press Q to finish (or SPACE to add more)"
        cv2.putText(disp, bottom, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (26, 180, 255), 2, cv2.LINE_AA)
        if time.time() - msg_t < 2.5 and msg:
            cv2.putText(disp, msg, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (26, 115, 217), 2, cv2.LINE_AA)
        if step < n_views:
            panel = side_panel(h, ang, fit, target, cam_note, short)
        else:
            panel = side_panel(h, None, None, target, cam_note, short)
            cv2.putText(panel, "all views done - press Q", (14, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (26, 160, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, np.hstack([disp, panel]))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord(" "):
            if corners is None:
                msg, msg_t = "board not found - not saved (whole board in view? less glare?)", time.time()
            else:
                n = (int(os.path.basename(saved[-1])[5:8]) + 1) if saved else 1
                p = os.path.join(images_dir, f"view_{n:03d}.png")
                cv2.imwrite(p, frame)
                saved.append(p)
                fb = view_feedback(corners, (w, h), ang, target) if step < n_views else f"tilt ~{tilt:.0f} deg"
                print(f"   saved {os.path.basename(p)}: {fb}")
                msg, msg_t = f"saved view {len(saved)}", time.time()
                step += 1
                if step < n_views:
                    text, short, target = instruction(step, n_views)
                    print("\n" + text)
                elif step == n_views:
                    print(f"\nAll {n_views} views done. Press Q in the window to finish (or SPACE to add more views).")
        elif key in (ord("n"), ord("N")) and step < n_views:
            print("   skipped")
            step += 1
            if step < n_views:
                text, short, target = instruction(step, n_views)
                print("\n" + text)
        elif key in (ord("u"), ord("U")) and saved:
            os.remove(saved.pop())
            step = max(0, step - 1)
            text, short, target = instruction(step, n_views)
            msg, msg_t = f"removed, {len(saved)} views left", time.time()
            print(f"   last view removed. Redo:\n{text}")
        elif key in (ord("h"), ord("H")):
            print(HELP)
    cam.stop()
    cv2.destroyAllWindows()
    print(f"\n{len(saved)} views in {images_dir}" + ("" if len(saved) >= 10 else "  - fewer than 10: the assignment asks for at least 10"))


def main():
    ap = argparse.ArgumentParser(description="Task 3: camera calibration with a chessboard.")
    ap.add_argument("mode", nargs="?", choices=["capture", "calibrate"])
    a = ap.parse_args()
    board = load_board()
    print(f"Task 3 - calibration.  Board {board['cols']}x{board['rows']} inner corners, square {board['square_used_mm']} mm "
          f"({'measured' if board.get('measured_square_mm') else 'nominal, not measured yet'}).")
    print("Press Enter to accept the [default].\n")
    mode = a.mode or ask_choice("What do you want to do?", ["capture", "calibrate"], "capture")
    images_dir = ask_folder("Folder for the calibration views", data_path("calibration", "images"))
    if mode == "capture":
        index = ask_number("Camera index", 0, int, 0)
        w, h = (int(v) for v in ask("Resolution WxH (use the same later)", "1280x720").lower().split("x"))
        n_views = ask_number("Number of calibration views (at least 10; 20-40 is better)", DEFAULT_VIEWS, int, 3)
        if n_views < 10:
            print("   NOTE: fewer than 10 views - the assignment asks for at least 10")
        capture_views(index, (w, h), images_dir, board, n_views)
        if ask("Calibrate now with these views? (y/n)", "y").lower().startswith("y"):
            mode = "calibrate"
    if mode == "calibrate":
        model = ask_choice("Distortion model: k1k2 = simplest, recommended; full = 5 coefficients", ["k1k2", "full"], "k1k2")
        out_dir = ask_folder("Folder for the calibration report", os.path.join(OUTPUT_DIR, "calibration"))
        cam_path = ask_folder("Camera file to write", data_path("calibration", "camera.npz"))
        run_calibration(images_dir, board, model, out_dir, cam_path)


if __name__ == "__main__":
    main()
