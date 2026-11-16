"""prep_data.py -- data and camera cut-outs for the paper's figures (run once, from the project folder):

    python paper/prep_data.py

Writes paper/data/*.csv (per-frame errors of E3) and paper/figures/photos/*.png.
Privacy rule: every camera picture written here is a tight crop around a marker (or the bonus path band):
no person and no room.
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
from config import OUTPUT_DIR, data_path, training_path          # noqa: E402
from mvlib import marker_object_points                           # noqa: E402
from mvtools import (MarkerPoseEstimator, cols_to_pose, load_camera, load_markers_cfg, load_truth, pose_errors,
                     truth_pose)                                  # noqa: E402
from netpose import NetPose                                      # noqa: E402
import t3_pose                                                   # noqa: E402

OUT_D, OUT_P = os.path.join(ROOT, "paper", "data"), os.path.join(ROOT, "paper", "figures", "photos")
GREEN, ORANGE, RED = (60, 200, 60), (26, 150, 255), (40, 40, 235)
cfg = load_markers_cfg()
K, dist, _ = load_camera()
L = cfg["L"]
obj = marker_object_points(L).astype(np.float64)
net = NetPose(training_path("models", "crop_resnet18_v1_ft3.pt"), K=K, dist=dist)


def outline(T):
    return cv2.projectPoints(obj, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, dist)[0].reshape(-1, 2)


def frames_of(name, idx=None):
    cap, out, i = cv2.VideoCapture(data_path("recordings", name + ".avi")), [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if idx is None or i in idx:
            out.append(f)
        i += 1
    cap.release()
    return out


def crop_around(frame, uv, pad=0.17, size=260):
    """Square crop around a marker outline uv (4x2), padded by pad x the marker's apparent size, resized to size px."""
    c = uv.mean(0)
    half = (0.5 + pad) * max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1]))
    H, W = frame.shape[:2]
    side = int(round(2 * half))
    x0, y0 = int(round(c[0] - half)), int(round(c[1] - half))
    x0, y0 = max(0, min(W - side, x0)), max(0, min(H - side, y0))
    return cv2.resize(frame[y0:y0 + side, x0:x0 + side], (size, size), interpolation=cv2.INTER_CUBIC), (x0, y0, side)


def draw_poly(img, uv, box, col, th=2, size=260):
    x0, y0, side = box
    p = (uv - [x0, y0]) * size / side
    cv2.polylines(img, [np.round(p).astype(np.int32)], True, col, th, cv2.LINE_AA)


# ---------------------------------------------------------------------------------- E3: per-frame errors
def e3_errors():
    for name in ("E3_motion_01", "E3_motion_02", "E3_motion_03"):
        video = data_path("recordings", name + ".avi")
        truth = load_truth(data_path("recordings", name + ".truth.csv"))
        cols = {}
        for kind, n in (("classical", None), ("network", net)):
            rows, _ = t3_pose.process_video(video, cfg, K, dist, True, n, verbose=False)
            t = np.array([float(r["t_s"]) for r in rows])
            T = [cols_to_pose(r, "FM") for r in rows]
            e = np.array([pose_errors(Ti, truth_pose(truth, ti)) for Ti, ti in zip(T, t)])
            x = np.array([1000 * Ti[0, 3] if Ti is not None else np.nan for Ti in T])
            cols[kind] = (t, x, e)
        t = cols["classical"][0]
        xt = np.array([1000 * truth_pose(truth, ti)[0, 3] for ti in t])
        a = np.column_stack([t, xt, cols["classical"][1], cols["network"][1], cols["classical"][2], cols["network"][2]])
        np.savetxt(os.path.join(OUT_D, name + "_errors.csv"), a, delimiter=",", fmt="%.4f",
                   header="t_s,x_truth_mm,x_classical_mm,x_network_mm,et_classical_mm,eR_classical_deg,et_network_mm,eR_network_deg", comments="")
        print(name, "median e_t classical / network:", np.nanmedian(a[:, 4]).round(2), np.nanmedian(a[:, 6]).round(2))


# ---------------------------------------------------------------------------------- camera cut-outs of the markers
def marker_views():
    """Tight crops of M (and F) in the test conditions; classical outline green, network outline orange."""
    picks = [("E2_d40_t0", 20), ("E2_d75_t40", 20), ("E4_dim", 20), ("E4_cover", 20), ("E4_blur_2", None)]
    for name, k in picks:
        fr = frames_of(name)
        est = MarkerPoseEstimator(cfg, K, dist, camera_still=True)
        res = [est.process(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in fr]
        if k is None:                                                 # the most blurred frame: lowest sharpness inside M's window
            sharp = []
            for f in fr[::3]:
                T = net.predict_one(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
                c, box = crop_around(f, outline(T))
                sharp.append(cv2.Laplacian(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            k = 3 * int(np.argmin(sharp))
        f = fr[k]
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        Tn = net.predict_one(g)
        uv_n = outline(Tn)
        raw, box = crop_around(f, uv_n)
        cv2.imwrite(os.path.join(OUT_P, f"view_{name}_raw.png"), raw)
        both = raw.copy()
        if res[k]["T_CM"] is not None:
            draw_poly(both, res[k]["corners"][cfg["id_moving"]], box, GREEN, 3)
        draw_poly(both, uv_n, box, ORANGE, 2)
        cv2.imwrite(os.path.join(OUT_P, f"view_{name}_both.png"), both)
        print(name, "frame", k, "classical M:", res[k]["T_CM"] is not None)
        if name == "E2_d40_t0" and res[k]["T_CF"] is not None:
            rawF, boxF = crop_around(f, res[k]["corners"][cfg["id_fixed"]])
            cv2.imwrite(os.path.join(OUT_P, "view_F_raw.png"), rawF)


def hard_case_examples():
    """The software degradations of the hard-case test, shown on one clean crop of M (E2_d40_t0)."""
    f = frames_of("E2_d40_t0", {20})[0]
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    est = MarkerPoseEstimator(cfg, K, dist, camera_still=True)
    r = est.process(g)
    uv = r["corners"][cfg["id_moving"]]
    rng = np.random.default_rng(0)

    def cover(g, frac, k=0):
        c, p = uv.mean(0), uv[k]
        q = p + (c - p) * 2 * np.sqrt(frac)
        x0, x1, y0, y1 = int(min(p[0], q[0])) - 6, int(max(p[0], q[0])) + 6, int(min(p[1], q[1])) - 6, int(max(p[1], q[1])) + 6
        g = g.copy()
        g[max(0, y0):y1, max(0, x0):x1] = 120
        return g

    def mblur(g, n):
        k = np.zeros((n, n), np.float32)
        k[n // 2, :] = 1.0 / n
        return cv2.filter2D(g, -1, k)

    def lowres(g, s):
        a = cv2.resize(g, (g.shape[1] // s, g.shape[0] // s), interpolation=cv2.INTER_AREA)
        return cv2.resize(a, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)

    def dim(g, a):
        return np.clip(g.astype(np.float32) * a + rng.normal(0, 3, g.shape), 0, 255).astype(np.uint8)
    cases = [("clean", g), ("cover30", cover(g, 0.30)), ("blur31", mblur(g, 31)), ("res6", lowres(g, 6)), ("light025", dim(g, 0.25)),
             ("light010", dim(g, 0.10))]
    for tag, img in cases:
        c, _ = crop_around(img, uv)
        cv2.imwrite(os.path.join(OUT_P, f"hard_{tag}.png"), c)


def synthetic_examples():
    X = np.load(training_path("synthetic", "synth_v1", "X_train.npy"), mmap_mode="r")
    for j, i in enumerate((3, 11, 27, 42)):
        cv2.imwrite(os.path.join(OUT_P, f"synth_full_{j}.png"), np.asarray(X[i]))
    Xc = np.load(training_path("synthetic", "crop_v1", "X_train.npy"), mmap_mode="r")
    for j, i in enumerate((1, 5, 9, 14, 22, 31)):
        cv2.imwrite(os.path.join(OUT_P, f"synth_crop_{j}.png"), np.asarray(Xc[i]))
    print("synthetic shapes", X.shape, Xc.shape)


def white_margin():
    src = cv2.imread(os.path.join(OUT_D, "m_zoom_source.jpg"))
    cv2.imwrite(os.path.join(OUT_P, "margin_M.png"), src[66:536, 138:500])
    cv2.imwrite(os.path.join(OUT_P, "margin_F.png"), src[66:570, 690:1136])


def bonus_strip():
    """Four moments of one pass behind the box (E7_box_5cms_2), drawn here without any text in the picture:
    0 M measured (green outline)   1, 2 M hidden, predicted pose (red outline)   3 M measured again (green outline).
    Privacy: only the band of the path (from just right of M's start position to F) and a small box around a measured M are kept;
    the band starts 4 px above the markers, so the fingers that hold M from above are not in the picture."""
    import csv
    from t7_animate import pixelate_outside, work_area
    name = "E7_box_5cms_2"
    rows = list(csv.DictReader(open(os.path.join(OUTPUT_DIR, "bonus", name + "_predicted_poses.csv"), newline="")))
    fr0 = int(rows[0]["frame"])
    meas = np.array([int(r["measured"]) for r in rows]); pred = np.array([int(r["predicted"]) for r in rows]); lost = np.array([int(r["lost"]) for r in rows])
    t = np.array([float(r["t_s"]) for r in rows])
    hid = (meas == 0) & (pred == 1) & (lost == 0)
    runs, i = [], 0                                                    # runs of predicted frames, at least 1.25 s long, after 0.5 s of measurements
    while i < len(rows):
        if hid[i]:
            j = i
            while j + 1 < len(rows) and meas[j + 1] == 0:
                j += 1
            k = i
            while k + 1 <= j and hid[k + 1]:
                k += 1
            if t[k] - t[i] >= 1.25 and i >= 15 and meas[i - 15:i].all() and j + 1 < len(rows):
                runs.append((i, k, j))
            i = j + 1
        else:
            i += 1
    i, k, j = runs[0]
    pick = [i - 3, int(np.searchsorted(t, t[i] + 0.4)), int(np.searchsorted(t, t[i] + 1.2)), j + 2]
    print("bonus frames: hidden from %.2f s for %.2f s (predicted for %.2f s); frames at %s s" % (t[i], t[j] - t[i], t[k] - t[i], np.round(t[pick] - t[i], 2)))
    frames = frames_of(name)
    est = MarkerPoseEstimator(cfg, K, dist, camera_still=True)
    res = [est.process(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames]
    corners = [r["corners"] for r in res]
    x0, y0, x1, y1 = work_area(corners, frames[0].shape, cfg["id_moving"])
    pts = np.vstack([c for d in corners for c in d.values()])
    y0 = int(np.percentile(pts[:, 1], 1)) - 4                           # top of the markers: nothing above them is kept
    y1 = min(y1, int(np.percentile(pts[:, 1], 99)) + 16)                # ... and only the white margin below them
    fpts = np.vstack([d[cfg["id_fixed"]] for d in corners if cfg["id_fixed"] in d])
    x1 = min(x1, int(np.percentile(fpts[:, 0], 99)) + 14)               # ... and nothing right of F's sheet
    last_CF, CF = None, []
    for r in res:
        last_CF = r["T_CF"] if r["T_CF"] is not None else last_CF
        CF.append(last_CF)
    for n, q in enumerate(pick):
        f = frames[fr0 + q]
        boxes, r = [(x0, y0, x1, y1)], res[fr0 + q]
        uv_m = r["corners"].get(cfg["id_moving"])
        if uv_m is not None:                                           # a measured M outside the band: a small box around it
            boxes.append((uv_m[:, 0].min() - 10, max(y0, uv_m[:, 1].min() - 4), uv_m[:, 0].max() + 10, uv_m[:, 1].max() + 10))
        out = pixelate_outside(f, boxes, cut=True)
        out[out[:, :, 0] == 38] = 255 if False else out[out[:, :, 0] == 38]
        if uv_m is not None:
            cv2.polylines(out, [uv_m.astype(np.int32)], True, GREEN, 3, cv2.LINE_AA)
        elif rows[q]["FM_x_mm"] not in ("", None) and CF[fr0 + q] is not None:
            Tc = CF[fr0 + q] @ cols_to_pose(rows[q], "FM")
            cv2.polylines(out, [outline(Tc).astype(np.int32)], True, RED, 3, cv2.LINE_AA)
        xa = int(min(b[0] for b in boxes)) if n == 0 else x0
        xa = min(xa, x0)
        crop = out[y0:y1, max(0, int(min(bx[0] for bx in boxes)) if uv_m is not None else x0):x1]
        crop = out[y0:y1, min(x0, max(0, int(min(bx[0] for bx in boxes)))):x1]
        cv2.imwrite(os.path.join(OUT_P, f"bonus_{n}.png"), crop)
    for n in (4,):                                                      # the older five-frame set is gone
        f = os.path.join(OUT_P, f"bonus_{n}.png")
        if os.path.exists(f):
            os.remove(f)


def t4_tracks():
    """T4_motion_1: pose of M in F per frame, classical and network (x y z [mm], roll pitch yaw [deg])."""
    from mvtools import rpy_deg
    video = data_path("recordings", "T4_motion_1.avi")
    out = []
    for n in (None, net):
        rows, _ = t3_pose.process_video(video, cfg, K, dist, True, n, verbose=False)
        t = np.array([float(r["t_s"]) for r in rows])
        a = np.full((len(rows), 6), np.nan)
        for i, r in enumerate(rows):
            T = cols_to_pose(r, "FM")
            if T is not None:
                a[i, :3], a[i, 3:] = 1000 * T[:3, 3], rpy_deg(T[:3, :3])
        out.append(a)
    np.savetxt(os.path.join(OUT_D, "T4_motion_1_tracks.csv"), np.column_stack([t, out[0], out[1]]), delimiter=",", fmt="%.3f",
               header="t_s," + ",".join(f"{c}_{m}" for m in ("classical", "network") for c in ("x", "y", "z", "roll", "pitch", "yaw")), comments="")
    d = np.linalg.norm(out[0][:, :3] - out[1][:, :3], axis=1)
    print("T4: frames", len(t), "classical output %.0f %%" % (100 * np.isfinite(out[0][:, 0]).mean()), "median difference %.1f mm" % np.nanmedian(d))



def all_errors():
    """Per-frame errors of both methods on every test recording with a truth file (E1-E4)."""
    import glob
    lines = ["recording,frame,et_classical_mm,eR_classical_deg,et_network_mm,eR_network_deg,xc,yc,zc,xn,yn,zn"]

    def fl(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")
    for tp in sorted(glob.glob(data_path("recordings", "E[1-4]*.truth.csv"))):
        name = os.path.basename(tp)[:-len(".truth.csv")]
        truth = load_truth(tp)
        e, pos = [], []
        for n in (None, net):
            rows, _ = t3_pose.process_video(data_path("recordings", name + ".avi"), cfg, K, dist, True, n, verbose=False)
            e.append(np.array([pose_errors(cols_to_pose(r, "FM"), truth_pose(truth, float(r["t_s"]))) for r in rows]))
            pos.append(np.array([[fl(r[f"FM_{c}_mm"]) for c in "xyz"] for r in rows]))
        for i in range(len(e[0])):
            lines.append(f"{name},{i},{e[0][i, 0]:.3f},{e[0][i, 1]:.3f},{e[1][i, 0]:.3f},{e[1][i, 1]:.3f}," + ",".join(f"{v:.3f}" for v in list(pos[0][i]) + list(pos[1][i])))
        us = [100 * np.mean((x[:, 0] < 20) & (x[:, 1] < 10)) for x in e]
        print(f"{name:14s} n {len(e[0]):4d}  classical {np.nanmedian(e[0][:, 0]):6.1f} mm {np.nanmedian(e[0][:, 1]):5.2f} deg usable {us[0]:5.1f} %   "
              f"network {np.nanmedian(e[1][:, 0]):6.1f} mm {np.nanmedian(e[1][:, 1]):5.2f} deg usable {us[1]:5.1f} %")
    open(os.path.join(OUT_D, "test_errors.csv"), "w").write("\n".join(lines) + "\n")



def calibration():
    """Re-runs the calibration on the 20 saved chessboard views (same flags as t3_calibrate.py) and saves what the
    calibration figure needs: every corner with its residual, per-view numbers, one board view masked and rectified."""
    import glob
    import t3_calibrate as tc
    board = tc.load_board()
    objp = tc.board_points(board)
    files = sorted(glob.glob(data_path("calibration", "images", "view_*.png")))
    corners, grays = [], []
    for f in files:
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        c = tc.find_corners(g, board)
        assert c is not None, f
        corners.append(c); grays.append(g)
    size = (grays[0].shape[1], grays[0].shape[0])
    cal = tc.calibrate(objp, corners, size, "k1k2")
    held = tc.held_out_check(objp, corners, size, "k1k2")
    print("calibration: rms %.4f px  fx %.2f fy %.2f cx %.2f cy %.2f  k1 %.4f k2 %.4f" % (cal["rms"], cal["K"][0, 0], cal["K"][1, 1], cal["K"][0, 2], cal["K"][1, 2], cal["dist"][0], cal["dist"][1]))
    lines = ["view,idx,u,v,du,dv"]
    vl = ["view,rms_px,tilt_deg,distance_cm,held_out,held_out_rms_px"]
    for i, (c, rv, tv) in enumerate(zip(corners, cal["rvecs"], cal["tvecs"])):
        pr = cv2.projectPoints(objp, rv, tv, cal["K"], cal["dist"])[0].reshape(-1, 2)
        d = pr - c.reshape(-1, 2)
        for k, (uv, dd) in enumerate(zip(c.reshape(-1, 2), d)):
            lines.append(f"{i + 1},{k},{uv[0]:.3f},{uv[1]:.3f},{dd[0]:.4f},{dd[1]:.4f}")
        ho = held["views"].index(i) if i in held["views"] else -1
        vl.append(f"{i + 1},{cal['per_view_rms'][i]:.4f},{cal['tilt_deg'][i]:.2f},{100 * float(np.linalg.norm(tv)):.1f},{int(ho >= 0)},{held['rms_per_view'][ho] if ho >= 0 else ''}")
    open(os.path.join(OUT_D, "calib_corners.csv"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(OUT_D, "calib_views.csv"), "w").write("\n".join(vl) + "\n")
    # one view: the board region only (everything outside the outer squares is blanked), and the same view rectified
    i = int(np.argmax(cal["tilt_deg"]))
    g, c = grays[i], corners[i].reshape(-1, 2)
    cols, rows = board["cols"], board["rows"]
    grid = np.array([[x, y] for y in range(rows) for x in range(cols)], np.float32)
    Hm, _ = cv2.findHomography(grid, c)
    m = 0.85                                                           # the outer ring of squares is one square wide: stay inside it
    quad = cv2.perspectiveTransform(np.float32([[[-m, -m]], [[cols - 1 + m, -m]], [[cols - 1 + m, rows - 1 + m]], [[-m, rows - 1 + m]]]), Hm).reshape(-1, 2)
    mask = np.zeros_like(g); cv2.fillConvexPoly(mask, np.int32(np.round(quad)), 255)
    view = cv2.cvtColor(np.where(mask > 0, g, 205).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for q in c:
        cv2.circle(view, (int(round(q[0])), int(round(q[1]))), 4, (60, 200, 60), 1, cv2.LINE_AA)
    x0, y0, x1, y1 = int(quad[:, 0].min()) - 6, int(quad[:, 1].min()) - 6, int(quad[:, 0].max()) + 6, int(quad[:, 1].max()) + 6
    cv2.imwrite(os.path.join(OUT_P, "calib_view.png"), view[max(0, y0):y1, max(0, x0):x1])
    S = 44
    dst = (grid + m) * S
    Hr, _ = cv2.findHomography(c, dst)
    rect = cv2.warpPerspective(g, Hr, (int((cols - 1 + 2 * m) * S), int((rows - 1 + 2 * m) * S)), flags=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(OUT_P, "calib_rect.png"), rect)
    print("calibration view shown:", os.path.basename(files[i]), "tilt %.1f deg, rms %.2f px" % (cal["tilt_deg"][i], cal["per_view_rms"][i]))


if __name__ == "__main__":
    what = sys.argv[1:] or ["e3", "views", "hard", "synth", "margin", "bonus", "t4", "errors", "calib"]
    if "e3" in what:
        e3_errors()
    if "views" in what:
        marker_views()
    if "hard" in what:
        hard_case_examples()
    if "synth" in what:
        synthetic_examples()
    if "margin" in what:
        white_margin()
    if "bonus" in what:
        bonus_strip()
    if "t4" in what:
        t4_tracks()
    if "errors" in what:
        all_errors()
    if "calib" in what:
        calibration()
    print("done ->", OUT_D, OUT_P)
