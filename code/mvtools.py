"""mvtools.py -- helpers shared by the task scripts (t1 ... t6).  Not run directly.

  * terminal questions with defaults (ask, ask_folder, ask_number, ask_choice)
  * markers.json / camera.npz loading
  * MarkerPoseEstimator: per frame  ^C T_F, ^C T_M and the relative pose ^F T_M  (classical, or M from a network)
  * poses <-> CSV rows, ground-truth files, error metrics

Conventions are those of mvlib.py: T_AB maps coordinates of {B} into {A}; metres inside the code,
millimetres and degrees in files and tables.  Roll-pitch-yaw is ZYX:  R = Rz(yaw) Ry(pitch) Rx(roll).
"""
from __future__ import annotations

import csv
import json
import re
import os

import cv2
import numpy as np

from config import OUTPUT_DIR, data_path, output_path
from mvlib import (MarkerDetector, T_from_rvec_tvec, inv_T, make_T, marker_object_points, relative_pose,
                   reproj_rms, rot_error_deg, rotvec, expm_so3)

# ----------------------------------------------------------------------------------------------
# questions
# ----------------------------------------------------------------------------------------------
def ask(question, default):
    """Prompt with a default; Enter accepts it. Without a terminal (EOF) the default is used."""
    try:
        answer = input(f"{question} [{default}]: ").strip().strip('"').strip("'")
    except EOFError:
        answer = ""
    return answer or str(default)


def ask_folder(question, default):
    return os.path.abspath(os.path.expanduser(ask(question, default)))


def ask_number(question, default, cast=float, minimum=None):
    while True:
        try:
            v = cast(float(ask(question, default).replace(",", ".")))
            if minimum is None or v >= minimum:
                return v
        except ValueError:
            pass
        print("   enter a number" + ("" if minimum is None else f" >= {minimum}"))


def ask_choice(question, choices, default):
    """choices: list of strings; returns the chosen string (typed in full or as its first letter/number)."""
    shown = " / ".join(choices)
    while True:
        a = ask(f"{question} ({shown})", default).lower()
        for i, c in enumerate(choices):
            if a in (c.lower(), str(i + 1), c.lower()[0]):
                return c
        print(f"   choose one of: {shown}")


def ask_yes(question, default="y"):
    return ask(question + " (y/n)", default).lower().startswith("y")


# ----------------------------------------------------------------------------------------------
# configuration files
# ----------------------------------------------------------------------------------------------
def _finish_cfg(cfg):
    side_mm = cfg.get("measured_side_mm") or cfg["nominal_side_mm"]
    return dict(cfg, dict_id=getattr(cv2.aruco, cfg["dictionary"]), L=side_mm / 1000.0,
                side_is_measured=bool(cfg.get("measured_side_mm")))


def load_markers_cfg(path=None):
    """data/markers.json plus dict_id, L [m] (measured side if present, else nominal).
    If the file does not exist but marker images from t1_generate_markers.py are in output/markers (or its subfolders),
    they are recognised and adopted (and data/markers.json is written)."""
    path = path or data_path("markers.json")
    if not os.path.exists(path):
        sets = discover_marker_sets([os.path.join(OUTPUT_DIR, "markers")])
        if not sets:
            raise SystemExit(f"{path} not found and no marker images found in {os.path.join(OUTPUT_DIR, 'markers')}: "
                             "run t1_generate_markers.py (Task 1) first")
        print(f"data/markers.json was missing: using the markers found in {sets[0]['folder']} "
              f"({sets[0]['dictionary']}, F = id {sets[0]['id_fixed']}, M = id {sets[0]['id_moving']}, {sets[0]['source']})")
        return _finish_cfg(adopt_marker_set(sets[0]))
    return _finish_cfg(json.load(open(path)))


# ----------------------------------------------------------------------------------------------
# finding marker images that were generated earlier (other sessions, other folders)
# ----------------------------------------------------------------------------------------------
_MARKER_FILE = re.compile(r"^marker_([FM])_id(\d+)\.png$")


def _dictionary_names_small_first():
    names = [n for n in dir(cv2.aruco) if n.startswith("DICT_") and isinstance(getattr(cv2.aruco, n), int)]

    def key(n):
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, n))
        return d.markerSize, len(d.bytesList), n
    return sorted(names, key=key)


def _read_ids(gray, dict_name):
    if max(gray.shape) > 800:
        f = 800.0 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name)), cv2.aruco.DetectorParameters())
    _, ids, _ = det.detectMarkers(gray)
    return [] if ids is None else ids.ravel().tolist()


def _printed_side_mm(path):
    """Side of the black square of a marker image, from its pixel size and the dpi stored in the PNG (None if unknown)."""
    from PIL import Image
    im = Image.open(path)
    dpi = im.info.get("dpi")
    ys, xs = np.where(np.array(im.convert("L")) < 128)
    if dpi is None or len(xs) == 0:
        return None
    return round(float(xs.max() - xs.min() + 1) / dpi[0] * 25.4, 1)


def discover_marker_sets(folders, max_depth=3):
    """Marker pairs (marker_F_id<N>.png + marker_M_id<N>.png in the same folder) found in `folders` and their subfolders.
    The dictionary is read from marker_info.json next to the images if there is one, otherwise recognised by letting OpenCV
    read the images with each predefined dictionary.  Newest first."""
    sets, seen = [], set()
    for root in folders:
        if not os.path.isdir(root):
            continue
        base_depth = os.path.abspath(root).count(os.sep)
        for dirpath, dirs, files in os.walk(root):
            if os.path.abspath(dirpath).count(os.sep) - base_depth >= max_depth:
                dirs[:] = []
            if os.path.abspath(dirpath) in seen:
                continue
            found = {}
            for f in files:
                m = _MARKER_FILE.match(f)
                if m:
                    p = os.path.join(dirpath, f)
                    if m.group(1) not in found or os.path.getmtime(p) > os.path.getmtime(found[m.group(1)][1]):
                        found[m.group(1)] = (int(m.group(2)), p)
            if "F" not in found or "M" not in found:
                continue
            seen.add(os.path.abspath(dirpath))
            (id_f, f_png), (id_m, m_png) = found["F"], found["M"]
            info, source, dict_name = {}, "", None
            side_path = os.path.join(dirpath, "markers_info.json")
            if os.path.exists(side_path):
                info = json.load(open(side_path))
                if info.get("id_fixed") == id_f and info.get("id_moving") == id_m:
                    dict_name, source = info.get("dictionary"), "from markers_info.json"
            if dict_name is None:                                       # recognise it from the images
                gf = cv2.imread(f_png, cv2.IMREAD_GRAYSCALE)
                gm = cv2.imread(m_png, cv2.IMREAD_GRAYSCALE)
                for name in _dictionary_names_small_first():
                    n = len(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name)).bytesList)
                    if id_f < n and id_m < n and _read_ids(gf, name) == [id_f] and _read_ids(gm, name) == [id_m]:
                        dict_name, source = name, "dictionary recognised from the images"
                        break
            if dict_name is None:
                continue
            side = info.get("nominal_side_mm") if source == "from markers_info.json" else None
            sets.append(dict(folder=dirpath, id_fixed=id_f, id_moving=id_m, dictionary=dict_name, source=source,
                             nominal_side_mm=side or _printed_side_mm(f_png) or 80.0,
                             quiet_zone_cells=info.get("quiet_zone_cells", 2), f_png=f_png, m_png=m_png,
                             mtime=max(os.path.getmtime(f_png), os.path.getmtime(m_png))))
    return sorted(sets, key=lambda s: -s["mtime"])


def adopt_marker_set(s):
    """Write data/markers.json for a discovered set (the hand-measured side is kept only if nothing else changed)."""
    path = data_path("markers.json")
    new = dict(dictionary=s["dictionary"], id_fixed=s["id_fixed"], id_moving=s["id_moving"], nominal_side_mm=s["nominal_side_mm"])
    measured = None
    if os.path.exists(path):
        old = json.load(open(path))
        if all(old.get(k) == v for k, v in new.items()):
            measured = old.get("measured_side_mm")
    cfg = dict(**new, measured_side_mm=measured, quiet_zone_cells=s.get("quiet_zone_cells", 2),
               note="replace measured_side_mm (null) with the calliper measurement of the PRINTED black square")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(cfg, open(path, "w"), indent=1)
    return cfg


def select_marker_config(extra_folders=()):
    """The marker configuration for a script: data/markers.json if it matches the images, otherwise a choice among the
    marker sets found in output/markers (and `extra_folders`, subfolders included).  Asks only when there is a real choice."""
    cur_path = data_path("markers.json")
    cur = json.load(open(cur_path)) if os.path.exists(cur_path) else None
    sets = discover_marker_sets([os.path.join(OUTPUT_DIR, "markers"), *extra_folders])
    key = lambda c: (c["dictionary"], c["id_fixed"], c["id_moving"])
    others = [s for s in sets if cur is None or key(s) != key(cur)]
    if cur is not None and not others:
        return _finish_cfg(cur)
    if cur is None and not sets:
        folder = ask_folder("No markers found. Folder with marker_F_id*.png / marker_M_id*.png (Enter = quit)", "-")
        sets = discover_marker_sets([folder]) if os.path.isdir(folder) else []
        if not sets:
            raise SystemExit("no marker images found: run t1_generate_markers.py (Task 1) first")
        others = sets
    options = ([("current data/markers.json", cur)] if cur is not None else []) + [
        (f"found in {s['folder']}  ({s['source']}, newest first)", s) for s in others]
    print("Marker sets available:")
    for i, (label, c) in enumerate(options, 1):
        print(f"  {i}) {c['dictionary']}, F = id {c['id_fixed']}, M = id {c['id_moving']}, side {c.get('nominal_side_mm')} mm  -  {label}")
    choice = ask_number("Which marker set do you use", 1, int, 1)
    chosen = options[min(choice, len(options)) - 1][1]
    if chosen is not cur:
        adopt_marker_set(chosen)
        print("   data/markers.json updated.")
    return load_markers_cfg()


def default_camera_path():
    return data_path("calibration", "camera.npz")


def save_camera(path, K, dist, size, rms, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, K=K, dist=np.asarray(dist).ravel(), width=size[0], height=size[1], rms=rms)
    info = dict(K=np.asarray(K).tolist(), dist=np.asarray(dist).ravel().tolist(), width=int(size[0]), height=int(size[1]),
                rms_px=float(rms), **(extra or {}))
    json.dump(info, open(os.path.splitext(path)[0] + ".json", "w"), indent=1)


def load_camera(path=None):
    """Returns K (3x3), dist (n,), (width, height)."""
    path = path or default_camera_path()
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: calibrate the camera first (t3_calibrate.py)")
    d = np.load(path)
    return d["K"].astype(float), d["dist"].astype(float).ravel(), (int(d["width"]), int(d["height"]))


# ----------------------------------------------------------------------------------------------
# rotations and poses
# ----------------------------------------------------------------------------------------------
def rpy_deg(R):
    """Roll-pitch-yaw (ZYX) in degrees."""
    pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
    return np.degrees([np.arctan2(R[2, 1], R[2, 2]), pitch, np.arctan2(R[1, 0], R[0, 0])])


def R_from_rpy_deg(roll, pitch, yaw):
    r, p, y = np.radians([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


POSE_COLS = ("x_mm", "y_mm", "z_mm", "rx", "ry", "rz")        # translation [mm], rotation vector [rad]


def pose_to_cols(prefix, T):
    """4x4 pose (or None) -> {prefix_x_mm, ..., prefix_rz}; NaN when the pose is missing."""
    if T is None:
        return {f"{prefix}_{c}": np.nan for c in POSE_COLS}
    v = list(1000.0 * T[:3, 3]) + list(rotvec(T[:3, :3]))
    return {f"{prefix}_{c}": float(x) for c, x in zip(POSE_COLS, v)}


def cols_to_pose(row, prefix):
    """Inverse of pose_to_cols; None when missing."""
    try:
        v = [float(row[f"{prefix}_{c}"]) for c in POSE_COLS]
    except (KeyError, ValueError):
        return None
    if not np.all(np.isfinite(v)):
        return None
    return make_T(expm_so3(v[3:]), np.array(v[:3]) / 1000.0)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = fieldnames or list(rows[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------------------------
# classical pose of one marker, with the planar ambiguity made visible
# ----------------------------------------------------------------------------------------------
AMBIGUITY_RATIO = 1.5      # second/best reprojection error below this: the planar pose is ambiguous


def marker_pose_candidates(corners, K, dist, L):
    """Both IPPE_SQUARE solutions (each refined with LM), best reprojection error first: [(rms_px, T_CM), ...]."""
    obj = marker_object_points(L)
    img = np.asarray(corners, np.float64).reshape(4, 2)
    n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    out = []
    for rv, tv in zip(rvecs, tvecs):
        rv, tv = rv.copy(), tv.copy()
        cv2.solvePnPRefineLM(obj, img, K, dist, rv, tv)
        out.append((reproj_rms(obj, img, K, dist, rv, tv), T_from_rvec_tvec(rv, tv)))
    out.sort(key=lambda c: c[0])
    return out


class MarkerPoseEstimator:
    """Per frame: detect F and M, estimate ^C T_F and ^C T_M, return the relative pose ^F T_M.

    * planar ambiguity: when the two candidate poses of a marker have similar reprojection error
      (ratio < AMBIGUITY_RATIO) the candidate closest in rotation to the previous frame is used;
    * F missing: with `camera_still` the last ^C T_F is held, otherwise the frame has no relative pose;
    * M from a network: pass `net` (netpose.NetPose); then only F is classical.
    """

    def __init__(self, cfg, K, dist, camera_still=True, net=None):
        self.cfg, self.K, self.dist, self.L = cfg, K, dist, cfg["L"]
        self.id_f, self.id_m = cfg["id_fixed"], cfg["id_moving"]
        self.det = MarkerDetector(cfg["dict_id"], subpixel=True)
        self.camera_still, self.net = camera_still, net
        self.prev = {}                      # last accepted ^C T per marker id
        self.T_CF_last = None

    def _pose(self, mid, corners):
        cands = marker_pose_candidates(corners, self.K, self.dist, self.L)
        best_rms, best = cands[0]
        ratio = cands[1][0] / max(best_rms, 1e-9) if len(cands) > 1 else np.inf
        ambiguous = ratio < AMBIGUITY_RATIO
        if ambiguous and mid in self.prev:
            R_prev = self.prev[mid][:3, :3]
            rms, T = min(cands, key=lambda c: rot_error_deg(R_prev, c[1][:3, :3]))
            best_rms, best = rms, T
        self.prev[mid] = best
        return best, best_rms, ratio, ambiguous

    def process(self, gray, frame_for_net=None):
        found = self.det.detect(gray)
        r = dict(F_found=self.id_f in found, M_found=self.id_m in found, F_held=False, M_source="classical",
                 amb_F=False, amb_M=False, rms_F=np.nan, rms_M=np.nan, T_CF=None, T_CM=None, T_FM=None, corners=found)
        if r["F_found"]:
            r["T_CF"], r["rms_F"], _, r["amb_F"] = self._pose(self.id_f, found[self.id_f])
            self.T_CF_last = r["T_CF"]
        elif self.camera_still and self.T_CF_last is not None:
            r["T_CF"], r["F_held"] = self.T_CF_last, True
        if self.net is not None:
            r["T_CM"], r["M_source"] = self.net.predict_one(gray), "network"
            r["M_found"] = True
        elif r["M_found"]:
            r["T_CM"], r["rms_M"], _, r["amb_M"] = self._pose(self.id_m, found[self.id_m])
        if r["T_CF"] is not None and r["T_CM"] is not None:
            r["T_FM"] = relative_pose(r["T_CF"], r["T_CM"])
        return r


def result_to_row(i, t, r):
    row = dict(frame=i, t_s=round(float(t), 4), F_found=int(r["F_found"]), M_found=int(r["M_found"]), F_held=int(r["F_held"]),
               amb_F=int(r["amb_F"]), amb_M=int(r["amb_M"]), rms_F_px=r["rms_F"], rms_M_px=r["rms_M"], M_source=r["M_source"])
    for pre, key in (("FM", "T_FM"), ("CM", "T_CM"), ("CF", "T_CF")):
        row.update(pose_to_cols(pre, r[key]))
    return row


# ----------------------------------------------------------------------------------------------
# ground truth (hand-measured pose of M in F) and metrics
# ----------------------------------------------------------------------------------------------
TRUTH_COLS = ("t_s", "x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg")


def load_truth(path):
    """CSV with columns t_s,x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg (M in F).
    One row = a static pose (used for every frame); several rows = a known motion, interpolated in time."""
    rows = read_csv(path)
    a = np.array([[float(r[c]) for c in TRUTH_COLS[1:]] for r in rows])
    t = np.array([float(r["t_s"]) if r.get("t_s") not in (None, "") else 0.0 for r in rows])
    return t, a


def truth_pose(truth, t_query):
    """^F T_M truth at time t_query [s]."""
    t, a = truth
    if len(t) == 1:
        v = a[0]
    else:
        v = np.array([np.interp(t_query, t, a[:, k]) for k in range(6)])      # (rotation: small-motion interpolation)
    return make_T(R_from_rpy_deg(*v[3:]), v[:3] / 1000.0)


def pose_errors(T_est, T_true):
    """(translation error [mm], rotation error [deg]); NaN if the estimate is missing."""
    if T_est is None:
        return np.nan, np.nan
    return 1000.0 * float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3])), rot_error_deg(T_true[:3, :3], T_est[:3, :3])


def summarize(values):
    """median, 95th percentile, RMS, max of the finite values."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return dict(n=0, median=np.nan, p95=np.nan, rmse=np.nan, max=np.nan)
    return dict(n=len(v), median=float(np.median(v)), p95=float(np.percentile(v, 95)),
                rmse=float(np.sqrt(np.mean(v ** 2))), max=float(v.max()))


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {path}")
    return cap


def recording_times(path, n_frames, fps_default=30.0):
    """Frame times [s]: from <name>.csv written by t3_record.py if present, else frame index / fps."""
    side = os.path.splitext(path)[0] + ".csv"
    if os.path.exists(side):
        t = [float(r["t_s"]) for r in read_csv(side)]
        if len(t) >= n_frames:
            return np.array(t[:n_frames])
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_default
    cap.release()
    return np.arange(n_frames) / (fps if fps > 1 else fps_default)
