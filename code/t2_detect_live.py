"""t2_detect_live.py -- Task 2: detect the markers live with the laptop camera.

  python t2_detect_live.py

Asks (Enter = [default]): output folder, mode (live | guided | matrix), camera index, resolution; then Enter starts the camera.
  live    live view with outlines and ids; T runs one 10-frame test (you type distance and tilt)
  matrix  guided detection-limit test: distances x tilts (default 30, 50, 75, 100 cm x 0 ... 85 deg); for each cell the
          panel shows the marker's measured distance and tilt next to the target and a fit bar; SPACE tests the cell
          (see t2_matrix.py); results in detection_matrix.csv / .md
  guided  guided live demo of the task check (see t2_guided.py): the camera detects all the time, nothing is saved automatically;
          a step list says when to take a screenshot (S) and a recording (R ... R); report guided_report.md
Dictionary and ids come from data/markers.json (written by t1_generate_markers.py). If that file is missing or does not
match, marker images found in output/markers (or in --markers-folder, subfolders included, also from earlier sessions)
are recognised from the images themselves and offered; the choice is written back to data/markers.json.

Keys (click the video window first):
  SPACE  pause / resume the picture      C  camera off / on          S  screenshot (annotated)
  T      10-frame detection test: asks distance [cm] and tilt [deg], appends a row to detection_tests.csv
  L      detection limits: the guided distance x tilt matrix (t2_matrix.py), then back to the live view
  G      the guided live demo (t2_guided.py), then back to the live view
  H      print this help in the terminal                              Q or Esc  stop and quit

Sub-pixel corner refinement is on (MarkerDetector in mvlib.py). The frame rate shown is the loop rate.
Test without a camera:  python t2_detect_live.py --image ..\\output\\markers\\marker_F_id0.png
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from config import data_path, OUTPUT_DIR
from mvtools import ask, ask_choice, ask_folder, ask_number, ask_yes, select_marker_config
from mvlib import MarkerDetector

WINDOW = "Task 2 - live marker detection"
N_TEST, PASS_MIN = 10, 9                  # assignment: detection succeeds in >= 9 of 10 frames
GREEN, RED, ORANGE, WHITE = (60, 200, 60), (50, 50, 230), (26, 115, 217), (255, 255, 255)
HELP = """
Keys (click the video window first):
  SPACE  pause / resume          C  camera off / on          S  screenshot
  T      10-frame detection test (asks distance and tilt, saves a row to the CSV)
  L      detection LIMITS: the guided distance x tilt matrix (back to the live view when done)
  G      GUIDED live demo: step list, you take the screenshots (S) and the recording (R) (back to the live view when done)
  H      this help               Q or Esc  stop and quit
"""


# ----------------------------------------------------------------------------------------------
# video sources
# ----------------------------------------------------------------------------------------------
class Camera:
    def __init__(self, index, width, height):
        self.index, self.size, self.cap = index, (width, height), None

    def start(self):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        self.cap = cap
        return True

    def read(self):
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def stop(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = None

    @property
    def running(self):
        return self.cap is not None


class ImageSource(Camera):
    """A still image played as video: lets you test the program without a camera."""
    def __init__(self, path):
        self.img = cv2.imread(path)
        if self.img is None:
            raise SystemExit(f"cannot read image {path}")
        self.cap = None
        self.size = (self.img.shape[1], self.img.shape[0])

    def start(self):
        self.cap = True
        return True

    def read(self):
        return self.img.copy() if self.cap else None

    def stop(self):
        self.cap = None


# ----------------------------------------------------------------------------------------------
# detection and drawing
# ----------------------------------------------------------------------------------------------
def side_px(corners):
    """Mean side length of the detected quadrilateral [px]."""
    c = np.asarray(corners, float).reshape(4, 2)
    return float(np.mean(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)))


def annotate(frame, found, roles, fps, status=""):
    """Draw outlines, ids and an information bar on a copy of the frame."""
    img = frame.copy()
    for mid, corners in found.items():
        c = corners.astype(np.int32)
        color = GREEN if mid in roles else RED
        cv2.polylines(img, [c], True, color, 2, cv2.LINE_AA)
        cv2.circle(img, tuple(c[0]), 6, color, -1)                      # first corner (top-left of the marker)
        label = f"{roles[mid]} id {mid}" if mid in roles else f"id {mid} (not ours)"
        cv2.putText(img, label, (c[:, 0].min(), max(20, c[:, 1].min() - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    seen = " ".join(f"{roles[i]}:{'yes' if i in found else 'no'}" for i in roles)
    bar = f"{fps:4.1f} fps   {img.shape[1]}x{img.shape[0]}   {seen}   {status}"
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (30, 30, 30), -1)
    cv2.putText(img, bar, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    return img


def measure(source, detector, ids, n=N_TEST, warmup=3):
    """Read n fresh frames; per marker id: number of frames in which it was detected and its mean side in pixels."""
    for _ in range(warmup):                                              # drop stale frames in the camera buffer
        source.read()
    hits, sides, got, t0 = {i: 0 for i in ids}, {i: [] for i in ids}, 0, time.time()
    while got < n:
        frame = source.read()
        if frame is None:
            continue
        got += 1
        found = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        for i in ids:
            if i in found:
                hits[i] += 1
                sides[i].append(side_px(found[i]))
    fps = n / (time.time() - t0)
    return hits, {i: (float(np.mean(s)) if s else None) for i, s in sides.items()}, fps


def append_row(csv_path, row):
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def detection_test(source, detector, cfg, roles, size, cells, csv_path):
    print("\n--- 10-frame detection test ---")
    dist = ask_number("Distance camera -> marker [cm] (ruler)", 50)
    tilt = ask_number("Tilt of the marker [deg] (0 = facing the camera)", 0)
    input("Hold the marker still, then press Enter here to capture 10 frames... ")
    hits, sides, fps = measure(source, detector, list(roles))
    row = dict(time=datetime.now().isoformat(timespec="seconds"), distance_cm=dist, tilt_deg=tilt,
               frames=N_TEST, resolution=f"{size[0]}x{size[1]}", fps=round(fps, 1), dictionary=cfg["dictionary"])
    for mid, role in roles.items():
        px = sides[mid]
        row.update({f"{role}_id": mid, f"{role}_detected": hits[mid], f"{role}_pass": hits[mid] >= PASS_MIN,
                    f"{role}_side_px": None if px is None else round(px, 1),
                    f"{role}_px_per_cell": None if px is None else round(px / cells, 2)})
        print(f"  {role} (id {mid}): detected in {hits[mid]}/{N_TEST} frames  ->  "
              f"{'PASS' if hits[mid] >= PASS_MIN else 'FAIL'}" + ("" if px is None else f"   side {px:.0f} px = {px / cells:.1f} px/cell"))
    append_row(csv_path, row)
    print(f"  saved to {csv_path}\n")


def parse_list(text):
    return [float(x) for x in text.replace(";", ",").split(",") if x.strip()]


def ask_list(question, default):
    """A comma-separated list of numbers; asks again on a wrong answer instead of stopping the program."""
    while True:
        try:
            v = parse_list(ask(question, default))
            if v:
                return v
        except ValueError:
            pass
        print("   numbers separated by commas, e.g. " + default)


def ask_matrix_settings():
    import t2_matrix as tm
    print("Detection-limit matrix.  Press Enter to accept the [default].")
    dists = ask_list("   distances [cm], comma separated (your set-up: max 1 m)", ",".join(f"{d:g}" for d in tm.DEFAULT_DISTANCES))
    tilts = ask_list("   tilts [deg], comma separated", ",".join(f"{t:g}" for t in tm.DEFAULT_TILTS))
    role = ask_choice("   marker used for the test", ["M", "F"], "M")
    skip = ask_yes("   after a clear FAIL (5/10 or less), skip the larger tilts at that distance?", "y")
    return dists, tilts, role, skip


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Task 2: live marker detection.")
    ap.add_argument("--image", help="use this image instead of the camera (test without a camera)")
    ap.add_argument("--markers-folder", help="also look for generated markers (marker_F_id*.png, marker_M_id*.png) in this folder")
    ap.add_argument("--mode", choices=["live", "guided", "matrix"], help="live view, the guided live demo, or the guided limit experiments")
    ap.add_argument("--fresh", action="store_true", help="guided mode: delete the step files of an earlier session and start over")
    ap.add_argument("--max-frames", type=int, help=argparse.SUPPRESS)    # development: quit after N frames
    a = ap.parse_args()

    # markers: data/markers.json, or marker images found in output/markers (other sessions, other folders)
    extra = [a.markers_folder] if a.markers_folder else []
    cfg = select_marker_config(extra)
    dict_id = getattr(cv2.aruco, cfg["dictionary"])
    cells = cv2.aruco.getPredefinedDictionary(dict_id).markerSize + 2
    roles = {cfg["id_fixed"]: "F", cfg["id_moving"]: "M"}
    side = cfg.get("measured_side_mm") or cfg["nominal_side_mm"]
    print(f"Task 2 - live detection.  Dictionary {cfg['dictionary']}, F = id {cfg['id_fixed']}, M = id {cfg['id_moving']}, "
          f"side {side} mm ({'measured' if cfg.get('measured_side_mm') else 'nominal, not measured yet'}).")
    print("Press Enter to accept the [default].\n")

    out_dir = ask_folder("1) Folder for screenshots and test results", os.path.join(OUTPUT_DIR, "detection"))
    os.makedirs(out_dir, exist_ok=True)
    print("2) Mode:  live    = live view (keys: T one test, L limit experiments, G guided demo)\n"
          "          guided  = guided live demo: the camera detects, a step list tells you when to screenshot (S) and record (R)\n"
          "          matrix  = detection-limit experiments (distance x tilt, guided)")
    mode = a.mode or ask_choice("   Mode", ["live", "guided", "matrix"], "live")
    if mode == "matrix":
        dists, tilts, role, skip = ask_matrix_settings()
    fresh = bool(a.fresh)
    gdir = os.path.join(out_dir, "guided")
    if mode == "guided" and not fresh and os.path.isdir(gdir):
        old = sorted(f for f in os.listdir(gdir) if f.startswith("step"))
        if old:
            fresh = ask_choice(f"   earlier guided-demo files found ({', '.join(old)}): continue with them, or start fresh (delete them)?",
                               ["continue", "fresh"], "continue") == "fresh"
    if a.image:
        source = ImageSource(a.image)
    else:
        index = int(ask("3) Camera index", 0))
        w, h = (int(v) for v in ask("4) Resolution WxH (same as the calibration)", "1280x720").lower().split("x"))
        source = Camera(index, w, h)
    print(HELP if mode == "live" else "")
    input("Press Enter to start the camera... ")
    if not source.start():
        raise SystemExit("cannot open the camera (wrong index? used by another program?)")
    if mode == "matrix":
        size = source.size if a.image else (int(source.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(source.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        import t2_matrix as tm
        tm.run_matrix(source, MarkerDetector(dict_id, subpixel=True), cfg, size, out_dir, dists, tilts, role, skip)
        source.stop()
        return
    if mode == "guided":
        size = source.size if a.image else (int(source.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(source.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        import t2_guided
        t2_guided.run_guided(source, MarkerDetector(dict_id, subpixel=True), cfg, roles, size, out_dir, WINDOW, keep_window=False, fresh=fresh)
        source.stop()
        return

    detector = MarkerDetector(dict_id, subpixel=True)
    csv_path = os.path.join(out_dir, "detection_tests.csv")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    shown, paused, fps, last, n_frames = None, False, 0.0, time.time(), 0
    status = ""

    while True:
        if source.running and not paused:
            frame = source.read()
            if frame is not None:
                found = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                now = time.time()
                fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1.0 / max(now - last, 1e-6)
                last = now
                shown = annotate(frame, found, roles, fps, status="")
        elif not source.running:
            shown = np.zeros((480, 854, 3), np.uint8)
            cv2.putText(shown, "camera off - press C to start", (170, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
        if shown is not None:
            view = shown.copy()
            if paused:
                cv2.putText(view, "PAUSED (SPACE to resume)", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ORANGE, 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, view)

        key = cv2.waitKey(1) & 0xFF
        n_frames += 1
        if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if a.max_frames and n_frames >= a.max_frames:
            break
        if key == ord(" "):
            paused = not paused
        elif key in (ord("c"), ord("C")):
            if source.running:
                source.stop()
                print("camera off")
            elif source.start():
                paused, fps = False, 0.0
                print("camera on")
            else:
                print("cannot open the camera")
        elif key in (ord("s"), ord("S")) and shown is not None:
            path = os.path.join(out_dir, f"screenshot_{datetime.now():%Y%m%d_%H%M%S}.png")
            cv2.imwrite(path, shown)
            print("screenshot ->", path)
        elif key in (ord("h"), ord("H")):
            print(HELP)
        elif key in (ord("g"), ord("G")):
            if source.running:
                import t2_guided
                size = source.size if a.image else (int(source.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(source.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                try:
                    t2_guided.run_guided(source, detector, cfg, roles, size, out_dir, WINDOW)
                except Exception as e:                                  # never lose the camera because of the demo
                    print(f"   the guided demo stopped: {type(e).__name__}: {e}")
                print("\nback in the live view." + HELP)
                last = time.time()
            else:
                print("start the camera first (C)")
        elif key in (ord("l"), ord("L")):
            if source.running:
                # default plan, no terminal questions: the keys stay in this window (custom plan: --mode matrix)
                import t2_matrix as tm
                print("\n--- detection-limit experiments with the default plan (custom distances/tilts: python t2_detect_live.py --mode matrix) ---")
                size = source.size if a.image else (int(source.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(source.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                try:
                    tm.run_matrix(source, detector, cfg, size, out_dir, tm.DEFAULT_DISTANCES, tm.DEFAULT_TILTS, "M", True,
                                  window=WINDOW, keep_window=True)
                except Exception as e:                                  # never lose the camera because of the experiments
                    print(f"   the limit experiments stopped: {type(e).__name__}: {e}")
                print("\nback in the live view." + HELP)
                last = time.time()
            else:
                print("start the camera first (C)")
        elif key in (ord("t"), ord("T")):
            if source.running:
                detection_test(source, detector, cfg, roles, source.size if a.image else
                               (int(source.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(source.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                               cells, csv_path)
                last = time.time()
            else:
                print("start the camera first (C)")

    source.stop()
    cv2.destroyAllWindows()
    print("stopped.")


if __name__ == "__main__":
    main()
