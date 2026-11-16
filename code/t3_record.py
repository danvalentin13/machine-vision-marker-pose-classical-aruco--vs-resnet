"""t3_record.py -- record the camera for Tasks 3-6 (training frames, tests, motion sequences).

  python t3_record.py

Asks (Enter = [default]): experiment tag, folder, recording name, notes, [truth of M in F], number of frames, camera, resolution.
Truth (static recordings, e.g. E1 / E2): the pose of M in the frame of F measured by hand, written to <name>.truth.csv:
  right [cm]  : M's centre to the RIGHT of F's centre, along F's surface (- = left)          -> x of M in F
  up [cm]     : M's centre ABOVE F's centre (- = below)                                    -> y
  front [cm]  : M's centre IN FRONT of F's printed surface, toward the viewer (- = behind)  -> z
  turn [deg]  : M turned about the vertical axis relative to F; + = M's right edge farther back  -> pitch (rotation about F's y)
  (F's frame: x right, y up, z out of the printed side; both boards upright, M's "up" the same as F's.)
Frames: with a number (e.g. 100) the recording stops by itself after that many frames.
Keys (click the video window first):   R  start / stop recording     D  show / hide marker detection
                                       S  screenshot                  Q or Esc  quit
Writes  <folder>/<name>.avi        the video (MJPG, one file per recording; frames are not resized)
        <folder>/<name>.csv        frame index and time stamp of every frame [s]
        <folder>/<name>.json       resolution, tag, notes, fps, number of frames
The folder is data/training/real for tag 'train' (frames for fine-tuning) and data/recordings for all test recordings.
Each recording is one unit for the train/test split: never use frames of the same recording for training and testing.
Ground truth (if you have it) goes in data/recordings/<name>.truth.csv  (see README, Task 6).
"""
import argparse
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np

from config import data_path, training_path
from mvtools import MarkerDetector, ask, ask_choice, ask_folder, ask_number, ask_yes, load_markers_cfg
from t2_detect_live import Camera

TAGS = ["E1_static", "E2_dist_tilt", "E3_motion", "E4_robust", "calib_view", "train", "demo", "other"]
HELP = """
Keys (click the video window first):
  R  start / stop recording     D  show / hide marker detection
  S  screenshot                 Q or Esc  quit
"""


def ask_truth(cfg):
    """Asks what to measure with the ruler, in plain words, and converts it into M's pose in F's frame."""
    side_cm = cfg["L"] * 100
    print(f"""
   TRUTH - measure with the ruler.  Top view, camera at the bottom; F and M stand upright, printed sides toward the camera.

           black square        gap         black square
          |#########|                   |#########|
          |<- {side_cm:3.1f} ->|<------ gap ------>|    gap = from the edge of one BLACK square
            side L                              to the nearest edge of the other one
                                                (not the white border, not the board)
          Simplest: both boards against one straight edge -> same plane, nothing turned (answers 4 and 5 = 0).
""")
    while True:
        side = ask_choice("     1. Is M to the LEFT or to the RIGHT of F (as seen from the camera)?", ["left", "right"], "left")
        gap = ask_number("     2. GAP between the two BLACK squares, edge to edge, along F's surface [cm]", 20.0, float, 0.0)
        up = ask_number("     3. HEIGHT difference of the CENTRES of the black squares [cm] (+ = M's centre higher than F's, 0 = same height)", 0.0)
        front = ask_number("     4. How far M's printed surface is IN FRONT of F's printed surface, toward the camera [cm]\n"
                           "        (0 = both boards against the same straight edge; - = M is behind)", 0.0)
        turn = ask_number("     5. ANGLE between the two boards [deg] (0 = parallel / same plane;\n"
                          "        + = M turned so that its RIGHT edge is farther from the camera than its left edge)", 0.0)
        x = (gap + side_cm) * (1 if side == "right" else -1)
        print(f"     -> M's centre is {abs(x):.1f} cm to the {side} of F's centre (gap {gap:g} + side {side_cm:.1f}), "
              f"{up:+g} cm higher, {front:+g} cm in front, turned {turn:+g} deg")
        if ask_yes("     correct?", "y"):
            return dict(x=x, y=up, z=front, turn=turn)
        print("     again:")


def main():
    argparse.ArgumentParser(description="Record the camera for Tasks 3-6.").parse_args()
    cfg = load_markers_cfg()
    print("Record the camera.  Press Enter to accept the [default].\n")
    tag = ask(f"1) Experiment tag ({', '.join(TAGS)})", "other")
    out_dir = ask_folder("2) Folder for the recording (training frames are kept apart from the test recordings)",
                         training_path("real") if tag == "train" else data_path("recordings"))
    name = ask("3) Recording name", f"rec_{datetime.now():%Y%m%d_%H%M%S}")
    notes = ask("4) Notes (e.g. M at grid position C3, blur, dim light, partial occlusion)", "")
    print("   Optional measurements for the report (Enter = not measured):")
    dist_cm = ask("   a) distance from the CAMERA LENS (small dot above the screen) to the CENTRE of M's black square [cm]", "-")
    tilt_deg = ask("   b) angle of M's printed surface to the camera [deg] (0 = M faces the camera straight on)", "-")
    truth = None
    if not tag.startswith(("E3", "E7", "train", "calib")):
        if ask_yes("   c) write a TRUTH file (where M is relative to F, measured by hand) for this recording?", "y" if tag.startswith(("E1", "E2")) else "n"):
            truth = ask_truth(cfg)
    elif tag.startswith(("E3", "E7")):
        print("   (motion recordings: write the known path in <name>.truth.csv by hand, several rows with t_s - see README T6)")
    n_max = ask("   number of frames to record (Enter = until you press R again)", "100" if tag.startswith("E1") else "-")
    n_max = None if n_max.strip() in ("-", "") else int(n_max)
    index = ask_number("5) Camera index", 0, int, 0)
    w, h = (int(v) for v in ask("6) Resolution WxH (same as the calibration)", "1280x720").lower().split("x"))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, name)
    if os.path.exists(base + ".avi"):
        raise SystemExit(f"{base}.avi exists: choose another name")

    cam = Camera(index, w, h)
    print(HELP)
    input("Press Enter to start the camera... ")
    if not cam.start():
        raise SystemExit("cannot open the camera")
    det = MarkerDetector(cfg["dict_id"], subpixel=True)
    win = "Record - R start/stop"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    writer, times, t_start, show_det, n = None, [], 0.0, True, 0
    last_t, fps = time.time(), 0.0
    while True:
        frame = cam.read()
        if frame is None:
            continue
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-6) if fps else 1.0
        last_t = now
        if writer is not None:
            writer.write(frame)
            times.append(now - t_start)
            if n_max and len(times) >= n_max:                       # stops by itself after the asked number of frames
                print(f"{n_max} frames recorded")
                break
        disp = frame.copy()
        if show_det:
            for mid, c in det.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)).items():
                cv2.polylines(disp, [c.astype(np.int32)], True, (60, 200, 60), 2, cv2.LINE_AA)
                cv2.putText(disp, f"id {mid}", tuple(c[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 200, 60), 2, cv2.LINE_AA)
        bar = (f"REC {len(times)}{'' if not n_max else '/' + str(n_max)} frames  {len(times) / max(times[-1], 1e-6):.0f} fps" if writer is not None
               else "ready - press R to record" + ("" if not n_max else f" ({n_max} frames, stops by itself)"))
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 30), (30, 30, 30), -1)
        cv2.putText(disp, f"{bar}   {disp.shape[1]}x{disp.shape[0]}   camera loop {fps:.0f} fps", (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (60, 60, 255) if writer is not None else (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord("r"), ord("R")):
            if writer is None:
                size = (frame.shape[1], frame.shape[0])
                writer = cv2.VideoWriter(base + ".avi", cv2.VideoWriter_fourcc(*"MJPG"), 30.0, size)
                times, t_start = [], time.time()
                print("recording ->", base + ".avi")
            else:
                break
        elif key in (ord("d"), ord("D")):
            show_det = not show_det
        elif key in (ord("s"), ord("S")):
            p = os.path.join(out_dir, f"{name}_screenshot_{datetime.now():%H%M%S}.png")
            cv2.imwrite(p, frame)
            print("screenshot ->", p)

    cam.stop()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
        fps_real = len(times) / times[-1] if times and times[-1] > 0 else 0.0
        with open(base + ".csv", "w") as f:
            f.write("frame,t_s\n" + "\n".join(f"{i},{t:.4f}" for i, t in enumerate(times)) + "\n")
        json.dump(dict(name=name, tag=tag, notes=notes, distance_cm=None if dist_cm == "-" else float(dist_cm),
                       tilt_deg=None if tilt_deg == "-" else float(tilt_deg), resolution=f"{w}x{h}", frames=len(times), duration_s=round(times[-1], 2),
                       fps=round(fps_real, 2), dictionary=cfg["dictionary"], created=datetime.now().isoformat(timespec="seconds")),
                  open(base + ".json", "w"), indent=1)
        print(f"saved {len(times)} frames ({times[-1]:.1f} s, {fps_real:.1f} fps) -> {base}.avi / .csv / .json")
        if truth is not None:
            with open(base + ".truth.csv", "w") as f:
                f.write("t_s,x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg\n")
                f.write(f"0,{10 * truth['x']:g},{10 * truth['y']:g},{10 * truth['z']:g},0,{truth['turn']:g},0\n")
            print(f"truth (M in F: x {10 * truth['x']:g} mm, y {10 * truth['y']:g} mm, z {10 * truth['z']:g} mm, turned {truth['turn']:g} deg) -> {base}.truth.csv")
    else:
        print("nothing recorded.")


if __name__ == "__main__":
    main()
