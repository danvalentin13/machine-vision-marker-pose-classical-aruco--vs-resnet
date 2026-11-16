"""t6_record.py -- Task 6: record one TEST recording with its hand-measured truth, from one command (no questions).

  python t6_record.py --name E2_d60_t30 --tag E2_dist_tilt --truth -200 0 0 0 30 0 --distance 60 --tilt 30 --notes "..."
  python t6_record.py --check                    live check only: what the camera measures now, nothing is recorded

  --truth x y z roll pitch yaw   the MEASURED pose of M in the frame of F [mm, deg] (ruler / protractor - never from the camera)
                                 x: + = M to the right of F (as the camera sees it), y: + = M higher, z: + = M nearer the camera;
                                 pitch = M turned about the vertical axis.  Same convention as the E1 recordings.
  --frames N                     how many frames (default 100, about 3 s); the recording stops by itself
The window shows the camera with F and M and a table: the truth you entered, what the classical pipeline measures right now,
and the difference.  That is only a CHECK against typing and sign mistakes (a large difference is red); the truth file holds
your numbers.  Below it: how each marker faces the camera and how steady the measurement is.
Keys:  R record (stops by itself)    Q close
Writes data/recordings/<name>.avi / .csv / .json / .truth.csv  - then run t6_evaluate.py.
"""
import argparse
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import OUTPUT_DIR, data_path
from mvtools import MarkerPoseEstimator, load_camera, load_markers_cfg, rpy_deg
from t2_detect_live import Camera

GREEN_BGR = (60, 200, 60)
BG, CARD, LINE = (30, 32, 36), (42, 45, 51), (78, 82, 90)                 # RGB (the panel is drawn with PIL)
WHITE, GREY, ORANGE, GREEN, RED = (236, 236, 236), (160, 164, 170), (255, 170, 40), (80, 210, 90), (240, 80, 70)
ROWS = (("x", "+ = M right of F", "mm", 30), ("y", "+ = M higher", "mm", 30), ("z", "+ = M nearer camera", "mm", 40),
        ("roll", "lean forward / back", "deg", 15), ("pitch", "turn about vertical", "deg", 15), ("yaw", "rotation in the picture", "deg", 15))


def font(size, bold=False, mono=False):
    for name in (("consolab.ttf" if bold else "consola.ttf") if mono else ("segoeuib.ttf" if bold else "segoeui.ttf"), "arial.ttf"):
        try:
            return ImageFont.truetype(os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", name), size)
        except OSError:
            pass
    return ImageFont.load_default()


class Panel:
    """The table next to the camera picture, drawn with real fonts so that rows and columns line up."""

    def __init__(self, h, w):
        self.h, self.w = h, w
        k = h / 720.0                                                 # everything scales with the height of the camera frame
        self.k = k
        self.f_title, self.f_sub = font(int(30 * k), True), font(int(19 * k))
        self.f_head, self.f_row, self.f_hint = font(int(19 * k), True), font(int(21 * k)), font(int(15 * k))
        self.f_num, self.f_big = font(int(22 * k), mono=True), font(int(24 * k), True)

    def table(self, d, x0, y0, widths, header, rows, row_h):
        """A grid: header + rows; a cell is text or (text, colour[, font]).  Numbers (mono font) are right-aligned."""
        k, x1 = self.k, x0 + sum(widths)
        d.rectangle([x0, y0, x1, y0 + row_h], fill=(58, 62, 70))
        for i, (text, wd) in enumerate(zip(header, widths)):
            xa = x0 + sum(widths[:i])
            if i == 0:
                d.text((xa + 10 * k, y0 + row_h / 2), text, font=self.f_head, fill=ORANGE, anchor="lm")
            else:
                d.text((xa + wd - 12 * k, y0 + row_h / 2), text, font=self.f_head, fill=ORANGE, anchor="rm")
        y = y0 + row_h
        for n, row in enumerate(rows):
            if n % 2 == 0:
                d.rectangle([x0, y, x1, y + row_h], fill=CARD)
            for i, (cell, wd) in enumerate(zip(row, widths)):
                text, col, fnt = (cell + (None,) * 3)[:3] if isinstance(cell, tuple) else (cell, WHITE, None)
                xa = x0 + sum(widths[:i])
                if i == 0:
                    d.text((xa + 10 * k, y + row_h / 2), text, font=fnt or self.f_row, fill=col or WHITE, anchor="lm")
                else:
                    d.text((xa + wd - 12 * k, y + row_h / 2), text, font=fnt or self.f_num, fill=col or WHITE, anchor="rm")
            y += row_h
        d.rectangle([x0, y0, x1, y], outline=LINE, width=max(1, int(2 * k)))
        for i in range(1, len(widths)):
            xa = x0 + sum(widths[:i])
            d.line([xa, y0, xa, y], fill=LINE, width=1)
        for n in range(1, len(rows) + 1):
            d.line([x0, y0 + n * row_h, x1, y0 + n * row_h], fill=LINE, width=1)
        return y

    def render(self, a, truth, m, tilt, jit, phase, frac, n_rec, seen_pct):
        k, W = self.k, self.w
        img = Image.new("RGB", (W, self.h), BG)
        d = ImageDraw.Draw(img)
        pad = int(22 * k)
        d.text((pad, int(14 * k)), "T6 live check" if a.check else f"T6 test recording   {a.name}", font=self.f_title, fill=WHITE)
        sub = "nothing is recorded" if a.check else f"experiment {a.tag}" + (f"    camera distance {a.distance:g} cm" if a.distance is not None else "") + \
              (f"    tilt {a.tilt:g} deg" if a.tilt is not None else "")
        d.text((pad, int(56 * k)), sub, font=self.f_sub, fill=GREY)
        # table 1: the pose of M in the frame of F
        inner = W - 2 * pad
        widths = [int(inner * f) for f in (0.34, 0.22, 0.22, 0.22)]
        rows, bad = [], False
        for i, (name, hint, unit, tol) in enumerate(ROWS):
            t_ok, m_ok = np.isfinite(truth[i]), m is not None
            diff = (m[i] - truth[i]) if (t_ok and m_ok) else None
            off = diff is not None and abs(diff) > tol
            bad = bad or off
            rows.append(((f"{name}   ({unit})", WHITE), (f"{truth[i]:+.1f}" if t_ok else "-", WHITE), (f"{m[i]:+.1f}" if m_ok else "-", WHITE),
                         ("-" if diff is None else f"{diff:+.1f}", GREY if diff is None else RED if off else GREEN)))
        d.text((pad, int(92 * k)), "Pose of M measured in the directions of F", font=self.f_head, fill=WHITE)
        y = self.table(d, pad, int(122 * k), widths, ("", "truth (yours)", "camera now", "difference"), rows, int(34 * k))
        d.text((pad, y + int(6 * k)), "x: + = M right of F     y: + = M higher     z: + = M nearer the camera     pitch = turn about the vertical axis",
               font=self.f_hint, fill=GREY)
        # table 2: how each marker faces the camera
        y += int(36 * k)
        d.text((pad, y), "How each marker faces the camera   (0 = straight on)", font=self.f_head, fill=WHITE)
        col = lambda v: GREEN if abs(v) < 8 else ORANGE
        if tilt is not None:
            rows2 = [(("F   (deg)", WHITE), (f"{tilt[0]:+.1f}", col(tilt[0])), (f"{tilt[1]:+.1f}", col(tilt[1])), (f"{tilt[4]:.1f} cm", WHITE)),
                     (("M   (deg)", WHITE), (f"{tilt[2]:+.1f}", col(tilt[2])), (f"{tilt[3]:+.1f}", col(tilt[3])),
                      (f"+-{jit:.1f} mm", GREEN if jit < 1.5 else ORANGE))]
        else:
            rows2 = [(("F   (deg)", WHITE), "-", "-", "-"), (("M   (deg)", WHITE), "-", "-", "-")]
        y = self.table(d, pad, y + int(30 * k), widths, ("", "up / down", "left / right", "camera / steady"), rows2, int(34 * k))
        # the verdict
        y += int(14 * k)
        if m is None:
            msg, c = "F and M must both be seen (green outlines).", ORANGE
        elif a.check or not np.isfinite(truth).all():
            msg, c = "live values - compare them with your ruler.", GREY
        elif bad:
            msg, c = "RED difference: check the sign, the number or the setup.", RED
        else:
            msg, c = "truth and camera agree roughly: ok.", GREEN
        d.text((pad, y), msg, font=self.f_big, fill=c)
        if not a.check and m is not None and not bad:
            d.text((pad, y + int(32 * k)), "(small differences are normal: they are the error the experiment measures)", font=self.f_hint, fill=GREY)
        # the recording
        y = self.h - int(118 * k)
        d.line([pad, y - int(12 * k), W - pad, y - int(12 * k)], fill=LINE, width=1)
        if a.check:
            d.text((pad, y), "LIVE CHECK - nothing is recorded.   Q closes.", font=self.f_big, fill=ORANGE)
        elif phase == "ready":
            d.text((pad, y), "Nothing moves.   Press R to record.", font=self.f_big, fill=ORANGE)
            d.text((pad, y + int(34 * k)), f"{a.frames} frames (about {a.frames / 30:.0f} s); it stops by itself.   Q closes.", font=self.f_sub, fill=GREY)
        elif phase == "rec":
            d.text((pad, y), f"RECORDING   {100 * frac:.0f} % done", font=self.f_big, fill=GREEN)
            d.rectangle([pad, y + int(38 * k), W - pad, y + int(62 * k)], outline=LINE, width=2)
            d.rectangle([pad + 2, y + int(38 * k) + 2, pad + 2 + int((inner - 4) * frac), y + int(62 * k) - 2], fill=GREEN)
        else:
            d.text((pad, y), "DONE - SAVED.   Press Q to close.", font=self.f_big, fill=GREEN)
            d.text((pad, y + int(34 * k)), f"{n_rec} frames, M measured in {seen_pct:.0f} % of them", font=self.f_sub, fill=WHITE)
        if a.notes:
            d.text((pad, self.h - int(26 * k)), a.notes[:int(W / (7.6 * k))], font=self.f_hint, fill=GREY)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description="Task 6: record a test recording with its measured truth.")
    ap.add_argument("--check", action="store_true", help="live check only: shows what the camera measures, records nothing")
    ap.add_argument("--name", default="check")
    ap.add_argument("--tag", default="other", choices=["E1_static", "E2_dist_tilt", "E3_motion", "E4_robust", "other"])
    ap.add_argument("--truth", type=float, nargs=6, default=None, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"))
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--distance", type=float, default=None, help="camera to markers [cm] (for the E2 table)")
    ap.add_argument("--tilt", type=float, default=None, help="tilt of M [deg] (for the E2 table)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()
    if not a.check and a.truth is None:
        ap.error("--truth is needed (or --check for a live check without recording)")
    cfg = load_markers_cfg()
    K, dist, _ = load_camera()
    base = data_path("recordings", a.name)
    live_dir = os.path.join(OUTPUT_DIR, "evaluation")
    os.makedirs(live_dir, exist_ok=True)
    if not a.check and os.path.exists(base + ".avi"):
        raise SystemExit(f"{base}.avi exists: choose another --name")
    cam = Camera(a.camera, 1280, 720)
    if not cam.start():
        raise SystemExit("cannot open the camera (close other camera windows; check the camera privacy key)")
    est = MarkerPoseEstimator(cfg, K, dist, camera_still=True)
    win = "T6 - live check  (Q close)" if a.check else f"T6 - {a.name}  (R record, Q close)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    truth = np.array(a.truth if a.truth is not None else [np.nan] * 6)
    phase, writer, times, t0, hist, tilt_hist, seen, last_auto, panel, m = "ready", None, [], 0.0, [], [], 0, 0.0, None, None
    missed = 0
    print("Window open: live check, nothing is recorded." if a.check else f"Window open: check the table, press R.  {a.frames} frames -> {base}.avi")
    while True:
        frame = cam.read()
        if frame is None:
            continue
        r = est.process(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if phase == "rec":
            writer.write(frame)
            times.append(time.time() - t0)
            seen += r["T_FM"] is not None
        disp = frame.copy()
        for mid, c in r["corners"].items():
            cv2.polylines(disp, [c.astype(np.int32)], True, GREEN_BGR, 3, cv2.LINE_AA)
            cv2.putText(disp, "F" if mid == cfg["id_fixed"] else "M" if mid == cfg["id_moving"] else "?", tuple(c.reshape(-1, 2)[0].astype(int) - [0, 14]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, GREEN_BGR, 3, cv2.LINE_AA)
        if r["T_FM"] is not None:
            hist.append(np.concatenate([1000 * r["T_FM"][:3, 3], rpy_deg(r["T_FM"][:3, :3])]))
            hist[:] = hist[-30:]
            nF, nM = r["T_CF"][:3, 2], r["T_CM"][:3, 2]                # how each board faces the camera (0 = straight on)
            tilt_hist.append([np.degrees(np.arctan2(nF[1], -nF[2])), np.degrees(np.arctan2(nF[0], -nF[2])),
                              np.degrees(np.arctan2(nM[1], -nM[2])), np.degrees(np.arctan2(nM[0], -nM[2])), 100 * r["T_CF"][2, 3]])
            tilt_hist[:] = tilt_hist[-30:]
            missed = 0
        else:
            missed += 1
            if missed > 10:                                           # M (or F) is not measured any more: no stale numbers in the table
                hist.clear(); tilt_hist.clear()
        m = np.median(hist, 0) if hist else None
        tilt = np.median(tilt_hist, 0) if tilt_hist else None
        jit = float(np.std(np.array(hist)[:, :3], 0).max()) if len(hist) > 2 else 0.0
        if panel is None:
            panel = Panel(disp.shape[0], int(disp.shape[0] * 1.12))       # the table is as high as the camera frame and scales with it
        view = np.hstack([disp, panel.render(a, truth, m, tilt, jit, phase, len(times) / a.frames, len(times), 100 * seen / max(1, len(times)))])
        cv2.imshow(win, view)
        if time.time() - last_auto > 2.0:                             # the latest view, for a helper to look at
            last_auto = time.time()
            cv2.imwrite(os.path.join(live_dir, "t6_live.jpg"), view, [cv2.IMWRITE_JPEG_QUALITY, 92])
            json.dump(dict(camera=None if m is None else [round(float(v), 1) for v in m], tilt_F_M_dist=None if tilt is None else [round(float(v), 1) for v in tilt],
                           steady_mm=round(jit, 2), phase=phase), open(os.path.join(live_dir, "t6_live.json"), "w"))
        key = cv2.waitKey(1) & 0xFF
        if phase == "rec" and len(times) >= a.frames:
            writer.release()
            n = len(times)
            open(base + ".csv", "w").write("frame,t_s\n" + "\n".join(f"{i},{t:.4f}" for i, t in enumerate(times)) + "\n")
            open(base + ".truth.csv", "w").write("t_s,x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg\n0," + ",".join(f"{v:g}" for v in truth) + "\n")
            json.dump(dict(name=a.name, tag=a.tag, notes=a.notes, distance_cm=a.distance, tilt_deg=a.tilt,
                           resolution=f"{frame.shape[1]}x{frame.shape[0]}", frames=n, duration_s=round(times[-1], 2), fps=round(n / times[-1], 2),
                           dictionary=cfg["dictionary"], created=datetime.now().isoformat(timespec="seconds"),
                           camera_check=(None if m is None else [round(float(v), 1) for v in m])), open(base + ".json", "w"), indent=1)
            print(f"saved: {n} frames, M measured in {100 * seen / n:.0f} %; truth {truth.tolist()}; camera check {None if m is None else np.round(m, 1).tolist()} -> {a.name}.avi")
            phase = "done"
        if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord("r"), ord("R")) and phase == "ready" and not a.check:
            writer = cv2.VideoWriter(base + ".avi", cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (frame.shape[1], frame.shape[0]))
            times, t0, seen, phase = [], time.time(), 0, "rec"
            print("recording ...")
    if phase == "rec" and writer is not None:
        writer.release()
        os.remove(base + ".avi")
        print("closed before the end: nothing saved")
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
