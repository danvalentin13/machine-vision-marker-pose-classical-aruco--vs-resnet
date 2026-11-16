"""t4_record_guided.py -- Task 4: guided recording of M moving (for the animation), instructions on the camera window.

  python t4_record_guided.py [--name T4_motion_1] [--seconds 30]

The window shows the camera with the detected markers and, on the right, what to do now:
  READY      hold M in your hand, F stays where it is, both markers green  -> press R to start
  RECORDING  a completion bar; the instruction changes by itself: left/right, closer/farther, turn, up/down, slow circle
             SPACE pause / resume (paused: nothing saved, the bar waits - e.g. to detach M);  S stop and save now
             warnings: M not seen (bring it back), F not seen, F moved (leave F alone)
  DONE       stops and saves by itself; Q closes the window
Writes data/recordings/<name>.avi / .csv / .json (tag demo), like t3_record.py.  Then: t3_pose.py and t4_animate.py export.
"""
import argparse
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np

from config import data_path
from mvtools import MarkerDetector, load_markers_cfg
from t2_detect_live import Camera

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, ORANGE, GREEN, RED = (235, 235, 235), (150, 150, 150), (26, 160, 255), (60, 200, 60), (60, 60, 235)
MOVES = [("move M slowly LEFT and RIGHT", "sideways, about 20 cm each way"),
         ("move M CLOSER and FARTHER", "toward the camera and back"),
         ("TURN M a little left and right", "up to about 30 degrees"),
         ("LIFT M up and down a bit", "about 10 cm"),
         ("move M in a slow CIRCLE", "keep it in the picture")]
# moves for the TRAINING recordings (--plan train): variety is what the network needs
MOVES_TRAIN = [("bring M CLOSE (25-35 cm) and move it around", "near the camera, all over"),
               ("take M FAR (70-90 cm) and move it around", "far from the camera"),
               ("move M to the LEFT and RIGHT edges of the picture", "and the top / bottom"),
               ("TILT M strongly (40-60 deg) in different directions", "left, right, up, down"),
               ("ROTATE M like a steering wheel", "slowly, both ways"),
               ("hold M RIGHT NEXT TO F (5-15 cm gap), flat", "slide it around F: left, right, above, below"),
               ("free: move M anywhere, any angle", "keep it in the picture")]
# the bonus recordings (--plan bonus): M slides along the ruler from the left stop to the right stop, several passes
MOVES_BONUS = [("REST 2 s at the start - SLIDE to the end - REST 2 s - go back", "repeat; S = stop and save when you have enough passes")]
F_MOVED_PX = 40.0


def motion_schedule(t, T, R0=3.0, rest=2.0):
    """E3 schedule: rest R0 - forward T - rest - back T - rest - forward ...  Returns (step, position 0..1 along the path, countdown).
    step: 0 first rest (countdown), 1 forward, 2 rest at the end, 3 back, 4 rest at the start."""
    if t < R0:
        return 0, 0.0, int(np.ceil(R0 - t))
    tc = (t - R0) % (2 * T + 2 * rest)
    if tc < T:
        return 1, tc / T, 0
    if tc < T + rest:
        return 2, 1.0, 0
    if tc < 2 * T + rest:
        return 3, 1.0 - (tc - T - rest) / T, 0
    return 4, 0.0, 0


def motion_breakpoints(t_end, T, R0=3.0, rest=2.0):
    """(time, position 0..1) at every corner of the schedule up to t_end: the truth is linear in between."""
    out, t = [(0.0, 0.0), (R0, 0.0)], R0
    while t < t_end + 2 * T + 2 * rest:
        out += [(t + T, 1.0), (t + T + rest, 1.0), (t + 2 * T + rest, 0.0), (t + 2 * T + 2 * rest, 0.0)]
        t += 2 * T + 2 * rest
    return out


def beep(freq, ms=120):
    """A short beep without stopping the camera loop (Windows only; silent elsewhere)."""
    try:
        import threading
        import winsound
        threading.Thread(target=winsound.Beep, args=(int(freq), int(ms)), daemon=True).start()
    except Exception:
        pass


def big(img, text, y, col, scale=1.3, th=3):
    (tw, thh), _ = cv2.getTextSize(text, FONT, scale, th)
    x = (img.shape[1] - tw) // 2
    cv2.rectangle(img, (x - 16, y - thh - 14), (x + tw + 16, y + 14), (20, 20, 20), -1)
    cv2.putText(img, text, (x, y), FONT, scale, col, th, cv2.LINE_AA)


def panel(h, phase, t_rec, seconds, found, roles, warn, stats, width=420, moves=None, title="T4 - RECORD M MOVING"):
    moves = moves or MOVES
    p = np.full((h, width, 3), 32, np.uint8)

    def put(y, text, col=WHITE, scale=0.55, th=1):
        cv2.putText(p, text, (14, y), FONT, scale, col, th, cv2.LINE_AA)
    put(30, title, WHITE, 0.68, 2)
    y = 70
    if phase == "ready":
        put(y, "GET READY", ORANGE, 0.65, 2); y += 30
        for line in ("1. F stays where it is - do not touch it.", "2. Both markers must be GREEN.", "3. Press R to start recording.",
                     "", "SPACE = PAUSE / RESUME at any time:", "  paused = nothing is saved and the bar", "  waits (e.g. to detach M). No hurry.",
                     "", "It stops by itself when the bar is full;", "the instruction changes by itself."):
            put(y, line); y += 24
    elif phase in ("rec", "paused"):
        if phase == "paused":
            put(y, f"PAUSED   {100 * min(1, t_rec / seconds):.0f} % done", ORANGE, 0.62, 2); y += 18
        else:
            put(y, f"RECORDING   {100 * min(1, t_rec / seconds):.0f} % done", GREEN, 0.62, 2); y += 18
        cv2.rectangle(p, (14, y), (width - 14, y + 18), (90, 90, 90), 1)
        cv2.rectangle(p, (15, y + 1), (15 + int((width - 30) * min(1, t_rec / seconds)), y + 17), GREEN, -1); y += 46
        k = min(int(t_rec // (seconds / len(moves))), len(moves) - 1)
        for i, (m, sub) in enumerate(moves):
            col = GREEN if i < k else (ORANGE if i == k else GREY)
            put(y, f"{'[x]' if i < k else '[>]' if i == k else '[ ]'} {m}", col, 0.5, 2 if i == k else 1); y += 24
        y += 6
        put(y, ("SPACE = resume" if phase == "paused" else "SPACE = pause") + "    S = stop and save now", ORANGE, 0.55, 2)
    else:
        put(y, "DONE - SAVED", GREEN, 0.68, 2); y += 32
        for line in stats:
            put(y, line); y += 24
        y += 10
        put(y, "Press Q to close the window.", ORANGE, 0.58, 2)
    y = h - 110
    put(y, "NOW", ORANGE, 0.6, 2); y += 26
    for mid, role in roles.items():
        ok = mid in found
        put(y, f"{role} (id {mid}): {'seen' if ok else 'NOT SEEN'}", GREEN if ok else RED, 0.55, 2); y += 24
    if warn:
        put(y, warn, RED, 0.5, 2)
    return p


def main():
    ap = argparse.ArgumentParser(description="Task 4: guided recording of M moving.")
    ap.add_argument("--name", default=None, help="recording name (default: T4_motion_1, or the next free real_v2_NN with --plan train)")
    ap.add_argument("--truth-start", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="motion plan: measured position of M in F at the start stop [mm]")
    ap.add_argument("--truth-end", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="motion plan: measured position of M in F at the end stop [mm]")
    ap.add_argument("--truth-angles", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("ROLL", "PITCH", "YAW"), help="motion plan: measured angles of M in F [deg]")
    ap.add_argument("--plan", choices=["t4", "train", "bonus", "motion"], default="t4",
                    help="t4: the T4 moves; train: varied moves for the network's real training frames; bonus: passes along the ruler (E7)")
    ap.add_argument("--notes", default="", help="extra text for the recording's notes (setup, speed, obstacle ...)")
    ap.add_argument("--folder", default=None, help="where to save (default: data/recordings, or training/<set> with --plan train)")
    ap.add_argument("--set", default="real_v2", help="training set name with --plan train (folder training/<set>, recordings <set>_NN), e.g. real_v3")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--travel", type=float, default=273.0, help="bonus plan: travel of M from the start stop to the end stop [mm]")
    ap.add_argument("--centre", type=float, default=377.0, help="bonus plan: centre-to-centre distance M - F at the start stop [mm]")
    ap.add_argument("--pass-s", type=float, default=2.7, help="bonus plan: how long one slide from the start stop to the end stop should take [s]")
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()
    cfg = load_markers_cfg()
    roles = {cfg["id_fixed"]: "F", cfg["id_moving"]: "M"}
    from config import training_path
    train = a.plan == "train"
    folder = a.folder or (training_path(a.set) if train else data_path("recordings"))
    os.makedirs(folder, exist_ok=True)
    if a.name is None:
        if train:
            k = 1
            while os.path.exists(os.path.join(folder, f"{a.set}_{k:02d}.avi")):
                k += 1
            a.name = f"{a.set}_{k:02d}"
        elif a.plan == "motion":
            k = 1
            while os.path.exists(os.path.join(folder, f"E3_motion_{k:02d}.avi")):
                k += 1
            a.name = f"E3_motion_{k:02d}"
        elif a.plan == "bonus":
            k = 1
            while os.path.exists(os.path.join(folder, f"E7_pass_{k:02d}.avi")):
                k += 1
            a.name = f"E7_pass_{k:02d}"
        else:
            a.name = "T4_motion_1"
    moves = MOVES_TRAIN if train else MOVES_BONUS if a.plan == "bonus" else MOVES
    title = (f"TRAINING FRAMES - {a.name}" if train else f"BONUS - {a.name}" if a.plan == "bonus" else
             f"T6 E3 KNOWN MOTION - {a.name}" if a.plan == "motion" else "T4 - RECORD M MOVING")
    if a.plan == "motion":
        if a.truth_start is None or a.truth_end is None:
            ap.error("--plan motion needs --truth-start and --truth-end (measured positions of M in F at the two stops)")
        a.travel = float(np.linalg.norm(np.array(a.truth_end) - np.array(a.truth_start)))
        moves = MOVES_BONUS
    base = os.path.join(folder, a.name)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    if os.path.exists(base + ".avi"):
        raise SystemExit(f"{base}.avi exists: choose another --name")
    cam = Camera(a.camera, 1280, 720)
    if not cam.start():
        raise SystemExit("cannot open the camera (close other camera windows)")
    det = MarkerDetector(cfg["dict_id"], subpixel=True)
    win = f"{title}  (R start, Q close)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    phase, writer, times, t0, f0, stats = "ready", None, [], 0.0, None, []
    t_pause, stop_now, pacer_step = 0.0, False, None
    ghost0, ghost_d = None, None
    hits = {i: 0 for i in roles}
    print(f"Window open: follow the instructions on it. Recording {a.seconds:g} s -> {base}.avi")
    while True:
        frame = cam.read()
        if frame is None:
            continue
        found = det.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        disp = frame.copy()
        for mid, c in found.items():
            col = GREEN if mid in roles else RED
            cv2.polylines(disp, [c.astype(np.int32)], True, col, 3, cv2.LINE_AA)
            cv2.putText(disp, f"{roles.get(mid, '?')} id {mid}", tuple(c[0].astype(int) - [0, 12]), FONT, 0.9, col, 2, cv2.LINE_AA)
        warn, t_rec = "", 0.0
        if phase == "rec":
            t_rec = time.time() - t0
            writer.write(frame)
            times.append(t_rec)
            for i in roles:
                hits[i] += i in found
            if cfg["id_fixed"] in found:
                c = found[cfg["id_fixed"]].reshape(-1, 2).mean(axis=0)
                f0 = c if f0 is None else f0
                if np.linalg.norm(c - f0) > F_MOVED_PX:
                    warn = "F MOVED - leave F alone!"
            k = min(int(t_rec // (a.seconds / len(moves))), len(moves) - 1)
            if a.plan == "motion":                                    # E3: the outline goes to the end AND back at constant speed (truth = this schedule)
                step, pos, count = motion_schedule(t_rec, a.pass_s)
                if (step, count) != pacer_step:
                    pacer_step = (step, count)
                    beep(600, 70) if step == 0 else beep(1400 if step in (1, 3) else 900)
                txt, col = [(f"HOLD STILL at the start - GO in {count}", WHITE), ("GO  -  keep M in the outline", GREEN), ("HOLD STILL at the end", WHITE),
                            ("BACK  -  keep M in the outline", GREEN), ("HOLD STILL at the start", WHITE)][step]
                big(disp, txt, 70, col, 1.1, 3)
                if step == 0:
                    big(disp, str(count), 250, ORANGE, 4.0, 8)
                if step in (0, 4) and cfg["id_moving"] in found:          # M rests at the start: remember where, and its size in pixels
                    ghost0 = found[cfg["id_moving"]].reshape(-1, 2).astype(np.float64)
                    wpx = (np.linalg.norm(ghost0[0] - ghost0[1]) + np.linalg.norm(ghost0[3] - ghost0[2])) / 2
                    away = -1.0 if (cfg["id_fixed"] not in found or ghost0[:, 0].mean() < found[cfg["id_fixed"]].reshape(-1, 2)[:, 0].mean()) else 1.0
                    sgn = away if abs(a.truth_end[0]) > abs(a.truth_start[0]) else -away      # the end stop is farther from F / nearer to F
                    ghost_d = np.array([sgn * a.travel * wpx / (1000 * cfg["L"]), 0.0])
                if ghost0 is not None:
                    gp = (ghost0 + pos * ghost_d).astype(np.int32)
                    cv2.polylines(disp, [gp], True, (255, 255, 0), 3, cv2.LINE_AA)
                    if step in (1, 3):
                        cv2.putText(disp, "keep M here", tuple(gp[0] - [0, 12]), FONT, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            elif a.plan == "bonus":                                   # pacer: rest - GO (bar sweeps in --pass-s) - rest - back, repeated
                R0 = 3.0                                              # rest at the start, with the countdown 3 - 2 - 1
                cyc = R0 + a.pass_s + 2.0 + 3.0
                n_pass, tc = int(t_rec // cyc) + 1, t_rec % cyc
                step = 0 if tc < R0 else 1 if tc < R0 + a.pass_s else 2 if tc < R0 + 2.0 + a.pass_s else 3
                count = int(np.ceil(R0 - tc)) if step == 0 else 0
                if (step, count) != pacer_step:                       # a tick for 3 - 2 - 1, a beep when GO starts and when M should arrive
                    pacer_step = (step, count)
                    if step == 0:
                        beep(600, 70)
                    elif step in (1, 2):
                        beep(1400 if step == 1 else 900)
                txt, col = [(f"pass {n_pass}: HOLD STILL - GO in {count}", WHITE), (f"pass {n_pass}: GO - keep M in the outline", GREEN),
                            (f"pass {n_pass}: HOLD STILL at the end", WHITE), ("bring M back to the start", ORANGE)][step]
                big(disp, txt, 70, col, 1.1, 3)
                if step == 0:                                         # the countdown, large
                    big(disp, str(count), 250, ORANGE, 4.0, 8)
                if step == 0 and cfg["id_moving"] in found and cfg["id_fixed"] in found:      # M rests at the start: remember where
                    ghost0 = found[cfg["id_moving"]].reshape(-1, 2).astype(np.float64)
                    ghost_d = (found[cfg["id_fixed"]].reshape(-1, 2).mean(0) - ghost0.mean(0)) * (a.travel / a.centre)
                if step == 1 and ghost0 is not None:                  # the ghost: where M should be now (constant speed) - keep M on it
                    gp = (ghost0 + (tc - R0) / a.pass_s * ghost_d).astype(np.int32)
                    cv2.polylines(disp, [gp], True, (255, 255, 0), 3, cv2.LINE_AA)
                    cv2.putText(disp, "keep M here", tuple(gp[0] - [0, 12]), FONT, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                if step == 1:                                         # the pacing bar: follow it with M
                    g = (tc - R0) / a.pass_s
                    xa, xb, yp = disp.shape[1] // 8, 7 * disp.shape[1] // 8, 160
                    cv2.rectangle(disp, (xa - 4, yp - 4), (xb + 4, yp + 44), (20, 20, 20), -1)
                    cv2.rectangle(disp, (xa, yp), (xa + int((xb - xa) * g), yp + 40), GREEN, -1)
                    cv2.rectangle(disp, (xa, yp), (xb, yp + 40), WHITE, 2)
            else:
                big(disp, moves[k][0], 70, ORANGE, 1.1, 3)
            frac = min(1.0, t_rec / a.seconds)                         # completion bar instead of a countdown
            x0, x1, yb = disp.shape[1] // 4, 3 * disp.shape[1] // 4, 110
            cv2.rectangle(disp, (x0 - 4, yb - 4), (x1 + 4, yb + 26), (20, 20, 20), -1)
            cv2.rectangle(disp, (x0, yb), (x0 + int((x1 - x0) * frac), yb + 22), GREEN, -1)
            cv2.rectangle(disp, (x0, yb), (x1, yb + 22), WHITE, 2)
            if cfg["id_moving"] not in found and a.plan != "bonus":     # (bonus: M is meant to disappear behind the obstacle)
                warn = warn or "M NOT SEEN - bring M back into the picture"
                big(disp, "M not seen - bring it back", disp.shape[0] - 40, RED, 1.0, 3)
            if t_rec >= a.seconds or stop_now:                        # stop by itself (or S: stop and save now)
                writer.release()
                n = len(times)
                open(base + ".csv", "w").write("frame,t_s\n" + "\n".join(f"{i},{t:.4f}" for i, t in enumerate(times)) + "\n")
                if a.plan == "motion":                                # the truth: the schedule of the outline between the two measured stops
                    ang = ",".join(f"{v:g}" for v in a.truth_angles)
                    rows_t = ["t_s,x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg"]
                    for tb, pb in motion_breakpoints(times[-1], a.pass_s):
                        pt = np.array(a.truth_start) + pb * (np.array(a.truth_end) - np.array(a.truth_start))
                        rows_t.append(f"{tb:.3f}," + ",".join(f"{v:.1f}" for v in pt) + "," + ang)
                    open(base + ".truth.csv", "w").write("\n".join(rows_t) + "\n")
                json.dump(dict(name=a.name, tag="train" if train else "E7_occlusion" if a.plan == "bonus" else "E3_motion" if a.plan == "motion" else "demo",
                               notes=(a.notes + "; " if a.notes else "") + ("bonus E7: F still, M slides along the ruler from the left stop to the right stop (rest - slide - rest), several passes"
                                      if a.plan == "bonus" else f"network training frames ({a.set}): F still, M moved by hand - close, far, edges, strong tilts, in-plane rotation, free"
                                      if train else "T4: F still, M moved by hand (guided: left/right, closer/farther, turn, up/down, circle)"),
                               distance_cm=None, tilt_deg=None, resolution=f"{frame.shape[1]}x{frame.shape[0]}", frames=n,
                               duration_s=round(times[-1], 2), fps=round(n / times[-1], 2), dictionary=cfg["dictionary"],
                               created=datetime.now().isoformat(timespec="seconds")), open(base + ".json", "w"), indent=1)
                stats = [f"{n} frames, {times[-1]:.1f} s ({n / times[-1]:.0f} fps)",
                         f"F seen in {100 * hits[cfg['id_fixed']] / n:.0f} % of the frames",
                         f"M seen in {100 * hits[cfg['id_moving']] / n:.0f} % of the frames", f"-> {a.name}.avi"]
                print("saved:", "; ".join(stats))
                phase = "done"
        elif phase == "paused":                                       # nothing is written, the bar waits
            t_rec = t_pause - t0
            k = min(int(t_rec // (a.seconds / len(moves))), len(moves) - 1)
            big(disp, "PAUSED - take your time. SPACE to resume", 70, ORANGE, 1.0, 3)
            big(disp, "next: " + moves[k][0], 130, WHITE, 0.8, 2)
        elif phase == "ready":
            both = all(i in found for i in roles)
            big(disp, "Both markers green? Press R to start" if both else "Show F and M to the camera", 70, GREEN if both else ORANGE, 1.1, 3)
        else:
            big(disp, "DONE - saved. Press Q to close", 70, GREEN, 1.1, 3)
        cv2.imshow(win, np.hstack([disp, panel(disp.shape[0], phase, t_rec, a.seconds, found, roles, warn, stats, moves=moves, title=title)]))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            if phase in ("rec", "paused") and len(times) > 150:      # closed while recording: keep what was recorded (5 s or more)
                if phase == "paused":
                    t0 += time.time() - t_pause
                phase, stop_now = "rec", True
                print("closed while recording: saving what was recorded")
                continue
            break
        if key in (ord("s"), ord("S")) and phase in ("rec", "paused") and len(times) > 30:
            if phase == "paused":
                t0 += time.time() - t_pause
            phase, stop_now = "rec", True                            # saved on the next frame
            print("stop requested: saving")
        elif key == ord(" ") and phase == "rec":                     # pause (e.g. to detach M)
            phase, t_pause = "paused", time.time()
            print("paused")
        elif key == ord(" ") and phase == "paused":                  # resume: the paused time does not count
            t0 += time.time() - t_pause
            phase = "rec"
            print("resumed")
        elif key in (ord("r"), ord("R")) and phase == "ready":
            writer = cv2.VideoWriter(base + ".avi", cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (frame.shape[1], frame.shape[0]))
            times, t0, f0, hits, phase = [], time.time(), None, {i: 0 for i in roles}, "rec"
            print("recording ...")
    if phase in ("rec", "paused") and writer is not None:                        # closed early: keep nothing half-written
        writer.release()
        os.remove(base + ".avi")
        print("closed before the end: nothing saved")
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
