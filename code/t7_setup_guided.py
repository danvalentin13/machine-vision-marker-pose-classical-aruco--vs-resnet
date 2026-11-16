"""t7_setup_guided.py -- Bonus: live check of the occlusion setup (F fixed on the right, M slides along a ruler toward F).

  python t7_setup_guided.py [--centre 377] [--travel 280] [--obstacle 155 235]

The window shows the camera with F and M and, on the right, a checklist with live numbers (green = ok, red = fix it):
  both markers seen / room to the picture's edges / camera distance / centre-to-centre distance against the ruler /
  same depth / same height / boards parallel.
On the picture: the path M's centre will take (line from M toward F, --travel mm long), where M ends (dashed square) and the
band the obstacle has to cover (ruler readings --obstacle, measured from M's left corner at the start).
Lengths in mm.  Keys:  S  save a picture and print the numbers      Q  close
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from config import OUTPUT_DIR
from mvtools import MarkerPoseEstimator, load_camera, load_markers_cfg
from t2_detect_live import Camera

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, ORANGE, GREEN, RED = (235, 235, 235), (150, 150, 150), (26, 160, 255), (60, 200, 60), (60, 60, 235)
MARGIN_PX = 50


def project(P, K, dist):
    return cv2.projectPoints(np.asarray(P, np.float64).reshape(-1, 3), np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser(description="Bonus: live check of the occlusion setup.")
    ap.add_argument("--centre", type=float, default=377.0, help="ruler: centre-to-centre distance of M (at the start) and F [mm]")
    ap.add_argument("--travel", type=float, default=280.0, help="how far M slides, start stop to end stop [mm]")
    ap.add_argument("--obstacle", type=float, nargs=2, default=[155.0, 235.0], help="ruler readings the obstacle covers, from M's left corner at the start [mm]")
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()
    cfg = load_markers_cfg()
    K, dist, _ = load_camera()
    L = 1000 * cfg["L"]
    cam = Camera(a.camera, 1280, 720)
    if not cam.start():
        raise SystemExit("cannot open the camera (close other camera windows; check the camera privacy key)")
    est = MarkerPoseEstimator(cfg, K, dist, camera_still=False)
    win = "Bonus - setup check  (S save, Q close)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    out_dir = os.path.join(OUTPUT_DIR, "bonus")
    os.makedirs(out_dir, exist_ok=True)
    print("Window open: follow the checklist on it.")
    hist, last_auto = [], 0.0
    while True:
        frame = cam.read()
        if frame is None:
            continue
        r = est.process(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        disp = frame.copy()
        H, W = frame.shape[:2]
        for mid, c in r["corners"].items():
            cv2.polylines(disp, [c.astype(np.int32)], True, GREEN, 3, cv2.LINE_AA)
        checks, vals = [], {}
        TF, TM = r["T_CF"], r["T_CM"]
        seen = TF is not None and TM is not None
        checks.append(("F and M both seen", seen, "" if seen else ("F missing" if TF is None else "M missing")))
        if seen:
            tF, tM = 1000 * TF[:3, 3], 1000 * TM[:3, 3]
            d = tF - tM
            dist_cc = float(np.linalg.norm(d))
            u = d / dist_cc                                           # the direction M will slide: from M toward F
            hist.append((dist_cc, tF.copy(), tM.copy(), TF[:3, 2].copy(), TM[:3, 2].copy()))
            hist[:] = hist[-15:]
            dist_cc = float(np.median([h[0] for h in hist]))
            tFm, tMm = np.median([h[1] for h in hist], 0), np.median([h[2] for h in hist], 0)
            nF, nM = np.mean([h[3] for h in hist], 0), np.mean([h[4] for h in hist], 0)
            turn = float(np.degrees(np.arccos(np.clip(nF @ nM / np.linalg.norm(nF) / np.linalg.norm(nM), -1, 1))))
            dz, dy = float(tMm[2] - tFm[2]), float(tFm[1] - tMm[1])   # + = M farther from the camera / M higher than F
            pts = np.vstack([c.reshape(-1, 2) for c in r["corners"].values()])
            mleft, mright, mtop, mbot = pts[:, 0].min(), W - pts[:, 0].max(), pts[:, 1].min(), H - pts[:, 1].max()
            room = min(mleft, mright, mtop, mbot)
            side = ["left", "right", "top", "bottom"][int(np.argmin([mleft, mright, mtop, mbot]))]
            checks.append((f"room to the picture edge: {room:.0f} px", room >= MARGIN_PX,
                           "" if room >= MARGIN_PX else f"too close at the {side}: move the laptop back / re-aim"))
            zF = float(tFm[2]) / 10
            checks.append((f"camera to F: {zF:.0f} cm", 50 <= zF <= 75, "" if 50 <= zF <= 75 else "aim for 55-65 cm"))
            ok = abs(dist_cc - a.centre) <= 6
            checks.append((f"centre to centre: {dist_cc:.0f} mm (ruler {a.centre:.0f})", ok, "" if ok else f"{dist_cc - a.centre:+.0f} mm against the ruler"))
            checks.append((f"depth: M {abs(dz):.0f} mm {'behind' if dz > 0 else 'in front of'} F", abs(dz) <= 20,
                           "" if abs(dz) <= 20 else "bring M and F into one plane"))
            checks.append((f"height: M {abs(dy):.0f} mm {'above' if dy > 0 else 'below'} F", abs(dy) <= 15,
                           "" if abs(dy) <= 15 else "level M with F"))
            checks.append((f"boards turned {turn:.1f} deg to each other", turn <= 6, "" if turn <= 6 else "make F and M parallel"))
            vals = dict(centre_mm=dist_cc, depth_mm=dz, height_mm=dy, turn_deg=turn, camera_to_F_cm=zF, room_px=float(room))
            # the path of M's centre, where M ends, and the band the obstacle has to cover (in M's plane)
            up = TM[:3, :3] @ np.array([0.0, -1.0, 0.0])             # along M's side edge (marker y axis)
            if abs(up @ u) > 0.7:                                    # M mounted turned by 90 deg: use its other in-plane axis
                up = TM[:3, :3] @ np.array([1.0, 0.0, 0.0])
            up = up - (up @ u) * u
            up /= np.linalg.norm(up)
            p0, p1 = tM, tM + a.travel * u
            a0, a1 = project([p0 / 1000, p1 / 1000], K, dist)
            cv2.arrowedLine(disp, tuple(a0.astype(int)), tuple(a1.astype(int)), ORANGE, 3, cv2.LINE_AA, tipLength=0.04)
            end = [(p1 + sx * L / 2 * u + sy * L / 2 * up) / 1000 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            cv2.polylines(disp, [project(end, K, dist).astype(np.int32)], True, ORANGE, 2, cv2.LINE_AA)
            cv2.putText(disp, "M ends here", tuple(project([end[3]], K, dist)[0].astype(int) + [0, -10]), FONT, 0.7, ORANGE, 2, cv2.LINE_AA)
            left0 = tM - L / 2 * u                                    # M's left corner at the start = ruler zero
            band = [(left0 + s * u + sy * 0.75 * L * up) / 1000 for s, sy in ((a.obstacle[0], -1), (a.obstacle[1], -1), (a.obstacle[1], 1), (a.obstacle[0], 1))]
            bp = project(band, K, dist).astype(np.int32)
            ov = disp.copy()
            cv2.fillPoly(ov, [bp], RED)
            disp = cv2.addWeighted(ov, 0.3, disp, 0.7, 0)
            cv2.polylines(disp, [bp], True, RED, 2, cv2.LINE_AA)
            cv2.putText(disp, "obstacle here", tuple(bp[3] + [0, -10]), FONT, 0.7, RED, 2, cv2.LINE_AA)
        else:
            hist.clear()
        p = np.full((H, 470, 3), 32, np.uint8)

        def put(y, text, col=WHITE, scale=0.55, th=1):
            cv2.putText(p, text, (14, y), FONT, scale, col, th, cv2.LINE_AA)
        put(34, "BONUS - SETUP CHECK", WHITE, 0.75, 2)
        y = 80
        for text, ok, hint in checks:
            put(y, ("[ok] " if ok else "[fix] ") + text, GREEN if ok else RED, 0.55, 2)
            y += 24
            if hint:
                put(y, "      " + hint, ORANGE, 0.5)
                y += 24
            y += 8
        allok = bool(checks) and all(c[1] for c in checks)
        y = max(y + 10, 400)
        put(y, "ALL OK - ready to record" if allok else "fix the red lines", GREEN if allok else ORANGE, 0.65, 2)
        y += 40
        for line in ("orange arrow: the path of M's centre", "orange square: where M ends (right stop)",
                     "red band: where the obstacle must stand", "   (between the camera and the ruler)",
                     "", "start position: M still at the left stop,", "its left corner at the ruler's 0.", "", "S = save picture + numbers     Q = close"):
            put(y, line, GREY if line.startswith(" ") else WHITE, 0.5)
            y += 22
        cv2.imshow(win, np.hstack([disp, p]))
        if time.time() - last_auto > 2.0:                             # the latest view, refreshed every 2 s (for a helper to look at)
            last_auto = time.time()
            cv2.imwrite(os.path.join(out_dir, "setup_live.jpg"), np.hstack([disp, p]))
            json.dump(dict(vals, all_ok=bool(checks) and all(c[1] for c in checks), red=[c[0] for c in checks if not c[1]]),
                      open(os.path.join(out_dir, "setup_live.json"), "w"))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord("s"), ord("S")):
            f = os.path.join(out_dir, f"setup_{time.strftime('%H%M%S')}.png")
            cv2.imwrite(f, np.hstack([disp, p]))
            print("saved", f, "|", "; ".join(f"{k} {v:.1f}" for k, v in vals.items()), "| all ok" if allok else "| not all ok")
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
