"""t2_matrix.py -- Task 2: guided detection-limit test as a matrix  distance x tilt  (used by t2_detect_live.py, mode 'matrix').

For every cell of the matrix the program says where to put the marker (distance, tilt about its vertical axis). A panel on the
right shows the marker's distance and tilt measured live from its pose (calibrated camera + measured side L) next to the target,
and a FIT bar that turns red when both match. SPACE runs the 10-frame test of the cell (PASS = detected in >= 9 of 10 frames).
The measured values are a guide: the reference for the report are the ruler and protractor values (= the targets you set).

Keys (click the window first):  SPACE  test this cell     N  skip this cell     U  back to the previous cell (redo it)
                                Q / Esc  stop (the results so far are kept)
Writes  <folder>/detection_matrix.csv   one row per cell: target distance and tilt, detections, PASS/FAIL, marker size [px],
                                        px per cell, distance and tilt estimated from the pose, fps
        <folder>/detection_matrix.md    the matrix as a table and the limits (largest distance at tilt 0, largest tilt per distance)
"""
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np

from config import data_path
from mvtools import marker_pose_candidates

DEFAULT_DISTANCES = (30, 50, 75, 100)                      # cm: spans a 1 m set-up (~27 -> ~8 px per cell for a 77 mm marker)
DEFAULT_TILTS = (0, 30, 45, 60, 70, 75, 80, 85)            # deg: dense at large tilts, where detection fails
N_TEST, PASS_MIN = 10, 9
CLEAR_FAIL = 5                                             # skip larger tilts only after <= 5/10 detections
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, ORANGE, GREEN, RED = (235, 235, 235), (150, 150, 150), (26, 160, 255), (60, 200, 60), (60, 60, 235)


def guide_camera(size):
    """camera.npz if it exists for this image size, else an approximate laptop camera (~70 deg field of view)."""
    path = data_path("calibration", "camera.npz")
    if os.path.exists(path):
        d = np.load(path)
        if (int(d["width"]), int(d["height"])) == tuple(size):
            return d["K"].astype(float), d["dist"].astype(float).ravel(), "calibrated camera (camera.npz)"
    f = size[0] / (2 * np.tan(np.radians(35)))
    return np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]]), np.zeros(5), "APPROXIMATE camera (no camera.npz)"


def view_estimate(corners, K, dist, L):
    """Distance camera -> marker centre [cm] and tilt of the marker to the line of sight [deg], from the marker pose."""
    T = marker_pose_candidates(corners, K, dist, L)[0][1]
    t, n = T[:3, 3], T[:3, 2]
    ray = t / np.linalg.norm(t)
    return 100.0 * float(np.linalg.norm(t)), float(np.degrees(np.arccos(min(1.0, abs(float(n @ ray))))))


def side_px(corners):
    c = np.asarray(corners, float).reshape(4, 2)
    return float(np.mean(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)))


def fit(target, est):
    """Score 0..1 and flags: distance within max(3 cm, 5 %), tilt within 5 deg (0 deg target: up to 10 deg, the
    tilt of a marker seen straight on is poorly determined: planar ambiguity)."""
    (d, a), (ed, ea) = target, est
    tol_d = max(3.0, 0.05 * d)
    ok_d = abs(ed - d) <= tol_d
    ok_a = ea <= 10.0 if a == 0 else abs(ea - a) <= 5.0
    s_d = 1.0 if ok_d else max(0.0, 1 - (abs(ed - d) - tol_d) / 20.0)
    s_a = 1.0 if ok_a else max(0.0, 1 - (max(0.0, ea - 10.0) if a == 0 else abs(ea - a) - 5.0) / 20.0)
    return (s_d + s_a) / 2, ok_d, ok_a


def run_cell(source, detector, mid, K, dist, L, cells, warmup=3):
    """10 fresh frames: detections of marker `mid`, mean size [px], median estimated distance and tilt, fps."""
    for _ in range(warmup):
        source.read()
    hits, sides, ests, got, t0 = 0, [], [], 0, time.time()
    while got < N_TEST:
        frame = source.read()
        if frame is None:
            continue
        got += 1
        found = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if mid in found:
            hits += 1
            sides.append(side_px(found[mid]))
            ests.append(view_estimate(found[mid], K, dist, L))
    px = float(np.mean(sides)) if sides else None
    e = np.median(np.array(ests), axis=0) if ests else (None, None)
    return dict(detected=hits, passed=hits >= PASS_MIN, side_px=px, px_per_cell=None if px is None else px / cells,
                est_distance_cm=None if e[0] is None else float(e[0]), est_tilt_deg=None if e[1] is None else float(e[1]),
                fps=N_TEST / (time.time() - t0))


def panel(h, k, n, target, est, found, sz, cells, results, dists, tilts, cam_note, role, mid, width=420):
    p = np.full((h, width, 3), 32, np.uint8)

    def put(y, text, col=WHITE, scale=0.55, th=1):
        cv2.putText(p, text, (14, y), FONT, scale, col, th, cv2.LINE_AA)
    put(30, "DETECTION LIMITS - MATRIX", WHITE, 0.68, 2)
    put(52, f"pose from: {cam_note}", GREY, 0.4)
    if k >= n:
        put(90, "ALL CELLS DONE - press Q", ORANGE, 0.65, 2)
    else:
        d, a = target
        put(88, f"TEST {k + 1}/{n}   marker {role} (id {mid})", ORANGE, 0.6, 2)
        put(114, f"target distance  {d:g} cm   (ruler: lens -> marker)")
        put(138, f"target tilt      {a:g} deg  " + ("(facing the camera)" if a == 0 else "(turn about its vertical axis)"))
        put(176, "MEASURED (guide)", ORANGE, 0.6, 2)
        if not found:
            put(202, "marker not detected", RED, 0.58, 2)
            put(226, "at the limit? hold still and press SPACE", GREY, 0.48)
            put(248, "to test this cell anyway", GREY, 0.48)
        else:
            score, ok_d, ok_a = fit(target, est)
            put(202, f"distance ~{est[0]:.0f} cm")
            cv2.putText(p, "OK" if ok_d else "--", (width - 48, 202), FONT, 0.55, GREEN if ok_d else RED, 2, cv2.LINE_AA)
            put(226, f"tilt     ~{est[1]:.0f} deg")
            cv2.putText(p, "OK" if ok_a else "--", (width - 48, 226), FONT, 0.55, GREEN if ok_a else RED, 2, cv2.LINE_AA)
            put(250, f"size     {sz:.0f} px  = {sz / cells:.1f} px per cell", GREY, 0.5)
            y0, x0, x1 = 272, 14, width - 14
            cv2.rectangle(p, (x0, y0), (x1, y0 + 26), (90, 90, 90), 1)
            full = ok_d and ok_a
            col = RED if full else (int(120 - 60 * score), int(120 + 60 * score), 200)
            cv2.rectangle(p, (x0 + 1, y0 + 1), (x0 + 1 + int((x1 - x0 - 2) * score), y0 + 25), col, -1)
            put(y0 + 50, "IN PLACE - hold still, press SPACE" if full else f"{100 * score:.0f} % - adjust the marker", RED if full else WHITE, 0.56, 2)
    # the matrix: P pass, F fail, s skipped, . to do; the current cell framed
    gy = 360
    put(gy, "MATRIX  (rows: cm, columns: deg)", ORANGE, 0.5, 2)
    cw, x_start = max(30, (width - 90) // max(1, len(tilts))), 74
    for j, a in enumerate(tilts):
        cv2.putText(p, f"{a:g}", (x_start + j * cw, gy + 24), FONT, 0.42, GREY, 1, cv2.LINE_AA)
    for i, d in enumerate(dists):
        y = gy + 48 + i * 24
        cv2.putText(p, f"{d:g}", (14, y), FONT, 0.45, GREY, 1, cv2.LINE_AA)
        for j, a in enumerate(tilts):
            r = results.get((d, a))
            ch, col = (".", GREY) if r is None else (("P", GREEN) if r.get("passed") else ("s", GREY) if r.get("note") else ("F", RED))
            x = x_start + j * cw
            cv2.putText(p, ch, (x + 4, y), FONT, 0.5, col, 2, cv2.LINE_AA)
            if k < n and (d, a) == target:
                cv2.rectangle(p, (x - 2, y - 17), (x + 20, y + 6), ORANGE, 2)
    return p


def write_results(path_csv, path_md, results, dists, tilts, meta):
    rows = []
    for d in dists:
        for a in tilts:
            r = results.get((d, a))
            if r is None:
                continue
            rows.append(dict(target_distance_cm=d, target_tilt_deg=a, marker=meta["role"], marker_id=meta["mid"], frames=N_TEST,
                             detected="" if r.get("note") else r["detected"], passed="" if r.get("note") else r["passed"],
                             side_px="" if r.get("side_px") is None else round(r["side_px"], 1),
                             px_per_cell="" if r.get("px_per_cell") is None else round(r["px_per_cell"], 2),
                             est_distance_cm="" if r.get("est_distance_cm") is None else round(r["est_distance_cm"], 1),
                             est_tilt_deg="" if r.get("est_tilt_deg") is None else round(r["est_tilt_deg"], 1),
                             fps="" if r.get("fps") is None else round(r["fps"], 1), note=r.get("note", ""),
                             resolution=meta["resolution"], dictionary=meta["dictionary"], side_mm=meta["side_mm"], time=r.get("time", "")))
    if rows:
        with open(path_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    L = [f"# Task 2 - detection limits ({meta['dictionary']}, marker {meta['role']} id {meta['mid']}, side {meta['side_mm']} mm, {meta['resolution']})", "",
         f"Cell = frames with a detection out of {N_TEST}; PASS = at least {PASS_MIN}. Distance from the camera lens, tilt about the marker's vertical axis.", "",
         "| distance [cm] | px per cell at 0 deg | " + " | ".join(f"{a:g} deg" for a in tilts) + " | largest tilt PASS |",
         "|---|---|" + "---|" * len(tilts) + "---|"]
    for d in dists:
        cells_txt, best = [], None
        for a in tilts:
            r = results.get((d, a))
            if r is None:
                cells_txt.append("-")
            elif r.get("note"):
                cells_txt.append("not tested")
            else:
                cells_txt.append(f"{r['detected']}/{N_TEST} " + ("PASS" if r["passed"] else "FAIL"))
                if r["passed"]:
                    best = a if best is None else max(best, a)
        r0 = results.get((d, tilts[0])) if tilts else None
        ppc = "-" if not r0 or r0.get("px_per_cell") is None else f"{r0['px_per_cell']:.1f}"
        L.append(f"| {d:g} | {ppc} | " + " | ".join(cells_txt) + f" | {'-' if best is None else f'{best:g} deg'} |")
    zero = [d for d in dists if results.get((d, 0)) and results[(d, 0)].get("passed")]
    L += [""]
    if 0 in tilts:
        if zero and max(zero) == max(dists):
            L.append(f"- **Distance limit (tilt 0):** detected at all tested distances up to {max(dists):g} cm (the largest of the set-up): "
                     "the limit was NOT reached.")
            r = results[(max(dists), 0)]
            if r.get("px_per_cell"):
                L.append(f"  At {max(dists):g} cm the marker has {r['px_per_cell']:.1f} px per cell. With the lecture rule (failure near 2 px per cell) "
                         f"and px per cell ~ 1/distance, the expected limit is about {max(dists) * r['px_per_cell'] / 2 / 100:.1f} m.")
        elif zero:
            L.append(f"- **Distance limit (tilt 0):** {max(zero):g} cm (largest distance with PASS).")
        else:
            L.append("- **Distance limit (tilt 0):** no PASS at tilt 0.")
    for d in dists:
        ok = [a for a in tilts if results.get((d, a)) and results[(d, a)].get("passed")]
        if ok:
            L.append(f"- **Tilt limit at {d:g} cm:** {max(ok):g} deg (largest tilt with PASS).")
    fps = [r["fps"] for r in results.values() if r.get("fps")]
    if fps:
        L.append(f"- Frame rate during the tests: {np.median(fps):.0f} fps (median).")
    open(path_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    return "\n".join(L)


def run_matrix(source, detector, cfg, size, out_dir, dists=DEFAULT_DISTANCES, tilts=DEFAULT_TILTS, role="M", skip_after_fail=True,
               window="Task 2 - detection limits", auto=False, keep_window=False):
    """The guided matrix.  auto=True runs every cell without waiting for keys (tests)."""
    mid = cfg["id_moving"] if role == "M" else cfg["id_fixed"]
    cells = cv2.aruco.getPredefinedDictionary(cfg["dict_id"]).markerSize + 2
    K, dist, cam_note = guide_camera(size)
    L = cfg["L"]
    tests = [(d, a) for d in dists for a in tilts]
    results, k = {}, 0
    meta = dict(role=role, mid=mid, dictionary=cfg["dictionary"], side_mm=round(L * 1000, 1), resolution=f"{size[0]}x{size[1]}")
    path_csv, path_md = os.path.join(out_dir, "detection_matrix.csv"), os.path.join(out_dir, "detection_matrix.md")

    def instruction(i):
        """(terminal text, short text for the window) of test i; says only what changes from the previous test."""
        d, a = tests[i]
        turn = "facing the camera straight on (tilt 0)" if a == 0 else f"turned {a:g} deg about its vertical axis (protractor)"
        same_d = i > 0 and tests[i - 1][0] == d
        if same_d:
            text = f"Keep marker {role} at {d:g} cm. Now " + ("turn it back to face the camera straight on (tilt 0)." if a == 0
                                                             else f"turn it to {a:g} deg about its vertical axis (protractor).")
        else:
            text = f"Move marker {role} (id {mid}) to {d:g} cm from the camera lens (ruler), {turn}."
        short = f"Test {i + 1}/{len(tests)}: {d:g} cm, " + ("straight on" if a == 0 else f"turn {a:g} deg") + " - then SPACE"
        return f"Test {i + 1}/{len(tests)}: {text}\n           Fine-tune with the panel (FIT bar red), hold still and press SPACE.", short

    def say(i):
        if i < len(tests):
            print("\n" + instruction(i)[0])
        else:
            print("\nAll tests done. Press Q in the window to finish.")

    print(f"\nExperiment plan: {len(dists)} distances x {len(tilts)} tilts = {len(tests)} tests, marker {role} (id {mid}); "
          f"pose shown from the {cam_note}.")
    for d in dists:
        print(f"   at {d:g} cm: tilt " + ", ".join(f"{a:g}" for a in tilts) + " deg")
    print("For each test: put the marker where the instruction says, press SPACE, 10 frames are tested (PASS = 9 or more).\n"
          "Keys (in the window): SPACE test, N skip, U back one test, Q stop (results so far are kept).")
    say(k)
    if not auto:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    while True:
        if auto and hasattr(source, "set_target") and k < len(tests):
            source.set_target(*tests[k])
        frame = source.read()
        if frame is None:
            continue
        disp = frame.copy()
        found = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        est, sz = None, 0.0
        if mid in found:
            c = found[mid]
            cv2.polylines(disp, [c.astype(np.int32)], True, GREEN, 2, cv2.LINE_AA)
            est, sz = view_estimate(c, K, dist, L), side_px(c)
        for other, cc in found.items():
            if other != mid:
                cv2.polylines(disp, [cc.astype(np.int32)], True, GREY, 1, cv2.LINE_AA)
        target = tests[k] if k < len(tests) else tests[-1]
        h_, w_ = disp.shape[:2]                                              # the instruction at the bottom of the video
        cv2.rectangle(disp, (0, h_ - 36), (w_, h_), (30, 30, 30), -1)
        cv2.putText(disp, instruction(k)[1] if k < len(tests) else "All tests done - press Q", (10, h_ - 12), FONT, 0.7,
                    (26, 180, 255), 2, cv2.LINE_AA)
        if auto:
            key = ord(" ") if k < len(tests) else ord("q")
        else:
            cv2.imshow(window, np.hstack([disp, panel(disp.shape[0], k, len(tests), target, est, est is not None, sz, cells, results,
                                                      dists, tilts, cam_note, role, mid)]))
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                key = ord("q")
        if key in (ord("q"), ord("Q"), 27):
            break
        if key == ord(" ") and k < len(tests):
            d, a = tests[k]
            r = run_cell(source, detector, mid, K, dist, L, cells)
            r["time"] = datetime.now().isoformat(timespec="seconds")
            results[(d, a)] = r
            print(f"   -> detected {r['detected']}/{N_TEST}  {'PASS' if r['passed'] else 'FAIL'}"
                  + ("" if r["side_px"] is None else f"   {r['px_per_cell']:.1f} px/cell, measured ~{r['est_distance_cm']:.0f} cm, ~{r['est_tilt_deg']:.0f} deg"))
            k += 1
            if r["detected"] <= CLEAR_FAIL and skip_after_fail:      # clear FAIL: larger tilts at this distance not tested
                while k < len(tests) and tests[k][0] == d:              # (a borderline 6-8/10 is noisy: keep testing)
                    results[tests[k]] = dict(note="not tested (larger tilt after a clear FAIL)")
                    print(f"   skipped {tests[k][1]:g} deg at {d:g} cm (after a clear FAIL, <= {CLEAR_FAIL}/10)")
                    k += 1
            write_results(path_csv, path_md, results, dists, tilts, meta)
            say(k)
        elif key in (ord("n"), ord("N")) and k < len(tests):
            results[tests[k]] = dict(note="skipped by the user")
            k += 1
            say(k)
        elif key in (ord("u"), ord("U")) and k > 0:
            k -= 1
            while k > 0 and results.get(tests[k], {}).get("note", "").startswith("not tested"):
                results.pop(tests[k], None)
                k -= 1
            results.pop(tests[k], None)
            print("   back one cell")
            say(k)
    if not auto and not keep_window:
        cv2.destroyWindow(window)
    summary = write_results(path_csv, path_md, results, dists, tilts, meta)
    print("\n" + summary + f"\n\nmatrix -> {path_csv}\nsummary -> {path_md}")
    return results
