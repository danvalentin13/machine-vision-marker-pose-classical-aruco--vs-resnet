"""t2_guided.py -- Task 2: the GUIDED live demo (t2_detect_live.py, mode 'guided', or key G in the live view).

The camera runs and detects all the time; NOTHING is saved automatically.  A step list on the right says what to show and
when to take a screenshot (S) or a recording (R = start, R again = stop).
  1  both markers   F and M in view together, still                     -> screenshot (S)
  2  M moving       F still, M moved slowly (closer, farther, sideways)  -> recording (R ... R), at least 5 s
  3  M tilted       M turned about 45 deg (>= 30 measured), F in view    -> screenshot (S)
  4  M far          M at about 1 m (>= 80 cm measured), F in view        -> screenshot (S)
* Resumes: step files saved in an earlier session count as done; it continues at the first missing step.
* R records the "M moving" step whatever step is shown, as long as that step is not done yet.
* A screenshot is checked (both ids; the tilt of step 3; the distance of step 4); if it does not fit you are told and the step stays open.
* The report is built from the files on disk:   python t2_guided.py --report     rebuilds it without the camera.
Keys (click the window first):  S screenshot   R start / stop recording   N next step   B previous step   Q / Esc finish
Writes  <folder>/guided/step1_both.png, step2_moving.avi, step3_tilted.png, step4_far.png, guided_report.md
"""
import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, ORANGE, GREEN, RED = (235, 235, 235), (150, 150, 150), (26, 160, 255), (60, 200, 60), (60, 60, 235)
MIN_REC_S, MIN_TILT, MIN_FAR_CM = 5.0, 30.0, 80.0
F_STILL_PX, F_REF_PX = 40.0, 60.0          # F may shift at most this much [px] during the recording / since step 1
M_MOVE_CM, M_MOVE_PX = 10.0, 80.0          # M must move at least this much (closer/farther [cm] or sideways [px])


def steps(cfg):
    f, m = cfg["id_fixed"], cfg["id_moving"]
    return [
        dict(name="both", action="S", title="Both markers",
             text=f"Hold F (id {f}) and M (id {m}) in view together, about 40-60 cm from the camera, still. "
                  "When both have a green outline, press S to take the screenshot."),
        dict(name="moving", action="R", title="M moving",
             text=f"Keep F (id {f}) still. Press R to START recording, then move M (id {m}) slowly - closer, farther, sideways - "
                  f"for about 10 s (at least {MIN_REC_S:g} s), then press R to STOP."),
        dict(name="tilted", action="S", title="M tilted ~45 deg",
             text=f"Turn M (id {m}) about 45 deg about its vertical axis (the panel shows its tilt, at least {MIN_TILT:g}), "
                  "F stays in view. Take your time. When both are green and still, press S."),
        dict(name="far", action="S", title="M far (~1 m)",
             text=f"Move M (id {m}) to about 1 m from the camera (the panel shows its distance, at least {MIN_FAR_CM:g} cm), "
                  "F stays in view, M facing the camera. When both are green and still, press S."),
    ]


def step_file(gdir, i, st):
    return os.path.join(gdir, f"step{i + 1}_{st['name']}.{'png' if st['action'] == 'S' else 'avi'}")


def wrap(text, width_chars):
    lines, cur = [], ""
    for w in text.split():
        if len(cur) + len(w) + 1 > width_chars:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + ([cur] if cur else [])


# ----------------------------------------------------------------------------------------------
# checking a step and the report (from the files on disk)
# ----------------------------------------------------------------------------------------------
def check_shot(name, found, ours, est):
    """Problems of a screenshot for step `name` (empty list = fits)."""
    probs = [f"id {i} not detected" for i in sorted(ours) if i not in found]
    if set(found) - ours:
        probs.append(f"foreign ids {sorted(set(found) - ours)}")
    if name == "tilted" and (est is None or est[1] < MIN_TILT):
        probs.append(f"M tilt {'?' if est is None else f'~{est[1]:.0f}'} deg, at least {MIN_TILT:g} wanted")
    if name == "far" and (est is None or est[0] < MIN_FAR_CM):
        probs.append(f"M distance {'?' if est is None else f'~{est[0]:.0f}'} cm, at least {MIN_FAR_CM:g} wanted")
    return probs


def centre(corners):
    return np.asarray(corners, float).reshape(-1, 2).mean(axis=0)


def wrong_marker_checks(name, found, cfg, K, dist, est_m, f_ref):
    """Safeguards for the screenshot steps: was F (the fixed marker) moved / turned instead of M?"""
    from t2_matrix import view_estimate
    probs, idf = [], cfg["id_fixed"]
    if idf in found:
        if f_ref is not None and np.linalg.norm(centre(found[idf]) - f_ref) > F_REF_PX:
            probs.append(f"F (id {idf}) has moved since step 1 by {np.linalg.norm(centre(found[idf]) - f_ref):.0f} px - "
                         "F is the FIXED marker: put it back and move only M")
        est_f = view_estimate(found[idf], K, dist, cfg["L"])
        if name == "tilted" and (est_m is None or est_m[1] < MIN_TILT) and est_f[1] >= MIN_TILT:
            probs.append(f"it looks like F was turned (F tilt ~{est_f[1]:.0f} deg), not M - turn M (id {cfg['id_moving']}) instead")
        if name == "far" and (est_m is None or est_m[0] < MIN_FAR_CM) and est_f[0] >= MIN_FAR_CM:
            probs.append(f"it looks like F was moved away (F ~{est_f[0]:.0f} cm), not M - move M (id {cfg['id_moving']}) instead")
    return probs


def analyse(path, st, cfg, det, K, dist):
    """Re-detect the markers in a step file: the facts the report states."""
    from t2_matrix import view_estimate
    ours, mid = {cfg["id_fixed"], cfg["id_moving"]}, cfg["id_moving"]
    if st["action"] == "S":
        found = det.detect(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        est = view_estimate(found[mid], K, dist, cfg["L"]) if mid in found else None
        return dict(kind="S", ids=sorted(set(found) & ours), problems=check_shot(st["name"], found, ours, est), est=est)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    hits, foreign, n, dists = {i: 0 for i in ours}, set(), 0, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        found = det.detect(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        for i in ours:
            hits[i] += i in found
        foreign |= set(found) - ours
        if mid in found and n % 5 == 0:
            dists.append(view_estimate(found[mid], K, dist, cfg["L"])[0])
    cap.release()
    rates = {i: 100.0 * hits[i] / max(n, 1) for i in ours}
    dur = n / fps
    probs = ([f"only {dur:.1f} s, at least {MIN_REC_S:g} wanted"] if dur < MIN_REC_S else []) + \
            [f"id {i} detected in only {rates[i]:.0f} % of the frames" for i in sorted(ours) if rates[i] < 90] + \
            ([f"foreign ids {sorted(foreign)}"] if foreign else [])
    return dict(kind="R", frames=n, duration=dur, rates=rates, foreign=sorted(foreign), problems=probs,
                dist_range=(min(dists), max(dists)) if dists else None)


def write_report(gdir, cfg, size, fps_med=None, log=print):
    from mvlib import MarkerDetector
    from t2_matrix import guide_camera
    det = MarkerDetector(cfg["dict_id"], subpixel=True)
    K, dist, _ = guide_camera(size)
    plan = steps(cfg)
    L = [f"# Task 2 - guided live demo ({datetime.now():%Y-%m-%d %H:%M})", "",
         f"Dictionary {cfg['dictionary']}, F = id {cfg['id_fixed']}, M = id {cfg['id_moving']}, side {cfg['L'] * 1000:g} mm, "
         f"{size[0]}x{size[1]}, sub-pixel corner refinement on." + ("" if not fps_med else f" Frame rate: **{fps_med:.0f} fps** (median, live)."),
         "Every row is re-checked from the saved file.", "", "| step | file | result |", "|---|---|---|"]
    all_ok, results = True, {}
    for i, st in enumerate(plan):
        p = step_file(gdir, i, st)
        if not os.path.exists(p):
            L.append(f"| {i + 1}. {st['title']} | - | not done |")
            all_ok = False
            continue
        r = analyse(p, st, cfg, det, K, dist)
        results[st["name"]] = r
        all_ok &= not r["problems"]
        if r["kind"] == "S":
            txt = "both ids read correctly" if not [x for x in r["problems"] if "id" in x] else "; ".join(r["problems"])
            if r["est"] is not None:
                txt += f"; M ~{r['est'][0]:.0f} cm, tilt ~{r['est'][1]:.0f} deg"
            if r["problems"] and "both ids read correctly" in txt:
                txt += " - " + "; ".join(r["problems"])
        else:
            txt = (f"{r['duration']:.1f} s, {r['frames']} frames: F detected in {r['rates'][cfg['id_fixed']]:.0f} %, "
                   f"M in {r['rates'][cfg['id_moving']]:.0f} % of the frames"
                   + ("" if r["dist_range"] is None else f"; M moved {r['dist_range'][0]:.0f}-{r['dist_range'][1]:.0f} cm")
                   + ("" if not r["foreign"] else f"; foreign ids {r['foreign']}")
                   + ("" if not r["problems"] else " - " + "; ".join(r["problems"])))
        L.append(f"| {i + 1}. {st['title']} | [{os.path.basename(p)}]({os.path.basename(p)}) | {txt} |")
    L += ["", f"- Both ids read correctly in every step: **{'yes' if all_ok else 'not yet - see the table'}**."]
    path = os.path.join(gdir, "guided_report.md")
    os.makedirs(gdir, exist_ok=True)
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    log("\n" + "\n".join(L[5:]) + f"\n\nreport -> {path}")
    return results, all_ok


# ----------------------------------------------------------------------------------------------
# the live demo
# ----------------------------------------------------------------------------------------------
def panel(h, plan, k, done, found_now, roles, est, fps, rec, cfg=None, width=430):
    p = np.full((h, width, 3), 32, np.uint8)

    def put(y, text, col=WHITE, scale=0.52, th=1, x=14):
        cv2.putText(p, text, (x, y), FONT, scale, col, th, cv2.LINE_AA)
    put(30, "GUIDED LIVE DEMO (Task 2)", WHITE, 0.66, 2)
    put(52, "nothing is saved until you press S or R", GREY, 0.42)
    y = 84
    for i, st in enumerate(plan):
        mark, col = ("[x]", GREEN) if done.get(st["name"]) else (("[>]", ORANGE) if i == k else ("[ ]", GREY))
        what = "screenshot S" if st["action"] == "S" else "recording R ... R"
        put(y, f"{mark} {i + 1}. {st['title']}  -  {what}", col, 0.5, 2 if i == k else 1)
        y += 24
    y += 12
    if k < len(plan):
        put(y, f"STEP {k + 1}/{len(plan)}", ORANGE, 0.6, 2)
        y += 24
        for line in wrap(plan[k]["text"], 46):
            put(y, line, WHITE, 0.5)
            y += 21
    else:
        put(y, "ALL STEPS DONE", GREEN, 0.62, 2)
        y += 24
        put(y, "press Q to finish and write the report", WHITE, 0.5)
        y += 21
    y += 14
    put(y, "NOW", ORANGE, 0.6, 2)
    y += 24
    for mid, role in roles.items():
        ok = mid in found_now
        put(y, f"{role} (id {mid}): {'detected' if ok else 'not detected'}", GREEN if ok else RED, 0.52, 2 if ok else 1)
        y += 22
    if est is not None:
        put(y, f"M: ~{est[0]:.0f} cm, tilt ~{est[1]:.0f} deg", WHITE, 0.5)
        y += 22
    put(y, f"{fps:.0f} fps", GREY, 0.5)
    if rec is not None and rec.get("n"):
        y += 26
        still = rec["f_shift"] <= F_STILL_PX
        put(y, f"F: {'still' if still else 'MOVING - keep F still!'}  (shift {rec['f_shift']:.0f} px)", GREEN if still else RED, 0.52, 2)
        y += 22
        m_cm = (max(rec["m_d"]) - min(rec["m_d"])) if rec["m_d"] else 0.0
        put(y, f"M moved: {m_cm:.0f} cm closer/farther" + ("  - move M more" if m_cm < M_MOVE_CM else ""), WHITE if m_cm >= M_MOVE_CM else ORANGE, 0.5)
    if rec is not None:
        cv2.circle(p, (width - 90, h - 40), 10, RED, -1)
        cv2.putText(p, f"REC {time.time() - rec['t0']:.0f} s", (width - 72, h - 33), FONT, 0.6, RED, 2, cv2.LINE_AA)
    put(h - 14, "S screenshot  R rec  N next  B back  Q finish", GREY, 0.45)
    return p


def checkpoint(k, n, title, ok, detail=""):
    """The CHECKPOINT line in the terminal after a step."""
    bar = "=" * 78
    if ok:
        print(f"\n{bar}\n  CHECKPOINT  step {k + 1}/{n}  {title.upper()}:  DONE - OK   {detail}\n{bar}")
    else:
        print(f"\n  CHECKPOINT  step {k + 1}/{n}  {title.upper()}:  NOT OK  -  {detail}")


def draw_banner(img, text, color):
    """Large message in the middle of the video (only on screen, never in the saved files)."""
    h, w = img.shape[:2]
    scale, th = 1.6, 4
    (tw, tht), _ = cv2.getTextSize(text, FONT, scale, th)
    x0, y0 = (w - tw) // 2 - 30, h // 2 - tht - 30
    cv2.rectangle(img, (x0, y0), (x0 + tw + 60, y0 + tht + 60), (20, 20, 20), -1)
    cv2.rectangle(img, (x0, y0), (x0 + tw + 60, y0 + tht + 60), color, 4)
    cv2.putText(img, text, (x0 + 30, y0 + tht + 30), FONT, scale, color, th, cv2.LINE_AA)
    return img


def run_guided(source, detector, cfg, roles, size, out_dir, window, keep_window=True, auto_keys=None, fresh=False):
    """auto_keys: list of keys (one per frame, None = no key) to drive the demo without a person (tests).
    fresh: delete the step files of an earlier session first (start over)."""
    from t2_detect_live import annotate
    from t2_matrix import guide_camera, view_estimate
    gdir = os.path.join(out_dir, "guided")
    os.makedirs(gdir, exist_ok=True)
    K, dist, _ = guide_camera(size)
    ours, mid_m = set(roles), cfg["id_moving"]
    plan = steps(cfg)
    if fresh:
        for i, st in enumerate(plan):
            if os.path.exists(step_file(gdir, i, st)):
                os.remove(step_file(gdir, i, st))
        print("   earlier step files deleted: starting fresh")
    done = {st["name"]: os.path.exists(step_file(gdir, i, st)) for i, st in enumerate(plan)}     # resume
    ok_steps = {}                                                                                 # steps that passed their check
    f_ref = None                                                                                  # F's image position at step 1
    banner = None                                                                                 # (text, colour, until)
    k = next((i for i, st in enumerate(plan) if not done[st["name"]]), len(plan))
    fps, last, fps_all, rec = 0.0, time.time(), [], None
    auto = iter(auto_keys) if auto_keys is not None else None

    def say(i):
        if i < len(plan):
            print(f"\nStep {i + 1}/{len(plan)} - {plan[i]['title'].upper()}: {plan[i]['text']}")
            return
        missing = [f"{j + 1} ({st['title']})" for j, st in enumerate(plan) if not done[st["name"]]]
        print("\nAll steps done. Press Q in the window to finish and write the report (or keep looking)." if not missing else
              f"\nEnd of the list, but not done yet: step {', '.join(missing)}. Press B to go back to it, or Q to finish anyway.")

    def next_open(i):
        return next((j for j in range(i, len(plan)) if not done[plan[j]["name"]]),
                    next((j for j in range(len(plan)) if not done[plan[j]["name"]]), len(plan)))

    print("\n--- GUIDED LIVE DEMO: the camera runs and detects; nothing is saved until you press S (screenshot) or R (recording). ---")
    print("Keys (in the window): S screenshot, R start/stop recording, N next step, B previous step, Q finish.")
    if any(done.values()):
        print("Found from an earlier session (kept): " + ", ".join(f"step {i + 1} {st['title']}" for i, st in enumerate(plan) if done[st["name"]])
              + ".  To redo one, go to it with B / N and save it again.")
    say(k)
    if auto is None:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    while True:
        frame = source.read()
        if frame is None:
            continue
        now = time.time()
        inst = 1.0 / max(now - last, 1e-6)
        last = now
        fps = inst if not fps else 0.9 * fps + 0.1 * inst
        fps_all.append(inst)
        found = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        shown = annotate(frame, found, roles, fps, status="REC" if rec else "")
        est = view_estimate(found[mid_m], K, dist, cfg["L"]) if mid_m in found else None
        if rec is not None:
            rec["writer"].write(shown)
            rec["n"] += 1
            for i in ours:
                rec["hits"][i] += i in found
            rec["foreign"] |= set(found) - ours
            if cfg["id_fixed"] in found:
                c = centre(found[cfg["id_fixed"]])
                rec["f0"] = c if rec["f0"] is None else rec["f0"]
                rec["f_shift"] = max(rec["f_shift"], float(np.linalg.norm(c - rec["f0"])))
            if mid_m in found:
                rec["m_c"].append(centre(found[mid_m]))
                if est is not None:
                    rec["m_d"].append(est[0])
        disp = shown.copy()
        if banner is not None and time.time() < banner[2]:
            draw_banner(disp, banner[0], banner[1])
        if auto is None:
            cv2.imshow(window, np.hstack([disp, panel(shown.shape[0], plan, k, done, found, roles, est, fps, rec, cfg)]))
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                key = ord("q")
        else:
            key = next(auto, "q")
            key = 255 if key is None else ord(key)
        if key in (ord("q"), ord("Q"), 27):
            break
        step = plan[k] if k < len(plan) else None
        if key in (ord("s"), ord("S")):
            if step is not None and step["action"] == "S":
                path = step_file(gdir, k, step)
                cv2.imwrite(path, shown)
                probs = check_shot(step["name"], found, ours, est)
                if not [x for x in probs if "not detected" in x]:
                    probs += wrong_marker_checks(step["name"], found, cfg, K, dist, est, None if step["name"] == "both" else f_ref)
                if step["name"] == "both" and cfg["id_fixed"] in found and not probs:
                    f_ref = centre(found[cfg["id_fixed"]])             # where F has to stay from now on
                done[step["name"]] = True
                print(f"   screenshot -> {path}")
                if probs:
                    ok_steps[step["name"]] = False
                    checkpoint(k, len(plan), step["title"], False, "; ".join(probs) + ".  Press S again (it replaces it), or N to keep it.")
                    banner = (f"STEP {k + 1} NOT OK - see the terminal", RED, time.time() + 3)
                else:
                    ok_steps[step["name"]] = True
                    checkpoint(k, len(plan), step["title"], True, "both ids read" + ("" if est is None else f", M ~{est[0]:.0f} cm, tilt ~{est[1]:.0f} deg"))
                    banner = (f"STEP {k + 1} OK", GREEN, time.time() + 2.5)
                    k = next_open(k + 1)
                    if k >= len(plan) and all(ok_steps.get(st["name"]) for st in plan):
                        banner = ("ALL STEPS OK - press Q", GREEN, time.time() + 1e9)
                    say(k)
            else:
                path = os.path.join(gdir, f"extra_{datetime.now():%H%M%S}.png")
                cv2.imwrite(path, shown)
                print(f"   extra screenshot -> {path}" + ("" if step is None else
                      f"\n   (this step needs a RECORDING: press R to start, move M, press R to stop)"))
        elif key in (ord("r"), ord("R")):
            if rec is None:
                i_rec = next((i for i, st in enumerate(plan) if st["action"] == "R" and (not done[st["name"]] or i == k)), None)
                path = step_file(gdir, i_rec, plan[i_rec]) if i_rec is not None else os.path.join(gdir, f"extra_{datetime.now():%H%M%S}.avi")
                rec = dict(path=path, i=i_rec, t0=time.time(), n=0, hits={i: 0 for i in ours}, foreign=set(), f0=None, f_shift=0.0, m_c=[], m_d=[],
                           writer=cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (shown.shape[1], shown.shape[0])))
                banner = ("REC", RED, time.time() + 1.0)
                print(f"   recording" + (f" for step {i_rec + 1} ({plan[i_rec]['title']})" if i_rec is not None else " (extra)")
                      + f" ... move M slowly, press R to stop -> {path}")
            else:
                rec["writer"].release()
                dur, n = time.time() - rec["t0"], max(rec["n"], 1)
                rates = {i: 100.0 * rec["hits"][i] / n for i in ours}
                print(f"   recording stopped: {dur:.1f} s, {rec['n']} frames -> {rec['path']}")
                if rec["i"] is not None:
                    st, i_rec = plan[rec["i"]], rec["i"]
                    done[st["name"]] = True
                    probs = ([f"only {dur:.1f} s (at least {MIN_REC_S:g})"] if dur < MIN_REC_S else []) + \
                            [f"{roles[i]} (id {i}) detected in only {rates[i]:.0f} % of the frames (at least 90)" for i in sorted(ours) if rates[i] < 90] + \
                            ([f"foreign ids {sorted(rec['foreign'])}"] if rec["foreign"] else [])
                    m_c = np.array(rec["m_c"]) if rec["m_c"] else np.zeros((0, 2))
                    m_px = float(np.max(np.linalg.norm(m_c - m_c.mean(axis=0), axis=1)) * 2) if len(m_c) else 0.0
                    m_cm = float(max(rec["m_d"]) - min(rec["m_d"])) if rec["m_d"] else 0.0
                    m_moved = m_cm >= M_MOVE_CM or m_px >= M_MOVE_PX
                    if rec["f_shift"] > F_STILL_PX and not m_moved:
                        probs.append(f"you moved F (id {cfg['id_fixed']}) instead of M (id {cfg['id_moving']}): F shifted {rec['f_shift']:.0f} px, M hardly moved")
                    else:
                        if rec["f_shift"] > F_STILL_PX:
                            probs.append(f"F (id {cfg['id_fixed']}) moved during the recording ({rec['f_shift']:.0f} px, at most {F_STILL_PX:g}): keep F still, move only M")
                        if not m_moved:
                            probs.append(f"M (id {cfg['id_moving']}) hardly moved ({m_cm:.0f} cm closer/farther, {m_px:.0f} px sideways): move M more")
                    if f_ref is not None and rec["f0"] is not None and np.linalg.norm(rec["f0"] - f_ref) > F_REF_PX:
                        probs.append(f"F has moved since step 1 ({np.linalg.norm(rec['f0'] - f_ref):.0f} px): put F back where it was")
                    detail = (f"{dur:.0f} s, F detected in {rates[cfg['id_fixed']]:.0f} % and still (max shift {rec['f_shift']:.0f} px), "
                              f"M in {rates[cfg['id_moving']]:.0f} % and moved {m_cm:.0f} cm / {m_px:.0f} px")
                    if probs:
                        ok_steps[st["name"]] = False
                        checkpoint(i_rec, len(plan), st["title"], False, "; ".join(probs) + ".  Press R to record it again (it replaces it), or N to keep it.")
                        banner = (f"STEP {i_rec + 1} NOT OK - see the terminal", RED, time.time() + 3)
                    else:
                        ok_steps[st["name"]] = True
                        checkpoint(i_rec, len(plan), st["title"], True, detail)
                        banner = (f"STEP {i_rec + 1} OK", GREEN, time.time() + 2.5)
                        if k == i_rec:
                            k = next_open(k + 1)
                            if k >= len(plan) and all(ok_steps.get(x["name"]) for x in plan):
                                banner = ("ALL STEPS OK - press Q", GREEN, time.time() + 1e9)
                            say(k)
                        else:
                            print(f"   (still at step {k + 1})")
                rec = None
        elif key in (ord("n"), ord("N")) and k < len(plan):
            k += 1
            say(k)
        elif key in (ord("b"), ord("B")) and k > 0:
            k -= 1
            say(k)
    if rec is not None:
        rec["writer"].release()
    fps_med = float(np.median(fps_all[10:])) if len(fps_all) > 20 else (float(np.median(fps_all)) if fps_all else None)
    print("\nchecking the saved files ...")
    write_report(gdir, cfg, size, fps_med)
    if auto is None and not keep_window:
        cv2.destroyWindow(window)
    return done, fps_med


def main():
    ap = argparse.ArgumentParser(description="Rebuild the report of the guided live demo from the saved files.")
    ap.add_argument("--report", action="store_true", help="rebuild output/detection/guided/guided_report.md (no camera)")
    ap.add_argument("--folder", help="the detection output folder (default output/detection)")
    a = ap.parse_args()
    from config import OUTPUT_DIR
    from mvtools import load_markers_cfg
    gdir = os.path.join(a.folder or os.path.join(OUTPUT_DIR, "detection"), "guided")
    if not a.report:
        print("Run the demo with:  python t2_detect_live.py --mode guided      (rebuild the report:  python t2_guided.py --report)")
        return
    write_report(gdir, load_markers_cfg(), (1280, 720))


if __name__ == "__main__":
    main()
