"""t7_passes.py -- Bonus: find the passes of M along the ruler in a recording and measure them (classical poses).

  python t7_passes.py E7_nocover_slow [--min-travel 150]

A pass = M rests at the start stop, slides toward F, rests at the end stop.  For every pass: when it starts / ends, the
centre-to-centre distance to F at both rests (compare with the ruler), the travel, the mean speed, how steady the speed is
(spread over the middle 80 % of the pass), how straight the path is, and for how long M was not measured (hidden).
Writes output/bonus/<name>_passes.csv, <name>_track.csv (t, distance to F, position along the path) and <name>_passes.png.
"""
import argparse
import os

import cv2
import numpy as np

from config import OUTPUT_DIR, data_path
from mvtools import MarkerPoseEstimator, load_camera, load_markers_cfg, write_csv


def track(video, cfg, K, dist):
    """t [s], M's position in F [mm] (NaN where M or F is not measured)."""
    t = np.loadtxt(os.path.splitext(video)[0] + ".csv", delimiter=",", skiprows=1)[:, 1]
    est = MarkerPoseEstimator(cfg, K, dist, camera_still=True)
    cap = cv2.VideoCapture(video)
    P = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        r = est.process(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        P.append(1000 * r["T_FM"][:3, 3] if r["T_FM"] is not None else np.full(3, np.nan))
    cap.release()
    n = min(len(t), len(P))
    return t[:n], np.array(P[:n])


def find_passes(t, P, min_travel=150.0, rest_speed=15.0, rest_s=0.5):
    """Passes toward F.  Returns the path axis (origin, unit direction pointing toward F), s(t) along it, and a list of
    (i_start, i_end): the last resting frame before the slide and the first resting frame after it."""
    ok = np.isfinite(P).all(1)
    c = P[ok].mean(0)
    u = np.linalg.svd(P[ok] - c)[2][0]                               # the main direction of motion
    if np.linalg.norm(P[ok][np.argmax((P[ok] - c) @ u)]) > np.linalg.norm(P[ok][np.argmin((P[ok] - c) @ u)]):
        u = -u                                                        # s grows toward F (F is the origin of these coordinates)
    s = (P - c) @ u
    si = np.interp(t, t[ok], s[ok])                                   # bridged over hidden frames, for the rest detection only
    k = max(3, int(round(0.2 / np.median(np.diff(t)))) | 1)
    sm = np.convolve(np.pad(si, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    v = np.gradient(sm, t)
    rest = np.abs(v) < rest_speed
    runs, i = [], 0                                                   # resting runs of at least rest_s
    while i < len(t):
        if rest[i]:
            j = i
            while j + 1 < len(t) and rest[j + 1]:
                j += 1
            if t[j] - t[i] >= rest_s:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    passes = []
    for (a0, a1), (b0, b1) in zip(runs[:-1], runs[1:]):
        if np.median(sm[b0:b1 + 1]) - np.median(sm[a0:a1 + 1]) >= min_travel:      # moved toward F by at least min_travel
            passes.append((a1, b0, (a0, a1), (b0, b1)))
    return c, u, s, passes


def main():
    ap = argparse.ArgumentParser(description="Bonus: measure the passes of M along the ruler.")
    ap.add_argument("name", help="recording name in data/recordings (without extension)")
    ap.add_argument("--min-travel", type=float, default=150.0, help="smallest travel that counts as a pass [mm]")
    a = ap.parse_args()
    cfg = load_markers_cfg()
    K, dist, _ = load_camera()
    t, P = track(data_path("recordings", a.name + ".avi"), cfg, K, dist)
    c, u, s, passes = find_passes(t, P, a.min_travel)
    D = np.linalg.norm(P, axis=1)                                     # centre-to-centre distance M - F
    out = os.path.join(OUTPUT_DIR, "bonus")
    os.makedirs(out, exist_ok=True)
    rows = []
    print(f"{a.name}: {len(t)} frames, {t[-1]:.1f} s, M measured in {100 * np.isfinite(D).mean():.0f} % of the frames; {len(passes)} passes toward F\n")
    print("pass  start   end   | centre-centre at start -> end | travel  duration  speed (spread)     | off the line | hidden")
    for n, (i0, i1, ra, rb) in enumerate(passes, 1):
        d0, d1 = np.nanmedian(D[ra[0]:ra[1] + 1]), np.nanmedian(D[rb[0]:rb[1] + 1])
        p0, p1 = np.nanmedian(P[ra[0]:ra[1] + 1], 0), np.nanmedian(P[rb[0]:rb[1] + 1], 0)
        travel = float(np.linalg.norm(p1 - p0))
        ul = (p1 - p0) / travel                                       # this pass's own line: rest position to rest position
        seg = slice(i0, i1 + 1)
        ok = np.isfinite(P[seg]).all(1)
        sl = (P[seg] - p0) @ ul
        mid = ok & (sl > 0.1 * travel) & (sl < 0.9 * travel)
        tm, sm_ = t[seg][mid], sl[mid]
        speed = float(np.polyfit(tm, sm_, 1)[0]) if mid.sum() > 3 else np.nan           # mm/s over the middle 80 %
        vv = np.gradient(sm_, tm) if mid.sum() > 5 else np.array([np.nan])
        lat = np.linalg.norm((P[seg][ok] - p0) - np.outer(sl[ok], ul), axis=1)
        hidden = float(np.sum(np.diff(t[seg])[~ok[1:]])) if (~ok).any() else 0.0
        rows.append(dict(**{"pass": n}, t_start_s=round(float(t[i0]), 3), t_end_s=round(float(t[i1]), 3), centre_start_mm=round(float(d0), 1),
                         centre_end_mm=round(float(d1), 1), travel_mm=round(travel, 1), duration_s=round(float(t[i1] - t[i0]), 2),
                         speed_mm_s=round(speed, 1), speed_std_mm_s=round(float(np.nanstd(vv)), 1), off_line_rms_mm=round(float(np.sqrt(np.mean(lat ** 2))), 1),
                         off_line_max_mm=round(float(lat.max()), 1), hidden_s=round(hidden, 2),
                         start_x_mm=round(float(p0[0]), 1), start_y_mm=round(float(p0[1]), 1), start_z_mm=round(float(p0[2]), 1),
                         end_x_mm=round(float(p1[0]), 1), end_y_mm=round(float(p1[1]), 1), end_z_mm=round(float(p1[2]), 1)))
        r = rows[-1]
        print(f"{n:3d}  {r['t_start_s']:6.2f} {r['t_end_s']:6.2f} |      {d0:6.1f} -> {d1:6.1f} mm       | {travel:5.0f} mm  {r['duration_s']:5.2f} s  "
              f"{speed:5.0f} mm/s (+-{r['speed_std_mm_s']:.0f}) | rms {r['off_line_rms_mm']:.1f} mm  | {hidden:.2f} s")
    if rows:
        write_csv(os.path.join(out, a.name + "_passes.csv"), rows)
        sp = np.array([r["speed_mm_s"] for r in rows])
        print(f"\nmean of the passes: speed {np.nanmean(sp):.0f} mm/s (pass to pass +-{np.nanstd(sp):.0f}), travel {np.mean([r['travel_mm'] for r in rows]):.0f} mm, "
              f"centre-centre {np.mean([r['centre_start_mm'] for r in rows]):.1f} -> {np.mean([r['centre_end_mm'] for r in rows]):.1f} mm")
    write_csv(os.path.join(out, a.name + "_track.csv"),
              [dict(t_s=round(float(a_), 4), centre_mm=("" if not np.isfinite(b) else round(float(b), 2)), along_mm=("" if not np.isfinite(c_) else round(float(c_), 2)))
               for a_, b, c_ in zip(t, D, s)])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, D, ".", ms=2, color="#1F3A5F")
    for n, (i0, i1, ra, rb) in enumerate(passes, 1):
        ax.axvspan(t[i0], t[i1], color="#D9731A", alpha=0.2)
        ax.text((t[i0] + t[i1]) / 2, np.nanmax(D), str(n), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("time [s]"); ax.set_ylabel("centre-to-centre distance M - F [mm]"); ax.grid(alpha=0.3)
    ax.set_title(f"{a.name}: passes toward F (orange); gaps = M not measured")
    fig.tight_layout(); fig.savefig(os.path.join(out, a.name + "_passes.png"), dpi=130)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
