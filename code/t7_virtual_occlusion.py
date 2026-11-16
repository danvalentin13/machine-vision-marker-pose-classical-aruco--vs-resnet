"""t7_virtual_occlusion.py -- Bonus: test the occlusion predictor on passes recorded WITHOUT an obstacle, hidden in software.

  python t7_virtual_occlusion.py E7_nocover_slow_2 E7_nocover_fast [--hide 78 235]

In these recordings the camera sees M all the way, so the true pose is known in every frame (classical measurement).
A virtual obstacle hides M while its left corner is between the two ruler readings --hide (measured from the start stop;
default 78-235 mm = an 8 cm obstacle covering 155-235 mm, M 77 mm wide).  The predictor (t7_bonus_occlusion.OcclusionPredictor:
constant-velocity Kalman + alpha-beta on SO(3)) gets no measurement there; its output is compared with the measured pose and with
the baseline 'hold the last pose'.  This truth does not depend on the hand moving at constant speed.
Writes output/bonus/virtual_occlusion.csv (one row per pass), virtual_occlusion.md and virtual_occlusion.png.
"""
import argparse
import os

import cv2
import numpy as np

from config import OUTPUT_DIR, data_path
from mvlib import rot_error_deg
from mvtools import MarkerPoseEstimator, load_camera, load_markers_cfg, write_csv
from t7_bonus_occlusion import OcclusionPredictor
from t7_passes import find_passes

TIMES = (0.25, 0.5, 1.0, 1.5, 2.0)


def poses(video, cfg, K, dist):
    """t [s] and ^F T_M per frame (None where not measured)."""
    t = np.loadtxt(os.path.splitext(video)[0] + ".csv", delimiter=",", skiprows=1)[:, 1]
    est = MarkerPoseEstimator(cfg, K, dist, camera_still=True)
    cap = cv2.VideoCapture(video)
    T = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        r = est.process(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        T.append(None if r["T_FM"] is None else r["T_FM"].copy())
    cap.release()
    n = min(len(t), len(T))
    return t[:n], T[:n]


def main():
    ap = argparse.ArgumentParser(description="Bonus: the predictor on uncovered passes hidden in software.")
    ap.add_argument("names", nargs="+", help="recordings in data/recordings (without extension)")
    ap.add_argument("--hide", type=float, nargs=2, default=[78.0, 235.0], help="M is hidden while its left corner is between these ruler readings [mm]")
    ap.add_argument("--line", action="store_true", help="predictor: when M disappears keep the velocity only along the line it was moving on (straight-path option)")
    ap.add_argument("--rot-tau", type=float, default=None, help="predictor: time constant of the turning-speed fade-out [s] (default: the predictor's own; -1 = no fade-out)")
    ap.add_argument("--tag", default="", help="suffix of the output files (to compare settings)")
    a = ap.parse_args()
    kw = {} if a.rot_tau is None else dict(rot_tau=None if a.rot_tau < 0 else a.rot_tau)
    if a.line:
        kw["line_s"] = 0.6
    cfg = load_markers_cfg()
    K, dist, _ = load_camera()
    out = os.path.join(OUTPUT_DIR, "bonus")
    os.makedirs(out, exist_ok=True)
    rows, curves = [], []
    for name in a.names:
        t, T = poses(data_path("recordings", name + ".avi"), cfg, K, dist)
        P = np.array([1000 * x[:3, 3] if x is not None else np.full(3, np.nan) for x in T])
        dt = float(np.median(np.diff(t)))
        _, _, _, passes = find_passes(t, P)
        for n, (i0, i1, ra, rb) in enumerate(passes, 1):
            p0, p1 = np.nanmedian(P[ra[0]:ra[1] + 1], 0), np.nanmedian(P[rb[0]:rb[1] + 1], 0)
            ul = (p1 - p0) / np.linalg.norm(p1 - p0)
            sl = (P - p0) @ ul                                        # = ruler reading of M's left corner (0 at the start stop)
            idx = np.arange(ra[0], rb[1] + 1)
            hid = [k for k in idx if i0 <= k <= i1 and np.isfinite(sl[k]) and a.hide[0] <= sl[k] <= a.hide[1]]
            if len(hid) < 3:
                continue
            h0, h1 = hid[0], hid[-1]                                  # one hidden episode: first to last frame inside the band
            pr = OcclusionPredictor(dt, cfg["L"], **kw)
            outs = {k: pr.step(None if h0 <= k <= h1 else T[k]) for k in idx}
            hold = outs[h0 - 1]["T"]
            e_p, e_h, th, lost_at = [], [], [], None
            for k in range(h0, h1 + 1):
                if T[k] is None:
                    continue
                o = outs[k]
                if o["lost"] and lost_at is None:
                    lost_at = (k - h0 + 1) * dt
                Tp = o["T"] if o["T"] is not None else None
                e_p.append((np.nan, np.nan) if Tp is None else (1000 * np.linalg.norm(Tp[:3, 3] - T[k][:3, 3]), rot_error_deg(T[k][:3, :3], Tp[:3, :3])))
                e_h.append((1000 * np.linalg.norm(hold[:3, 3] - T[k][:3, 3]), rot_error_deg(T[k][:3, :3], hold[:3, :3])))
                th.append((k - h0 + 1) * dt)
            e_p, e_h, th = np.array(e_p), np.array(e_h), np.array(th)
            speed = float(np.polyfit(t[h0:h1 + 1][np.isfinite(sl[h0:h1 + 1])], sl[h0:h1 + 1][np.isfinite(sl[h0:h1 + 1])], 1)[0])
            v_before = float(np.polyfit(t[max(ra[1], h0 - 8):h0], sl[max(ra[1], h0 - 8):h0], 1)[0]) if h0 - max(ra[1], h0 - 8) > 2 else np.nan
            ko = next((k for k in range(h1 + 1, rb[1] + 1) if np.isfinite(outs[k]["jump_mm"])), None)
            row = dict(recording=name, **{"pass": n}, hidden_s=round(float(th[-1]), 2), speed_hidden_mm_s=round(speed, 0),
                       speed_just_before_mm_s=round(v_before, 0), visible_before_s=round(float(t[h0] - t[i0]), 2))
            for x in TIMES:
                j = np.argmin(np.abs(th - x)) if th[-1] >= x - dt else None
                row[f"pred_mm_{x:g}s"] = "" if j is None or not np.isfinite(e_p[j, 0]) else round(float(e_p[j, 0]), 1)
                row[f"hold_mm_{x:g}s"] = "" if j is None else round(float(e_h[j, 0]), 1)
            row.update(pred_mm_end=("" if not np.isfinite(e_p[-1, 0]) else round(float(e_p[-1, 0]), 1)), hold_mm_end=round(float(e_h[-1, 0]), 1),
                       pred_deg_end=("" if not np.isfinite(e_p[-1, 1]) else round(float(e_p[-1, 1]), 2)), hold_deg_end=round(float(e_h[-1, 1]), 2),
                       pred_mm_max=round(float(np.nanmax(e_p[:, 0])), 1), sigma3_mm_end=round(3 * float(outs[h1]["sigma_mm"]), 0),
                       lost_after_s=("" if lost_at is None else round(lost_at, 2)),
                       jump_mm=("" if ko is None else round(float(outs[ko]["jump_mm"]), 1)))
            for thr in (20, 40):                                    # horizon: the hidden time at which the error first passes thr mm
                ip, ih = np.flatnonzero(e_p[:, 0] > thr), np.flatnonzero(e_h[:, 0] > thr)
                row[f'pred_horizon_s_{thr}mm'] = '' if len(ip) == 0 else round(float(th[ip[0]]), 2)
                row[f'hold_horizon_s_{thr}mm'] = '' if len(ih) == 0 else round(float(th[ih[0]]), 2)
            rows.append(row)
            curves.append((name, n, th, e_p[:, 0], e_h[:, 0]))
    if not rows:
        raise SystemExit("no passes found")
    write_csv(os.path.join(out, f"virtual_occlusion{a.tag}.csv"), rows)

    def med(rs, key):
        v = [float(r[key]) for r in rs if r[key] != ""]
        return f"{np.median(v):.1f}" if v else "-"
    lines = ["# Bonus - predictor on uncovered passes hidden in software", "",
             f"Virtual obstacle: M hidden while its left corner is at {a.hide[0]:g}-{a.hide[1]:g} mm on the ruler. Truth = the classical pose measured in the hidden frames.",
             "Predictor = constant-velocity Kalman (position) + alpha-beta on SO(3) (rotation); baseline = hold the last pose. Medians over the passes.", "",
             "| recording | passes | speed while hidden [mm/s] | hidden [s] | e_t predictor / hold at 0.25 s | at 0.5 s | at 1 s | at the end | e_R at the end | lost |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    print(f"\n{'recording':22s} pass  speed  hidden | error at the end of the hidden time: predictor / hold | max pred | 3 sigma | lost | jump")
    for name in a.names:
        rs = [r for r in rows if r["recording"] == name]
        for r in rs:
            print(f"{name:22s} {r['pass']:3d}  {r['speed_hidden_mm_s']:5.0f}  {r['hidden_s']:5.2f} s |   {str(r['pred_mm_end']):>6s} / {r['hold_mm_end']:6.1f} mm,  "
                  f"{str(r['pred_deg_end']):>5s} / {r['hold_deg_end']:5.2f} deg      | {r['pred_mm_max']:6.1f} mm | {r['sigma3_mm_end']:4.0f} mm | {str(r['lost_after_s']) or '-':>4s} | {r['jump_mm']}")
        if rs:
            lines.append(f"| {name} | {len(rs)} | {med(rs, 'speed_hidden_mm_s')} | {med(rs, 'hidden_s')} | {med(rs, 'pred_mm_0.25s')} / {med(rs, 'hold_mm_0.25s')} mm | "
                         f"{med(rs, 'pred_mm_0.5s')} / {med(rs, 'hold_mm_0.5s')} | {med(rs, 'pred_mm_1s')} / {med(rs, 'hold_mm_1s')} | "
                         f"{med(rs, 'pred_mm_end')} / {med(rs, 'hold_mm_end')} | {med(rs, 'pred_deg_end')} / {med(rs, 'hold_deg_end')} deg | "
                         f"{sum(r['lost_after_s'] != '' for r in rs)} of {len(rs)} |")
    lines += ['', 'Horizon = hidden time at which the position error first passes the threshold (median over the passes that pass it; n = how many do).', '',
              '| recording | 20 mm: predictor | 20 mm: hold | 40 mm: predictor | 40 mm: hold |', '|---|---|---|---|---|']
    for name in a.names:
        rs = [r for r in rows if r['recording'] == name]
        cell = lambda key: f"{med(rs, key)} s (n {sum(r[key] != '' for r in rs)} of {len(rs)})" if any(r[key] != '' for r in rs) else f'not reached (0 of {len(rs)})'
        lines.append(f"| {name} | {cell('pred_horizon_s_20mm')} | {cell('hold_horizon_s_20mm')} | {cell('pred_horizon_s_40mm')} | {cell('hold_horizon_s_40mm')} |")
    open(os.path.join(out, f"virtual_occlusion{a.tag}.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[5:]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(a.names), figsize=(6.5 * len(a.names), 4.2), squeeze=False)
    for ax, name in zip(axes[0], a.names):
        for (nm, n, th, ep, eh) in curves:
            if nm == name:
                ax.plot(th, ep, "-", color="#1F3A5F", lw=1.4, label="predictor" if n == 1 else None)
                ax.plot(th, eh, "--", color="#D9731A", lw=1.2, label="hold the last pose" if n == 1 else None)
        ax.set_title(f"{name}: position error while hidden (one line per pass)", fontsize=10)
        ax.set_xlabel("time hidden [s]"); ax.set_ylabel("error against the measured pose [mm]"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"virtual_occlusion{a.tag}.png"), dpi=130)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
