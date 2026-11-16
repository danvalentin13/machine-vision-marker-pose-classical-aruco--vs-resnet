"""t7_real_occlusion.py -- Bonus: the occlusion predictor on passes behind a REAL fixed obstacle.

  python t7_real_occlusion.py E7_cover_slow_4 E7_cover_fast E7_cover_slow_3 [--min-runup 60]

A pass = M rests at the start stop, slides along the ruler behind the obstacle, rests at the end stop (see t7_passes.py).
While M is hidden there is no measurement, so the truth is known exactly only at the moment M is seen again:
  error at reappearance = distance between the predicted pose and the first measured pose after the obstacle
for the predictor (constant-velocity Kalman + alpha-beta on SO(3), t7_bonus_occlusion.OcclusionPredictor) and for the baseline
'hold the last pose'.  In between, the errors are given against a straight line at constant speed from the last pose seen to the
first pose seen again (an assumption: the hand does not move at constant speed).
A pass is CLEAN when M was seen moving for at least --min-runup mm before it disappeared (the predictor needs that to know the
speed); passes where M was lost at the start (blur, fingers) are listed but kept out of the summary.
Writes output/bonus/real_occlusion.csv, real_occlusion.md, real_occlusion.png.
"""
import argparse
import os

import numpy as np

from config import OUTPUT_DIR, data_path
from mvlib import rot_error_deg
from mvtools import load_camera, load_markers_cfg, write_csv
from t7_bonus_occlusion import OcclusionPredictor
from t7_passes import find_passes
from t7_virtual_occlusion import poses

TIMES = (0.25, 0.5, 1.0, 1.5)


def main():
    ap = argparse.ArgumentParser(description="Bonus: the predictor on passes behind a real obstacle.")
    ap.add_argument("names", nargs="+", help="recordings in data/recordings (without extension)")
    ap.add_argument("--min-runup", type=float, default=60.0, help="a pass is clean if M was seen moving for at least this far before hiding [mm]")
    ap.add_argument("--line", action="store_true", help="predictor: when M disappears keep the velocity only along the line it was moving on (straight-path option)")
    ap.add_argument("--rot-tau", type=float, default=None, help="predictor: fade-out of the turning speed [s] (default: the predictor's own)")
    a = ap.parse_args()
    kw = {} if a.rot_tau is None else dict(rot_tau=None if a.rot_tau < 0 else a.rot_tau)
    if a.line:
        kw["line_s"] = 0.6
    cfg = load_markers_cfg()
    K, dist, _ = load_camera()
    out = os.path.join(OUTPUT_DIR, "bonus")
    os.makedirs(out, exist_ok=True)
    rows = []
    for name in a.names:
        t, T = poses(data_path("recordings", name + ".avi"), cfg, K, dist)
        P = np.array([1000 * x[:3, 3] if x is not None else np.full(3, np.nan) for x in T])
        dt = float(np.median(np.diff(t)))
        _, _, _, passes = find_passes(t, P)
        for n, (i0, i1, ra, rb) in enumerate(passes, 1):
            p0, p1 = np.nanmedian(P[ra[0]:ra[1] + 1], 0), np.nanmedian(P[rb[0]:rb[1] + 1], 0)
            ul = (p1 - p0) / np.linalg.norm(p1 - p0)
            sl = (P - p0) @ ul                                        # ruler reading of M's left corner (0 at the start stop)
            bad = [k for k in range(i0, i1 + 1) if T[k] is None]
            if not bad:
                continue
            runs, k0 = [], bad[0]                                     # runs of frames without a measurement
            for u, v in zip(bad[:-1], bad[1:]):
                if v != u + 1:
                    runs.append((k0, u)); k0 = v
            runs.append((k0, bad[-1]))
            r0, r1 = max(runs, key=lambda x: x[1] - x[0])             # the obstacle = the LONGEST run (single dropped frames around it
            ia, ib = r0 - 1, r1 + 1                                   # are detector flicker): last frame seen before / first frame seen after
            pr = OcclusionPredictor(dt, cfg["L"], **kw)
            outs = {k: pr.step(T[k]) for k in range(ra[0], rb[1] + 1)}
            hold, o = outs[ia]["T"], outs[ib]
            hidden_s = float(t[ib] - t[ia])
            row = dict(recording=name, **{"pass": n}, clean=int(sl[ia] >= a.min_runup), last_seen_mm=round(float(sl[ia]), 0), seen_again_mm=round(float(sl[ib]), 0),
                       hidden_s=round(hidden_s, 2), speed_hidden_mm_s=round(float((sl[ib] - sl[ia]) / hidden_s), 0),
                       pred_mm=round(float(o["innov_mm"]), 1), hold_mm=round(1000 * float(np.linalg.norm(hold[:3, 3] - T[ib][:3, 3])), 1),
                       pred_deg=round(float(o["innov_deg"]), 2), hold_deg=round(float(rot_error_deg(T[ib][:3, :3], hold[:3, :3])), 2),
                       sigma3_mm=round(3 * float(outs[ib - 1]["sigma_mm"]), 0))
            lost = [k for k in range(ia + 1, ib) if outs[k]["lost"]]
            row["lost_after_s"] = "" if not lost else round(float(t[lost[0]] - t[ia]), 2)
            row["event"] = o["event"]
            pa, pb = T[ia][:3, 3], T[ib][:3, 3]                       # in between: against a straight line at constant speed
            for x in TIMES:
                k = ia + int(round(x / dt))
                if k < ib and outs[k]["T"] is not None:
                    pt = pa + (pb - pa) * (t[k] - t[ia]) / hidden_s
                    row[f"pred_mm_{x:g}s"] = round(1000 * float(np.linalg.norm(outs[k]["T"][:3, 3] - pt)), 1)
                    row[f"hold_mm_{x:g}s"] = round(1000 * float(np.linalg.norm(hold[:3, 3] - pt)), 1)
                else:
                    row[f"pred_mm_{x:g}s"] = row[f"hold_mm_{x:g}s"] = ""
            rows.append(row)
    if not rows:
        raise SystemExit("no hidden passes found")
    write_csv(os.path.join(out, "real_occlusion.csv"), rows)
    print(f"\n{'recording':18s} pass clean | last seen -> again | hidden  speed | AT REAPPEARANCE: predictor / hold      | 3 sigma | lost")
    for r in rows:
        print(f"{r['recording']:18s} {r['pass']:3d}  {'yes' if r['clean'] else ' no'}  |  {r['last_seen_mm']:4.0f} -> {r['seen_again_mm']:4.0f} mm  | {r['hidden_s']:4.2f} s {r['speed_hidden_mm_s']:4.0f} | "
              f"{r['pred_mm']:6.1f} / {r['hold_mm']:6.1f} mm, {r['pred_deg']:5.1f} / {r['hold_deg']:5.1f} deg | {r['sigma3_mm']:4.0f} mm | {r['lost_after_s'] or '-'}")

    def med(rs, key):
        v = [float(r[key]) for r in rs if r[key] != ""]
        return f"{np.median(v):.1f}" if v else "-"
    clean = [r for r in rows if r["clean"]]
    groups = [("slower (under 125 mm/s)", [r for r in clean if r["speed_hidden_mm_s"] < 125]),
              ("faster (125 mm/s and more)", [r for r in clean if r["speed_hidden_mm_s"] >= 125])]
    lines = ["# Bonus - predictor behind a real obstacle", "",
             f"Recordings: {', '.join(a.names)}. {len(rows)} hidden passes, {len(clean)} clean (M seen moving for at least {a.min_runup:g} mm before hiding).",
             "Error at reappearance = predicted pose against the first measured pose after the obstacle (exact truth). Medians over the clean passes.",
             "Errors while hidden are against a straight line at constant speed between the last and the next measured pose (assumption).", "",
             "| speed group | passes | speed [mm/s] | hidden [s] | e_t at reappearance: predictor / hold | e_R at reappearance | e_t at 0.5 s | e_t at 1 s | lost before reappearing |",
             "|---|---|---|---|---|---|---|---|---|"]
    for label, rs in groups:
        if rs:
            lines.append(f"| {label} | {len(rs)} | {med(rs, 'speed_hidden_mm_s')} | {med(rs, 'hidden_s')} ({min(r['hidden_s'] for r in rs):.2f}-{max(r['hidden_s'] for r in rs):.2f}) | "
                         f"{med(rs, 'pred_mm')} / {med(rs, 'hold_mm')} mm | {med(rs, 'pred_deg')} / {med(rs, 'hold_deg')} deg | "
                         f"{med(rs, 'pred_mm_0.5s')} / {med(rs, 'hold_mm_0.5s')} | {med(rs, 'pred_mm_1s')} / {med(rs, 'hold_mm_1s')} | {sum(r['lost_after_s'] != '' for r in rs)} of {len(rs)} |")
    open(os.path.join(out, "real_occlusion.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[6:]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for rs, mk, lab in ((clean, "o", "clean passes"), ([r for r in rows if not r["clean"]], "x", "M lost before the obstacle")):
        if rs:
            ax.scatter([r["hidden_s"] for r in rs], [r["pred_mm"] for r in rs], marker=mk, color="#1F3A5F", label=f"predictor ({lab})")
            ax.scatter([r["hidden_s"] for r in rs], [r["hold_mm"] for r in rs], marker=mk, color="#D9731A", label=f"hold the last pose ({lab})")
    ax.set_xlabel("time hidden [s]"); ax.set_ylabel("position error at reappearance [mm]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Real obstacle: error when M is seen again", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "real_occlusion.png"), dpi=130)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
