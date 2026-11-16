"""t7_bonus_occlusion.py -- Bonus (+10): predict the pose of M while a fixed obstacle hides it.

  python t7_bonus_occlusion.py

Setup (assignment): F stays visible; M passes behind a fixed obstacle at least 5 times, at two speeds, hidden for 0.5-2 s each time;
record it with a KNOWN path at constant speed (turntable / ruler / rail at a measured speed) and write that path in <name>.truth.csv
(several rows t_s,x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg of M in F, see README T6).  Tag the recording E7_occlusion.
Then run t3_pose.py on it (classical) to get <name>_poses.csv: frames without a measurement are the hidden ones.

What this script does with  output/pose/<name>_poses.csv  +  data/recordings/<name>.truth.csv:
  * finds the hidden episodes (runs of frames without a pose of M, 0.3-4 s, after the tracker is running)
  * runs the PREDICTOR: constant-velocity Kalman filter (translation) + alpha-beta filter on SO(3) (rotation); while M is hidden the
    filter only predicts and its position uncertainty (sigma = sqrt(trace P)) grows; the pose is declared LOST when 3 sigma > lost_factor * L
  * RE-ACQUISITION: the first measurements are gated against the prediction (chi-square gate on the position, an angle gate on the rotation to
    reject planar-ambiguity flips); after a lost pose the tracker restarts from the measurement.  The jump of the output pose is reported.
  * compares with the truth and with the baseline "hold the last pose"
Writes  <output>/episodes.csv          one row per hidden episode: duration, e_t at 0.25/0.5/1/2 s (predictor and hold), lost time, jump
        <output>/summary.md            tables per speed: error against time hidden, horizon where the error passes the threshold, jump
        <output>/occlusion_curves.png  e_t and e_R against time hidden, both speeds, with the filter's 3 sigma
        <output>/<name>_predicted_poses.csv   output poses with predicted / lost / sigma columns: give it to t4_animate.py (dashed segment + uncertainty)
"""
import argparse
import glob
import os

import numpy as np

from config import data_path, OUTPUT_DIR
from mvlib import AlphaBetaSO3, ConstantVelocityKalman, make_T, rot_error_deg
from mvtools import (ask, ask_folder, ask_number, cols_to_pose, load_markers_cfg, load_truth, pose_errors, pose_to_cols, read_csv,
                     summarize, truth_pose, write_csv)

TIMES = (0.25, 0.5, 1.0, 2.0)               # hidden times reported in the table [s]


class OcclusionPredictor:
    """Kalman (position) + alpha-beta (rotation) predictor with growing uncertainty, a lost-pose rule and gated re-acquisition."""

    def __init__(self, dt, L, sigma_a=0.15, sigma_m=0.004, lost_factor=2.0, rot_gate_deg=30.0, rot_tau=0.1, line_s=None):
        # rot_tau [s]: while M is hidden the estimated turning speed fades out with this time constant (0 = hold the last
        # orientation at once, None = keep turning at the last speed).  The turning speed is estimated from noisy rotations
        # (hand wobble), so extrapolating it for a second makes the rotation worse than holding it (E7 uncovered passes).
        # line_s [s]: when M disappears, the velocity is kept only along the direction M moved over the last line_s seconds
        # (a marker on a rail / ruler moves on a straight line).  The velocity components across that line are estimated from
        # measurement noise.  Tested on the E7 recordings (line_s 0.6): better at long hidden times and slow speed (55 -> 40 mm after
        # 2.9 s at 5 cm/s), slightly worse at 10-14 cm/s (50 -> 57, 96 -> 106 mm): no clear gain, so it is OFF by default (None).
        self.dt, self.L, self.lost_factor, self.rot_gate, self.rot_tau = dt, L, lost_factor, rot_gate_deg, rot_tau
        self.line_s, self.recent = line_s, []
        self.kf, self.rf = ConstantVelocityKalman(dt, sigma_a, sigma_m), AlphaBetaSO3(dt)
        self.init, self.lost, self.rej, self.hidden_s = False, False, 0, 0.0

    def sigma_mm(self):
        return 1000.0 * float(np.sqrt(np.trace(self.kf.P[:3, :3])))

    def pose(self):
        return make_T(self.rf.R, self.kf.position)

    def _start(self, T):
        self.kf.init(T[:3, 3]); self.rf.init(T[:3, :3])
        self.recent = []
        self.init, self.lost, self.rej, self.hidden_s = True, False, 0, 0.0

    def step(self, T):
        """T: measured ^F T_M (4x4) or None.  Returns a dict: T (output pose or None), predicted, lost, sigma_mm, event, innovation, jump."""
        out = dict(T=None, predicted=False, lost=False, sigma_mm=np.nan, event="", innov_mm=np.nan, innov_deg=np.nan, jump_mm=np.nan, jump_deg=np.nan)
        if not self.init:
            if T is not None:
                self._start(T)
                out.update(T=T.copy(), sigma_mm=self.sigma_mm(), event="start")
            return out
        self.kf.predict(); self.rf.predict()
        pre, was_hidden, accepted = self.pose(), self.hidden_s > 0, False
        if T is not None:
            out["innov_mm"] = 1000.0 * float(np.linalg.norm(T[:3, 3] - self.kf.position))
            out["innov_deg"] = rot_error_deg(self.rf.R, T[:3, :3])
            if self.lost:
                self._start(T); accepted = True; out["event"] = "re-acquired after LOST: restarted from the measurement"
            elif out["innov_deg"] <= self.rot_gate and self.kf.update(T[:3, 3]):
                self.rf.update(T[:3, :3]); self.rej, accepted = 0, True
                if was_hidden:
                    out["event"] = "re-acquired (gated)"
            else:
                self.rej += 1
                out["event"] = "measurement rejected by the gate"
                if self.rej >= 5:                                           # locked onto a wrong pose: restart (as in the lecture)
                    self._start(T); accepted = True; out["event"] = "restarted after 5 rejections"
        if accepted:
            if was_hidden:
                post = self.pose()
                out["jump_mm"] = 1000.0 * float(np.linalg.norm(post[:3, 3] - pre[:3, 3]))
                out["jump_deg"] = rot_error_deg(pre[:3, :3], post[:3, :3])
            self.hidden_s = 0.0
            if self.line_s:                                                # the last positions seen: the direction of motion
                self.recent = (self.recent + [self.kf.position])[-max(3, int(round(self.line_s / self.dt))):]
        else:
            if self.hidden_s == 0 and self.line_s and len(self.recent) >= 3:   # M has just disappeared: keep the velocity along the line
                d = self.recent[-1] - self.recent[0]
                if np.linalg.norm(d) >= 0.010:
                    u = d / np.linalg.norm(d)
                    self.kf.x[3:6] = (self.kf.x[3:6] @ u) * u
            self.hidden_s += self.dt
            if self.rot_tau is not None:                                   # fade the turning speed out while predicting
                self.rf.w = self.rf.w * (0.0 if self.rot_tau <= 0 else float(np.exp(-self.dt / self.rot_tau)))
            if 3.0 * self.sigma_mm() / 1000.0 > self.lost_factor * self.L and not self.lost:
                self.lost = True
                out["event"] = "pose LOST (3 sigma > %g L)" % self.lost_factor
        out.update(T=None if self.lost else self.pose(), predicted=not accepted, lost=self.lost, sigma_mm=self.sigma_mm())
        return out


def run_predictor(rows, dt, L, **kw):
    pr = OcclusionPredictor(dt, L, **kw)
    meas = [cols_to_pose(r, "FM") for r in rows]
    return meas, [pr.step(T) for T in meas]


def find_episodes(meas, dt, min_s=0.3, max_s=4.0):
    """(start, end) frame indices [start, end) of runs without a measurement, after a measurement and before the next one."""
    eps, i, n = [], 0, len(meas)
    while i < n:
        if meas[i] is None and i > 0 and any(m is not None for m in meas[:i]):
            j = i
            while j < n and meas[j] is None:
                j += 1
            if j < n and min_s <= (j - i) * dt <= max_s:
                eps.append((i, j))
            i = j
        else:
            i += 1
    return eps


def analyse(name, rows, truth, L, dt, **kw):
    meas, outs = run_predictor(rows, dt, L, **kw)
    t = np.array([float(r["t_s"]) for r in rows])
    p_true = np.array([truth_pose(truth, x)[:3, 3] for x in t])
    speed = float(np.median(np.linalg.norm(np.gradient(p_true, t, axis=0), axis=1))) * 1000.0 if len(t) > 2 else np.nan   # mm/s
    episodes = []
    for (a, b) in find_episodes(meas, dt):
        hold = outs[a - 1]["T"]
        ks = np.arange(a, b)
        th = (ks - a + 1) * dt
        e = [pose_errors(outs[k]["T"], truth_pose(truth, t[k])) for k in ks]
        h = [pose_errors(hold, truth_pose(truth, t[k])) for k in ks]
        lost_k = next((k for k in ks if outs[k]["lost"]), None)
        after = outs[b]
        episodes.append(dict(recording=name, start_frame=a, end_frame=b, hidden_s=(b - a) * dt, speed_mm_s=speed, th=th,
                             et=np.array([x[0] for x in e]), er=np.array([x[1] for x in e]),
                             et_hold=np.array([x[0] for x in h]), er_hold=np.array([x[1] for x in h]),
                             sigma3=np.array([3 * outs[k]["sigma_mm"] for k in ks]),
                             lost_s=None if lost_k is None else (lost_k - a + 1) * dt,
                             innov_mm=after["innov_mm"], innov_deg=after["innov_deg"], jump_mm=after["jump_mm"], jump_deg=after["jump_deg"],
                             reacq=after["event"] or "measurement accepted"))
    return meas, outs, episodes, speed


def at_time(ep, key, s):
    """Value of an error series at hidden time s [s] (nearest frame), NaN if the episode is shorter."""
    th = ep["th"]
    frame = th[1] - th[0] if len(th) > 1 else 0.05
    if s > th[-1] + 0.5 * frame:                      # (time stamps are rounded: allow half a frame)
        return np.nan
    return float(ep[key][int(np.argmin(np.abs(th - s)))])


def horizon(eps, key, thr, grid):
    """First hidden time at which the MEDIAN error over episodes passes thr [s] (nan if never within the grid)."""
    for s in grid:
        v = [at_time(e, key, s) for e in eps]
        v = [x for x in v if np.isfinite(x)]
        if v and np.median(v) > thr:
            return float(s)
    return np.nan


def curves_figure(groups, path, thr_mm, dt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.arange(dt, 2.01, dt)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    cols = ("#1F3A5F", "#D9731A", "#1E8C85", "#B83227")
    for (label, eps), c in zip(groups.items(), cols):
        def med(key, q=50):
            return np.array([np.nanpercentile([at_time(e, key, s) for e in eps], q) if any(np.isfinite(at_time(e, key, s)) for e in eps) else np.nan for s in grid])
        for a, key, hold in ((ax[0], "et", "et_hold"), (ax[1], "er", "er_hold")):
            a.plot(grid, med(key), color=c, lw=1.6, label=f"predictor, {label}")
            a.fill_between(grid, med(key), med(key, 90), color=c, alpha=0.12, lw=0)
            a.plot(grid, med(hold), color=c, lw=1.0, ls="--", label=f"hold last pose, {label}")
        ax[0].plot(grid, med("sigma3"), color=c, lw=0.8, ls=":")
    ax[0].axhline(thr_mm, color="gray", lw=0.6); ax[0].text(0.02, thr_mm * 1.05, f"{thr_mm:g} mm", color="gray", fontsize=7)
    ax[0].set_ylabel("translation error [mm]   (dotted: filter 3 sigma)"); ax[1].set_ylabel("rotation error [deg]")
    for a in ax:
        a.set_xlabel("time hidden [s]"); a.grid(alpha=0.3); a.set_xlim(0, 2); a.set_ylim(bottom=0)
    ax[0].set_ylim(0, max(250, 3 * thr_mm)); ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def summary_md(eps_all, thr_mm, thr_deg, lost_factor, path):
    speeds = sorted({round(e["speed_mm_s"], -1) for e in eps_all})
    L = ["# Bonus - prediction through an obstacle", "",
         f"Predictor: constant-velocity Kalman (position) + alpha-beta on SO(3) (rotation). Pose declared LOST when 3 sigma > {lost_factor:g} L. "
         f"Threshold for the horizon: {thr_mm:g} mm / {thr_deg:g} deg.", "",
         f"{len(eps_all)} hidden episodes from {len({e['recording'] for e in eps_all})} recording(s).", ""]
    grid = np.arange(0.05, 2.01, 0.05)
    for sp in speeds:
        eps = [e for e in eps_all if round(e["speed_mm_s"], -1) == sp]
        L += [f"## Speed {sp:.0f} mm/s  ({len(eps)} episodes, hidden {min(e['hidden_s'] for e in eps):.2f}-{max(e['hidden_s'] for e in eps):.2f} s)", "",
              "| hidden time [s] | predictor e_t [mm] | hold e_t [mm] | predictor e_R [deg] | hold e_R [deg] | filter 3 sigma [mm] | episodes |", "|---|---|---|---|---|---|---|"]
        for s in TIMES:
            def m(k):
                v = [at_time(e, k, s) for e in eps]
                v = [x for x in v if np.isfinite(x)]
                return (f"{np.median(v):.1f}", len(v)) if v else ("-", 0)
            L.append(f"| {s:g} | {m('et')[0]} | {m('et_hold')[0]} | {m('er')[0]} | {m('er_hold')[0]} | {m('sigma3')[0]} | {m('et')[1]} |")
        lost = [e["lost_s"] for e in eps if e["lost_s"] is not None]
        L += ["", f"- horizon (median error passes {thr_mm:g} mm): predictor {horizon(eps, 'et', thr_mm, grid):.2f} s, hold {horizon(eps, 'et_hold', thr_mm, grid):.2f} s  (nan = not within 2 s)",
              f"- horizon (median error passes {thr_deg:g} deg): predictor {horizon(eps, 'er', thr_deg, grid):.2f} s, hold {horizon(eps, 'er_hold', thr_deg, grid):.2f} s",
              f"- pose declared lost in {len(lost)}/{len(eps)} episodes" + (f" (median after {np.median(lost):.2f} s hidden)" if lost else ""),
              "- re-acquisition (first measurement after the obstacle): jump of the output pose "
              f"median {np.nanmedian([e['jump_mm'] for e in eps]):.1f} mm / {np.nanmedian([e['jump_deg'] for e in eps]):.1f} deg, "
              f"max {np.nanmax([e['jump_mm'] for e in eps]):.1f} mm / {np.nanmax([e['jump_deg'] for e in eps]):.1f} deg; "
              f"measurement vs prediction (innovation) median {np.nanmedian([e['innov_mm'] for e in eps]):.1f} mm", ""]
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def episodes_csv(eps_all, path):
    rows = []
    for e in eps_all:
        r = dict(recording=e["recording"], start_frame=e["start_frame"], hidden_s=round(e["hidden_s"], 2), speed_mm_s=round(e["speed_mm_s"], 1))
        for s in TIMES:
            r[f"pred_et_{s:g}s_mm"] = round(at_time(e, "et", s), 1)
            r[f"hold_et_{s:g}s_mm"] = round(at_time(e, "et_hold", s), 1)
        r.update(lost_s=e["lost_s"], innovation_mm=round(e["innov_mm"], 1), jump_mm=round(e["jump_mm"], 1) if np.isfinite(e["jump_mm"]) else "",
                 jump_deg=round(e["jump_deg"], 1) if np.isfinite(e["jump_deg"]) else "", reacquisition=e["reacq"])
        rows.append(r)
    write_csv(path, rows)


def write_predicted_csv(path, rows, outs, meas):
    out = []
    for r, o, m in zip(rows, outs, meas):
        out.append(dict(frame=r["frame"], t_s=r["t_s"], measured=int(m is not None), predicted=int(o["predicted"] and not o["lost"]),
                        lost=int(o["lost"]), sigma_mm=round(o["sigma_mm"], 2) if np.isfinite(o["sigma_mm"]) else "", **pose_to_cols("FM", o["T"])))
    write_csv(path, out)


def evaluate(pose_csvs, truth_dir, L, out_dir, lost_factor=2.0, sigma_a=0.15, thr_mm=20.0, thr_deg=5.0, log=print):
    os.makedirs(out_dir, exist_ok=True)
    eps_all = []
    for pc in pose_csvs:
        name = os.path.basename(pc).replace("_poses.csv", "")
        tp = os.path.join(truth_dir, name + ".truth.csv")
        if not os.path.exists(tp):
            log(f"  {name}: no truth file, skipped")
            continue
        truth = load_truth(tp)
        if len(truth[0]) < 2:
            log(f"  {name}: the truth file has one row (static pose): the occlusion test needs the known path, skipped")
            continue
        rows = read_csv(pc)
        dt = float(np.median(np.diff([float(r["t_s"]) for r in rows])))
        meas, outs, eps, speed = analyse(name, rows, truth, L, dt, lost_factor=lost_factor, sigma_a=sigma_a)
        write_predicted_csv(os.path.join(out_dir, name + "_predicted_poses.csv"), rows, outs, meas)
        log(f"  {name}: speed {speed:.0f} mm/s, {len(eps)} hidden episodes " + ", ".join(f"{e['hidden_s']:.1f} s" for e in eps))
        eps_all += eps
    if not eps_all:
        return []
    speeds = sorted({round(e["speed_mm_s"], -1) for e in eps_all})
    dt = float(np.median(np.diff([float(r["t_s"]) for r in read_csv(pose_csvs[0])])))
    groups = {f"{sp:.0f} mm/s": [e for e in eps_all if round(e["speed_mm_s"], -1) == sp] for sp in speeds}
    episodes_csv(eps_all, os.path.join(out_dir, "episodes.csv"))
    summary_md(eps_all, thr_mm, thr_deg, lost_factor, os.path.join(out_dir, "summary.md"))
    curves_figure(groups, os.path.join(out_dir, "occlusion_curves.png"), thr_mm, dt)
    return eps_all


def main():
    argparse.ArgumentParser(description="Bonus: prediction through an obstacle.").parse_args()
    cfg = load_markers_cfg()
    print("Bonus - prediction through an obstacle.  Press Enter to accept the [default].\n")
    pose_dir = ask_folder("1) Folder with the *_poses.csv files (from t3_pose.py)", os.path.join(OUTPUT_DIR, "pose"))
    files = sorted(glob.glob(os.path.join(pose_dir, "*_poses.csv")))
    names = ask("2) Which recordings (names separated by commas, or 'all')", "all")
    if names.lower() != "all":
        keep = [n.strip() for n in names.split(",")]
        files = [f for f in files if os.path.basename(f).replace("_poses.csv", "") in keep]
    truth_dir = ask_folder("3) Folder with the truth files", data_path("recordings"))
    lost_factor = ask_number("4) Declare the pose LOST when 3 sigma exceeds this many marker sides L", 2.0, float, 0.1)
    sigma_a = ask_number("5) Process noise: how abruptly M accelerates [m/s^2]", 0.15, float, 0.001)
    thr_mm = ask_number("6) Error threshold for the horizon [mm]", 20.0, float, 0.1)
    out_dir = ask_folder("7) Output folder", os.path.join(OUTPUT_DIR, "occlusion"))
    print(f"\nL = {cfg['L'] * 1000:g} mm; lost when 3 sigma > {lost_factor * cfg['L'] * 1000:g} mm\n")
    eps = evaluate(files, truth_dir, cfg["L"], out_dir, lost_factor, sigma_a, thr_mm)
    if not eps:
        raise SystemExit("no hidden episodes found: check that M is really hidden for 0.3-4 s in the recording and that the truth file has several rows")
    print(f"\n{len(eps)} episodes -> {out_dir}\n  summary.md, episodes.csv, occlusion_curves.png, <name>_predicted_poses.csv (use it in t4_animate.py)")


if __name__ == "__main__":
    main()
