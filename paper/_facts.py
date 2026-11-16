"""Numbers quoted in the paper text, recomputed from the saved files (run: python paper/_facts.py)."""
import csv
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def rows(p):
    with open(os.path.join(ROOT, p), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


# ---- T2 cross-check
r = rows("output/detection/detection_matrix.csv")
td, ed = np.array([fl(x["target_distance_cm"]) for x in r]), np.array([fl(x["est_distance_cm"]) for x in r])
tt, et = np.array([fl(x["target_tilt_deg"]) for x in r]), np.array([fl(x["est_tilt_deg"]) for x in r])
ok = np.isfinite(ed)
print("T2 rows", len(r), "with estimate", ok.sum(), "fps median", np.nanmedian([fl(x["fps"]) for x in r]))
print("  distance diff: max |%.1f| cm mean %+.2f" % (np.nanmax(np.abs(ed - td)), np.nanmean(ed - td)))
for lo in (0, 30):
    m = ok & (tt >= lo)
    print("  tilt >= %d: max |%.1f| deg mean %+.2f (n %d)" % (lo, np.nanmax(np.abs(et - tt)[m]), np.nanmean((et - tt)[m]), m.sum()))
print("  tilt at 0 deg targets:", et[(tt == 0)])

# ---- test errors: pooled groups, usable, components
E = {}
for x in rows("paper/data/test_errors.csv"):
    E.setdefault(x["recording"], []).append([fl(x[k]) for k in ("et_classical_mm", "eR_classical_deg", "et_network_mm", "eR_network_deg", "xc", "yc", "zc", "xn", "yn", "zn")])
E = {k: np.array(v) for k, v in E.items()}
for g in ("E1", "E2", "E3", "E4"):
    for k, name in ((0, "classical"), (1, "network")):
        a = np.concatenate([E[n][:, 2 * k:2 * k + 2] for n in E if n.startswith(g)])
        f = np.isfinite(a[:, 0])
        print(f"{g} {name:9s}: e_t median {np.nanmedian(a[:, 0]):.1f} p95 {np.nanpercentile(a[f, 0], 95):.1f} | e_R median {np.nanmedian(a[:, 1]):.2f} p95 {np.nanpercentile(a[f, 1], 95):.2f} | "
              f"output {100 * f.mean():.1f} % usable {100 * np.mean((a[:, 0] < 20) & (a[:, 1] < 10)):.1f} % (frames {len(a)})")
print("E4 usable, mean over recordings: classical %.1f network %.1f" % tuple(
    np.mean([100 * np.mean((E[n][:, 2 * k] < 20) & (E[n][:, 2 * k + 1] < 10)) for n in E if n.startswith("E4")]) for k in (0, 1)))
for n in ("E4_cover", "E4_dim", "E4_clutter", "E2_d75_t40", "E1_turn40"):
    tr = rows(f"data/recordings/{n}.truth.csv")[0]
    truth = np.array([fl(tr["x_mm"]), fl(tr["y_mm"]), fl(tr["z_mm"])])
    for k, name in ((0, "classical"), (1, "network")):
        p = E[n][:, 4 + 3 * k:7 + 3 * k]
        p = p[np.isfinite(p).all(1)]
        if len(p):
            print(f"  {n} {name}: mean - truth = {np.round(p.mean(0) - truth, 1)} mm, jitter {np.sqrt(np.mean(np.sum((p - p.mean(0)) ** 2, 1))):.2f} mm")
for n in ("E3_motion_01", "E3_motion_02", "E3_motion_03"):
    a = E[n]
    print(f"  {n}: classical frames above 30 mm: {int(np.sum(a[:, 0] > 30))}, max {np.nanmax(a[:, 0]):.1f} (e_R there {a[np.nanargmax(a[:, 0]), 1]:.1f}); network max {np.nanmax(a[:, 2]):.1f}")
    for k, name in ((0, "classical"), (1, "network")):
        v = a[:, 2 * k]
        print(f"     {name}: RMSE {np.sqrt(np.nanmean(v ** 2)):.1f} median {np.nanmedian(v):.1f} max {np.nanmax(v):.1f} | e_R median {np.nanmedian(a[:, 2 * k + 1]):.2f} max {np.nanmax(a[:, 2 * k + 1]):.1f}")

# ---- T4: difference between the methods
a = np.loadtxt(os.path.join(ROOT, "paper/data/T4_motion_1_tracks.csv"), delimiter=",", skiprows=1)
d = np.linalg.norm(a[:, 1:4] - a[:, 7:10], axis=1)
f = np.isfinite(d)
print("T4: frames %d, duration %.1f s, classical pose in %.1f %%; position difference median %.1f mm p95 %.1f" % (len(a), a[-1, 0], 100 * f.mean(), np.nanmedian(d), np.nanpercentile(d[f], 95)))

# ---- bonus: real obstacle by speed group, software-hidden
r = rows("output/bonus/real_occlusion.csv")
sp = np.array([fl(x["speed_hidden_mm_s"]) for x in r]); hid = np.array([fl(x["hidden_s"]) for x in r])
pr, ho = np.array([fl(x["pred_mm"]) for x in r]), np.array([fl(x["hold_mm"]) for x in r])
pd_, hd = np.array([fl(x["pred_deg"]) for x in r]), np.array([fl(x["hold_deg"]) for x in r])
lost = np.array([fl(x["lost_after_s"]) for x in r])
print("bonus real: passes", len(r), "predictor better in", int(np.sum(pr < ho)), "| all: pred/hold median %.1f / %.1f mm" % (np.median(pr), np.median(ho)))
for lo, hi, name in ((0, 75, "4-7 cm/s"), (75, 125, "8-12 cm/s"), (125, 1e9, "13-25 cm/s")):
    m = (sp >= lo) & (sp < hi)
    lb = np.isfinite(lost[m]) & (lost[m] < hid[m])
    print(f"  {name}: n {m.sum()} speed {np.median(sp[m]):.0f} mm/s ({sp[m].min():.0f}-{sp[m].max():.0f}) hidden {np.median(hid[m]):.2f} s ({hid[m].min():.2f}-{hid[m].max():.2f}) "
          f"e_t {np.median(pr[m]):.1f} / {np.median(ho[m]):.1f} mm, better {int(np.sum(pr[m] < ho[m]))}, e_R {np.median(pd_[m]):.1f} / {np.median(hd[m]):.1f} deg, "
          f"lost before reappearing {int(lb.sum())} (after {np.nanmedian(lost[m]):.2f} s)")
    for key in ("0.5s", "1s"):
        a_, b_ = np.array([fl(x[f"pred_mm_{key}"]) for x in r])[m], np.array([fl(x[f"hold_mm_{key}"]) for x in r])[m]
        print(f"      at {key}: {np.nanmedian(a_):.1f} / {np.nanmedian(b_):.1f} mm (n {np.isfinite(a_).sum()})")
m = (hid >= 0.5) & (hid <= 2.0)
print("  hidden 0.5-2 s: n %d, pred/hold %.1f / %.1f mm, better in %d" % (m.sum(), np.median(pr[m]), np.median(ho[m]), int(np.sum(pr[m] < ho[m]))))
lb = np.isfinite(lost) & (lost < hid)
print("  lost before reappearing: %d of %d; lost_after median %.2f s" % (lb.sum(), len(r), np.nanmedian(lost)))
v = rows("output/bonus/virtual_occlusion.csv")
for rec in ("E7_nocover_slow_2", "E7_nocover_fast"):
    rr = [x for x in v if x["recording"] == rec]
    g = lambda k: np.array([fl(x[k]) for x in rr])
    print(f"virtual {rec}: n {len(rr)} speed {np.median(g('speed_hidden_mm_s')):.0f} hidden {np.median(g('hidden_s')):.2f} | "
          + " | ".join(f"{t}: {np.nanmedian(g(f'pred_mm_{t}')):.1f}/{np.nanmedian(g(f'hold_mm_{t}')):.1f}" for t in ("0.25s", "0.5s", "1s", "1.5s", "2s"))
          + f" | jump {np.nanmedian(g('jump_mm')):.1f} mm | lost after {np.nanmedian(g('lost_after_s')):.2f} s | horizon 20: {np.nanmedian(g('pred_horizon_s_20mm')):.2f}/{np.nanmedian(g('hold_horizon_s_20mm')):.2f} "
          f"40: {np.nanmedian(g('pred_horizon_s_40mm')):.2f}/{np.nanmedian(g('hold_horizon_s_40mm')):.2f}")
# ---- calibration
c = rows("paper/data/calib_views.csv")
rm = np.array([fl(x["rms_px"]) for x in c])
print("calibration per view: min %.2f max %.2f median %.2f; distance %.0f-%.0f cm; tilt %.0f-%.0f deg" % (
    rm.min(), rm.max(), np.median(rm), min(fl(x["distance_cm"]) for x in c), max(fl(x["distance_cm"]) for x in c), min(fl(x["tilt_deg"]) for x in c), max(fl(x["tilt_deg"]) for x in c)))
# ---- hard cases
h = json.load(open(os.path.join(ROOT, "output/network/hard_cases_E1.json")))
for k, v_ in h.items():
    print(f"hard {k:18s}: classical out {v_['classical_output']:.0f} usable {v_['classical_usable']:.0f} | ft2 {v_['_ft2']['usable']:.0f} ({v_['_ft2']['et_med']:.1f} mm / {v_['_ft2']['er_med']:.1f}) | ft3 {v_['_ft3']['usable']:.0f} ({v_['_ft3']['et_med']:.1f} / {v_['_ft3']['er_med']:.1f})")
