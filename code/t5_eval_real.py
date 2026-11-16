"""t5_eval_real.py -- Task 5 / 6: classical against the networks on the REAL test recordings (never used in training).

  python t5_eval_real.py

E1_p1..p5 (M fixed at 5 measured positions): e_t, e_R against the ruler truth, jitter, output rate, x/y/z bias, ms per frame.
T4_motion_1 (M moving, no truth): difference between each network and the classical pose.
Edit the dictionary `nets` to choose the networks (passes = 1: stage 1 places the crop; 2: the crop is re-centred once).
Writes output/network/real_E1_comparison_ft3.json.  Results of the three fine-tuning rounds: output/network/T5_runs.md.
"""
import os, sys, json
import numpy as np
import mvtools as mt, t3_pose
from config import data_path, training_path, OUTPUT_DIR
from netpose import NetPose

cfg = mt.load_markers_cfg(); K, dist, size = mt.load_camera()
M = lambda n, p: NetPose(training_path("models", n + ".pt"), K=K, dist=dist, passes=p)
nets = {"classical": None,
        "two-stage _ft2 (round 2), 2 pass": M("crop_resnet18_v1_ft2", 2),
        "whole-image _ft3 (round 3)": M("resnet18_v2_ft3", 1),
        "two-stage _ft3 (round 3), 1 pass": M("crop_resnet18_v1_ft3", 1),
        "two-stage _ft3 (round 3), 2 pass": M("crop_resnet18_v1_ft3", 2)}
recs = [f"E1_p{k}" for k in range(1, 6)]
res = {m: {} for m in nets}
for m, net in nets.items():
    for r in recs:
        rows, s = t3_pose.process_video(data_path("recordings", r + ".avi"), cfg, K, dist, True, net, verbose=False)
        truth = mt.load_truth(data_path("recordings", r + ".truth.csv"))
        Tt = mt.truth_pose(truth, 0)
        e = [mt.pose_errors(mt.cols_to_pose(x, "FM"), Tt) for x in rows]
        et, er = np.array([a for a, b in e]), np.array([b for a, b in e])
        P = np.array([[x["FM_x_mm"], x["FM_y_mm"], x["FM_z_mm"]] for x in rows], float)
        ok = np.isfinite(P).all(1)
        res[m][r] = dict(et=float(np.nanmedian(et)), er=float(np.nanmedian(er)), out=float(100 * ok.mean()),
                         jit=float(np.sqrt(np.mean(np.sum((P[ok] - P[ok].mean(0)) ** 2, 1)))) if ok.sum() > 1 else np.nan,
                         d=(P[ok].mean(0) - 1000 * Tt[:3, 3]).tolist(), ms=s["ms_per_frame"], all_et=et.tolist(), all_er=er.tolist())
        print(f"{m:34s} {r}: e_t {res[m][r]['et']:6.1f} mm  e_R {res[m][r]['er']:5.2f} deg  jitter {res[m][r]['jit']:5.2f} mm  output {res[m][r]['out']:.0f} %  "
              f"(x/y/z bias {res[m][r]['d'][0]:+.0f}/{res[m][r]['d'][1]:+.0f}/{res[m][r]['d'][2]:+.0f})")
print()
for m in nets:
    et = np.concatenate([res[m][r]["all_et"] for r in recs]); er = np.concatenate([res[m][r]["all_er"] for r in recs])
    print(f"{m:34s} ALL 500 frames: e_t median {np.nanmedian(et):.1f} (p95 {np.nanpercentile(et, 95):.1f}) mm, e_R median {np.nanmedian(er):.2f} "
          f"(p95 {np.nanpercentile(er, 95):.1f}) deg, jitter mean {np.mean([res[m][r]['jit'] for r in recs]):.2f} mm, {np.mean([res[m][r]['ms'] for r in recs]):.1f} ms/frame")
# T4 motion recording (no truth): does the network follow the classical pose?
print()
rows_c, _ = t3_pose.process_video(data_path("recordings", "T4_motion_1.avi"), cfg, K, dist, True, None, verbose=False)
for m in list(nets)[1:]:
    rows_n, s = t3_pose.process_video(data_path("recordings", "T4_motion_1.avi"), cfg, K, dist, True, nets[m], verbose=False)
    d_t, d_r = [], []
    for a, b in zip(rows_c, rows_n):
        Ta, Tb = mt.cols_to_pose(a, "FM"), mt.cols_to_pose(b, "FM")
        if Ta is not None and Tb is not None:
            x, y = mt.pose_errors(Tb, Ta); d_t.append(x); d_r.append(y)
    print(f"T4_motion_1 (M moving, 30 s, never seen in training): {m} vs classical on {len(d_t)} frames: "
          f"difference median {np.median(d_t):.1f} mm / {np.median(d_r):.2f} deg (p95 {np.percentile(d_t, 95):.0f} mm / {np.percentile(d_r, 95):.1f} deg)")
json.dump({m: {r: {k: v for k, v in x.items() if not k.startswith("all_")} for r, x in d.items()} for m, d in res.items()},
          open(os.path.join(OUTPUT_DIR, "network", "real_E1_comparison_ft3.json"), "w"), indent=1)
