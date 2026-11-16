"""extra_checks.py -- three computations quoted in the paper that no other program saves (run from the project folder):

    python paper/extra_checks.py

1. E5 at full resolution: both methods on the 1 000 synthetic test images of 1280 x 720 px (the evaluation script uses the
   640 x 360 test split); writes paper/data/e5_full.json.
2. E3: how far the hand lags behind the guide outline (position along the ruler against the schedule).
3. E3: the single-frame outliers of the classical pipeline (depth of M and reprojection error in and around those frames);
   writes paper/data/e3_outliers.json.
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
from config import data_path, training_path          # noqa: E402
from mvtools import MarkerPoseEstimator, load_camera, load_markers_cfg, summarize  # noqa: E402
from netpose import NetPose                           # noqa: E402
import t6_evaluate                                    # noqa: E402

OUT = os.path.join(ROOT, "paper", "data")
cfg = load_markers_cfg()
K, dist, _ = load_camera()


def e5_full():
    net = NetPose(training_path("models", "crop_resnet18_v1_ft3.pt"), K=K, dist=dist)
    out = {}
    for name in ("synth_v1", "synth_full_test"):
        res = t6_evaluate.synthetic_eval(training_path("synthetic", name), net, cfg, log=lambda s: None)
        out[name] = {}
        for m, d in res.items():
            et, er = d["et"], d["er"]
            ok = np.isfinite(et)
            out[name][m] = dict(output_pct=float(100 * ok.mean()), et_median=summarize(et)["median"], et_p95=summarize(et)["p95"],
                                er_median=summarize(er)["median"], er_p95=summarize(er)["p95"],
                                usable_pct=float(100 * np.mean((et < 20) & (er < 10))))
            print(f"E5 {name:16s} {m:9s}: output {out[name][m]['output_pct']:.1f} %  e_t {out[name][m]['et_median']:.2f} ({out[name][m]['et_p95']:.1f}) mm  "
                  f"e_R {out[name][m]['er_median']:.3f} ({out[name][m]['er_p95']:.2f}) deg  usable {out[name][m]['usable_pct']:.1f} %")
    json.dump(out, open(os.path.join(OUT, "e5_full.json"), "w"), indent=1)


def e3_lag():
    for k in (1, 2, 3):
        a = np.loadtxt(os.path.join(OUT, f"E3_motion_0{k}_errors.csv"), delimiter=",", skiprows=1)
        t, xt, xc = a[:, 0], a[:, 1], a[:, 2]
        v = np.gradient(xt, t)                                         # mm/s of the guide: -10 on the way out, +10 on the way back
        for name, m in (("way out", v < -5), ("way back", v > 5)):
            lag = -(xc[m] - xt[m]) / v[m]                              # s the measured position is behind the guide
            d = xc[m] - xt[m]
            print(f"E3 recording {k}, {name}: behind the guide by median {np.nanmedian(lag):.2f} s, largest {np.nanmax(lag):.2f} s "
                  f"({np.nanmax(np.abs(d)):.1f} mm along the ruler)")
        rest = np.abs(v) <= 5
        print(f"   at rest: |x - truth| median {np.nanmedian(np.abs(xc[rest] - xt[rest])):.1f} mm;  e_t median at rest {np.nanmedian(a[rest, 4]):.1f} mm, "
              f"while moving {np.nanmedian(a[~rest, 4]):.1f} mm")


def e3_outliers():
    out = {}
    for name in ("E3_motion_01", "E3_motion_02", "E3_motion_03"):
        est = MarkerPoseEstimator(cfg, K, dist, camera_still=True)
        cap = cv2.VideoCapture(data_path("recordings", name + ".avi"))
        z, rms, side = [], [], []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            r = est.process(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
            c = r["corners"].get(cfg["id_moving"])
            z.append(np.nan if r["T_CM"] is None else 1000 * r["T_CM"][2, 3])
            rms.append(r["rms_M"])
            side.append(np.nan if c is None else float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)])))
        z, rms, side = np.array(z), np.array(rms), np.array(side)
        med = np.array([np.nanmedian(z[max(0, i - 5):i + 6]) for i in range(len(z))])
        bad = np.where(z - med > 30)[0]
        out[name] = dict(frames=[int(i) for i in bad], depth_jump_mm=[float(z[i] - med[i]) for i in bad],
                         side_ratio=[float(side[i] / np.nanmedian(side[max(0, i - 5):i + 6])) for i in bad],
                         rms_px=[float(rms[i]) for i in bad], rms_px_median_all=float(np.nanmedian(rms)))
        print(f"{name}: {len(bad)} outlier frame(s) {list(bad)}; depth +{np.round(out[name]['depth_jump_mm'], 0)} mm; outline side x "
              f"{np.round(out[name]['side_ratio'], 3)}; reprojection {np.round(out[name]['rms_px'], 2)} px (median of all frames {out[name]['rms_px_median_all']:.2f} px)")
    json.dump(out, open(os.path.join(OUT, "e3_outliers.json"), "w"), indent=1)


if __name__ == "__main__":
    what = sys.argv[1:] or ["e5", "lag", "outliers"]
    if "e5" in what:
        e5_full()
    if "lag" in what:
        e3_lag()
    if "outliers" in what:
        e3_outliers()
