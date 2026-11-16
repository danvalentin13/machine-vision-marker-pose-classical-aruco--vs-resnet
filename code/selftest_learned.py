"""selftest_learned.py -- the Task 5 / Task 6 part of selftest.py (called from there; not run directly)."""
import json
import os

import cv2
import numpy as np

import mvtools as mt
import synth
from config import OUTPUT_DIR, data_path, training_path


def run(check, cfg):
    import torch
    import t5_make_data as md
    import t5_train
    import t6_evaluate as t6
    from netpose import NetPose, build_model, six_d_to_R
    from synthdata import load_set
    from mvlib import rot_error_deg

    print("T5  synthetic data set (tiny) and the network code")
    syn_dir = training_path("synthetic")
    d, meta = md.make_dataset(syn_dir, "tiny", cfg, md.PROVISIONAL["K"], md.PROVISIONAL["size"], 300, 60, 100, 320, (0.3, 1.2), 60.0, 0, True)
    X, R, t = load_set(d, "test")
    check("data set written", X.shape == (100, 180, 320) and R.shape == (100, 3, 3), f"images {X.shape}, {meta['generation_seconds']:.0f} s")
    est = mt.MarkerPoseEstimator(dict(cfg), np.array(meta["K_net"]), np.zeros(5), camera_still=False)
    et, er = [], []
    for i in range(len(X)):
        T = est.process(X[i])["T_CM"]
        if T is not None:
            et.append(1000 * np.linalg.norm(T[:3, 3] - t[i])); er.append(rot_error_deg(R[i].astype(float), T[:3, :3]))
    check("labels agree with the pixels (classical PnP on the synthetic images)", len(et) > 60 and np.median(et) < 15 and np.median(er) < 3,
          f"found {len(et)}/100, median {np.median(et):.1f} mm, {np.median(er):.2f} deg")

    a = torch.randn(64, 6)
    Rm = six_d_to_R(a)
    I = torch.eye(3).expand(64, 3, 3)
    check("6-D output gives valid rotation matrices", torch.allclose(Rm.transpose(1, 2) @ Rm, I, atol=1e-5) and torch.allclose(torch.linalg.det(Rm), torch.ones(64), atol=1e-5))

    models, out = training_path("models"), os.path.join(OUTPUT_DIR, "network", "selftest")
    hist, config = t5_train.train(d, "selftest_net", epochs=2, batch=32, max_steps=None, models_dir=models, out_dir=out, log=lambda *_: None)
    check("training loop runs and records the validation error (throwaway net, 2 epochs of 300 images)",
          len(hist) == 2 and np.isfinite(hist[-1]["loss"]) and hist[-1]["loss"] < hist[0]["loss"] * 1.5,
          f"loss {hist[0]['loss']:.2f} -> {hist[-1]['loss']:.2f}, device {config['device']}")
    check("checkpoint and curves written", os.path.exists(os.path.join(models, "selftest_net.pt")) and os.path.exists(os.path.join(out, "training_curves.png")))
    net = NetPose(os.path.join(models, "selftest_net.pt"), K=md.PROVISIONAL["K"], dist=np.zeros(5))
    T = net.predict_one(X[0])
    check("NetPose returns a 4x4 rigid transform", T.shape == (4, 4) and abs(np.linalg.det(T[:3, :3]) - 1) < 1e-5 and T[2, 3] > 0)

    print("T6  evaluation of both pipelines (the network here is the throwaway one: only the machinery is tested)")
    K, dist, _ = mt.load_camera(data_path("calibration", "camera_synth.npz"))
    ev = os.path.join(OUTPUT_DIR, "evaluation")
    log = []
    recs, synthres, cost, table = t6.evaluate(data_path("recordings"), cfg, K, dist, net, os.path.join(models, "selftest_net.pt"), d, ev, True, log.append)
    check("evaluation ran on the recordings with truth", len(recs) == 2 and "network" in recs[0], f"{len(recs)} recordings")
    et = t6.summarize(recs[0]["classical"]["et"])
    check("classical E3 error against hand-style truth", et["median"] < 10, f"median {et['median']:.1f} mm, p95 {et['p95']:.1f}, max {et['max']:.1f}")
    check("hidden frames count as no output", 80 < recs[0]["classical"]["output_pct"] < 98, f"output {recs[0]['classical']['output_pct']:.0f} %")
    check("E5 synthetic evaluation ran for both methods", synthres is not None and set(synthres) == {"classical", "network"})
    check("E6 cost numbers present", cost is not None and cost.get("network_cpu_ms", 0) > 0 and cost["loc_classical"] > 50,
          f"classical {cost['classical_fps']:.0f} fps, network {cost.get('network_fps_gpu', 0):.0f} fps GPU / {cost.get('network_fps_cpu', 0):.0f} fps CPU")
    files = ["comparison_table.md", "comparison_table.csv", "report.md", "summary.json", "worst_frames.csv"]
    check("report files written", all(os.path.exists(os.path.join(ev, f)) for f in files) and any(f.startswith("E3_") for f in os.listdir(ev)))
    print("\n" + open(os.path.join(ev, "comparison_table.md"), encoding="utf-8").read())
