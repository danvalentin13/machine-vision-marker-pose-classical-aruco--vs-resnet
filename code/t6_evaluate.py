"""t6_evaluate.py -- Task 6: compare the classical pipeline and the network on the SAME frames with the SAME truth.

  python t6_evaluate.py                                             asks its questions
  python t6_evaluate.py --network crop_resnet18_v1_ft3 --yes        one command, all defaults (the numbers of the report)

Asks (Enter = [default]): recordings folder, camera file, network file, synthetic data set (for E5), output folder.
Test recordings: data/recordings/<name>.avi  +  <name>.truth.csv (hand-measured pose of M in F)  +  <name>.json (tag, notes)
  <name>.truth.csv   columns  t_s,x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg   one row = static pose, several rows = known motion
  tags (set in t3_record.py):  E1_static  E2_dist_tilt  E3_motion  E4_robust
Errors of the relative pose ^F T_M:  e_t = |t_est - t_true| [mm],  e_R = geodesic angle of R_true^T R_est [deg]  (mvlib.rot_error_deg).
The classical pipeline: ArUco + IPPE + LM for F and M.   The network pipeline: the network for M, the classical pipeline for F.
No test frame is used for training or tuning.  Both pipelines see raw frames (no filter) so that the comparison is fair.
Writes  <output>/comparison_table.md / .csv        the table of the assignment (median and 95th percentile)
        <output>/report.md, summary.json           all experiments (E1-E6), per recording
        <output>/worst_frames.csv and worst/*.png  the worst frames of each method
        <output>/E3_<recording>.png                e_t(t) and e_R(t) of both methods
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

from config import data_path, OUTPUT_DIR, training_path
from mvlib import rot_error_deg
from mvtools import (MarkerPoseEstimator, ask, ask_folder, cols_to_pose, default_camera_path, load_camera, load_markers_cfg, load_truth,
                     pose_errors, read_csv, summarize, truth_pose, write_csv, rpy_deg)
from t3_pose import process_video

METHODS = ("classical", "network")


def fmt(s, key_a="median", key_b="p95", d=1):
    """'median (p95)' for the table, '-' if there is no data."""
    return "-" if s["n"] == 0 else f"{s[key_a]:.{d}f} ({s[key_b]:.{d}f})"


def jitter_mm(rows):
    """RMS scatter of the position around its own mean over the frames with a pose [mm]."""
    p = np.array([[r["FM_x_mm"], r["FM_y_mm"], r["FM_z_mm"]] for r in rows], float)
    p = p[np.isfinite(p).all(axis=1)]
    return float(np.sqrt(np.mean(np.sum((p - p.mean(0)) ** 2, axis=1)))) if len(p) > 1 else np.nan


def bias_mm(rows, truth):
    p = np.array([[r["FM_x_mm"], r["FM_y_mm"], r["FM_z_mm"]] for r in rows], float)
    ok = np.isfinite(p).all(axis=1)
    if ok.sum() == 0:
        return np.nan
    t_true = np.mean([1000 * truth_pose(truth, float(r["t_s"]))[:3, 3] for r, o in zip(rows, ok) if o], axis=0)
    return float(np.linalg.norm(p[ok].mean(0) - t_true))


def errors_from_rows(rows, truth):
    et, er = [], []
    for r in rows:
        a, b = pose_errors(cols_to_pose(r, "FM"), truth_pose(truth, float(r["t_s"])))
        et.append(a); er.append(b)
    return np.array(et), np.array(er)


def run_recordings(rec_dir, cfg, K, dist, net, still=True, log=print):
    """Both pipelines on every recording that has a truth file.  Returns a list of dicts."""
    out = []
    for truth_path in sorted(glob.glob(os.path.join(rec_dir, "*.truth.csv"))):
        name = os.path.basename(truth_path)[:-len(".truth.csv")]
        video = next((os.path.join(rec_dir, name + e) for e in (".avi", ".mp4") if os.path.exists(os.path.join(rec_dir, name + e))), None)
        if video is None:
            log(f"  {name}: no video, skipped")
            continue
        meta = json.load(open(os.path.join(rec_dir, name + ".json"))) if os.path.exists(os.path.join(rec_dir, name + ".json")) else {}
        truth = load_truth(truth_path)
        rec = dict(name=name, video=video, tag=meta.get("tag", "other"), notes=meta.get("notes", ""), distance_cm=meta.get("distance_cm"),
                   tilt_deg=meta.get("tilt_deg"))
        for m in METHODS:
            if m == "network" and net is None:
                continue
            rows, s = process_video(video, cfg, K, dist, still, net if m == "network" else None, verbose=False)
            et, er = errors_from_rows(rows, truth)
            rec[m] = dict(rows=rows, et=et, er=er, t=np.array([float(r["t_s"]) for r in rows]), summary=s, jitter=jitter_mm(rows),
                          bias=bias_mm(rows, truth), output_pct=100 * np.isfinite(et).mean())
        log(f"  {name} [{rec['tag']}]: " + "   ".join(
            f"{m}: e_t {summarize(rec[m]['et'])['median']:.1f} mm, e_R {summarize(rec[m]['er'])['median']:.2f} deg, output {rec[m]['output_pct']:.0f} %"
            for m in METHODS if m in rec))
        out.append(rec)
    return out


def pooled(recs, tags, method, key):
    v = [rec[method][key] for rec in recs if rec["tag"] in tags and method in rec]
    return np.concatenate(v) if v else np.array([])


def synthetic_eval(data_dir, net, cfg, log=print, max_images=None):
    """E5: both methods on held-out synthetic images with exact truth (camera frame ^C T_M)."""
    from synthdata import load_set
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    X, R, t = load_set(data_dir, "test")
    if max_images:
        X, R, t = X[:max_images], R[:max_images], t[:max_images]
    Kn = np.array(meta["K_net"])
    est = MarkerPoseEstimator(dict(cfg, id_moving=meta["id_moving"], L=meta["L_m"]), Kn, np.zeros(5), camera_still=False)
    res = {}
    et, er, t0 = [], [], time.perf_counter()
    for i in range(len(X)):
        T = est.process(X[i])["T_CM"]
        a = np.nan if T is None else 1000 * np.linalg.norm(T[:3, 3] - t[i])
        b = np.nan if T is None else rot_error_deg(R[i].astype(float), T[:3, :3])
        et.append(a); er.append(b)
    res["classical"] = dict(et=np.array(et), er=np.array(er), ms=1000 * (time.perf_counter() - t0) / len(X))
    if net is not None:
        pr, pt = [], []
        for s in range(0, len(X), 64):
            r_, t_ = net.predict_batch([np.ascontiguousarray(x) for x in X[s:s + 64]], undistort=False)   # synthetic: no lens distortion
            pr.append(r_); pt.append(t_)
        pr, pt = np.concatenate(pr), np.concatenate(pt)
        res["network"] = dict(et=1000 * np.linalg.norm(pt - t, axis=1), er=np.array([rot_error_deg(R[i].astype(float), pr[i]) for i in range(len(X))]))
    for m, d in res.items():
        log(f"  E5 {m}: output {100 * np.isfinite(d['et']).mean():.0f} %, e_t median {summarize(d['et'])['median']:.1f} mm, e_R median {summarize(d['er'])['median']:.2f} deg")
    return res


def count_lines(files):
    n = 0
    for f in files:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f)
        if os.path.exists(p):
            n += sum(1 for l in open(p, encoding="utf-8") if l.strip() and not l.strip().startswith("#"))
    return n


def cost_table(recs, net, net_path, sample_gray):
    c = dict(classical_ms_per_frame=float(np.mean([r["classical"]["summary"]["ms_per_frame"] for r in recs])) if recs else np.nan,
             loc_classical=count_lines(["mvlib.py", "t3_pose.py"]), loc_network=count_lines(["netpose.py", "synthdata.py", "t5_make_data.py", "t5_train.py"]))
    c["classical_fps"] = 1000 / c["classical_ms_per_frame"] if recs else np.nan
    if net is not None:
        c["network_gpu_ms"] = net.time_per_frame_ms(sample_gray) if net.device.type == "cuda" else np.nan
        from netpose import NetPose
        cpu = NetPose(net_path, device="cpu")
        c["network_cpu_ms"] = cpu.time_per_frame_ms(sample_gray, n=20)
        c["network_fps_gpu"], c["network_fps_cpu"] = 1000 / c["network_gpu_ms"], 1000 / c["network_cpu_ms"]
        c["pipeline_network_ms_per_frame"] = float(np.mean([r["network"]["summary"]["ms_per_frame"] for r in recs if "network" in r])) if recs else np.nan
        c["model_params_M"] = net.cfg["n_params"] / 1e6
        c["model_file_MB"] = os.path.getsize(net_path) / 1e6
        c["train_minutes"] = net.cfg["train_minutes"]
        c["train_device"] = net.cfg["device"]
        c["n_train_images"] = net.cfg["n_train"]
    return c


def worst_frames(recs, out_dir, k=5, save_images=3):
    os.makedirs(os.path.join(out_dir, "worst"), exist_ok=True)
    rows = []
    for rec in recs:
        for m in METHODS:
            if m not in rec:
                continue
            et = np.nan_to_num(rec[m]["et"], nan=-1)
            for rank, i in enumerate(np.argsort(-et)[:k]):
                if et[i] < 0:
                    continue
                rows.append(dict(recording=rec["name"], method=m, frame=int(i), e_t_mm=round(float(et[i]), 2), e_R_deg=round(float(rec[m]["er"][i]), 2)))
                if rank < save_images:
                    cap = cv2.VideoCapture(rec["video"]); cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read(); cap.release()
                    if ok:
                        cv2.putText(f, f"{m}  frame {i}  e_t {et[i]:.1f} mm  e_R {rec[m]['er'][i]:.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.imwrite(os.path.join(out_dir, "worst", f"{m}_{rec['name']}_frame{i}.png"), f)
    if rows:
        write_csv(os.path.join(out_dir, "worst_frames.csv"), rows)
    return rows


def motion_plots(recs, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for rec in recs:
        if rec["tag"] != "E3_motion":
            continue
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
        for m, c in zip(METHODS, ("#1F3A5F", "#D9731A")):
            if m in rec:
                ax[0].plot(rec[m]["t"], rec[m]["et"], color=c, lw=1, label=m); ax[1].plot(rec[m]["t"], rec[m]["er"], color=c, lw=1, label=m)
        ax[0].set_ylabel("e_t [mm]"); ax[1].set_ylabel("e_R [deg]")
        for a in ax:
            a.set_xlabel("t [s]"); a.grid(alpha=0.3); a.legend()
        fig.suptitle(rec["name"]); fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"E3_{rec['name']}.png"), dpi=120); plt.close(fig)


def build_report(recs, synth, cost, out_dir):
    E1, E3, E4 = ("E1_static",), ("E3_motion",), ("E4_robust",)
    table = []
    for m in METHODS:
        if not any(m in r for r in recs):
            continue
        s1t, s1r = summarize(pooled(recs, E1, m, "et")), summarize(pooled(recs, E1, m, "er"))
        jit = [r[m]["jitter"] for r in recs if r["tag"] in E1 and m in r and np.isfinite(r[m]["jitter"])]
        s3t, s3r = summarize(pooled(recs, E3, m, "et")), summarize(pooled(recs, E3, m, "er"))
        e4 = [r[m]["output_pct"] for r in recs if r["tag"] in E4 and m in r]
        fps = (cost["classical_fps"] if m == "classical" else 1000 / cost["pipeline_network_ms_per_frame"]) if cost else np.nan
        table.append({"Method": m.capitalize(), "E1 e_t [mm]": fmt(s1t), "E1 e_R [deg]": fmt(s1r, d=2), "E1 jitter [mm]": f"{np.mean(jit):.2f}" if jit else "-",
                      "E3 e_t [mm]": fmt(s3t), "E3 e_R [deg]": fmt(s3r, d=2), "E4 output [%]": f"{np.mean(e4):.0f}" if e4 else "-",
                      "E6 [fps]": "-" if not np.isfinite(fps) else f"{fps:.0f}"})
    write_csv(os.path.join(out_dir, "comparison_table.csv"), table) if table else None
    L = ["# Task 6 - comparison", "", "Real test set; median with the 95th percentile in brackets.", ""]
    if table:
        L += ["| " + " | ".join(table[0]) + " |", "|" + "---|" * len(table[0])] + ["| " + " | ".join(r.values()) + " |" for r in table]
    L += ["", "## E1 static accuracy (bias and jitter per recording)", "", "| recording | method | bias [mm] | jitter [mm] | e_t median [mm] | e_R median [deg] |", "|---|---|---|---|---|---|"]
    for r in recs:
        if r["tag"] == "E1_static":
            for m in METHODS:
                if m in r:
                    L.append(f"| {r['name']} | {m} | {r[m]['bias']:.2f} | {r[m]['jitter']:.2f} | {summarize(r[m]['et'])['median']:.2f} | {summarize(r[m]['er'])['median']:.2f} |")
    L += ["", "## E2 distance and tilt", "", "| recording | distance [cm] | tilt [deg] | method | output [%] | e_t median [mm] | e_R median [deg] |", "|---|---|---|---|---|---|---|"]
    for r in sorted((r for r in recs if r["tag"] == "E2_dist_tilt"), key=lambda r: (r["distance_cm"] or 0, r["tilt_deg"] or 0)):
        for m in METHODS:
            if m in r:
                L.append(f"| {r['name']} | {r['distance_cm']} | {r['tilt_deg']} | {m} | {r[m]['output_pct']:.0f} | {summarize(r[m]['et'])['median']:.1f} | {summarize(r[m]['er'])['median']:.2f} |")
    L += ["", "## E3 motion (known path)", "", "| recording | method | RMSE [mm] | median [mm] | max [mm] | e_R median [deg] | e_R max [deg] |", "|---|---|---|---|---|---|---|"]
    for r in recs:
        if r["tag"] == "E3_motion":
            for m in METHODS:
                if m in r:
                    a, b = summarize(r[m]["et"]), summarize(r[m]["er"])
                    L.append(f"| {r['name']} | {m} | {a['rmse']:.1f} | {a['median']:.1f} | {a['max']:.1f} | {b['median']:.2f} | {b['max']:.1f} |")
    L += ["", "## E4 robustness", "", "| recording | condition (notes) | method | output [%] | e_t median [mm] | e_R median [deg] |", "|---|---|---|---|---|---|"]
    for r in recs:
        if r["tag"] == "E4_robust":
            for m in METHODS:
                if m in r:
                    L.append(f"| {r['name']} | {r['notes']} | {m} | {r[m]['output_pct']:.0f} | {summarize(r[m]['et'])['median']:.1f} | {summarize(r[m]['er'])['median']:.2f} |")
    if synth:
        L += ["", "## E5 synthetic held-out set (exact truth, camera frame) and the synthetic-real gap", "", "| method | output [%] | e_t median (p95) [mm] | e_R median (p95) [deg] |", "|---|---|---|---|"]
        for m, d in synth.items():
            L.append(f"| {m} | {100 * np.isfinite(d['et']).mean():.0f} | {fmt(summarize(d['et']))} | {fmt(summarize(d['er']), d=2)} |")
    if cost:
        L += ["", "## E6 cost (same laptop)", ""] + [f"- {k}: {v:.2f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in cost.items()]
    open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    open(os.path.join(out_dir, "comparison_table.md"), "w", encoding="utf-8").write("\n".join(L[:L.index("## E1 static accuracy (bias and jitter per recording)") - 1]) + "\n")
    return table


def evaluate(rec_dir, cfg, K, dist, net, net_path, synth_dir, out_dir, still=True, log=print):
    os.makedirs(out_dir, exist_ok=True)
    log("recordings:")
    recs = run_recordings(rec_dir, cfg, K, dist, net, still, log)
    synth = synthetic_eval(synth_dir, net, cfg, log) if synth_dir and os.path.exists(os.path.join(synth_dir, "meta.json")) else None
    cost = None
    if recs:
        cap = cv2.VideoCapture(recs[0]["video"]); ok, f = cap.read(); cap.release()
        cost = cost_table(recs, net, net_path, cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) if ok else None
    worst_frames(recs, out_dir)
    motion_plots(recs, out_dir)
    table = build_report(recs, synth, cost, out_dir)
    json.dump(dict(table=table, cost=cost, synthetic={m: dict(output_pct=float(100 * np.isfinite(d['et']).mean()), et=summarize(d['et']), er=summarize(d['er']))
                                                         for m, d in (synth or {}).items()}), open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=float)
    return recs, synth, cost, table


def main():
    ap = argparse.ArgumentParser(description="Task 6: classical vs network.")
    ap.add_argument("--network", default="resnet18_v1", help="network in training/models (name without .pt), e.g. crop_resnet18_v1_ft3")
    ap.add_argument("--yes", action="store_true", help="no questions: every default is accepted (with --network this is one command)")
    a = ap.parse_args()
    if a.yes:                                                     # same defaults, nothing asked
        global ask, ask_folder
        ask = ask_folder = lambda text, default, *rest: default
    cfg = load_markers_cfg()
    print("Task 6 - evaluation.  Press Enter to accept the [default].\n")
    rec_dir = ask_folder("1) Folder with the test recordings (.avi + .truth.csv + .json)", data_path("recordings"))
    K, dist, size = load_camera(ask_folder("2) Camera file", default_camera_path()))
    net_path = ask_folder("3) Network file (type - to compare nothing but the classical pipeline)", training_path("models", a.network + ".pt"))
    net = None
    if os.path.exists(net_path):
        from netpose import NetPose
        net = NetPose(net_path, K=K, dist=dist)
    else:
        print("   no network file: only the classical pipeline is evaluated")
    sets = sorted(glob.glob(training_path("synthetic", "*", "meta.json")))
    synth_dir = ask_folder("4) Synthetic data set for E5 (its test split)", os.path.dirname(sets[-1]) if sets else training_path("synthetic", "synth_v1"))
    out_dir = ask_folder("5) Output folder", os.path.join(OUTPUT_DIR, "evaluation"))
    still = ask("6) Was the camera still in the recordings? (y/n)", "y").lower().startswith("y")
    print()
    recs, synth, cost, table = evaluate(rec_dir, cfg, K, dist, net, net_path, synth_dir, out_dir, still)
    print("\n" + "\n".join("  ".join(f"{k}: {v}" for k, v in r.items()) for r in table))
    print(f"\nreport -> {os.path.join(out_dir, 'report.md')}")


if __name__ == "__main__":
    main()
