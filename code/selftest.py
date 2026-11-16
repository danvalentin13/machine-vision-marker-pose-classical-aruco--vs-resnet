"""selftest.py -- end-to-end test of the whole pipeline on SYNTHETIC data (no camera, no printouts needed).

  python selftest.py            runs in a temporary folder and deletes it afterwards (your data/ and output/ are not touched)
  python selftest.py --keep     keeps the temporary folder and prints where it is

Steps:  T1 markers -> T3 chessboard + calibration of simulated views -> synthetic recording with exact truth ->
        T3 poses (compared with the truth) -> T4 animation -> bonus (prediction through an obstacle) -> T5 synthetic data set, label check, network code check
        (a throwaway net of a few steps, saved only in the temp folder; NOT a trained model) -> T6 evaluation of both pipelines.
A step prints PASS / FAIL with the numbers it checked.
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

# all folders of this run are temporary: set before config is imported
# (worker processes of the data generator re-import this file as __mp_main__: they must not create folders)
_ap = argparse.ArgumentParser()
_ap.add_argument("--keep", action="store_true")
_ap.add_argument("--quick", action="store_true", help="skip the slower steps (animation, network)")
if __name__ == "__main__":
    ARGS = _ap.parse_args()
    TMP = tempfile.mkdtemp(prefix="mv_selftest_")
    os.environ["MV_DATA_DIR"] = os.path.join(TMP, "data")
    os.environ["MV_OUTPUT_DIR"] = os.path.join(TMP, "output")
    os.environ["MV_TRAINING_DIR"] = os.path.join(TMP, "training")
else:
    ARGS, TMP = argparse.Namespace(keep=False, quick=True), None

import cv2                      # noqa: E402
import numpy as np              # noqa: E402
from PIL import Image           # noqa: E402

from config import data_path, output_path, DATA_DIR, OUTPUT_DIR      # noqa: E402
import mvtools as mt            # noqa: E402
import synth                    # noqa: E402
from mvlib import inv_T         # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# ----------------------------------------------------------------------------------------------
def step_t1():
    print("T1  markers")
    import t1_generate_markers as t1
    t1.generate(os.path.join(OUTPUT_DIR, "markers"), os.path.join(OUTPUT_DIR, "dictionary"), "DICT_6X6_250", 0, 23, 100.0)
    cfg = mt.load_markers_cfg()
    check("markers.json written", cfg["id_fixed"] == 0 and cfg["id_moving"] == 23, f"L = {cfg['L'] * 1000:g} mm")
    return cfg


def step_calibration():
    print("T3  chessboard + calibration (simulated views, known camera)")
    import t3_make_board as mb
    import t3_calibrate as tc
    img, dpi = mb.make_board_image(9, 6, 25.0)
    import json
    json.dump(dict(cols=9, rows=6, square_mm=25.0, measured_square_mm=None), open(data_path("calibration", "board.json"), "w")) \
        if os.makedirs(data_path("calibration"), exist_ok=True) is None else None
    board = tc.load_board()
    check("board corners found in the board image", cv2.findChessboardCornersSB(img, (9, 6))[0])
    H, W = img.shape
    mm_px = 25.4 / dpi
    page = np.array([[-W / 2, -H / 2, 0], [W / 2, -H / 2, 0], [W / 2, H / 2, 0], [-W / 2, H / 2, 0]]) * mm_px / 1000
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    rng = np.random.default_rng(3)
    imgdir = data_path("calibration", "images")
    os.makedirs(imgdir, exist_ok=True)
    from mvlib import expm_so3
    n = 0
    while n < 14:
        tilt, az = np.radians(rng.uniform(0 if n < 2 else 15, 45)), rng.uniform(0, 2 * np.pi)
        R = expm_so3(tilt * np.array([np.cos(az), np.sin(az), 0]))
        t = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.05, 0.05), rng.uniform(0.45, 0.7)])
        uv = (synth.K @ ((R @ page.T).T + t).T).T
        uv = uv[:, :2] / uv[:, 2:3]
        frame = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, np.float32(uv)), (synth.W, synth.H), flags=cv2.INTER_AREA, borderValue=200)
        frame = np.clip(frame.astype(float) + rng.normal(0, 2, frame.shape), 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(imgdir, f"view_{n + 1:03d}.png"), frame)
        n += 1
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        rep = tc.run_calibration(imgdir, board, "k1k2", output_path("calibration", "x"), data_path("calibration", "camera.npz"))
    err = abs(rep["fx"] - synth.K[0, 0]) / synth.K[0, 0] * 100
    check("calibration recovers fx", err < 1.0, f"fx {rep['fx']:.1f} vs {synth.K[0, 0]:.0f} ({err:.2f} %), RMS {rep['rms_px']:.2f} px")
    check("calibration recovers the principal point", abs(rep["cx"] - 640) < 3 and abs(rep["cy"] - 360) < 3, f"cx {rep['cx']:.1f} cy {rep['cy']:.1f}")
    check("held-out views have a similar error", rep["held_out"] is not None and rep["held_out"]["rms_mean"] < 2 * rep["rms_px"] + 0.2)


def make_recording(name, n_frames=150, fps=30.0, noise=3.0, hide=None):
    """Synthetic recording of the lecture scene with exact truth: video + <name>.csv + <name>.truth.csv + <name>.json."""
    import json
    rec_dir = data_path("recordings")
    os.makedirs(rec_dir, exist_ok=True)
    rend = synth.Renderer(0)
    T_CF = inv_T(synth.camera_pose(0.0, False))
    base = os.path.join(rec_dir, name)
    w = cv2.VideoWriter(base + ".avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (synth.W, synth.H))
    truth = []
    for k in range(n_frames):
        t = k / fps
        T_FM = synth.moving_marker_pose(t)
        poses = {synth.ID_FIXED: T_CF, synth.ID_MOVING: T_CF @ T_FM}
        if hide and hide[0] <= k < hide[1]:
            poses.pop(synth.ID_MOVING)
        w.write(cv2.cvtColor(rend.render(poses, noise_sigma=noise), cv2.COLOR_GRAY2BGR))
        truth.append((t, T_FM))
    w.release()
    open(base + ".csv", "w").write("frame,t_s\n" + "\n".join(f"{k},{k / fps:.4f}" for k in range(n_frames)) + "\n")
    mt.write_csv(base + ".truth.csv", [dict(t_s=round(t, 4), x_mm=1000 * T[0, 3], y_mm=1000 * T[1, 3], z_mm=1000 * T[2, 3],
                                            roll_deg=mt.rpy_deg(T[:3, :3])[0], pitch_deg=mt.rpy_deg(T[:3, :3])[1], yaw_deg=mt.rpy_deg(T[:3, :3])[2])
                                       for t, T in truth])
    json.dump(dict(name=name, tag="E3_motion", notes="synthetic", resolution=f"{synth.W}x{synth.H}", frames=n_frames, fps=fps),
              open(base + ".json", "w"))
    return base + ".avi"


def step_pose(cfg):
    print("T3  poses on a synthetic recording (exact truth)")
    import t3_pose
    mt.save_camera(data_path("calibration", "camera_synth.npz"), synth.K, np.zeros(5), (synth.W, synth.H), 0.0)
    video = make_recording("synth_motion", hide=(60, 75))
    K, dist, size = mt.load_camera(data_path("calibration", "camera_synth.npz"))
    cfg = dict(cfg, L=synth.L)                                                    # the synthetic markers are 100 mm
    t0 = time.time()
    rows, s = t3_pose.process_video(video, cfg, K, dist, True, None, verbose=False)
    truth = mt.load_truth(data_path("recordings", "synth_motion.truth.csv"))
    et, er = [], []
    for r in rows:
        T = mt.cols_to_pose(r, "FM")
        a, b = mt.pose_errors(T, mt.truth_pose(truth, float(r["t_s"])))
        et.append(a); er.append(b)
    et, er = np.array(et), np.array(er)
    check("M found in the frames where it is visible", s["M_found_pct"] > 85, f"{s['M_found_pct']:.0f} % found (15 frames hidden on purpose)")
    check("hidden frames give no pose (no invented values)", np.isnan(et[60:75]).all())
    check("relative position error", np.nanmedian(et) < 10, f"median {np.nanmedian(et):.1f} mm, max {np.nanmax(et):.1f} mm")
    check("relative rotation error", np.nanmedian(er) < 3, f"median {np.nanmedian(er):.2f} deg, max {np.nanmax(er):.1f} deg")
    t3_pose_csv = output_path("pose", "synth_motion_poses.csv")
    mt.write_csv(t3_pose_csv, [{c: r.get(c, "") for c in rows[0]} for r in rows], list(rows[0]))
    return video, t3_pose_csv, rows


def step_animation(video, poses_csv):
    print("T4  animation")
    import t4_animate as t4
    rows, t, poses = t4.load_poses_csv(poses_csv)
    out = os.path.join(OUTPUT_DIR, "animation", "synth")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    files = t4.export(rows, t, poses, video, out, ("mp4", "gif"), target_fps=10, with_filter=True, verbose=False)
    sizes = [os.path.getsize(f) for f in files]
    check("mp4 and gif written", len(files) == 2 and min(sizes) > 10_000, ", ".join(f"{os.path.basename(f)} {s / 1e6:.1f} MB" for f, s in zip(files, sizes)))
    cap = cv2.VideoCapture(out + ".mp4")
    ok, fr = cap.read()
    check("mp4 can be read back (video + 3D view + curves side by side)", ok and fr.shape[1] > 1200, f"first frame {fr.shape[1]}x{fr.shape[0]}" if ok else "")
    cap.release()
    filt, pred = t4.filter_poses(t, poses)
    check("filter marks the hidden frames as predicted", any(pred[60:75]) and sum(p is not None for p in filt) > len(filt) * 0.9)


def step_bonus(cfg):
    print("Bonus  prediction through an obstacle (M hidden for 0.5 s and for 1.5 s, exact truth)")
    import t3_pose
    import t4_animate as t4
    import t7_bonus_occlusion as t7
    K, dist, size = mt.load_camera(data_path("calibration", "camera_synth.npz"))
    cfg = dict(cfg, L=synth.L)
    make_recording("synth_occ", hide=(50, 95))
    rows, _ = t3_pose.process_video(data_path("recordings", "synth_occ.avi"), cfg, K, dist, True, None, verbose=False)
    mt.write_csv(output_path("pose", "synth_occ_poses.csv"), [{c: r.get(c, "") for c in rows[0]} for r in rows], list(rows[0]))
    out = os.path.join(OUTPUT_DIR, "occlusion")
    pcs = [output_path("pose", "synth_motion_poses.csv"), output_path("pose", "synth_occ_poses.csv")]
    eps = t7.evaluate(pcs, data_path("recordings"), synth.L, out, lost_factor=1.0, log=lambda *_: None)
    short = [e for e in eps if e["hidden_s"] < 1.0]
    long_ = [e for e in eps if e["hidden_s"] >= 1.0]
    check("both hidden episodes found", len(short) == 1 and len(long_) == 1, f"hidden {[round(e['hidden_s'], 2) for e in eps]} s")
    e = short[0]
    check("0.5 s: prediction beats 'hold the last pose'", t7.at_time(e, "et", 0.5) < 0.5 * t7.at_time(e, "et_hold", 0.5),
          f"predictor {t7.at_time(e, 'et', 0.5):.1f} mm vs hold {t7.at_time(e, 'et_hold', 0.5):.1f} mm")
    check("0.5 s: prediction error small, pose not declared lost", t7.at_time(e, "et", 0.5) < 30 and e["lost_s"] is None)
    check("re-acquisition after 0.5 s without a jump", e["jump_mm"] < 25 and e["jump_deg"] < 10, f"jump {e['jump_mm']:.1f} mm, {e['jump_deg']:.1f} deg ({e['reacq']})")
    l = long_[0]
    check("1.5 s: pose declared LOST when 3 sigma exceeds the threshold (set to 1 L here; the default is 2 L)", l["lost_s"] is not None and 0.3 < l["lost_s"] < 1.5,
          f"lost after {l['lost_s']} s; 3 sigma at 1 s = {t7.at_time(l, 'sigma3', 1.0):.0f} mm")
    check("tracker recovers after the lost pose", "re-acquired" in l["reacq"], l["reacq"])
    check("bonus outputs written", all(os.path.exists(os.path.join(out, f)) for f in ("summary.md", "episodes.csv", "occlusion_curves.png", "synth_occ_predicted_poses.csv")))
    rows2, t, poses = t4.load_poses_csv(os.path.join(out, "synth_occ_predicted_poses.csv"))
    files = t4.export(rows2, t, poses, None, os.path.join(OUTPUT_DIR, "animation", "synth_occ"), ("mp4",), target_fps=10, verbose=False)
    check("animation with predicted (dashed) segment and uncertainty", os.path.exists(files[0]) and os.path.getsize(files[0]) > 10_000)


def main():
    t0 = time.time()
    print(f"Self-test in {TMP}\n")
    try:
        cfg = step_t1()
        step_calibration()
        video, poses_csv, rows = step_pose(cfg)
        if not ARGS.quick:
            step_animation(video, poses_csv)
            step_bonus(cfg)
        if not ARGS.quick:
            import selftest_learned
            selftest_learned.run(check, dict(cfg, L=synth.L))
    finally:
        n_fail = sum(not ok for _, ok in RESULTS)
        print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed in {time.time() - t0:.0f} s")
        if ARGS.keep:
            print("kept:", TMP)
        else:
            shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
