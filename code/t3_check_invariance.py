"""t3_check_invariance.py -- Task 3, check B: move the LAPTOP, not the markers; ^F T_M must stay constant (camera invariance).

  python t3_check_invariance.py

Record F and M standing still while you move the laptop (t3_record.py), run t3_pose.py on the recording, then run this.
Asks (Enter = [default]): poses file (*_poses.csv from t3_pose.py), expected distance between the marker centres [mm] (optional).
Reports
  * how much the CAMERA moved: spread of ^C T_F (position of F seen from the camera) [mm] and of the viewing direction [deg]
  * how much the RELATIVE pose changed: ^F T_M position spread (RMS jitter, max deviation from the mean) [mm] and rotation [deg]
  * |^F t_M| (distance between the marker centres) against the expected value, if given
Writes  output/pose/<name>_invariance.png   camera motion and relative pose against time (same scale)
        output/pose/<name>_invariance.md    the numbers for the report
"""
import argparse
import glob
import os

import numpy as np

from config import OUTPUT_DIR
from mvlib import mean_rotation, rot_error_deg
from mvtools import ask, ask_folder, cols_to_pose, read_csv


def analyse(rows):
    t, cf, fm = [], [], []
    for r in rows:
        a, b = cols_to_pose(r, "CF"), cols_to_pose(r, "FM")
        if a is not None and b is not None and r.get("F_held") in ("0", 0, "", None):
            t.append(float(r["t_s"])); cf.append(a); fm.append(b)
    if len(fm) < 10:
        raise SystemExit(f"only {len(fm)} frames with both markers: record longer, keep F and M in view")
    t = np.array(t)
    p_cf = np.array([T[:3, 3] for T in cf]) * 1000
    p_fm = np.array([T[:3, 3] for T in fm]) * 1000
    R_cf_mean, R_fm_mean = mean_rotation([T[:3, :3] for T in cf]), mean_rotation([T[:3, :3] for T in fm])
    cam_rot = np.array([rot_error_deg(R_cf_mean, T[:3, :3]) for T in cf])
    fm_rot = np.array([rot_error_deg(R_fm_mean, T[:3, :3]) for T in fm])
    d_cf = np.linalg.norm(p_cf - p_cf.mean(0), axis=1)
    d_fm = np.linalg.norm(p_fm - p_fm.mean(0), axis=1)
    return dict(t=t, p_cf=p_cf, p_fm=p_fm, cam_rot=cam_rot, fm_rot=fm_rot, n=len(fm), n_rows=len(rows),
                cam_span_mm=float(np.max(np.linalg.norm(p_cf[:, None] - p_cf[None], axis=2))) if len(p_cf) < 3000 else float(np.ptp(p_cf, 0).max()),
                cam_rot_max=float(cam_rot.max()), cam_dist=(float(np.linalg.norm(p_cf, axis=1).min()), float(np.linalg.norm(p_cf, axis=1).max())),
                fm_mean=p_fm.mean(0), fm_jitter=float(np.sqrt(np.mean(d_fm ** 2))), fm_max=float(d_fm.max()),
                fm_rot_rms=float(np.sqrt(np.mean(fm_rot ** 2))), fm_rot_max=float(fm_rot.max()),
                fm_norm=float(np.linalg.norm(p_fm, axis=1).mean()), fm_norm_std=float(np.linalg.norm(p_fm, axis=1).std()))


def figure(a, path, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for k, (lab, col) in enumerate(zip("xyz", ("#D62728", "#2CA02C", "#1F77B4"))):
        ax[0].plot(a["t"], a["p_cf"][:, k] - a["p_cf"][:, k].mean(), color=col, lw=1, label=lab)
        ax[1].plot(a["t"], a["p_fm"][:, k] - a["p_fm"][:, k].mean(), color=col, lw=1, label=lab)
    ax[0].set_title("camera motion: position of F seen from the camera, minus its mean [mm]", fontsize=9)
    ax[1].set_title("relative pose: position of M in F, minus its mean [mm]", fontsize=9)
    for x in ax:
        x.set_xlabel("t [s]"); x.grid(alpha=0.3); x.legend(fontsize=8)
    fig.suptitle(f"{name}: camera invariance of the relative pose", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def report(a, name, expected_mm):
    L = [f"# Task 3 - camera invariance ({name})", "",
         f"{a['n']} of {a['n_rows']} frames with both markers (F not held). The markers stood still; the laptop was moved.", "",
         "| quantity | camera motion (^C T_F) | relative pose (^F T_M) |", "|---|---|---|",
         f"| position change | span {a['cam_span_mm']:.0f} mm (F seen at {a['cam_dist'][0] / 10:.0f}-{a['cam_dist'][1] / 10:.0f} cm) | "
         f"RMS jitter {a['fm_jitter']:.1f} mm, max deviation {a['fm_max']:.1f} mm |",
         f"| rotation change | up to {a['cam_rot_max']:.1f} deg | RMS {a['fm_rot_rms']:.2f} deg, max {a['fm_rot_max']:.1f} deg |", "",
         f"- Relative position of M in F (mean): x {a['fm_mean'][0]:.1f}, y {a['fm_mean'][1]:.1f}, z {a['fm_mean'][2]:.1f} mm; "
         f"distance between the centres {a['fm_norm']:.1f} mm (std {a['fm_norm_std']:.1f})."]
    if expected_mm:
        L.append(f"- Expected distance between the centres (measured by hand): {expected_mm:.1f} mm -> difference "
                 f"{a['fm_norm'] - expected_mm:+.1f} mm ({100 * (a['fm_norm'] - expected_mm) / expected_mm:+.1f} %).")
    ratio = a["cam_span_mm"] / max(a["fm_max"], 1e-6)
    verdict = ("shown: the camera moved " + f"{a['cam_span_mm']:.0f} mm / {a['cam_rot_max']:.0f} deg while ^F T_M stayed within "
               f"{a['fm_max']:.1f} mm / {a['fm_rot_max']:.1f} deg (ratio {ratio:.0f}:1)") if a["cam_span_mm"] > 50 or a["cam_rot_max"] > 10 \
        else "NOT shown: the camera hardly moved - move the laptop more (several cm and some degrees) and record again"
    L.append(f"- **Camera invariance: {verdict}.**")
    return "\n".join(L)


def main():
    argparse.ArgumentParser(description="Task 3 check B: camera invariance of the relative pose.").parse_args()
    print("Task 3, check B - camera invariance.  Press Enter to accept the [default].\n")
    newest = sorted(glob.glob(os.path.join(OUTPUT_DIR, "pose", "*_poses.csv")), key=os.path.getmtime)
    path = ask_folder("1) Poses file (from t3_pose.py)", newest[-1] if newest else os.path.join(OUTPUT_DIR, "pose", "name_poses.csv"))
    exp = ask("2) Expected distance between the marker centres [mm], measured by hand (Enter = skip)", "-")
    expected = None if exp.strip() in ("-", "") else float(exp.replace(",", "."))
    name = os.path.basename(path).replace("_poses.csv", "")
    a = analyse(read_csv(path))
    out = os.path.dirname(path)
    figure(a, os.path.join(out, name + "_invariance.png"), name)
    text = report(a, name, expected)
    open(os.path.join(out, name + "_invariance.md"), "w", encoding="utf-8").write(text + "\n")
    print("\n" + text + f"\n\nfigure -> {os.path.join(out, name + '_invariance.png')}\nreport -> {os.path.join(out, name + '_invariance.md')}")


if __name__ == "__main__":
    main()
