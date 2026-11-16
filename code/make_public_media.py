"""make_public_media.py -- animations that are safe to publish: only the computed 3D view and curves, NO camera picture.

  python make_public_media.py

The other animations (output/animation, output/bonus, output/evaluation/side_by_side_*) contain camera frames and are kept out of
the repository on purpose.  This script renders the same pose data without the camera panel:
  output/public/T4_motion_1_classical.mp4      T4: pose of M in the frame of F, classical pipeline
  output/public/T4_motion_1_network.mp4        T4: the same recording, M from the two-stage network
  output/public/E7_box_5cms_2_predicted.mp4    bonus: measured pose + predicted segments (dashed, with the 3 sigma circles)
and a half-size GIF of each.
"""
import os

import numpy as np

from config import OUTPUT_DIR, data_path, training_path
from mvtools import cols_to_pose, load_camera, load_markers_cfg
from t4_animate import export, load_poses_csv


def main():
    out = os.path.join(OUTPUT_DIR, "public")
    os.makedirs(out, exist_ok=True)
    cfg = load_markers_cfg()
    K, dist, _ = load_camera()
    done = []
    p = os.path.join(OUTPUT_DIR, "pose", "T4_motion_1_poses.csv")
    if os.path.exists(p):                                             # classical poses of the T4 recording (README step 4.2)
        rows, t, poses = load_poses_csv(p)
        done += export(rows, t, poses, None, os.path.join(out, "T4_motion_1_classical"), formats=("mp4", "gif"), with_filter=False, verbose=False)
    video = data_path("recordings", "T4_motion_1.avi")
    net_file = training_path("models", "crop_resnet18_v1_ft3.pt")
    if os.path.exists(video) and os.path.exists(net_file):            # the same recording with the network
        import t3_pose
        from netpose import NetPose
        rows, _ = t3_pose.process_video(video, cfg, K, dist, True, NetPose(net_file, K=K, dist=dist), verbose=False)
        t = np.loadtxt(data_path("recordings", "T4_motion_1.csv"), delimiter=",", skiprows=1)[:len(rows), 1]
        rows = rows[:len(t)]
        done += export(rows, t, [cols_to_pose(r, "FM") for r in rows], None, os.path.join(out, "T4_motion_1_network"), formats=("mp4", "gif"),
                       with_filter=False, verbose=False)
    p = os.path.join(OUTPUT_DIR, "bonus", "E7_box_5cms_2_predicted_poses.csv")
    if os.path.exists(p):                                             # bonus: measured + predicted (README step 7.7 writes this file)
        rows, t, poses = load_poses_csv(p)
        done += export(rows, t, poses, None, os.path.join(out, "E7_box_5cms_2_predicted"), formats=("mp4", "gif"), verbose=False)
    print("written (no camera picture in any of them):")
    for f in done:
        print("  ", f, f"{os.path.getsize(f) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
