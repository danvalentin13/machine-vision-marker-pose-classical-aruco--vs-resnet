"""t5_test_hard_cases.py -- usability: classical against the networks on test frames degraded in software.

  python t5_test_hard_cases.py

Takes every second frame of E1_p1..p5 (never used in training) and degrades it: part of M covered, motion blur, low resolution,
dim light.  Reference = the classical pose of the CLEAN frame; usable = within 20 mm and 10 deg of it.
Reports, per case: how often classical gives a pose / a usable pose, and the same for the two-stage networks (_ft2, _ft3).
Note: simulated degradations of the same kind as the training augmentation (t5_train.hard_cases) - real hard-case recordings
are the stronger test.  Writes output/network/hard_cases_E1.json and example images in output/network/figures.
"""
import sys, json, os
import cv2, numpy as np, mvtools as mt
from config import data_path, training_path, OUTPUT_DIR
from netpose import NetPose
from mvlib import marker_object_points, rot_error_deg
cfg = mt.load_markers_cfg(); K, dist, _ = mt.load_camera()
obj = marker_object_points(cfg["L"]).astype(np.float64)
rng = np.random.default_rng(0)
def cover(g, T, frac):
    uv = cv2.projectPoints(obj, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, dist)[0].reshape(-1, 2)
    c = uv.mean(0); k = rng.integers(4); p = uv[k]                       # from a corner toward the centre
    q = p + (c - p) * 2 * np.sqrt(frac)
    x0, x1, y0, y1 = int(min(p[0], q[0])) - 6, int(max(p[0], q[0])) + 6, int(min(p[1], q[1])) - 6, int(max(p[1], q[1])) + 6
    g = g.copy(); g[max(0, y0):y1, max(0, x0):x1] = 120; return g
def mblur(g, n):
    k = np.zeros((n, n), np.float32); k[n // 2, :] = 1.0 / n; return cv2.filter2D(g, -1, k)
def lowres(g, f):
    s = cv2.resize(g, (g.shape[1] // f, g.shape[0] // f), interpolation=cv2.INTER_AREA); return cv2.resize(s, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
def dim(g, a):
    return np.clip(g.astype(np.float32) * a + rng.normal(0, 3, g.shape), 0, 255).astype(np.uint8)
CASES = [("clean", lambda g, T: g), ("cover 15 % of M", lambda g, T: cover(g, T, 0.15)), ("cover 30 % of M", lambda g, T: cover(g, T, 0.30)),
         ("motion blur 15 px", lambda g, T: mblur(g, 15)), ("motion blur 31 px", lambda g, T: mblur(g, 31)),
         ("resolution / 4", lambda g, T: lowres(g, 4)), ("resolution / 6", lambda g, T: lowres(g, 6)),
         ("light x 0.25", lambda g, T: dim(g, 0.25)), ("light x 0.10", lambda g, T: dim(g, 0.10))]
frames, ref = [], []
for r in [f"E1_p{k}" for k in range(1, 6)]:
    est = mt.MarkerPoseEstimator(cfg, K, dist, camera_still=False); cap = cv2.VideoCapture(data_path("recordings", r + ".avi")); n = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); res = est.process(g)
        if n % 2 == 0 and res["T_CM"] is not None: frames.append(g); ref.append(res["T_CM"].copy())
        n += 1
nets = {"_ft2": NetPose(training_path("models", "crop_resnet18_v1_ft2.pt"), K=K, dist=dist, passes=2),
        "_ft3": NetPose(training_path("models", "crop_resnet18_v1_ft3.pt"), K=K, dist=dist, passes=2)}
def errs(R, t, T):
    return 1000 * np.linalg.norm(t - T[:3, 3]), rot_error_deg(T[:3, :3], R)
out = {}
print(f"{len(frames)} frames of E1_p1..p5; reference = classical pose of the clean frame; usable = within 20 mm and 10 deg of it\n")
print(f"{'case':20s} | classical: output  usable | " + " | ".join(f"{n}: usable  median" for n in nets))
for name, fn in CASES:
    G = [fn(g, T) for g, T in zip(frames, ref)]
    est = mt.MarkerPoseEstimator(cfg, K, dist, camera_still=False); got = us = 0
    for g, T in zip(G, ref):
        r = est.process(g)
        if r["T_CM"] is not None:
            got += 1; e = errs(r["T_CM"][:3, :3], r["T_CM"][:3, 3], T); us += e[0] < 20 and e[1] < 10
    row = dict(classical_output=100 * got / len(G), classical_usable=100 * us / len(G)); line = f"{name:20s} |        {row['classical_output']:5.1f} %  {row['classical_usable']:5.1f} % | "
    for n, net in nets.items():
        E = []
        for s in range(0, len(G), 32):
            R, t = net.predict_batch(G[s:s + 32])
            E += [errs(R[i], t[i], ref[s + i]) for i in range(len(R))]
        E = np.array(E); u = 100 * np.mean((E[:, 0] < 20) & (E[:, 1] < 10))
        row[n] = dict(usable=u, et_med=float(np.median(E[:, 0])), er_med=float(np.median(E[:, 1])))
        line += f"{u:5.1f} %  {np.median(E[:, 0]):5.1f} mm / {np.median(E[:, 1]):4.1f} deg | "
    out[name] = row; print(line)
    if name in ("cover 30 % of M", "motion blur 31 px", "resolution / 6", "light x 0.10"):
        cv2.imwrite(os.path.join(OUTPUT_DIR, "network", "figures", f"test_degraded_{name.split()[0]}.jpg"), G[10][150:570, 300:980])
json.dump(out, open(os.path.join(OUTPUT_DIR, "network", "hard_cases_E1.json"), "w"), indent=1)
