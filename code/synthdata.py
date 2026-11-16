"""synthdata.py -- renderer and generator of synthetic training / test images (used by t5_make_data.py, t6_evaluate.py, selftest.py).

A synthetic image shows the moving marker M (and, mostly, the fixed marker F as a distractor) on a random background, seen by a
pinhole camera K, with exact labels (R, t) = ^C T_M.  Randomised: pose, background, clutter, brightness, blur, noise, sometimes a
partial occlusion of M ("domain randomisation", Tobin et al. 2017).  The camera is the calibrated camera scaled to the network
input size; lens distortion is not simulated (real frames are undistorted before they go into the network, see netpose.py).
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from mvlib import expm_so3, marker_object_points

SS = 2                        # supersampling factor (anti-aliasing)
QUIET = 0.25                  # white quiet zone around the black square, in units of L (= 2 cells of a 6x6 marker)


class SyntheticRenderer:
    def __init__(self, dict_id, id_m, id_f, L, K, size):
        self.L, self.K, (self.W, self.H) = L, np.asarray(K, float), size
        d = cv2.aruco.getPredefinedDictionary(dict_id)
        pad = int(round(QUIET * 240))
        self.patch, self.id_m, self.id_f = {}, id_m, id_f
        for mid in (id_m, id_f):
            m = cv2.aruco.generateImageMarker(d, mid, 240)
            self.patch[mid] = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255).astype(np.float32)
        self.ps = self.patch[id_m].shape[0]
        half = L / 2 + QUIET * L
        self.phys = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
        self.src = np.float32([[-0.5, -0.5], [self.ps - 0.5, -0.5], [self.ps - 0.5, self.ps - 0.5], [-0.5, self.ps - 0.5]])
        self.mask_src = np.full((self.ps, self.ps), 255, np.uint8)

    # ---- poses --------------------------------------------------------------------------------
    def sample_pose(self, rng, z_range=(0.3, 1.2), tilt_max_deg=60.0):
        """Marker facing the camera, tilted by up to tilt_max about a random in-plane axis, any roll, centre in view."""
        Z = float(np.exp(rng.uniform(np.log(z_range[0]), np.log(z_range[1]))))
        a, phi = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
        tilt = np.radians(rng.uniform(0, tilt_max_deg))
        R = expm_so3(np.array([np.cos(a), np.sin(a), 0.0]) * tilt) @ np.diag([1.0, -1.0, -1.0]) @ expm_so3([0, 0, phi])
        fx = self.K[0, 0]
        m = 0.8 * fx * self.L / Z + 2                                  # keep the marker and its quiet zone inside the image
        u = rng.uniform(min(m, self.W / 2 - 1), max(self.W - 1 - m, self.W / 2))
        v = rng.uniform(min(m, self.H / 2 - 1), max(self.H - 1 - m, self.H / 2))
        return R, Z * np.linalg.solve(self.K, [u, v, 1.0])

    # ---- image ------------------------------------------------------------------------------------
    def _background(self, rng):
        W, H = self.W * SS, self.H * SS
        low = rng.normal(0, 1, (int(rng.integers(3, 12)), int(rng.integers(4, 16)))).astype(np.float32)
        img = rng.uniform(70, 200) + rng.uniform(5, 40) * cv2.resize(low, (W, H), interpolation=cv2.INTER_CUBIC)
        s = W / 320.0
        for _ in range(int(rng.integers(0, 7))):                      # clutter: rectangles and lines
            x0, y0 = rng.integers(0, [W, H])
            if rng.random() < 0.6:
                w, h = np.maximum(1, (rng.integers(10, 120, 2) * s).astype(int))
                cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + w), int(y0 + h)), float(rng.uniform(0, 255)), -1)
            else:
                x1, y1 = rng.integers(0, [W, H])
                cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), float(rng.uniform(0, 255)), max(1, int(rng.integers(1, 6) * s)))
        return img.astype(np.float32)

    def _draw(self, img, marker_id, R, t):
        P = (R @ self.phys.T).T + t
        if np.any(P[:, 2] <= 0.02):
            return img
        p = (self.K @ P.T).T
        dst = np.float32(p[:, :2] / p[:, 2:3] * SS + (SS - 1) / 2.0)
        Hm = cv2.getPerspectiveTransform(self.src, dst)
        size = (self.W * SS, self.H * SS)
        warped = cv2.warpPerspective(self.patch[marker_id], Hm, size, flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(self.mask_src, Hm, size, flags=cv2.INTER_NEAREST)
        return np.where(mask > 0, warped, img)

    def render(self, R, t, rng, distractor=True, occlude=False, z_range=(0.3, 1.2), tilt_max_deg=60.0):
        """uint8 gray image (H, W) of M at pose (R, t); F is drawn first as a distractor at a random pose."""
        img = self._background(rng)
        if distractor:
            Rf, tf = self.sample_pose(rng, z_range, tilt_max_deg)
            img = self._draw(img, self.id_f, Rf, tf)
        img = self._draw(img, self.id_m, R, t)
        if occlude:                                                    # grey patch over one corner of the black square
            c = marker_object_points(self.L).astype(float)
            ctr = c[int(rng.integers(0, 4))] * rng.uniform(0.55, 0.8)
            s = rng.uniform(0.35, 0.5) * self.L / 2
            sq = np.array([[ctr[0] - s, ctr[1] - s, 0], [ctr[0] + s, ctr[1] - s, 0], [ctr[0] + s, ctr[1] + s, 0], [ctr[0] - s, ctr[1] + s, 0]])
            q = (self.K @ ((R @ sq.T).T + t).T).T
            q = q[:, :2] / q[:, 2:3] * SS + (SS - 1) / 2.0
            cv2.fillConvexPoly(img, np.int32(np.round(q)), float(rng.uniform(60, 200)))
        img = img * rng.uniform(0.6, 1.25) + rng.uniform(-20, 20)
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2) * SS)
        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img + rng.normal(0, rng.uniform(1, 8), img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------------
# generation of whole sets (parallel), written straight into .npy files
# ----------------------------------------------------------------------------------------------
_RENDERER = None


def _init_worker(cfg):
    global _RENDERER
    _RENDERER = SyntheticRenderer(cfg["dict_id"], cfg["id_m"], cfg["id_f"], cfg["L"], np.array(cfg["K"]), tuple(cfg["size"]))


def _chunk(job):
    start, n, seed, par = job
    rng = np.random.default_rng(np.random.SeedSequence([seed, start]))
    r = _RENDERER
    X = np.zeros((n, r.H, r.W), np.uint8)
    Rs, ts = np.zeros((n, 3, 3), np.float32), np.zeros((n, 3), np.float32)
    for i in range(n):
        R, t = r.sample_pose(rng, par["z_range"], par["tilt_max_deg"])
        X[i] = r.render(R, t, rng, rng.random() < par["p_distractor"], rng.random() < par["p_occlude"], par["z_range"], par["tilt_max_deg"])
        Rs[i], ts[i] = R, t
    return start, X, Rs, ts


def generate_set(out_dir, split, n, seed, cfg, par, workers=None, chunk=100, progress=True):
    """Writes X_<split>.npy (N,H,W uint8), R_<split>.npy (N,3,3), t_<split>.npy (N,3) [m] into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    W, H = cfg["size"]
    X = np.lib.format.open_memmap(os.path.join(out_dir, f"X_{split}.npy"), mode="w+", dtype=np.uint8, shape=(n, H, W))
    Rs, ts = np.zeros((n, 3, 3), np.float32), np.zeros((n, 3), np.float32)
    jobs = [(s, min(chunk, n - s), seed, par) for s in range(0, n, chunk)]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    done = 0
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(cfg,)) as ex:
        for start, Xc, Rc, tc in ex.map(_chunk, jobs):
            X[start:start + len(Xc)], Rs[start:start + len(Xc)], ts[start:start + len(Xc)] = Xc, Rc, tc
            done += len(Xc)
            if progress:
                print(f"\r  {split}: {done}/{n}", end="", flush=True)
    X.flush()
    del X
    np.save(os.path.join(out_dir, f"R_{split}.npy"), Rs)
    np.save(os.path.join(out_dir, f"t_{split}.npy"), ts)
    if progress:
        print()


def load_set(out_dir, split, mmap=False):
    X = np.load(os.path.join(out_dir, f"X_{split}.npy"), mmap_mode="r" if mmap else None)
    return X, np.load(os.path.join(out_dir, f"R_{split}.npy")), np.load(os.path.join(out_dir, f"t_{split}.npy"))


def scaled_camera(K, size_cam, size_net):
    """Intrinsics of the camera when its image is resized to the network input size."""
    f = size_net[0] / size_cam[0]
    Kn = np.array(K, float).copy()
    Kn[:2] *= f
    return Kn


# ----------------------------------------------------------------------------------------------
# crops for the two-stage network: a square window around M, cut from the FULL-resolution camera image
# ----------------------------------------------------------------------------------------------
_FULL = _CROP = None


def _init_crop_worker(cfg):
    global _FULL, _CROP
    import copy
    _FULL = SyntheticRenderer(cfg["dict_id"], cfg["id_m"], cfg["id_f"], cfg["L"], np.array(cfg["K"]), tuple(cfg["size"]))
    _CROP = copy.copy(_FULL)


def _crop_chunk(job):
    """Each sample: M at a random pose in the full camera image; a window around it as stage 1 would cut it (centre and
    scale perturbed like stage-1 errors), rendered directly at full resolution and resized to S x S."""
    from netpose import CROP_FACTOR, CROP_MIN
    start, n, seed, par, S = job
    rng = np.random.default_rng(np.random.SeedSequence([seed, start, 7]))
    full, crop = _FULL, _CROP
    K, W, H = full.K, full.W, full.H
    X = np.zeros((n, S, S), np.uint8)
    Rs, ts, boxes = np.zeros((n, 3, 3), np.float32), np.zeros((n, 3), np.float32), np.zeros((n, 3), np.float32)
    for i in range(n):
        R, t = full.sample_pose(rng, par["z_range"], par["tilt_max_deg"])
        s = K[0, 0] * full.L / t[2]                                          # apparent marker side [px]
        u = K[0, 0] * t[0] / t[2] + K[0, 2] + rng.normal(0, par["jitter"] * s)
        v = K[1, 1] * t[1] / t[2] + K[1, 2] + rng.normal(0, par["jitter"] * s)
        side = int(np.clip(round(CROP_FACTOR * s * np.exp(rng.normal(0, par["jitter"]))), CROP_MIN, min(W, H)))
        u0, v0 = int(np.clip(round(u - side / 2), 0, W - side)), int(np.clip(round(v - side / 2), 0, H - side))
        Kc = K.copy()
        Kc[0, 2] -= u0
        Kc[1, 2] -= v0
        crop.K, crop.W, crop.H = Kc, side, side
        img = crop.render(R, t, rng, rng.random() < par["p_distractor"], rng.random() < par["p_occlude"], par["z_range"], par["tilt_max_deg"])
        X[i] = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
        Rs[i], ts[i], boxes[i] = R, t, (u0, v0, side)
    return start, X, Rs, ts, boxes


def generate_crop_set(out_dir, split, n, seed, cfg, par, S=160, workers=None, chunk=100, progress=True):
    """Writes X_<split>.npy (N,S,S uint8), R_<split>.npy, t_<split>.npy [m] and box_<split>.npy (u0, v0, side in camera pixels)."""
    os.makedirs(out_dir, exist_ok=True)
    X = np.lib.format.open_memmap(os.path.join(out_dir, f"X_{split}.npy"), mode="w+", dtype=np.uint8, shape=(n, S, S))
    Rs, ts, bx = np.zeros((n, 3, 3), np.float32), np.zeros((n, 3), np.float32), np.zeros((n, 3), np.float32)
    jobs = [(s, min(chunk, n - s), seed, par, S) for s in range(0, n, chunk)]
    done = 0
    with ProcessPoolExecutor(workers or max(1, (os.cpu_count() or 2) - 1), initializer=_init_crop_worker, initargs=(cfg,)) as ex:
        for start, Xc, Rc, tc, bc in ex.map(_crop_chunk, jobs):
            k = len(Xc)
            X[start:start + k], Rs[start:start + k], ts[start:start + k], bx[start:start + k] = Xc, Rc, tc, bc
            done += k
            if progress:
                print(f"\r  {split}: {done}/{n}", end="", flush=True)
    X.flush()
    del X
    np.save(os.path.join(out_dir, f"R_{split}.npy"), Rs)
    np.save(os.path.join(out_dir, f"t_{split}.npy"), ts)
    np.save(os.path.join(out_dir, f"box_{split}.npy"), bx)
    if progress:
        print()


def load_boxes(out_dir, split):
    p = os.path.join(out_dir, f"box_{split}.npy")
    return np.load(p) if os.path.exists(p) else None
