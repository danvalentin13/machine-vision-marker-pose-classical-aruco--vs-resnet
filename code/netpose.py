"""netpose.py -- the pose networks (ResNet-18 regressors) and their use at inference time.

Output of a network (9 numbers) for one image of M:
   6  rotation in the 6-D representation (Zhou et al., 2019): two 3-vectors, Gram-Schmidt -> rotation matrix R
   3  translation as (x/Z, y/Z, log Z):  ^C t_M = Z * (x/Z, y/Z, 1)
Two kinds of network (the checkpoint says which, config["kind"]):
   full   the whole camera image, grey, resized to net_size (e.g. 640x360)                         -> 1 input channel
   crop   TWO-STAGE: a 'full' network (stage 1) finds M; a crop around it, taken from the full-resolution
          (undistorted) camera image and resized to crop_size x crop_size, goes into this network      -> 3 input channels:
          grey + the normalised camera coordinates x/Z, y/Z of every crop pixel (where in the camera the crop came from)
The first conv layer of the ImageNet-pretrained ResNet-18 is summed over its colour channels for the grey channel;
the coordinate channels start at zero.  A network is valid only for the camera K and marker size L it was trained with.
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

from mvlib import make_T

MEAN, STD = 0.45, 0.225
CROP_FACTOR, CROP_MIN = 2.0, 48        # crop side = CROP_FACTOR x apparent marker side [px], at least CROP_MIN


def build_model(pretrained=True, in_ch=1):
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    try:
        m = torchvision.models.resnet18(weights=weights)
    except Exception as e:                                          # offline: pretrained weights cannot be downloaded
        print(f"  could not load pretrained weights ({type(e).__name__}); using random initialisation")
        m = torchvision.models.resnet18(weights=None)
    w = m.conv1.weight.data.sum(dim=1, keepdim=True)
    m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    m.conv1.weight.data.zero_()
    m.conv1.weight.data[:, :1] = w
    m.fc = nn.Linear(512, 9)
    nn.init.normal_(m.fc.weight, 0, 0.01)
    m.fc.bias.data = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, np.log(0.6)], dtype=torch.float32)   # identity-like rotation, Z = 0.6 m
    return m


def six_d_to_R(a):
    """(B,6) -> (B,3,3) rotation matrices (columns b1, b2, b3), Gram-Schmidt."""
    a1, a2 = a[:, 0:3], a[:, 3:6]
    b1 = a1 / torch.linalg.norm(a1, dim=1, keepdim=True).clamp_min(1e-8)
    a2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
    b2 = a2 / torch.linalg.norm(a2, dim=1, keepdim=True).clamp_min(1e-8)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=1)], dim=2)


def decode(out):
    """(B,9) network output -> R (B,3,3), t (B,3) [m]."""
    Z = torch.exp(out[:, 8])
    return six_d_to_R(out[:, :6]), torch.stack([out[:, 6] * Z, out[:, 7] * Z, Z], dim=1)


def targets(t):
    """Translation target (B,3): (x/Z, y/Z, log Z)."""
    return torch.stack([t[:, 0] / t[:, 2], t[:, 1] / t[:, 2], torch.log(t[:, 2])], dim=1)


def preprocess(x_uint8):
    """(B,H,W) uint8 (or float 0-255) tensor -> (B,1,H,W) normalised float."""
    return ((x_uint8.float() / 255.0 - MEAN) / STD).unsqueeze(1)


# ----------------------------------------------------------------------------------------------
# crops (two-stage network)
# ----------------------------------------------------------------------------------------------
def crop_box(t, K, L, size, factor=CROP_FACTOR):
    """Integer square window (u0, v0, side) [px] around the marker centre t (camera frame) in an image of size (W, H);
    side = factor x apparent marker side; the window is kept inside the image."""
    W, H = size
    if not np.all(np.isfinite(t)) or t[2] <= 0.05:
        side = min(W, H) // 2
        return (W - side) // 2, (H - side) // 2, side
    u, v = K[0, 0] * t[0] / t[2] + K[0, 2], K[1, 1] * t[1] / t[2] + K[1, 2]
    side = int(np.clip(round(factor * K[0, 0] * L / t[2]), CROP_MIN, min(W, H)))
    u0 = int(np.clip(round(u - side / 2), 0, W - side))
    v0 = int(np.clip(round(v - side / 2), 0, H - side))
    return u0, v0, side


def coord_maps(boxes, K, S):
    """(B,3) boxes (u0, v0, side) in camera pixels -> (B,2,S,S): x/Z and y/Z of every crop pixel."""
    j = (torch.arange(S, device=boxes.device, dtype=torch.float32) + 0.5) / S
    u = boxes[:, 0:1] + j[None, :] * boxes[:, 2:3]
    v = boxes[:, 1:2] + j[None, :] * boxes[:, 2:3]
    xn, yn = (u - float(K[0][2])) / float(K[0][0]), (v - float(K[1][2])) / float(K[1][1])
    return torch.stack([xn[:, None, :].expand(-1, S, -1), yn[:, :, None].expand(-1, -1, S)], dim=1)


def make_input(x, boxes=None, K=None):
    """Network input: grey channel, plus the two coordinate channels for a crop network."""
    g = preprocess(x)
    if boxes is None:
        return g
    return torch.cat([g, coord_maps(boxes.float(), K, x.shape[-1])], dim=1)


# ----------------------------------------------------------------------------------------------
class NetPose:
    """Loads a checkpoint written by t5_train.py and predicts ^C T_M from camera frames (any kind of network)."""

    def __init__(self, path, K=None, dist=None, device=None, passes=2):
        self.passes = max(1, int(passes))                                 # crop networks: 1 = stage 1 only places the window, 2 = re-centred once
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cfg = ck["config"]
        self.kind = self.cfg.get("kind", "full")
        self.model = build_model(pretrained=False, in_ch=3 if self.kind == "crop" else 1)
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(self.device).eval()
        self.net_size = tuple(self.cfg["net_size"])
        self.K_cam, self.size_cam = np.array(self.cfg["K_cam"], float), tuple(self.cfg["size_cam"])
        self.K, self.dist = (None if K is None else np.asarray(K, float)), (None if dist is None else np.asarray(dist, float))
        if self.K is not None:
            fx_train = self.cfg["K_cam"][0][0]
            if abs(self.K[0, 0] - fx_train) / fx_train > 0.05:
                print(f"  WARNING: this network was trained with fx = {fx_train:.0f} px, the camera file has fx = {self.K[0, 0]:.0f} px: "
                      "the translation will be wrong - retrain with the calibrated camera")
        self.stage1 = None
        if self.kind == "crop":
            p1 = os.path.join(os.path.dirname(os.path.abspath(path)), self.cfg["stage1"])
            self.stage1 = NetPose(p1, K, dist, device)
            self.crop_size, self.L = self.cfg["crop_size"], self.cfg["L_m"]

    def _undist(self, gray, undistort):
        if undistort and self.K is not None and self.dist is not None and np.any(self.dist != 0):
            return cv2.undistort(gray, self.K, self.dist)               # the networks were trained on distortion-free images
        return gray

    @torch.no_grad()
    def predict_batch(self, grays, undistort=True):
        """grays: list of grey camera frames (full resolution, or any size with the same aspect ratio).
        undistort=False for images that have no lens distortion (synthetic test images)."""
        if self.kind == "full":
            x = torch.from_numpy(np.stack([cv2.resize(self._undist(g, undistort), self.net_size, interpolation=cv2.INTER_AREA)
                                           for g in grays])).to(self.device)
            out = self.model(make_input(x).contiguous(memory_format=torch.channels_last))
        else:
            _, t1 = self.stage1.predict_batch(grays, undistort)          # stage 1: where is M, how big does it look
            gus = [self._undist(g, undistort) for g in grays]
            for p in range(self.passes):                                  # pass 2+: re-centre the window on the crop network's own answer
                crops, boxes = [], []                                     # (removes a stage-1 offset as long as M was partly in the window)
                for gu, t in zip(gus, t1):
                    H, W = gu.shape[:2]
                    f = W / self.size_cam[0]                              # this frame's scale against the training camera
                    Kf = self.K_cam.copy()
                    Kf[:2] *= f
                    u0, v0, side = crop_box(t, Kf, self.L, (W, H))
                    crops.append(cv2.resize(np.ascontiguousarray(gu[v0:v0 + side, u0:u0 + side]), (self.crop_size, self.crop_size),
                                            interpolation=cv2.INTER_AREA))
                    boxes.append([u0 / f, v0 / f, side / f])              # in pixels of the training camera
                x = torch.from_numpy(np.stack(crops)).to(self.device)
                b = torch.tensor(boxes, dtype=torch.float32, device=self.device)
                out = self.model(make_input(x, b, self.K_cam).contiguous(memory_format=torch.channels_last))
                if p < self.passes - 1:
                    t1 = decode(out.float())[1].cpu().numpy().astype(float)
        R, t = decode(out.float())
        return R.cpu().numpy().astype(float), t.cpu().numpy().astype(float)

    def predict_one(self, gray, undistort=True):
        R, t = self.predict_batch([gray], undistort)
        return make_T(R[0], t[0])

    def time_per_frame_ms(self, gray, n=50):
        """Latency for one frame (batch 1), after warm-up."""
        for _ in range(5):
            self.predict_one(gray)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            self.predict_one(gray)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return 1000 * (time.perf_counter() - t0) / n
