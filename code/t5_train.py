"""t5_train.py -- Task 5, step 2: train / fine-tune the ResNet-18 pose network.

  python t5_train.py

Asks (Enter = [default]): data set, model name, epochs, batch size, learning rate, loss weights, seed, pretrained weights.
Network:  ResNet-18 (ImageNet-pretrained), grey image in, 9 numbers out: 6-D rotation + (x/Z, y/Z, log Z)  (see netpose.py).
Loss:     w_rot * chordal rotation loss ||R_hat - R||_F^2   +   w_trans * L1 loss on (x/Z, y/Z, log Z).
          The chordal loss equals 4 (1 - cos e_R): a smooth function of the rotation error angle.
Reports per epoch: train loss, validation rotation error [deg] and translation error [mm] (median), time.
Writes  data/models/<model>.pt                          checkpoint (weights + camera + marker + settings, for netpose.py)
        output/network/<model>/train_log.json, training_curves.png
The validation set is separate synthetic data; the real test frames are NEVER used here (assignment rule).
Run with the GPU if available (CUDA); on CPU it is much slower.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from config import data_path, OUTPUT_DIR, training_path
from mvtools import ask, ask_folder, ask_number, ask_yes
from netpose import MEAN, STD, build_model, decode, make_input, preprocess, targets
from synthdata import load_boxes, load_set


def losses(out, R_true, t_true, w_rot, w_trans):
    from netpose import six_d_to_R
    Rh = six_d_to_R(out[:, :6])
    l_rot = ((Rh - R_true) ** 2).sum(dim=(1, 2)).mean()
    l_tr = (out[:, 6:9] - targets(t_true)).abs().sum(dim=1).mean()
    return w_rot * l_rot + w_trans * l_tr, l_rot.detach(), l_tr.detach()


def rotate_camera(x, R, t, K, max_deg=8.0, roll_deg=10.0):
    """Augmentation: the image as seen by the SAME camera turned a little (pitch/yaw up to max_deg, roll up to roll_deg).
    A pure camera rotation Ra maps the image exactly by the homography K Ra K^-1, and the labels become Ra R, Ra t -
    so the network cannot use 'where things are in the room' (it broke when the laptop was tilted)."""
    B, H, W = x.shape
    dev = x.device
    a = torch.deg2rad(torch.stack([torch.empty(B, device=dev).uniform_(-max_deg, max_deg),
                                   torch.empty(B, device=dev).uniform_(-max_deg, max_deg),
                                   torch.empty(B, device=dev).uniform_(-roll_deg, roll_deg)], 1))
    c, s = torch.cos(a), torch.sin(a)
    o, z = torch.ones(B, device=dev), torch.zeros(B, device=dev)
    Rx = torch.stack([o, z, z, z, c[:, 0], -s[:, 0], z, s[:, 0], c[:, 0]], 1).view(B, 3, 3)
    Ry = torch.stack([c[:, 1], z, s[:, 1], z, o, z, -s[:, 1], z, c[:, 1]], 1).view(B, 3, 3)
    Rz = torch.stack([c[:, 2], -s[:, 2], z, s[:, 2], c[:, 2], z, z, z, o], 1).view(B, 3, 3)
    Ra = Rz @ Ry @ Rx
    Kt = torch.as_tensor(np.asarray(K, np.float32), device=dev)
    Hinv = Kt @ Ra.transpose(1, 2) @ torch.linalg.inv(Kt)                    # new pixel -> old pixel
    v, u = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    p = torch.stack([u, v, torch.ones_like(u)], -1).view(1, -1, 3)            # (1, HW, 3)
    q = p @ Hinv.transpose(1, 2)                                              # (B, HW, 3)
    q = q[..., :2] / q[..., 2:3]
    grid = torch.stack([(q[..., 0] + 0.5) / W * 2 - 1, (q[..., 1] + 0.5) / H * 2 - 1], -1).view(B, H, W, 2)
    xw = torch.nn.functional.grid_sample(x.unsqueeze(1), grid, mode="bilinear", padding_mode="zeros", align_corners=False).squeeze(1)
    return xw, Ra @ R, (Ra @ t.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def hard_cases(x, t, L, K, boxes=None, crop_size=None, p=0.5):
    """Augmentation: turn a share p of the images into cases the classical detector cannot decode, keeping the label:
    0 partial cover (a patch over up to ~half of the marker)   1 motion blur (a streak in a random direction)
    2 low resolution (as if the marker were much farther)      3 dim, low-contrast light.
    x (B,H,W) float 0..255; t (B,3) marker position [m]; K = camera of the image (whole image) or of the full frame with
    boxes (B,3: u0, v0, side) and crop_size for crops - used to find the marker in each image."""
    B, H, W = x.shape
    dev = x.device
    K = torch.as_tensor(np.asarray(K, np.float32), device=dev)
    u = K[0, 0] * t[:, 0] / t[:, 2] + K[0, 2]
    v = K[1, 1] * t[:, 1] / t[:, 2] + K[1, 2]
    s = K[0, 0] * L / t[:, 2]                                     # the marker's side in pixels
    if boxes is not None:                                         # crop: full-frame pixels -> crop pixels
        k = crop_size / boxes[:, 2]
        u, v, s = (u - boxes[:, 0]) * k, (v - boxes[:, 1]) * k, s * k
    kind = torch.where(torch.rand(B, device=dev) < p, torch.randint(0, 4, (B,), device=dev), torch.full((B,), -1, device=dev))
    x = x.clone()
    for i in torch.nonzero(kind == 0).flatten().tolist():         # partial cover
        si = float(s[i])
        w, h = (float(a) * si for a in torch.empty(2).uniform_(0.25, 0.7))
        cu, cv = float(u[i]) + float(torch.empty(1).uniform_(-0.6, 0.6)) * si, float(v[i]) + float(torch.empty(1).uniform_(-0.6, 0.6)) * si
        x0, x1, y0, y1 = int(max(0, cu - w / 2)), int(min(W, cu + w / 2)), int(max(0, cv - h / 2)), int(min(H, cv + h / 2))
        if x1 > x0 and y1 > y0:
            x[i, y0:y1, x0:x1] = float(torch.empty(1).uniform_(0, 255)) + 12 * torch.randn(y1 - y0, x1 - x0, device=dev)
    m = kind == 1                                                 # motion blur: one streak per batch
    if m.any():
        n = int(torch.randint(2, 5, (1,))) * 2 + 1 if boxes is not None else int(torch.randint(1, 3, (1,))) * 2 + 1
        a = float(torch.empty(1).uniform_(0, np.pi))
        ker = torch.zeros(n, n, device=dev)
        for j in np.linspace(-(n - 1) / 2, (n - 1) / 2, 4 * n):
            ker[int(round((n - 1) / 2 + j * np.sin(a))), int(round((n - 1) / 2 + j * np.cos(a)))] = 1
        ker = (ker / ker.sum()).view(1, 1, n, n)
        x[m] = torch.nn.functional.conv2d(x[m].unsqueeze(1), ker, padding=n // 2).squeeze(1)
    m = kind == 2                                                 # low resolution
    if m.any():
        f = float(torch.empty(1).uniform_(1.5, 3.0)) if boxes is not None else float(torch.empty(1).uniform_(1.3, 2.0))
        small = torch.nn.functional.interpolate(x[m].unsqueeze(1), size=(max(8, int(H / f)), max(8, int(W / f))), mode="area")
        x[m] = torch.nn.functional.interpolate(small, size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
    m = kind == 3                                                 # dim, low contrast
    if m.any():
        n3 = int(m.sum())
        mean = x[m].mean(dim=(1, 2), keepdim=True)
        x[m] = (mean * torch.empty(n3, 1, 1, device=dev).uniform_(0.45, 1.0)
                + (x[m] - mean) * torch.empty(n3, 1, 1, device=dev).uniform_(0.3, 0.7))
    return x


@torch.no_grad()
def evaluate(model, X, R, t, device, batch=256, boxes=None, K=None):
    model.eval()
    er, et = [], []
    for s in range(0, len(X), batch):
        x = X[s:s + batch].to(device)
        b = None if boxes is None else boxes[s:s + batch].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(make_input(x, b, K).contiguous(memory_format=torch.channels_last))
        Rh, th = decode(out.float())
        Rt, tt = R[s:s + batch].to(device), t[s:s + batch].to(device)
        tr = (torch.bmm(Rt.transpose(1, 2), Rh).diagonal(dim1=1, dim2=2).sum(1) - 1) / 2
        er.append(torch.rad2deg(torch.acos(tr.clamp(-1, 1))).cpu())
        et.append((1000 * torch.linalg.norm(th - tt, dim=1)).cpu())
    er, et = torch.cat(er).numpy(), torch.cat(et).numpy()
    return dict(rot_med=float(np.median(er)), rot_p95=float(np.percentile(er, 95)),
                trans_med=float(np.median(et)), trans_p95=float(np.percentile(et, 95)))


def train(data_dir, model_name, epochs=15, batch=64, lr=5e-4, w_rot=1.0, w_trans=1.0, seed=0, pretrained=True,
          models_dir=None, out_dir=None, max_steps=None, save=True, log=print, gui=False, gui_headless=False, resume=True,
          strong_aug=None, clip=None, init_from=None, mix_with=None, keep_best=False, rot_aug=0.0, hard_aug=0.0):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":                                     # full GPU speed: fastest convolution algorithms, TF32 matmuls
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    log(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "  -- no GPU found, this will be slow"))
    Xtr, Rtr, ttr = (torch.from_numpy(np.ascontiguousarray(a)) for a in load_set(data_dir, "train"))
    Xva, Rva, tva = (torch.from_numpy(np.ascontiguousarray(a)) for a in load_set(data_dir, "val"))
    crop = meta.get("kind") == "crop"                             # two-stage network: crops + their windows in the camera image
    Btr = torch.from_numpy(load_boxes(data_dir, "train")) if crop else None
    Bva = torch.from_numpy(load_boxes(data_dir, "val")) if crop else None
    K_cam = meta["K_cam"]
    K_net_aug = meta["K_net"]                                     # the camera of the whole-image network's input (for rot_aug)
    if strong_aug is None:
        strong_aug = crop                                         # the crop network is trained with the stronger augmentation
    if clip is None:
        clip = 5.0 if crop else 0.0
    # images kept on the GPU only while they leave room for the network (16 GB card: a full card spills into shared memory
    # and training slows down ~100x); larger sets stay in RAM and only each batch is moved
    on_gpu = device.type == "cuda" and Xtr.nbytes < 2e9
    if on_gpu:
        Xtr, Rtr, ttr = Xtr.to(device), Rtr.to(device), ttr.to(device)
        Btr = None if Btr is None else Btr.to(device)
    log(f"train {len(Xtr)} images, val {len(Xva)} images, {tuple(Xtr.shape[1:])} px, data on {'GPU' if on_gpu else 'CPU'}")
    Xs = None
    if mix_with:                                                  # fine-tuning: every batch = half these images, half the synthetic set
        Xs, Rs_, ts_ = (torch.from_numpy(np.ascontiguousarray(a)) for a in load_set(mix_with, "train"))
        Bs = torch.from_numpy(load_boxes(mix_with, "train")) if crop else None
        if device.type == "cuda" and Xs.nbytes + (Xtr.nbytes if on_gpu else 0) < 3.5e9:
            Xs, Rs_, ts_ = Xs.to(device), Rs_.to(device), ts_.to(device)
            Bs = None if Bs is None else Bs.to(device)
        log(f"mixed with the synthetic set {os.path.basename(mix_with)} ({len(Xs)} images): half of every batch")
    if meta.get("provisional_camera"):
        log("NOTE: this data set uses the PROVISIONAL camera: the network is a prototype, retrain after the calibration.")

    model = build_model(pretrained, in_ch=3 if crop else 1).to(device).to(memory_format=torch.channels_last)
    if init_from:                                                 # fine-tuning: start from a trained network
        model.load_state_dict(torch.load(init_from, map_location=device, weights_only=False)["state_dict"])
        log(f"starting from {os.path.basename(init_from)}")
    log(f"network: {'two-stage CROP network (grey + 2 coordinate channels), stage 1 = ' + meta['stage1'] if crop else 'full-image network'}; "
        f"augmentation: {'strong (gain, offset, noise, blur)' if strong_aug else 'gain and offset'}"
        f"{f' + camera turned up to {rot_aug:g} deg (roll {1.25 * rot_aug:g})' if rot_aug and not crop else ''}{f' + hard cases in {100 * hard_aug:.0f} % (cover, motion blur, low resolution, dim light)' if hard_aug else ''}; gradient clipping: {clip if clip else 'off'}")
    n_params = sum(p.numel() for p in model.parameters())
    half = batch // 2 if Xs is not None else batch
    steps_per_epoch = max(1, len(Xtr) // half)
    total = epochs * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.1, anneal_strategy="cos")
    hist, t0, step = [], time.time(), 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    out_dir = out_dir or os.path.join(OUTPUT_DIR, "network", model_name)
    # checkpoint after every epoch: an interrupted run continues where it stopped (at most one epoch is lost)
    ckpt_path = os.path.join(models_dir or training_path("models"), model_name + "_checkpoint.pt")
    start_ep, minutes_before, best_resumed = 1, 0.0, None
    if save and resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        same = all(ck["settings"].get(k) == v for k, v in dict(epochs=epochs, batch=batch, lr=lr, w_rot=w_rot, w_trans=w_trans, seed=seed,
                                                                  data_set=meta["name"]).items())
        if same:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            g.set_state(ck["rng"].cpu()); hist, step, start_ep, minutes_before = ck["hist"], ck["step"], ck["epoch"] + 1, ck["minutes"]
            best_resumed = ck.get("best")
            log(f"RESUMING {model_name} from its checkpoint: epochs 1-{ck['epoch']} done, continuing with epoch {start_ep}")
        else:
            log(f"a checkpoint of {model_name} exists but with other settings: starting from scratch (it will be overwritten)")
    dash, stopped = None, False
    if gui:                                                       # live training window (t5_dashboard.py)
        from t5_dashboard import TrainDashboard, example_tiles
        dash = TrainDashboard(model_name, total if not max_steps else min(total, max_steps), epochs, w_rot, w_trans, out_dir,
                              headless=gui_headless)
        ex_idx = np.linspace(0, len(Xva) - 1, 4).astype(int)
        K_net, L_m = np.array(meta["K_net"]), meta["L_m"]
        for h in hist:                                            # resumed: show the epochs already done
            dash.ep.append(h["epoch"]); dash.vrot.append(h["val_rot_med"]); dash.vrot95.append(h["val_rot_p95"])
            dash.vtr.append(h["val_trans_med"]); dash.vtr95.append(h["val_trans_p95"])
    t0 -= minutes_before * 60
    best = best_resumed
    for ep in range(start_ep, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        acc = []
        for b in range(steps_per_epoch):
            idx = perm[b * half:(b + 1) * half]
            if on_gpu:
                idx = idx.to(device)
            x, R, t = Xtr[idx].to(device, non_blocking=True), Rtr[idx].to(device), ttr[idx].to(device)
            if Xs is not None:                                    # the synthetic half of the batch
                js = torch.randint(0, len(Xs), (batch - len(idx),), generator=g)
                js = js.to(Xs.device)
                x = torch.cat([x, Xs[js].to(device)]); R = torch.cat([R, Rs_[js].to(device)]); t = torch.cat([t, ts_[js].to(device)])
            if rot_aug and not crop:                              # the camera turned a little (image warped, labels turned with it)
                x, R, t = rotate_camera(x.float(), R.float(), t.float(), K_net_aug, rot_aug, 1.25 * rot_aug)
            x = x.float() * torch.empty(len(x), 1, 1, device=device).uniform_(0.85, 1.15) + torch.empty(len(x), 1, 1, device=device).uniform_(-10, 10)
            if strong_aug:                                        # sensor noise and blur, different every time
                x = x + torch.randn_like(x) * torch.empty(len(x), 1, 1, device=device).uniform_(0, 6)
                if torch.rand(1).item() < 0.3:
                    import torchvision.transforms.functional as TF
                    x = TF.gaussian_blur(x.unsqueeze(1), 5, float(torch.empty(1).uniform_(0.3, 1.3))).squeeze(1)
            bb = None if Btr is None else Btr[idx].to(device)
            if bb is not None and Xs is not None:
                bb = torch.cat([bb, Bs[js].to(device)])
            if hard_aug:                                          # cases the classical detector cannot decode (label stays valid)
                x = hard_cases(x, t, meta["L_m"], K_cam if crop else K_net_aug, bb, meta.get("crop_size"), hard_aug)
            x = x.clamp(0, 255)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(make_input(x, bb, K_cam).contiguous(memory_format=torch.channels_last))
            loss, l_rot, l_tr = losses(out.float(), R, t, w_rot, w_trans)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            sched.step()
            step += 1
            acc.append((loss.item(), l_rot.item(), l_tr.item()))
            if dash is not None and dash.step(step, ep, acc[-1][0], acc[-1][1], acc[-1][2], sched.get_last_lr()[0], batch):
                stopped = True
            if (max_steps and step >= max_steps) or stopped:
                break
        m = np.mean(acc, 0)
        val = evaluate(model, Xva, Rva, tva, device, boxes=Bva, K=K_cam)
        rec = dict(epoch=ep, loss=float(m[0]), rot_loss=float(m[1]), trans_loss=float(m[2]), minutes=(time.time() - t0) / 60, **{f"val_{k}": v for k, v in val.items()})
        hist.append(rec)
        score = val["rot_med"] / 5.0 + val["trans_med"] / 10.0          # 5 deg weighs as much as 10 mm
        if keep_best and (best is None or score < best[0]):           # keep the network of the best validation epoch
            best = (score, ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            log(f"   best so far: epoch {ep} (validation rot {val['rot_med']:.2f} deg, trans {val['trans_med']:.1f} mm)")
        eta = (rec["minutes"] - minutes_before) / (ep - start_ep + 1) * (epochs - ep)       # time of this session only (resumed runs)
        log(f"epoch {ep:3d}/{epochs}  loss {m[0]:.4f} (rot {m[1]:.4f} trans {m[2]:.4f})  val: rot {val['rot_med']:.2f} deg (p95 {val['rot_p95']:.1f}), "
            f"trans {val['trans_med']:.1f} mm (p95 {val['trans_p95']:.0f})   {rec['minutes']:.1f} min, ~{eta:.1f} min left")
        if dash is not None:
            ex_idx = np.sort(np.random.default_rng(ep).choice(len(Xva), min(4, len(Xva)), replace=False))   # 4 new validation frames every epoch
            model.eval()
            with torch.no_grad():
                xe = Xva[ex_idx].to(device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    oe = model(make_input(xe, None if Bva is None else Bva[ex_idx].to(device), K_cam).contiguous(memory_format=torch.channels_last))
                Re, te = decode(oe.float())
            if crop:                                              # the crop camera of every example
                Kx = []
                for (u0, v0, side) in Bva[ex_idx].numpy():
                    kk = meta["crop_size"] / side
                    Kc = np.array(K_cam, float); Kc[0, 2] -= u0; Kc[1, 2] -= v0; Kc[:2] *= kk
                    Kx.append(Kc)
            else:
                Kx = K_net
            tiles = example_tiles(Xva[ex_idx].numpy(), Rva[ex_idx].numpy(), tva[ex_idx].numpy(), Re.cpu().numpy(), te.cpu().numpy(), Kx, L_m)
            if dash.end_epoch(ep, val, tiles):
                stopped = True
        if save and not stopped and not max_steps:
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), rng=g.get_state(), hist=hist, step=step,
                            epoch=ep, minutes=(time.time() - t0) / 60, best=best,
                            settings=dict(epochs=epochs, batch=batch, lr=lr, w_rot=w_rot, w_trans=w_trans, seed=seed, data_set=meta["name"])),
                       ckpt_path)
        if (max_steps and step >= max_steps) or stopped:
            if stopped:
                log(f"stopped early by the user after epoch {ep} (step {step}); the model is saved as it is")
            break
    minutes = (time.time() - t0) / 60
    best_epoch = None
    if keep_best and best is not None:                            # save the best validation epoch, not the last one
        model.load_state_dict(best[2])
        best_epoch = best[1]
        log(f"keeping the network of epoch {best_epoch} (best validation), not the last epoch")
    config = dict(best_epoch=best_epoch,arch="resnet18", net_size=meta["net_size"], K_cam=meta["K_cam"], size_cam=meta["size_cam"], K_net=meta["K_net"], L_m=meta["L_m"],
                  dictionary=meta["dictionary"], id_moving=meta["id_moving"], provisional_camera=meta.get("provisional_camera", False),
                  data_set=meta["name"], epochs=epochs, batch=batch, lr=lr, w_rot=w_rot, w_trans=w_trans, seed=seed, pretrained=pretrained,
                  n_train=len(Xtr), n_params=int(n_params), train_minutes=minutes, device=str(device),
                  val=(hist[best_epoch - 1] if best_epoch else hist[-1]) if hist else None, stopped_early=stopped, epochs_done=len(hist),
                  kind="crop" if crop else "full", crop_size=meta.get("crop_size"), stage1=meta.get("stage1"), strong_aug=bool(strong_aug), clip=clip,
                  rot_aug_deg=rot_aug if not crop else 0.0, hard_aug=hard_aug, fine_tuned_from=os.path.basename(init_from) if init_from else None, mixed_with=os.path.basename(mix_with) if mix_with else None,
                  real_recordings=meta.get("from_recordings"), validation_recording=meta.get("validation_recording"))
    if dash is not None:
        log(f"dashboard -> {dash.close()}")
    if save:
        models_dir = models_dir or training_path("models")
        os.makedirs(models_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(models_dir, model_name + ".pt")
        torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()}, config=config), path)
        if os.path.exists(ckpt_path) and not stopped:
            os.remove(ckpt_path)                                  # finished: the checkpoint is no longer needed
        json.dump(dict(config=config, log=hist), open(os.path.join(out_dir, "train_log.json"), "w"), indent=1)
        curves(hist, os.path.join(out_dir, "training_curves.png"))
        log(f"\nmodel -> {path}\nlog and curves -> {out_dir}")
    return hist, config


def curves(hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    ax[0].plot(ep, [h["rot_loss"] for h in hist], label="rotation (chordal)"); ax[0].plot(ep, [h["trans_loss"] for h in hist], label="translation (L1)")
    ax[0].set_title("training loss"); ax[0].set_yscale("log"); ax[0].legend()
    ax[1].plot(ep, [h["val_rot_med"] for h in hist], label="median"); ax[1].plot(ep, [h["val_rot_p95"] for h in hist], label="95th percentile")
    ax[1].set_title("validation rotation error [deg]"); ax[1].set_yscale("log"); ax[1].legend()
    ax[2].plot(ep, [h["val_trans_med"] for h in hist], label="median"); ax[2].plot(ep, [h["val_trans_p95"] for h in hist], label="95th percentile")
    ax[2].set_title("validation translation error [mm]"); ax[2].set_yscale("log"); ax[2].legend()
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    argparse.ArgumentParser(description="Task 5: train the pose network.").parse_args()
    print("Task 5 - train the ResNet-18 pose network.  Press Enter to accept the [default].\n")
    sets = sorted(d for d in os.listdir(training_path("synthetic")) if os.path.exists(os.path.join(training_path("synthetic"), d, "meta.json"))) \
        if os.path.isdir(training_path("synthetic")) else []
    if not sets:
        raise SystemExit("no synthetic data set found: run t5_make_data.py first")
    print("data sets:", ", ".join(sets))
    name = ask("1) Data set", sets[-1])
    data_dir = os.path.join(training_path("synthetic"), name)
    is_crop = json.load(open(os.path.join(data_dir, "meta.json"))).get("kind") == "crop"
    if is_crop:
        print("   crop data set: the two-stage network is trained (stage 2; stage 1 is the full-image network stored with it)")
    model_name = ask("2) Model name", "crop_resnet18_v1" if is_crop else "resnet18_v1")
    epochs = ask_number("3) Epochs", 40 if is_crop else 15, int, 1)
    batch = ask_number("4) Batch size", 64, int, 1)
    lr = ask_number("5) Learning rate (peak)", 5e-4, float, 1e-8)
    w_rot = ask_number("6a) Weight of the rotation loss", 1.0, float, 0)
    w_trans = ask_number("6b) Weight of the translation loss", 1.0, float, 0)
    seed = ask_number("7) Seed", 0, int, 0)
    pretrained = ask_yes("8) Start from ImageNet-pretrained weights?", "y")
    gui = ask_yes("9) Show the live training window (losses, validation errors, example predictions)?", "y")
    print()
    train(data_dir, model_name, epochs, batch, lr, w_rot, w_trans, seed, pretrained, gui=gui)


if __name__ == "__main__":
    main()
