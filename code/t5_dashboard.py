"""t5_dashboard.py -- live training window for t5_train.py (an OpenCV window with matplotlib plots, refreshed about once a second).

Shows:  status (epoch, step, elapsed, ETA, images/s, learning rate, GPU memory)
        training loss per step: total and its two weighted parts (w_rot * rotation chordal loss, w_trans * translation L1 loss)
        validation error per epoch: rotation [deg] and translation [mm], median and 95th percentile
        learning-rate schedule
        4 validation images: true marker outline (green) and the network's prediction (orange), redrawn after every epoch
Keys (click the window first):  Q  stop after this step (the model is saved as it is)     S  save a screenshot of the dashboard
At the end the last dashboard is saved as output/network/<model>/dashboard.png; the example photos of every epoch go to output/network/<model>/examples/epoch_NNN.png.
"""
import os
import time

import cv2
import numpy as np

GREEN, ORANGE = (60, 200, 60), (26, 115, 217)


class TrainDashboard:
    def __init__(self, title, total_steps, epochs, w_rot, w_trans, out_dir, headless=False, every_s=1.0):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt, self.title, self.total, self.epochs = plt, title, total_steps, epochs
        self.w_rot, self.w_trans, self.out_dir, self.headless, self.every = w_rot, w_trans, out_dir, headless, every_s
        self.win = "T5 - training (Q stop and save, S screenshot)"
        self.steps, self.loss, self.lrot, self.ltr, self.lr = [], [], [], [], []
        self.ep, self.vrot, self.vrot95, self.vtr, self.vtr95 = [], [], [], [], []
        self.examples = None
        self.t0, self.last, self.epoch, self.n_img, self.stop = time.time(), 0.0, 0, 0, False
        self.batch = 1
        os.makedirs(out_dir, exist_ok=True)
        if not headless:
            cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.win, 1400, 760)

    # ---- data in -------------------------------------------------------------------------------
    def step(self, step, epoch, loss, l_rot, l_tr, lr, batch):
        self.steps.append(step); self.loss.append(loss); self.lrot.append(self.w_rot * l_rot); self.ltr.append(self.w_trans * l_tr)
        self.lr.append(lr); self.epoch, self.batch = epoch, batch
        if time.time() - self.last >= self.every:
            self.show()
        return self.stop

    def end_epoch(self, epoch, val, examples=None):
        self.ep.append(epoch); self.vrot.append(val["rot_med"]); self.vrot95.append(val["rot_p95"])
        self.vtr.append(val["trans_med"]); self.vtr95.append(val["trans_p95"])
        if examples is not None:
            self.examples = examples
            d = os.path.join(self.out_dir, "examples")             # the example photos of every epoch, kept for the report
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, f"epoch_{epoch:03d}.png"), examples)
        self.show()
        return self.stop

    # ---- drawing ---------------------------------------------------------------------------------
    @staticmethod
    def ema(v, a=0.05):
        out, m = [], None
        for x in v:
            m = x if m is None else (1 - a) * m + a * x
            out.append(m)
        return out

    def render(self):
        plt = self.plt
        fig = plt.figure(figsize=(14, 7.6), dpi=100)
        gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.25, 1], left=0.05, right=0.98, top=0.83, bottom=0.08, wspace=0.28, hspace=0.38)
        el = time.time() - self.t0
        s = self.steps[-1] if self.steps else 0
        eta = el / max(s, 1) * (self.total - s)
        ips = s * self.batch / max(el, 1e-6)
        gpu = ""
        try:
            import torch
            if torch.cuda.is_available():
                gpu = f"   GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
        except Exception:
            pass
        status = (f"{self.title}    epoch {self.epoch}/{self.epochs}   step {s}/{self.total} ({100 * s / max(self.total, 1):.0f} %)   "
                  f"elapsed {el / 60:.1f} min   ETA {eta / 60:.1f} min   {ips:.0f} images/s   lr {self.lr[-1] if self.lr else 0:.2e}{gpu}")
        status += (f"\nVALIDATION after epoch {self.ep[-1]}: rotation {self.vrot[-1]:.2f} deg, translation {self.vtr[-1]:.1f} mm"
                   f"   (best so far: {min(self.vrot):.2f} deg, {min(self.vtr):.1f} mm)" if self.ep else
                   "\nVALIDATION: computed at the end of every epoch - the first value appears after epoch 1")
        fig.suptitle(status + ("\nSTOPPING - saving the model ..." if self.stop else
                               f"\nloss = {self.w_rot:g} x rotation (chordal ||R_hat - R||^2)  +  {self.w_trans:g} x translation (L1 on x/Z, y/Z, log Z)"),
                     fontsize=10, x=0.02, ha="left")
        a = fig.add_subplot(gs[0, 0:2])
        if self.steps:
            for v, c, lab in ((self.loss, "k", "total"), (self.lrot, "#D62728", f"{self.w_rot:g} x rotation"), (self.ltr, "#1F77B4", f"{self.w_trans:g} x translation")):
                a.plot(self.steps, v, color=c, alpha=0.15, lw=0.6)
                a.plot(self.steps, self.ema(v), color=c, lw=1.6, label=lab)
            a.set_yscale("log")
            for e in range(1, self.epochs):
                a.axvline(e * self.total / self.epochs, color="grey", lw=0.4, alpha=0.5)
        a.set_title("training loss per step (thin: raw, thick: smoothed; grey lines: epochs)", fontsize=9)
        a.set_xlabel("step"); a.grid(alpha=0.3)
        if self.steps:
            a.legend(fontsize=8, loc="upper right")
        a = fig.add_subplot(gs[0, 2])
        if self.examples is not None:
            a.imshow(cv2.cvtColor(self.examples, cv2.COLOR_BGR2RGB))
        a.set_title(f"validation images after epoch {self.ep[-1] if self.ep else '-'}\ngreen = true, orange = network", fontsize=9)
        a.axis("off")
        for col, (m, p, lab) in enumerate(((self.vrot, self.vrot95, "rotation error [deg]"), (self.vtr, self.vtr95, "translation error [mm]"))):
            a = fig.add_subplot(gs[1, col])
            if self.ep:
                a.plot(self.ep, m, "o-", color="#1F3A5F", ms=4, label="median")
                a.plot(self.ep, p, "s--", color="#D9731A", ms=3, label="95th percentile")
                a.set_yscale("log")
                a.text(0.98, 0.95, f"last: {m[-1]:.2f} (p95 {p[-1]:.1f})", transform=a.transAxes, ha="right", va="top", fontsize=9)
            a.set_title("validation " + lab, fontsize=9); a.set_xlabel("epoch"); a.grid(alpha=0.3)
            if self.ep:
                a.legend(fontsize=8, loc="upper center")
            a.set_xlim(0.5, self.epochs + 0.5)
            from matplotlib.ticker import MaxNLocator
            a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a = fig.add_subplot(gs[1, 2])
        if self.steps:
            a.plot(self.steps, self.lr, color="#1E8C85")
        a.set_title("learning rate (one-cycle: warm-up, cosine decay)", fontsize=9); a.set_xlabel("step"); a.grid(alpha=0.3)
        a.set_xlim(0, self.total)
        fig.canvas.draw()
        img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(), cv2.COLOR_RGB2BGR)
        plt.close(fig)
        return img

    def show(self):
        self.last = time.time()
        img = self.render()
        self.img = img
        if self.headless:
            return
        cv2.imshow(self.win, img)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            self.stop = True
            print("   Q pressed: stopping after this step, the model will be saved")
        elif key in (ord("s"), ord("S")):
            p = os.path.join(self.out_dir, f"dashboard_{time.strftime('%H%M%S')}.png")
            cv2.imwrite(p, img)
            print("   dashboard screenshot ->", p)
        if cv2.getWindowProperty(self.win, cv2.WND_PROP_VISIBLE) < 1:          # window closed: keep training, stop showing
            self.headless = True

    def close(self):
        img = self.render()
        cv2.imwrite(os.path.join(self.out_dir, "dashboard.png"), img)
        if not self.headless:
            cv2.imshow(self.win, img)
            cv2.waitKey(1500)
            cv2.destroyWindow(self.win)
        return os.path.join(self.out_dir, "dashboard.png")


def example_tiles(X, R_true, t_true, R_pred, t_pred, K, L):
    """2x2 tiles: image with the true (green) and predicted (orange) marker outline."""
    from mvlib import marker_object_points
    obj = marker_object_points(L).astype(float)
    tiles = []
    for i in range(min(4, len(X))):
        Ki = np.asarray(K[i] if isinstance(K, list) else K, float)        # crops: every example has its own camera
        img = cv2.cvtColor(np.ascontiguousarray(X[i]), cv2.COLOR_GRAY2BGR)
        for R, t, col in ((R_true[i], t_true[i], GREEN), (R_pred[i], t_pred[i], ORANGE)):
            P = (np.asarray(R, float) @ obj.T).T + np.asarray(t, float)
            if np.all(P[:, 2] > 0.01):
                uv = (Ki @ P.T).T
                uv = uv[:, :2] / uv[:, 2:3]
                if np.all(np.abs(uv) < 1e5):
                    cv2.polylines(img, [uv.astype(np.int32)], True, col, 2, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) < 4:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
