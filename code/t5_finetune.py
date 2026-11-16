"""t5_finetune.py -- Task 5: fine-tune the networks on REAL training frames labelled by the classical pipeline.

  python t5_finetune.py                                         round 1: training/real_v2 -> *_ft.pt
  python t5_finetune.py --sets real_v2 real_v3 --suffix _ft2    round 2: the recordings of both folders -> *_ft2.pt
  python t5_finetune.py --sets real_v2 real_v3 --suffix _ft2 --yes     the same, all questions answered with the defaults
  python t5_finetune.py --sets real_v2 real_v3 real_v4 --suffix _ft3 --val real_v2_05 real_v3_05 real_v4_05 --yes
                                                                round 3: several validation recordings (one per kind of view)

The real training set of the assignment (slide 10): frames recorded with the camera (training/<set>/*.avi, tag 'train'),
labelled with the classical T3 poses (ArUco + IPPE + LM, ^C T_M).  The labels carry the classical errors, so these frames are
used for training only.  The TEST recordings (data/recordings: E1_*, T4_*, ...) are never used here.

Steps (a window shows steps 1-2 with a completion bar, then the training window of t5_dashboard.py opens)
  (every step keeps its result: labels, data sets and a training checkpoint after each epoch - a stopped or frozen run is
   simply started again with the same command and continues where it was;  --relabel / --rebuild / --restart redo a step)
  1  LABEL  every frame where M is detected and the classical pose fits its corners to <= MAX_RMS_PX.
            Labels are saved per recording and reused the next time (delete training/<set>/labels to label again).
  2  BUILD  two real data sets in the synthetic format, split by recording (one recording = validation, never trained on)
              full   the undistorted frame resized to 640x360                 -> fine-tunes stage 1 (whole image: where is M?)
              crop   a 224 px window around M, shifted/scaled at random       -> fine-tunes stage 2 (crop: exact pose)
  3  TRAIN  each network on real + synthetic images mixed (half / half per batch: the synthetic half keeps what was learned),
            low learning rate; the network of the best validation epoch is kept.
            Stage 1 also sees the camera turned a little (ROT_AUG_DEG): it must look at the marker, not at the room.
            Both see HARD_AUG of the images as hard cases (partial cover, motion blur, low resolution, dim light): the label
            stays valid, so the network learns poses where the classical detector gives none.

Writes  training/<set>/labels/<recording>_labels.npz             the classical labels
        training/synthetic/<sets>_full, <sets>_crop               the real data sets
        training/models/<stage1><suffix>.pt, <stage2><suffix>.pt  the fine-tuned networks (the crop network uses the fine-tuned stage 1)
        output/network/<network><suffix>/                         training log, curves, dashboard, example photos of every epoch
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

from config import training_path
from mvtools import MarkerPoseEstimator, ask, ask_number, ask_yes, load_camera, load_markers_cfg
from netpose import CROP_FACTOR, CROP_MIN

MAX_RMS_PX = 1.0             # a label is used only if the classical pose fits its corners to this reprojection error
ROT_AUG_DEG = 8.0            # stage 1: the camera turned up to 8 deg (roll 10) in training - no 'room' shortcut (tilted-laptop test)
HARD_AUG = 0.4              # share of the training images turned into cases classical cannot decode (cover, motion blur, low-res, dim)
CROP_JITTER = 0.25           # stage 2: window offsets of about 25 % of the marker size: it learns to find M anywhere in the window
SYNTHETIC = {"full": "synth_v1", "crop": "crop_v1"}          # the synthetic set mixed into every batch
WHITE, GREY, ORANGE, GREEN, RED = (235, 235, 235), (150, 150, 150), (26, 160, 255), (60, 200, 60), (60, 60, 235)


def rec_name(video):
    return os.path.splitext(os.path.basename(video))[0]


def frame_count(video):
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


# ---- the preparation window ---------------------------------------------------------------------------
class PrepWindow:
    """Live window for steps 1-2: which step, a completion bar, what happens, and one example frame in every 100."""
    STEPS = ["1  LABEL  the classical pipeline measures M in every frame",
             "2  BUILD  the real training sets (whole image + crop around M)",
             "3  TRAIN  fine-tuning - the training window opens"]

    def __init__(self, enabled=True):
        self.on, self.win, self.last, self.frame = enabled, "T5 - preparing the fine-tuning (Q hides this window)", 0.0, None
        if self.on:
            cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)

    def show(self, step, frac, title, lines, force=False):
        if not self.on or (not force and time.time() - self.last < 0.1):
            return
        self.last = time.time()
        img = np.full((560, 1280, 3), 32, np.uint8)

        def put(y, text, col=WHITE, scale=0.52, th=1):
            cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)
        put(36, "T5 FINE-TUNING - PREPARATION", WHITE, 0.8, 2)
        y = 80
        for i, s in enumerate(self.STEPS):
            put(y, ("[x] " if i < step else "[>] " if i == step else "[ ] ") + s,
                GREEN if i < step else ORANGE if i == step else GREY, 0.55, 2 if i == step else 1)
            y += 30
        y += 14
        put(y, title, ORANGE, 0.62, 2)
        y += 16
        frac = min(1.0, max(0.0, frac))
        cv2.rectangle(img, (20, y), (560, y + 22), (90, 90, 90), 1)
        cv2.rectangle(img, (21, y + 1), (21 + int(538 * frac), y + 21), GREEN, -1)
        put(y + 17, f"{100 * frac:.0f} %", WHITE if frac < 0.08 else (20, 20, 20), 0.5, 2)
        y += 52
        for text in lines:
            put(y, text)
            y += 26
        if self.frame is not None:
            img[180:540, 620:1260] = cv2.resize(self.frame, (640, 360))
        cv2.imshow(self.win, img)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q")) or cv2.getWindowProperty(self.win, cv2.WND_PROP_VISIBLE) < 1:
            self.close()

    def close(self):
        if self.on:
            self.on = False
            try:
                cv2.destroyWindow(self.win)
                cv2.waitKey(1)
            except cv2.error:
                pass


# ---- step 1: labels -----------------------------------------------------------------------------------
def label_recording(video, cfg, K, dist, ui=None, done_before=0, total_frames=None):
    """Classical labels (^C T_M) of one recording: (frame indices, R, t, video)."""
    from mvlib import marker_object_points
    obj = marker_object_points(cfg["L"]).astype(np.float64)
    name, n_all = rec_name(video), frame_count(video)
    est = MarkerPoseEstimator(cfg, K, dist, camera_still=False)
    cap = cv2.VideoCapture(video)
    idx, Rs, ts, n = [], [], [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        r = est.process(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        good = r["T_CM"] is not None and r["rms_M"] <= MAX_RMS_PX
        if good:
            idx.append(n); Rs.append(r["T_CM"][:3, :3]); ts.append(r["T_CM"][:3, 3])
        if ui is not None and ui.on:
            if n % 100 == 0:                                          # one example picture in every 100 frames
                if r["T_CM"] is not None:
                    uv, _ = cv2.projectPoints(obj, cv2.Rodrigues(r["T_CM"][:3, :3])[0], r["T_CM"][:3, 3], K, dist)
                    cv2.polylines(f, [uv.reshape(-1, 2).astype(np.int32)], True, GREEN if good else RED, 4, cv2.LINE_AA)
                ui.frame = f
            if n % 10 == 0:
                ui.show(0, (done_before + n) / max(1, total_frames or n_all), f"labelling {name}: frame {n} of {n_all}",
                        ["The classical pipeline (ArUco + IPPE + LM) measures M in every frame.",
                         "Those poses are the LABELS the network learns from.",
                         f"A frame is kept only if the fit is very good (reprojection <= {MAX_RMS_PX:g} px).", "",
                         f"kept so far in {name}: {len(idx)} of {n + 1} frames",
                         "picture: GREEN outline = kept,  RED = not used,  none = M not seen", "",
                         "Runs on the CPU - the GPU starts in step 3.", "(the picture shows one frame in every 100)"])
        n += 1
    cap.release()
    return np.array(idx, int), np.array(Rs, np.float32).reshape(-1, 3, 3), np.array(ts, np.float32).reshape(-1, 3), video


def label_recordings(rec_dir, cfg, K, dist, log=print, ui=None, done_before=0, total_frames=None, reuse=True):
    """Labels of every recording of a folder: {name: (frame indices, R, t, video)}.  Saved in <rec_dir>/labels and reused when
    they are newer than the video and were made with the same MAX_RMS_PX."""
    out = {}
    os.makedirs(os.path.join(rec_dir, "labels"), exist_ok=True)
    for video in sorted(glob.glob(os.path.join(rec_dir, "*.avi"))):
        name = rec_name(video)
        path = os.path.join(rec_dir, "labels", name + "_labels.npz")
        lab = None
        if reuse and os.path.exists(path) and os.path.getmtime(path) > os.path.getmtime(video):
            z = np.load(path)
            made_with = float(z["max_rms_px"]) if "max_rms_px" in z.files else 1.0     # older label files: made with 1.0 px
            if made_with == MAX_RMS_PX:
                lab = (z["frames"], z["R"], z["t"], video)
                log(f"  {name}: {len(lab[0])} labels reused from labels/{os.path.basename(path)}")
        if lab is None:
            lab = label_recording(video, cfg, K, dist, ui, done_before, total_frames)
            np.savez(path, frames=lab[0], R=lab[1], t=lab[2], max_rms_px=MAX_RMS_PX)
            log(f"  {name}: {len(lab[0])} of {frame_count(video)} frames labelled (M detected, reprojection error <= {MAX_RMS_PX:g} px)")
        done_before += frame_count(video)
        out[name] = lab
    return out


# ---- step 2: data sets --------------------------------------------------------------------------------
def build_sets(labels, K, dist, size, out_root, val_name, L=0.077, S=224, jitter=CROP_JITTER, seed=0, step=1, log=print,
               set_name="real_v2", ui=None):
    """Real 'full' and 'crop' data sets in the synthetic format; the recording(s) val_name (a name or a list of names)
    become the validation split - whole recordings, never single frames.
    L = marker side [m]; jitter = random offset of the crop window (fraction of the marker's size in the image)."""
    rng = np.random.default_rng(seed)
    val_names = [val_name] if isinstance(val_name, str) else list(val_name)
    W, H = size
    net_w, net_h = 640, int(round(640 * H / W))
    sets = {"full": {"train": [], "val": []}, "crop": {"train": [], "val": []}}
    n_total, n_done = sum(len(x[0][::step]) for x in labels.values()), 0
    for name, (idx, Rs, ts, video) in labels.items():
        split = "val" if name in val_names else "train"
        lab = {int(i): (R, t) for i, R, t in list(zip(idx, Rs, ts))[::step]}
        cap = cv2.VideoCapture(video)
        n = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if n in lab:
                g = cv2.undistort(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), K, dist)          # the networks see undistorted frames
                R, t = lab[n]
                sets["full"][split].append((cv2.resize(g, (net_w, net_h), interpolation=cv2.INTER_AREA), R, t, None))
                s = K[0, 0] * L / t[2]                                                    # the marker's side in pixels
                u = K[0, 0] * t[0] / t[2] + K[0, 2] + rng.normal(0, jitter * s)
                v = K[1, 1] * t[1] / t[2] + K[1, 2] + rng.normal(0, jitter * s)
                side = int(np.clip(round(CROP_FACTOR * s * np.exp(rng.normal(0, jitter))), CROP_MIN, min(W, H)))
                u0, v0 = int(np.clip(round(u - side / 2), 0, W - side)), int(np.clip(round(v - side / 2), 0, H - side))
                sets["crop"][split].append((cv2.resize(g[v0:v0 + side, u0:u0 + side], (S, S), interpolation=cv2.INTER_AREA), R, t, (u0, v0, side)))
                n_done += 1
                if ui is not None and ui.on:
                    if n_done % 100 == 1:                             # one example in every 100: the crop window on the frame
                        ui.frame = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                        cv2.rectangle(ui.frame, (u0, v0), (u0 + side, v0 + side), ORANGE, 4)
                    if n_done % 10 == 0:
                        ui.show(1, n_done / max(1, n_total), f"building the sets: {name} ({'VALIDATION' if split == 'val' else 'training'})",
                                ["Every kept frame is undistorted (the lens model from T3) and stored twice:",
                                 "  whole image 640x360  -> for stage 1 (where is M?)",
                                 "  224 px crop around M -> for stage 2 (exact pose);  ORANGE box in the picture",
                                 "The crop is shifted at random, like stage 1's real errors.", "",
                                 f"frames done: {n_done} of {n_total}",
                                 f"validation (never trained on): {', '.join(val_names)}", "",
                                 "(the picture shows one frame in every 100)"])
            n += 1
        cap.release()
    for split in ("train", "val"):
        if not sets["full"][split]:
            raise SystemExit(f"no labelled frames for the {split} split - check the recordings (validation: {', '.join(val_names)})")
    dirs = {}
    for kind in ("full", "crop"):
        d = os.path.join(out_root, f"{set_name}_{kind}")
        os.makedirs(d, exist_ok=True)
        for split in ("train", "val"):
            items = sets[kind][split]
            np.save(os.path.join(d, f"X_{split}.npy"), np.stack([x[0] for x in items]).astype(np.uint8))
            np.save(os.path.join(d, f"R_{split}.npy"), np.stack([x[1] for x in items]).astype(np.float32))
            np.save(os.path.join(d, f"t_{split}.npy"), np.stack([x[2] for x in items]).astype(np.float32))
            if kind == "crop":
                np.save(os.path.join(d, f"box_{split}.npy"), np.array([x[3] for x in items], np.float32))
        log(f"  real {kind} set: {len(sets[kind]['train'])} train, {len(sets[kind]['val'])} validation frames -> {d}")
        dirs[kind] = d
    return dirs, (net_w, net_h)


# ---- the guided run -----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Task 5: fine-tune on real frames labelled by the classical pipeline.")
    ap.add_argument("--sets", nargs="+", default=["real_v2"], help="training folders (training/<set>) whose recordings are used, e.g. real_v2 real_v3")
    ap.add_argument("--suffix", default="_ft", help="suffix of the new networks (e.g. _ft2: the earlier _ft networks are kept)")
    ap.add_argument("--val", nargs="+", default=None, help="validation recording(s), whole recordings never trained on "
                    "(default: the last recording); give one per kind of view, e.g. real_v2_05 real_v3_05 real_v4_05")
    ap.add_argument("--yes", action="store_true", help="answer every question with its default (and overwrite networks with the same suffix)")
    ap.add_argument("--relabel", action="store_true", help="label every recording again instead of reusing the saved labels")
    ap.add_argument("--rebuild", action="store_true", help="build the real data sets again instead of reusing the saved ones")
    ap.add_argument("--restart", action="store_true", help="forget the training checkpoints: train both stages from their first epoch")
    a = ap.parse_args()
    cfg = load_markers_cfg()
    K, dist, size = load_camera()
    vids = [v for s in a.sets for v in sorted(glob.glob(os.path.join(training_path(s), "*.avi")))]
    names = [rec_name(v) for v in vids]
    if len(vids) < 2:
        raise SystemExit(f"need at least 2 training recordings in {', '.join(a.sets)} (one is kept for validation)")
    set_name = "_".join(a.sets)
    print(f"Task 5 - fine-tuning on real frames.  {len(vids)} training recordings in {', '.join(a.sets)}; "
          f"the new networks get the suffix {a.suffix}.  Press Enter to accept the [default].\n")
    q = (lambda text, default, *rest: default) if a.yes else None            # --yes: no questions
    stage1 = (q or ask)("1) Stage-1 (whole-image) network to fine-tune", "resnet18_v2")
    stage2 = (q or ask)("2) Stage-2 (crop) network to fine-tune", "crop_resnet18_v1")
    val_names = (q or ask)("3) Recording(s) kept for validation, not trained on (several: separate with spaces)",
                           " ".join(a.val or [names[-1]])).replace(",", " ").split()
    epochs = (q or ask_number)("4) Fine-tuning epochs, stage 1 (whole image; slow, ~1.7 min per epoch)", 25, int, 1)
    epochs2 = (q or ask_number)("   Fine-tuning epochs, stage 2 (crop; fast, ~0.3 min per epoch)", 50, int, 1)
    lr = (q or ask_number)("5) Learning rate (peak; low for fine-tuning)", 1e-4, float, 1e-8)
    gui = True if a.yes else ask_yes("6) Show the live windows (preparation + training)?", "y")
    batch = (q or ask_number)("7) Batch size (half real, half synthetic; 128 keeps the GPU busy)", 128, int, 2)

    # everything that could stop the run is checked BEFORE the work starts
    for v in val_names:
        if v not in names:
            raise SystemExit(f"'{v}' is not one of the recordings: {', '.join(names)}")
    if len(set(val_names)) >= len(names):
        raise SystemExit("every recording is a validation recording: nothing left to train on")
    print()
    print(f"training recordings:   {', '.join(n for n in names if n not in val_names)}")
    print(f"validation recordings: {', '.join(val_names)}")
    for base in (stage1, stage2):
        if not os.path.exists(training_path("models", base + ".pt")):
            raise SystemExit(f"network not found: {training_path('models', base + '.pt')}")
    for syn in SYNTHETIC.values():
        if not os.path.exists(training_path("synthetic", syn, "meta.json")):
            raise SystemExit(f"synthetic set not found: {training_path('synthetic', syn)}  (t5_make_data.py / t5_make_crops.py)")
    for base in (stage1, stage2):                                 # never overwrite an earlier network by accident
        if os.path.exists(training_path("models", base + a.suffix + ".pt")) and not a.yes and not ask_yes(
                f"   {base}{a.suffix}.pt exists already: overwrite it? (n = stop; run again with another --suffix)", "n"):
            raise SystemExit("stopped: nothing changed")

    print("\nstep 1 - labelling the real frames with the classical pipeline ...")
    ui = PrepWindow(gui)
    total = sum(frame_count(v) for v in vids)
    labels, done = {}, 0
    for s in a.sets:
        lab = label_recordings(training_path(s), cfg, K, dist, ui=ui, done_before=done, total_frames=total, reuse=not a.relabel)
        done += sum(frame_count(x[3]) for x in lab.values())
        labels.update(lab)
    # the built sets are kept: the same recordings, labels and validation choice -> reused the next time (seconds, not minutes)
    info = dict(labels={n: int(len(x[0])) for n, x in sorted(labels.items())}, validation=sorted(val_names), crop_jitter=CROP_JITTER,
                max_rms_px=MAX_RMS_PX, marker_side_m=cfg["L"])
    dirs = {k: os.path.join(training_path("synthetic"), f"{set_name}_{k}") for k in ("full", "crop")}
    reuse = not a.rebuild and not a.relabel
    for d in dirs.values():
        f = os.path.join(d, "build_info.json")
        reuse = reuse and os.path.exists(f) and json.load(open(f)) == info and os.path.exists(os.path.join(d, "X_train.npy"))
    if reuse:
        print(f"step 2 - the real data sets are already built: reused ({', '.join(os.path.basename(d) for d in dirs.values())})")
    else:
        print("step 2 - building the real data sets ...")
        dirs, _ = build_sets(labels, K, dist, size, training_path("synthetic"), val_names, L=cfg["L"], set_name=set_name, ui=ui)
        for d in dirs.values():
            json.dump(info, open(os.path.join(d, "build_info.json"), "w"), indent=1)
    n_tr = sum(len(x[0]) for k, x in labels.items() if k not in val_names)
    n_va = sum(len(labels[k][0]) for k in val_names)
    ui.show(2, 1.0, "ready - the training window opens now",
            [f"{n_tr} real training frames, {n_va} validation frames ({', '.join(val_names)}).",
             "Every batch = half real frames + half synthetic images (keeps what was learned).",
             "First the whole-image network, then the crop network.", "", "This window closes by itself."], force=True)
    if ui.on:
        cv2.waitKey(1500)
    ui.close()

    import t5_train
    if a.restart:                                                 # flush the training only (labels and data sets are kept)
        for base in (stage1, stage2):
            ck = training_path("models", base + a.suffix + "_checkpoint.pt")
            if os.path.exists(ck):
                os.remove(ck)
                print(f"checkpoint of {base}{a.suffix} removed: it trains from epoch 1")
    for kind, base in (("full", stage1), ("crop", stage2)):
        syn = SYNTHETIC[kind]
        meta = json.load(open(training_path("synthetic", syn, "meta.json")))
        meta.update(name=os.path.basename(dirs[kind]), real=True, from_recordings=sorted(labels), validation_recording=", ".join(val_names))
        if kind == "crop":
            meta.update(stage1=stage1 + a.suffix + ".pt", crop_jitter_real=CROP_JITTER)   # the crop network works behind the NEW stage 1
        json.dump(meta, open(os.path.join(dirs[kind], "meta.json"), "w"), indent=1)
        print(f"\nstep 3 - fine-tuning {base} -> {base}{a.suffix} on real ({kind}) + synthetic ({syn}) ...")
        t5_train.train(dirs[kind], base + a.suffix, epochs=epochs if kind == "full" else epochs2, batch=batch, lr=lr, w_rot=1.0, w_trans=1.0, seed=0, pretrained=False,
                       gui=gui, keep_best=True, hard_aug=HARD_AUG, rot_aug=ROT_AUG_DEG if kind == "full" else 0.0,
                       init_from=training_path("models", base + ".pt"), mix_with=training_path("synthetic", syn))
    print(f"\ndone: {stage1}{a.suffix}.pt and {stage2}{a.suffix}.pt in {training_path('models')}")


if __name__ == "__main__":
    main()
