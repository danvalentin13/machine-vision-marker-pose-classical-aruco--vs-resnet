"""live.py -- one command for the live view with everything at once. Run it from the project folder:

    python live.py

It checks the set-up first and repairs what it can, then opens the camera window of code/t8_live_all.py:
GREEN = classical pipeline, RED = network, BLUE = occlusion predictor, with live position, angles and distances.

What it checks, in this order
  1. the project folder (this file must sit next to the folders code/ and data/)
  2. the Python environment: the project's .venv should be active and OpenCV (with ArUco) and NumPy must import;
     if something is missing it prints the exact commands and stops
  3. the markers: if data/markers.json or the marker images are missing, the two markers and the dictionary table are
     generated with the project's defaults (DICT_6X6_250, F = id 0, M = id 23, 80 mm); an existing measured size is kept
  4. the camera calibration: data/calibration/camera.npz (rebuilt from camera.json if only that exists); without any
     calibration an APPROXIMATE camera is used, so that the window still runs - distances are then only rough
  5. the network: training/models/<name>.pt; without it (or with an approximate camera) the window runs without the red outline

Options
  --check               only run the checks, do not open the camera
  --regenerate-markers  write the marker images and the dictionary table again
  everything else is passed on to t8_live_all.py, e.g.  --video E4_cover   --camera 1   --network NAME
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(ROOT, "code")
DEFAULTS = dict(dictionary="DICT_6X6_250", id_fixed=0, id_moving=23, side_mm=80.0)


def say(n, text):
    print(f"[{n}/5] {text}")


def stop(text):
    print("\nCANNOT START: " + text)
    raise SystemExit(1)


def check_folder():
    for d in ("code", "data"):
        if not os.path.isdir(os.path.join(ROOT, d)):
            stop(f"the folder '{d}' is missing next to live.py ({ROOT}). Keep live.py in the project folder.")
    say(1, f"project folder: {ROOT}")


def check_python():
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    venv = os.path.join(ROOT, ".venv")
    activate = r".venv\Scripts\Activate.ps1" if os.name == "nt" else "source .venv/bin/activate"
    missing = []
    try:
        import numpy  # noqa: F401
    except Exception:
        missing.append("numpy")
    try:
        import cv2
        if not hasattr(cv2, "aruco"):
            missing.append("opencv-contrib-python (this OpenCV has no ArUco module)")
    except Exception:
        missing.append("opencv-contrib-python")
    try:
        import PIL  # noqa: F401
        import matplotlib  # noqa: F401
    except Exception:
        missing.append("pillow / matplotlib")
    if missing:
        how = ([] if os.path.isdir(venv) else ["python -m venv .venv"]) + ([] if in_venv else [activate]) + [
            "python -m pip install -r requirements.txt",
            "python -m pip install torch torchvision    (only for the network; with an RTX 50-series card add  --index-url https://download.pytorch.org/whl/cu128)"]
        stop("these packages are missing: " + ", ".join(missing) + "\n  In the project folder run:\n    " + "\n    ".join(how))
    try:
        import torch
        tinfo = f"PyTorch {torch.__version__}, " + ("GPU " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU (the network runs on the processor, slower)")
    except Exception:
        tinfo = "PyTorch not installed: the window runs without the network"
    import cv2
    import numpy
    say(2, f"Python {sys.version.split()[0]}, OpenCV {cv2.__version__}, NumPy {numpy.__version__}; {tinfo}")
    if not in_venv:
        print(f"      note: the project's virtual environment is not active ({activate}); continuing with this Python because the packages are there")


def check_markers(regenerate):
    import json
    cfg_path = os.path.join(ROOT, "data", "markers.json")
    out = os.path.join(ROOT, "output", "markers")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else None
    have = cfg is not None and all(os.path.exists(os.path.join(out, f"marker_{r}_id{cfg[k]}.png")) for r, k in (("F", "id_fixed"), ("M", "id_moving")))
    if regenerate or not have:
        import t1_generate_markers as t1
        p = dict(DEFAULTS) if cfg is None else dict(dictionary=cfg["dictionary"], id_fixed=cfg["id_fixed"], id_moving=cfg["id_moving"], side_mm=cfg["nominal_side_mm"])
        print("      generating the markers and the dictionary table ...")
        t1.generate(out, os.path.join(ROOT, "output", "dictionary"), p["dictionary"], p["id_fixed"], p["id_moving"], p["side_mm"])
        cfg = json.load(open(cfg_path))
    side = cfg.get("measured_side_mm") or cfg["nominal_side_mm"]
    say(3, f"markers: {cfg['dictionary']}, F = id {cfg['id_fixed']}, M = id {cfg['id_moving']}, side {side:g} mm "
           f"({'measured' if cfg.get('measured_side_mm') else 'NOMINAL - measure the print and write measured_side_mm in data/markers.json'}); images in output\\markers")


def check_camera():
    """Returns (path of the camera file, exact?)."""
    import json
    import numpy as np
    npz = os.path.join(ROOT, "data", "calibration", "camera.npz")
    js = os.path.join(ROOT, "data", "calibration", "camera.json")
    if not os.path.exists(npz) and os.path.exists(js):
        d = json.load(open(js))
        np.savez(npz, K=np.array(d["K"], float), dist=np.array(d["dist"], float), width=d["width"], height=d["height"], rms=d.get("rms_px", 0.0))
        print("      camera.npz rebuilt from camera.json")
    if os.path.exists(npz):
        d = np.load(npz)
        say(4, f"camera calibration: fx {float(d['K'][0, 0]):.0f} px, {int(d['width'])} x {int(d['height'])} px, RMS {float(d['rms']):.2f} px")
        print("      (this is the calibration of the laptop camera used in the project; on another computer the distances are only as good as"
              "\n       the match between the cameras - recalibrate with README steps 3.1 and 3.2 for real measurements)")
        return npz, True
    approx = os.path.join(ROOT, "data", "calibration", "camera_approx.npz")
    os.makedirs(os.path.dirname(approx), exist_ok=True)
    w, h = 1280, 720
    f = 0.5 * w / np.tan(np.radians(74.0) / 2)                         # a typical laptop camera: 74 deg horizontal field of view
    np.savez(approx, K=np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]), dist=np.zeros(5), width=w, height=h, rms=0.0)
    say(4, "camera calibration NOT FOUND: using an approximate camera (distances are only rough, the network is switched off).\n"
           "      Calibrate with README steps 3.1 and 3.2 (code\\t3_calibrate.py) for real measurements.")
    return approx, False


def check_network(name, exact):
    path = os.path.join(ROOT, "training", "models", name + ".pt")
    if not exact:
        say(5, "network: off (it is only valid for the calibrated camera)")
        return False
    try:
        import torch  # noqa: F401
    except Exception:
        say(5, "network: off (PyTorch is not installed)")
        return False
    if not os.path.exists(path):
        say(5, f"network: {name}.pt not found in training\\models - the window runs without the red outline (train it with README steps 5.1 - 5.6)")
        return False
    say(5, f"network: {name}.pt ({os.path.getsize(path) / 1e6:.0f} MB)")
    return True


def main():
    args = sys.argv[1:]
    only_check = "--check" in args
    regen = "--regenerate-markers" in args
    args = [a for a in args if a not in ("--check", "--regenerate-markers")]
    name = args[args.index("--network") + 1] if "--network" in args and args.index("--network") + 1 < len(args) else "crop_resnet18_v1_ft3"
    print("Live view: classical (green), network (red), occlusion predictor (blue)\n")
    check_folder()
    check_python()
    sys.path.insert(0, CODE)
    check_markers(regen)
    cam_file, exact = check_camera()
    net_ok = check_network(name, exact)
    if only_check:
        print("\nAll checks done. Start the live view with:  python live.py")
        return
    print("\nOpening the camera window ... (click it, then S screenshot, N network on/off, C restart predictor, Q quit)\n")
    import t8_live_all
    sys.argv = ["t8_live_all.py", "--camera-file", cam_file] + ([] if net_ok else ["--no-network"]) + args
    t8_live_all.main()


if __name__ == "__main__":
    main()
