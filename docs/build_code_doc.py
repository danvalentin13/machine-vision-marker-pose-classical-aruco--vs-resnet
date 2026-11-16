"""build_code_doc.py -- writes docs/Code_Documentation.docx: the code of the project documented step by step, in the order and with
the numbers of README.md (steps 0.1 ... 8.1), with the inputs, outputs and links of every step and a reference of the shared file formats.

    .venv\\Scripts\\python.exe docs\\build_code_doc.py

Uses paper/docx_lib.py (house style) read-only and draws one diagram (docs/code_documentation_build/dataflow.py, checked with
paper/figcheck.py).  All facts were taken from the code in code/ and from read-only checks of the real files
(docs/code_documentation_build/inspect_files.py prints them).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
BUILD = os.path.join(HERE, "code_documentation_build")
sys.dont_write_bytecode = True                      # paper/ is used read-only: no __pycache__ written there
sys.path.insert(0, os.path.join(ROOT, "paper"))
sys.path.insert(0, BUILD)
from docx_lib import build  # noqa: E402
import dataflow  # noqa: E402

OUT = os.path.join(HERE, "Code_Documentation.docx")
FIG = os.path.join(BUILD, "dataflow.png")

NOT_VERIFIED = []          # filled while the steps are written: (step, text)

# ================================================================================================ shared phrases
REC_AVI = r"AVI, MJPG, 1280x720 BGR frames as delivered by the camera, header rate 30 fps (real rate about 30 fps, see the `.csv`)"
REC_CSV = r"CSV `frame,t_s`: frame index and time since recording start [s] (first row about 0.005-0.03 s)"
REC_JSON = (r"JSON, keys `name, tag, notes, distance_cm, tilt_deg, resolution, frames, duration_s, fps, dictionary, created` "
            r"({sec:fmt_rec})")
TRUTH_1 = r"CSV `t_s,x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg`, one row (`t_s` = 0): static pose of M in F [mm, deg]"
POSES = r"CSV, one row per frame, 31 columns (flags, reprojection errors, `FM_`, `CM_`, `CF_` poses, roll / pitch / yaw); see {tab:poses}"
MARKERS = r"`data\markers.json` (step 1.1, edited in 1.4)"
CAMERA = r"`data\calibration\camera.npz` (step 3.2)"
JSON_MK = r"JSON; `dictionary`, `id_fixed`, `id_moving`, `measured_side_mm` (L = 77 mm used); {tab:markers}"
NPZ_CAM = r"NPZ: `K` 3x3 float64, `dist` (5,) float64, `width`, `height`, `rms`; {tab:camera}"
SZ_MK = "254 B"
SZ_CAM = "1.3 KB"


def nv(step, text):
    NOT_VERIFIED.append((step, text))
    return f"*Not verified:* {text}"


# ================================================================================================ the steps
# Each step: num, title, cmd, script, what (2-5 sentences), funcs [list], opts [list], io [(IN/OUT, path, source or use, format, size)],
#            notes [extra paragraphs]
S = []


def step(num, title, cmd, script, what, funcs=(), opts=(), io=(), notes=()):
    S.append(dict(num=num, title=title, cmd=cmd, script=script, what=what, funcs=list(funcs), opts=list(opts), io=list(io), notes=list(notes)))


# ------------------------------------------------------------------------------------------------ Setup
step("0.1", "Open the project folder", r'cd "C:\Study\TUIASI\MLRC\an2\sem1\Machine Vision\Lab 2"', "(PowerShell command, no script)",
     "Makes the project root the working folder of the terminal. Steps 0.2-0.5 use paths relative to it (`.venv`, `requirements.txt`). "
     "It reads and writes no file.",
     notes=["Inputs: none. Outputs: none. Link: every new terminal starts with 0.1, 0.5 and 0.6."])

step("0.2", "Create the venv (once)", r"python -m venv .venv", "(Python standard module `venv`)",
     "Creates a private Python environment in the folder `.venv` next to `code`, so that the packages of this project do not mix with "
     "other projects. Only needed when `.venv` does not exist yet.",
     io=[("IN", "system Python", "installed on the laptop", "Python interpreter", "-"),
         ("OUT", r"`.venv\` (`Scripts\python.exe`, `Scripts\Activate.ps1`, `Lib\site-packages\`)", "used by 0.3-0.5 and by every later command",
          "folder with the interpreter and its packages", "not measured")],
     notes=[nv("0.2", "the Python version of `.venv` and the size of the folder were not inspected.")])

step("0.3", "Install the packages (once)", r".venv\Scripts\python.exe -m pip install -r requirements.txt", "(pip)",
     "Installs the packages listed in `requirements.txt` into `.venv`: NumPy, OpenCV with the contrib modules (ArUco), Matplotlib, Pillow. "
     "PyTorch is not in the file (see 0.4).",
     opts=["`requirements.txt` asks for `numpy>=2.0`, `opencv-contrib-python>=4.7`, `matplotlib>=3.8`, `pillow>=10.0`"],
     io=[("IN", "`requirements.txt`", "project file", "text, one requirement per line, comments with #", "0.3 KB"),
         ("OUT", r"`.venv\Lib\site-packages\`", "imported by every script (`cv2`, `numpy`, `matplotlib`, `PIL`)", "installed packages", "not measured")],
     notes=["The documentation builder also needs `python-docx` (it is importable in the existing `.venv`; it is not listed in `requirements.txt`)."])

step("0.4", "Install PyTorch with CUDA (once)",
     r".venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128", "(pip)",
     "Installs PyTorch and torchvision built for CUDA 12.8, for the RTX 5080 GPU. Only the learned pipeline needs them "
     "(`netpose.py`, `t5_train.py`, `t5_finetune.py`, `t5_eval_real.py`, `t5_test_hard_cases.py`, `t6_evaluate.py`, `t6_side_by_side.py`).",
     io=[("IN", "download.pytorch.org (index cu128)", "internet", "wheels", "-"),
         ("OUT", r"`.venv\Lib\site-packages\torch`, `torchvision`", "imported by the network scripts; the ImageNet ResNet-18 weights are downloaded "
          "by torchvision the first time `netpose.build_model(pretrained=True)` runs", "installed packages; the existing `.venv` has torch 2.11.0+cu128 "
          "(checked by import)", "not measured")])

step("0.5", "Activate the venv", r".venv\Scripts\Activate.ps1", "(PowerShell script created by 0.2)",
     "Puts `.venv\\Scripts` first on the PATH of this terminal, so that `python` in the later steps is the interpreter of the venv "
     "(the prompt starts with `(.venv)`). It writes no file.",
     notes=["Inputs: `.venv\\Scripts\\Activate.ps1` (step 0.2). Outputs: none (only the environment of the terminal)."])

step("0.6", "Go to the code folder", r"cd code", "(PowerShell command)",
     "Makes `code` the working folder: every later command is `python <script>.py` and the scripts import each other "
     "(`config`, `mvlib`, `mvtools`, ...) from this folder. The paths of the data do not depend on it: they come from `config.py`. "
     "A `can't open file ... No such file or directory` error means the terminal is in the wrong folder. Only `python live.py` (step 8.1) "
     "works from both folders: `live.py` in the project root, and a stub `code\\live.py` that runs it.",
     notes=["Inputs: none. Outputs: none."])

step("0.7", "Check the folders", r"python config.py", "`code\\config.py`",
     "Prints `PROJECT_ROOT`, `DATA_DIR`, `OUTPUT_DIR` and `TRAINING_DIR`. Every script imports its paths from this one file, so the check "
     "shows where everything is read and written. On import the three folders are created if they are missing.",
     funcs=[r"`config.py`: module level: each folder = environment variable `MV_DATA_DIR` / `MV_OUTPUT_DIR` / `MV_TRAINING_DIR`, else the "
            r"`<name>_OVERRIDE` value, else `<project>\data`, `<project>\output`, `<project>\training`; `os.makedirs(..., exist_ok=True)`",
            r"`config.py: data_path(*parts)`, `training_path(*parts)`: join a path inside the folder",
            r"`config.py: output_path(*parts)`: the same inside `output`, and creates the parent folders"],
     opts=["no options; `DATA_DIR_OVERRIDE`, `OUTPUT_DIR_OVERRIDE`, `TRAINING_DIR_OVERRIDE` = None (edit to move the folders)"],
     io=[("IN", "environment variables `MV_DATA_DIR`, `MV_OUTPUT_DIR`, `MV_TRAINING_DIR`", "optional, set by the user (and by `selftest.py`)", "text", "-"),
         ("OUT", "terminal: four absolute paths", "read by the user", "text", "-"),
         ("OUT", r"`data\`, `output\`, `training\` (created only if missing)", "every later step", "folders", "-")])

step("0.8", "Self-test without hardware", r"python selftest.py --quick", "`code\\selftest.py` (+ `synth.py`), option `--quick`",
     "Runs the core of the classical pipeline on simulated data in a temporary folder (`mv_selftest_<random>` in the system temp folder): "
     "the three `MV_..._DIR` variables are pointed there before `config` is imported, so `data`, `output` and `training` of the project are "
     "not touched. It generates the markers (T1), renders 14 chessboard views with a known camera (fx 900 px) and calibrates (T3), renders a "
     "synthetic recording with exact truth (150 frames, M hidden in frames 60-74) and checks the poses against the truth. With `--quick` the "
     "animation, the bonus (occlusion predictor) and the network part are skipped (they run in step 0.9). Each check prints PASS / FAIL; the "
     "exit code is 1 if one fails.",
     funcs=[r"`selftest.py: main()`: with `--quick` runs `step_t1()`, `step_calibration()`, `step_pose()` only",
            r"`selftest.py: step_t1()`, `step_calibration()`, `step_pose()`: one block of checks each",
            r"`selftest.py: make_recording()`: synthetic `.avi` + `.csv` + `.truth.csv` + `.json` with `synth.Renderer` (lecture scene, 100 mm markers)",
            r"`synth.py: Renderer`, `moving_marker_pose(t)`, `camera_pose(t, moving)`: the lecture's synthetic scene (K fx = fy = 900 px, 1280x720)",
            r"calls into the real scripts: `t1_generate_markers.generate`, `t3_calibrate.run_calibration`, `t3_pose.process_video`"],
     opts=["`--quick` (given here): skip the animation, the bonus and the network part", "`--keep`: keep the temporary folder and print its path"],
     io=[("IN", "none from the project", "everything is generated", "-", "-"),
         ("OUT", r"`%TEMP%\mv_selftest_<random>\data`, `output`, `training`", "nothing later (deleted at the end unless `--keep`)",
          "the same file types as steps 1.1, 1.2, 3.2, 3.5, at small size", "temporary"),
         ("OUT", "terminal: PASS / FAIL per check, `n/m checks passed`", "the user", "text", "-")])

step("0.9", "Self-test of the network code", r"python selftest.py", "`code\\selftest.py` (+ `synth.py`, `selftest_learned.py`)",
     "The same self-test as 0.8 without `--quick`: after the classical checks it also exports the animation (T4), runs the occlusion predictor "
     "on two synthetic recordings with M hidden for 0.5 s and 1.5 s (bonus), and calls `selftest_learned.run()` for the network code: a tiny "
     "synthetic set (300 / 60 / 100 images, 320 px wide), a check that the labels agree with the pixels (classical PnP on the synthetic images), "
     "a check that the 6-D output gives rotation matrices, a throw-away training of 2 epochs, a `NetPose` prediction and a small run of "
     "`t6_evaluate.evaluate`. Everything is written into the temporary folder and deleted at the end.",
     funcs=[r"`selftest.py: step_animation()`: `t4_animate.export` (MP4 + GIF) and `filter_poses` on the synthetic recording",
            r"`selftest.py: step_bonus()`: `t7_bonus_occlusion.evaluate` (prediction against hold, lost rule, re-acquisition)",
            r"`selftest_learned.py: run(check, cfg)`: the network checks listed above; `check` is the PASS / FAIL printer of `selftest.py`",
            r"`netpose.py: six_d_to_R()`: tested for orthonormality and determinant 1 on 64 random 6-D vectors",
            r"calls into `t5_make_data.make_dataset`, `t5_train.train`, `netpose.NetPose`, `t6_evaluate.evaluate`"],
     opts=["no `--quick`; `--keep` as in 0.8"],
     io=[("IN", "none from the project", "everything is generated (ImageNet weights downloaded by torchvision if not cached)", "-", "-"),
         ("OUT", r"temporary `output\animation\synth.mp4` / `.gif`, `output\occlusion\`, `training\synthetic\tiny`, `training\models\selftest_net.pt`, "
          r"`output\network\selftest\`, `output\evaluation\`", "nothing later (deleted with the temporary folder unless `--keep`)",
          "same formats as 4.3, 5.1, 5.2, 6.4", "temporary"),
         ("OUT", "terminal: PASS / FAIL per check, `n/m checks passed`, the comparison table of the throw-away network", "the user", "text", "-")],
     notes=["`selftest_learned.py` has no `if __name__ == \"__main__\"` block: it is not run on its own, only through `selftest.py`."])

# ------------------------------------------------------------------------------------------------ T1
step("1.1", "Generate the two markers", r"python t1_generate_markers.py", "`code\\t1_generate_markers.py`",
     "Draws marker F (fixed, id 0) and marker M (moving, id 23) from OpenCV's predefined dictionary `DICT_6X6_250` with a white quiet zone "
     "of 2 cells, as PNG files whose stored dpi makes the black square exactly the asked side when printed at 100 %. It also writes the "
     "whole dictionary (bit table and a contact sheet) and the configuration `data\\markers.json` read by every later step. "
     "A self-test reads both ids back from the saved images.",
     funcs=[r"`t1_generate_markers.py: make_marker_image(dictionary, id, side_mm)`: `cv2.aruco.generateImageMarker` with a whole number of "
            r"pixels per cell at about 300 dpi, quiet zone added with `cv2.copyMakeBorder`; returns the image and its dpi",
            r"`t1_generate_markers.py: write_dictionary_files()`: bit pattern of every id (`getBitsFromByteList`) to `dictionary.csv` / `.json`",
            r"`t1_generate_markers.py: write_dictionary_sheet()`: A4 pages (6 x 9 markers, 150 dpi), ids 0 and 23 framed orange",
            r"`t1_generate_markers.py: generate()`: writes everything, keeps `measured_side_mm` only if dictionary, ids and side are unchanged",
            r"`mvtools.py: ask()`, `ask_folder()`: terminal questions with a [default]"],
     opts=["asked (or as options): `--markers-out` [output\\markers], `--dict-out` [output\\dictionary], `--dictionary` [DICT_6X6_250], "
           "`--id-fixed` [0], `--id-moving` [23], `--side-mm` [80]", "constants: `QUIET_CELLS` = 2, `TARGET_DPI` = 300"],
     io=[("IN", "OpenCV predefined dictionary", "`cv2.aruco`", "in memory", "-"),
         ("OUT", r"`output\markers\marker_F_id0.png`, `marker_M_id23.png`", "1.3 (print sheet); recognised by `mvtools.discover_marker_sets` (2.1) if "
          "`markers.json` is missing", "PNG 8-bit grey 1416x1416, dpi 299.72 stored; black square 944 px = 80 mm", "5 KB each"),
         ("OUT", r"`output\markers\markers_info.json`", "`mvtools.discover_marker_sets` (dictionary of found marker images)",
          "JSON `dictionary, id_fixed, id_moving, nominal_side_mm, quiet_zone_cells`", "123 B"),
         ("OUT", r"`output\dictionary\dictionary.csv`, `dictionary.json`", "report only", "CSV `id`, `bits_6x6_row_major (1 = white, 0 = black)` "
          "(250 rows); JSON `dictionary, grid, n_markers, note`", "10 KB; 180 B"),
         ("OUT", r"`output\dictionary\dictionary_sheet.pdf`, `sheet\page_1..5.png`", "report only", "PDF 5 pages A4 150 dpi; PNG 1240x1754 RGB",
          "0.99 MB"),
         ("OUT", r"`data\markers.json`", "every later step through `mvtools.load_markers_cfg()`; edited in 1.4", JSON_MK, SZ_MK)])

step("1.2", "Make the chessboard", r"python t3_make_board.py", "`code\\t3_make_board.py`",
     "Draws the calibration target: a chessboard with 9 x 6 inner corners (10 x 7 squares) of 25 mm on an A4 landscape page, at a dpi "
     "chosen so that a square is a whole number of pixels. It checks that OpenCV finds all corners in the image and writes "
     "`board.json`, which tells the calibration the corner count and the square size.",
     funcs=[r"`t3_make_board.py: make_board_image(cols, rows, square_mm)`: the board image and its dpi; refuses a board that does not fit A4",
            r"`t3_make_board.py: main()`: asks, saves the PNG with its dpi, self-test with `cv2.findChessboardCornersSB`, writes `board.json` "
            r"(keeps `measured_square_mm` only if cols, rows and square are unchanged)",
            r"`mvtools.py: ask_number()`, `ask_folder()`"],
     opts=["asked: inner corners columns [9], rows [6], square [25 mm], folder [output\\calibration]; DPI constant 300"],
     io=[("OUT", r"`output\calibration\chessboard_9x6_25mm.png`", "1.3 (the newest `chessboard_<...>.png` goes on page 3)",
          "PNG 8-bit grey 3505x2478, dpi 299.72", "19.5 KB"),
         ("OUT", r"`data\calibration\board.json`", "3.1 (capture guide) and 3.2 (`t3_calibrate.load_board`); edited in 1.4",
          "JSON `cols, rows, square_mm, measured_square_mm, note`; {tab:board}", "200 B")])

step("1.3", "Make the print sheet", r"python print_sheet.py", "`code\\print_sheet.py`",
     "Puts marker F, marker M and the chessboard, each untouched and centred, on A4 pages whose pixel size matches the image dpi, and saves "
     "them as one PDF at exact physical size. Printed at 100 % (actual size) the black squares are 80 mm and the chessboard squares 25 mm.",
     funcs=[r"`print_sheet.py: page_with(img_path, caption, landscape)`: white A4 page at the image dpi, image pasted in the centre, caption at the bottom",
            r"`print_sheet.py: main()`: reads `markers.json` for the ids, finds the newest chessboard, writes the PNG pages and the 3-page PDF"],
     opts=["asked: folder [output\\print]"],
     io=[("IN", r"`data\markers.json`", "step 1.1", "JSON (`id_fixed`, `id_moving`, `dictionary`, `nominal_side_mm`)", SZ_MK),
         ("IN", r"`output\markers\marker_F_id0.png`, `marker_M_id23.png`", "step 1.1", "PNG with dpi", "5 KB each"),
         ("IN", r"`output\calibration\chessboard_9x6_25mm.png`", "step 1.2 (newest `chessboard_<...>.png`)", "PNG with dpi", "19.5 KB"),
         ("OUT", r"`output\print\print_sheet.pdf`", "printed by hand, then measured in 1.4", "PDF, 3 pages: F, M (A4 portrait), board (A4 landscape)", "430 KB"),
         ("OUT", r"`output\print\page_1_marker_F.png`, `page_2_marker_M.png`, `page_3_chessboard.png`", "none (copies of the pages)",
          "PNG grey, 2478x3505 (portrait) / 3505x2478, dpi 299.72", "20-25 KB each")])

step("1.4", "Print, measure and write the measured sizes", r"notepad ..\data\markers.json", "(text editor; no script)",
     "The printed squares are measured with a calliper and the values are typed into the two configuration files: "
     "`measured_side_mm` = 77.0 in `markers.json` and `measured_square_mm` = 24.0 in `board.json` (the printer scaled the page to about 96 %). "
     "From now on every pose uses L = 77 mm (`mvtools._finish_cfg`: L = measured side / 1000, else the nominal side) and the calibration "
     "uses 24 mm squares (`t3_calibrate.load_board`: `square_used_mm`).",
     io=[("IN", "printed sheet, calliper", "step 1.3 + hand measurement", "-", "-"),
         ("OUT", r"`data\markers.json` (`measured_side_mm`: 77.0)", "every step that computes a pose (marker side L in `marker_object_points`), "
          "the synthetic renderer (5.1, 5.3) and the checkpoints (`L_m` = 0.077)", "JSON; {tab:markers}", SZ_MK),
         ("OUT", r"`data\calibration\board.json` (`measured_square_mm`: 24.0)", "3.2: metric size of the board, hence the scale of all translations",
          "JSON; {tab:board}", "200 B")],
     notes=[r"`output\calibration\measurements.csv` / `.json` record these measurements (columns `item, what, intended_mm, measured_mm, "
            r"difference_mm, scale_pct, within_1mm, date`); no script in `code\` writes them (searched): they are hand-written."])

# ------------------------------------------------------------------------------------------------ T2
step("2.1", "Live detection", r"python t2_detect_live.py --mode live", "`code\\t2_detect_live.py`",
     "Opens the camera (DirectShow, 1280x720) and detects ArUco markers in every frame with sub-pixel corner refinement; the window shows "
     "the outlines, the ids with their role (F / M) and the loop frame rate. Keys start the other T2 tools from the same window: T runs one "
     "10-frame detection test, L the detection-limit matrix (2.3), G the guided demo (2.2). It computes no pose.",
     funcs=[r"`mvtools.py: select_marker_config()`: `markers.json` if it matches the marker images in `output\markers`, else a choice among the found sets "
            r"(`discover_marker_sets`, `adopt_marker_set` rewrite `markers.json`)",
            r"`t2_detect_live.py: Camera`: `cv2.VideoCapture(index, CAP_DSHOW)` with the asked size; `ImageSource` replays one image (`--image`)",
            r"`mvlib.py: MarkerDetector.detect(gray)`: `cv2.aruco.ArucoDetector` with `CORNER_REFINE_SUBPIX`; returns `id -> (4,2)` corners",
            r"`t2_detect_live.py: annotate()`: outlines, ids, side in pixels, fps bar",
            r"`t2_detect_live.py: measure()`, `detection_test()`: 10 fresh frames (3 dropped first), hits per marker, PASS = 9 of 10, appends a CSV row"],
     opts=["`--mode` live / guided / matrix (asked if missing)", "`--image <png>`: test without camera", "`--markers-folder`: more places to look for marker images",
           "`--fresh`: guided mode, delete earlier step files", "asked: folder [output\\detection], camera index [0], resolution [1280x720]",
           "keys: SPACE pause, C camera on/off, S screenshot, T test, L limits, G guided, H help, Q / Esc quit"],
     io=[("IN", "camera (index 0)", "laptop camera", "BGR frames 1280x720", "-"),
         ("IN", r"`data\markers.json` (or `output\markers\` images + `markers_info.json`)", "steps 1.1 / 1.4", JSON_MK, SZ_MK),
         ("OUT", r"`output\detection\screenshot_<YYYYmmdd_HHMMSS>.png` (key S)", "report only (camera image)", "PNG 1280x720 BGR, annotated", "about 1 MB"),
         ("OUT", r"`output\detection\detection_tests.csv` (key T, appended)", "report only",
          "CSV `time, distance_cm, tilt_deg, frames, resolution, fps, dictionary`, then per role F / M: `<role>_id, _detected, _pass, _side_px, _px_per_cell`",
          "one row per test")],
     notes=[nv("2.1", "`detection_tests.csv` does not exist in the project; its columns are taken from `detection_test()` only.")])

step("2.2", "Guided live demo", r"python t2_detect_live.py --mode guided", "`code\\t2_detect_live.py` -> `code\\t2_guided.py`",
     "The camera detects all the time; a step list on the right asks for four pieces of evidence of the T2 check: both markers (screenshot), "
     "M moving for at least 5 s (recording), M tilted at least 30 deg (screenshot), M at least 80 cm away (screenshot). A screenshot is "
     "checked before it counts (both ids; tilt and distance estimated from the marker pose). At the end the report is built from the saved files.",
     funcs=[r"`t2_guided.py: run_guided()`: the live loop, keys S (screenshot), R (start / stop recording), N / B (next / previous step), Q",
            r"`t2_guided.py: steps(cfg)`, `step_file()`: the four steps and their file names; earlier step files count as done (resume)",
            r"`t2_guided.py: check_shot()`, `wrong_marker_checks()`: is the screenshot valid for its step",
            r"`t2_matrix.py: view_estimate(corners, K, dist, L)`: distance [cm] and tilt [deg] of M from `marker_pose_candidates`",
            r"`t2_matrix.py: guide_camera(size)`: `camera.npz` if it exists for this size, else an approximate camera (70 deg field of view)",
            r"`t2_guided.py: analyse()`, `write_report()`: re-detect every saved file and write `guided_report.md`"],
     opts=["constants: `MIN_REC_S` 5 s, `MIN_TILT` 30 deg, `MIN_FAR_CM` 80 cm; `--fresh` deletes the earlier step files"],
     io=[("IN", "camera", "laptop camera", "BGR frames 1280x720", "-"),
         ("IN", r"`data\markers.json`", "1.1 / 1.4", JSON_MK, SZ_MK),
         ("IN", r"`data\calibration\camera.npz` (if present)", "step 3.2 (else an approximate camera)", NPZ_CAM, SZ_CAM),
         ("OUT", r"`output\detection\guided\step1_both.png`, `step3_tilted.png`, `step4_far.png`", "2.4 (re-checked); report", "PNG 1280x720, annotated camera image", "1.1 MB each"),
         ("OUT", r"`output\detection\guided\step2_moving.avi`", "2.4 (re-checked); report", "AVI MJPG 30 fps, 1280x720, the annotated frames (1014 frames)", "78 MB"),
         ("OUT", r"`output\detection\guided\guided_report.md`", "report", "Markdown table `step | file | result`", "0.8 KB"),
         ("OUT", r"`output\detection\guided\extra_<HHMMSS>.png` / `.avi`", "none", "extra shots when all steps are done", "-")])

step("2.3", "Detection limits (distance x tilt)", r"python t2_detect_live.py --mode matrix", "`code\\t2_detect_live.py` -> `code\\t2_matrix.py`",
     "A guided matrix experiment: for every cell (distance x tilt of M) the window says where to put the marker and shows its distance and "
     "tilt estimated from the pose next to the target; SPACE tests the cell with 10 frames (PASS = detected in at least 9). After a clear fail "
     "(5 of 10 or less) the larger tilts at that distance are skipped. The reference values for the report are the ruler and protractor targets.",
     funcs=[r"`t2_detect_live.py: ask_matrix_settings()`: distances, tilts, marker, skip rule",
            r"`t2_matrix.py: run_matrix()`: the guided loop (SPACE test, N skip, U redo, Q stop); results written after every cell",
            r"`t2_matrix.py: run_cell()`: 10 frames after 3 dropped; hits, side in pixels, pixels per cell, fps",
            r"`t2_matrix.py: view_estimate()`, `fit()`, `panel()`: live distance / tilt against the target and the FIT bar",
            r"`t2_matrix.py: write_results()`: `detection_matrix.csv` and the Markdown summary with the limits"],
     opts=["distances [30, 50, 75, 100 cm] (`DEFAULT_DISTANCES`)", "tilts [0, 30, 45, 60, 70, 75, 80, 85 deg] (`DEFAULT_TILTS`)",
           "marker [M]", "skip after a clear fail [y] (`CLEAR_FAIL` = 5)", "`N_TEST` 10, `PASS_MIN` 9"],
     io=[("IN", "camera", "laptop camera", "BGR frames 1280x720", "-"),
         ("IN", r"`data\markers.json`", "1.1 / 1.4", JSON_MK, SZ_MK),
         ("IN", r"`data\calibration\camera.npz`", "3.2 (approximate camera if missing)", NPZ_CAM, SZ_CAM),
         ("OUT", r"`output\detection\detection_matrix.csv`", "report (T2 limits)",
          "CSV, one row per cell: `target_distance_cm, target_tilt_deg, marker, marker_id, frames, detected, passed, side_px, px_per_cell, "
          "est_distance_cm, est_tilt_deg, fps, note, resolution, dictionary, side_mm, time`", "3.1 KB (32 cells)"),
         ("OUT", r"`output\detection\detection_matrix.md`", "report", "Markdown: the matrix and the distance / tilt limits", "-")],
     notes=[r"`output\detection\T2_summary.md` (named in the README results table) is not written by any script in `code\` (searched): it is a hand-written summary."])

step("2.4", "Rebuild the demo report", r"python t2_guided.py --report", "`code\\t2_guided.py`",
     "Rewrites `guided_report.md` from the files saved in step 2.2, without the camera: every screenshot is detected again (ids, distance and "
     "tilt of M) and the recording is replayed (detection rate of F and M, duration, range of distances of M).",
     funcs=[r"`t2_guided.py: main()`: `write_report(gdir, load_markers_cfg(), (1280, 720))`",
            r"`t2_guided.py: analyse(path, step, cfg, det, K, dist)`: facts of one step file", r"`t2_matrix.py: guide_camera()`, `view_estimate()`"],
     opts=["`--report` (without it the script only prints how to run the demo)", "`--folder` [output\\detection]"],
     io=[("IN", r"`output\detection\guided\step1_both.png` ... `step4_far.png`, `step2_moving.avi`", "step 2.2", "PNG / AVI (see 2.2)", "about 82 MB"),
         ("IN", r"`data\markers.json`, `data\calibration\camera.npz`", "1.1 / 1.4, 3.2", "JSON; NPZ", "-"),
         ("OUT", r"`output\detection\guided\guided_report.md`", "report (T2 evidence)", "Markdown table `step | file | result`", "0.8 KB")])

# ------------------------------------------------------------------------------------------------ T3
step("3.1", "Capture the calibration views", r"python t3_calibrate.py capture", "`code\\t3_calibrate.py` (mode capture)",
     "A guided capture of chessboard views: for each view the terminal and the window say the wanted distance (close / medium / far), image "
     "region and tilt; a side panel compares the measured board distance, region and tilt with the target. SPACE saves the frame only "
     "when the board is found. After the last view the script offers to calibrate at once (then it continues as step 3.2).",
     funcs=[r"`t3_calibrate.py: capture_views(index, size, images_dir, board, n_views)`: the live loop; older views can be moved to `old_<date>`",
            r"`t3_calibrate.py: instruction(i, n)`, `PLAN`: 24 instructions (distance, region, tilt direction, angle)",
            r"`t3_calibrate.py: find_corners()`: `cv2.findChessboardCornersSB` with `NORMALIZE_IMAGE | ACCURACY`",
            r"`t3_calibrate.py: board_angles()`, `match()`, `side_panel()`, `view_feedback()`: live distance, region, tilt and the FIT bar",
            r"`t3_calibrate.py: guide_camera()`: approximate camera before the first calibration"],
     opts=["asked: images folder [data\\calibration\\images], camera index [0], resolution [1280x720], number of views [20]",
           "keys: SPACE save, U remove the last view, N skip, H help, Q finish"],
     io=[("IN", "camera", "laptop camera", "BGR frames 1280x720", "-"),
         ("IN", r"`data\calibration\board.json`", "steps 1.2 / 1.4", "JSON; {tab:board}", "200 B"),
         ("OUT", r"`data\calibration\images\view_001.png` ... `view_020.png`", "3.2 (calibration)", "PNG 1280x720 RGB, the full camera frame", "about 1.2 MB each, 24 MB")])

step("3.2", "Calibrate the camera", r"python t3_calibrate.py calibrate", "`code\\t3_calibrate.py` (mode calibrate)",
     "Finds the inner corners in every saved view, builds the board's 3D corner grid with the measured square size and runs "
     "`cv2.calibrateCamera` with the `k1k2` model (k1, k2 only: tangential terms and k3 fixed at zero). It reports the RMS reprojection error, "
     "the per-view errors and tilts, and a held-out check (every 5th view left out, its pose re-estimated with the reduced calibration). "
     "This project: 20 views, RMS 1.18 px, fx 846.6 px, fy 845.2 px, cx 641.1, cy 361.8, k1 0.092, k2 -0.135, horizontal field of view 74.2 deg.",
     funcs=[r"`t3_calibrate.py: load_board()`: `board.json` + `square_used_mm` (measured, else nominal)",
            r"`t3_calibrate.py: board_points(board)`: (54, 3) float32 corner grid in metres, Z = 0",
            r"`t3_calibrate.py: calibrate(objp, corners, size, model)`: `cv2.calibrateCamera`, per-view RMS by `cv2.projectPoints`, board tilt per view",
            r"`t3_calibrate.py: held_out_check()`: calibrate without views 4, 9, 14, 19 (0-based), `cv2.solvePnP` + reprojection on them",
            r"`t3_calibrate.py: run_calibration()`: the whole step, warnings (fewer than 10 views, too little tilt, square not measured)",
            r"`t3_calibrate.py: report_figure()`: coverage, per-view RMS, tilt per view", r"`mvtools.py: save_camera()`: writes `camera.npz` and `camera.json`"],
     opts=["asked: images folder, distortion model [k1k2] or full (k1 k2 p1 p2 k3), report folder [output\\calibration], camera file "
           "[data\\calibration\\camera.npz]", "`HELD_OUT_EVERY` = 5"],
     io=[("IN", r"`data\calibration\images\view_<NNN>.png`", "step 3.1", "PNG 1280x720 (other sizes skipped)", "24 MB"),
         ("IN", r"`data\calibration\board.json`", "steps 1.2 / 1.4", "JSON; {tab:board}", "200 B"),
         ("OUT", r"`data\calibration\camera.npz`", "every later step that computes a pose or a network input (3.3-7.8) through `mvtools.load_camera()`; "
          "`K` is copied into the data sets and checkpoints (`K_cam`)", NPZ_CAM, SZ_CAM),
         ("OUT", r"`data\calibration\camera.json`", "human-readable copy; not read by the code", "JSON `K, dist, width, height, rms_px, model, n_views, square_mm`", "393 B"),
         ("OUT", r"`output\calibration\report.json`", "report (T3 numbers)", "JSON `model, n_views, image_size, rms_px, fx, fy, cx, cy, hfov_deg, dist, "
          "per_view_rms_max, worst_view, held_out` (`views, rms_per_view, rms_mean, train_rms, fx_train`), `square_used_mm, warnings`", "747 B"),
         ("OUT", r"`output\calibration\per_view.csv`", "report", "CSV `view, rms_px, tilt_deg, held_out` (20 rows)", "565 B"),
         ("OUT", r"`output\calibration\calibration_report.png`", "report", "PNG, 3 panels (coverage, per-view RMS, tilt)", "78 KB")])

step("3.3", "Record the 5 measured positions (check A)", r"python t6_record.py --name E1_p1 --tag E1_static --truth -277 0 0 0 0 0", "`code\\t6_record.py`",
     "Records one static test recording with its hand-measured truth, from one command. The window shows the camera with F and M and a "
     "table: the truth typed on the command line, what the classical pipeline measures now (median of the last 30 frames) and the difference "
     "(a check against typing and sign mistakes only; the truth file holds the typed numbers). R records `--frames` frames and stops by itself. "
     "Run once per position: `E1_p1` ... `E1_p5`, truth x = -(gap + 77) mm.",
     funcs=[r"`t6_record.py: main()`: camera loop, recording, writing the four files",
            r"`mvtools.py: MarkerPoseEstimator.process(gray)`: live relative pose (see 3.5)", r"`mvtools.py: rpy_deg(R)`: roll, pitch, yaw (ZYX) in degrees",
            r"`t6_record.py: Panel.render()`: the comparison table, how each marker faces the camera, steadiness (drawn with PIL)"],
     opts=["`--name` [check], `--tag` [other] (E1_static, E2_dist_tilt, E3_motion, E4_robust, other)",
           "`--truth X Y Z ROLL PITCH YAW` [mm, deg] (required unless `--check`)", "`--frames` [100]", "`--distance` [cm], `--tilt` [deg] (stored for the E2 table)",
           "`--notes` ['']", "`--camera` [0]; the camera is opened at 1280x720"],
     io=[("IN", "camera", "laptop camera", "BGR frames 1280x720", "-"),
         ("IN", MARKERS + "; " + CAMERA, "steps 1.1 / 1.4, 3.2", "JSON; NPZ", "-"),
         ("IN", "`--truth` values", "hand-measured (ruler / protractor)", "6 numbers [mm, deg]", "-"),
         ("OUT", r"`data\recordings\E1_p1.avi`", "3.5, 5.7, 5.8, 6.4 (E1)", REC_AVI + "; 100 frames", "7.5 MB"),
         ("OUT", r"`data\recordings\E1_p1.csv`", "`mvtools.recording_times` (3.5, 6.4), 6.5", REC_CSV, "1.1 KB"),
         ("OUT", r"`data\recordings\E1_p1.json`", "6.4 (`tag` E1_static selects the E1 pool)", REC_JSON + " plus `camera_check` (6 numbers)", "0.4 KB"),
         ("OUT", r"`data\recordings\E1_p1.truth.csv`", "5.7, 6.4 (`mvtools.load_truth`)", TRUTH_1 + "; e.g. `0,-277,0,0,0,0,0`", "65 B"),
         ("OUT", r"`output\evaluation\t6_live.jpg`, `t6_live.json` (every 2 s)", "none (a snapshot for a helper)", "JPEG of the window; JSON "
          "`camera, tilt_F_M_dist, steady_mm, phase`", "315 KB; 130 B")],
     notes=["The existing `E1_p1.json` has the key `recorded_by` ('claude (headless, same format as t3_record.py)') and no `camera_check`: "
            "it was written by another recorder than the current `t6_record.py`; the file format is the same."])

step("3.4", "Record the camera-invariance sequence (check B)", r"python t3_record.py", "`code\\t3_record.py`",
     "A general recorder that asks its settings in the terminal: experiment tag, folder, name, notes, optional distance / tilt, an optional "
     "hand-measured truth (asked in plain words: left or right, gap between the black squares, height, depth, angle) and the number of frames. "
     "R starts and stops; D shows the detection. For check B the markers stand still and the laptop is moved slowly.",
     funcs=[r"`t3_record.py: ask_truth(cfg)`: converts the ruler answers to x = (gap + L) with the sign of the side, y, z [cm], turn [deg] -> pitch",
            r"`t3_record.py: main()`: questions, camera loop, writer; truth row written as `0, 10x, 10y, 10z, 0, turn, 0` [mm, deg]",
            r"`t2_detect_live.py: Camera`; `mvlib.py: MarkerDetector`"],
     opts=["asked: tag [other] (`TAGS`: E1_static, E2_dist_tilt, E3_motion, E4_robust, calib_view, train, demo, other); folder [training\\real for tag train, "
           "else data\\recordings]; name [rec_<date_time>]; notes; distance and tilt [-]; truth file [y for E1 / E2 tags, else n; not asked for E3, E7, train, calib]; "
           "frames [100 for E1, else until R]; camera [0]; resolution [1280x720]", "README answers: tag `other`, name `invariance_1`"],
     io=[("IN", "camera", "laptop camera", "BGR frames 1280x720", "-"),
         ("IN", MARKERS, "steps 1.1 / 1.4", JSON_MK, SZ_MK),
         ("OUT", r"`data\recordings\invariance_1.avi`", "3.5 (camera still = n), then 3.6", REC_AVI + "; 1499 frames, 50 s", "96.8 MB"),
         ("OUT", r"`data\recordings\invariance_1.csv`", "3.5 (`recording_times`)", REC_CSV, "19 KB"),
         ("OUT", r"`data\recordings\invariance_1.json`", "documentation (notes, distance)", REC_JSON, "0.3 KB"),
         ("OUT", r"`data\recordings\<name>.truth.csv` (only if asked)", "6.4", TRUTH_1, "about 60 B"),
         ("OUT", r"`<folder>\<name>_screenshot_<HHMMSS>.png` (key S)", "none", "PNG camera frame", "-")],
     notes=[r"The existing `invariance_1.json` has `tag` = `demo` (not `other` as in the README answer); the tag of this recording is not used by any later step. "
            r"`data\recordings\rec_20261115_175752` (tag other, with a truth file) was also made with this script; step 6.4 evaluates it because it has a truth file."])

step("3.5", "Poses of a recording", r"python t3_pose.py", "`code\\t3_pose.py`",
     "Runs the classical pipeline over every frame of a recording: ArUco detection of F and M, the pose of each marker in the camera "
     "(`SOLVEPNP_IPPE_SQUARE`, both solutions refined with Levenberg-Marquardt), and the relative pose T_FM = inverse(T_CF) T_CM. "
     "When the two IPPE solutions fit almost equally well (ratio below 1.5) the one closest in rotation to the previous frame is kept; with "
     "camera still = y a missing F is replaced by the last F. Optionally M comes from a network (`netpose.NetPose`) and only F is classical.",
     funcs=[r"`t3_pose.py: process_video(video, cfg, K, dist, camera_still, net, annotated_path)`: loop over frames; returns rows and a summary "
            r"(found %, relative pose %, ambiguous frames, F held frames, ms per frame); reused by 5.7, 6.4, 6.6",
            r"`mvtools.py: MarkerPoseEstimator`: `process(gray)` -> flags, T_CF, T_CM, T_FM; `_pose()` handles the planar ambiguity",
            r"`mvtools.py: marker_pose_candidates(corners, K, dist, L)`: `cv2.solvePnPGeneric(IPPE_SQUARE)` + `cv2.solvePnPRefineLM`, sorted by RMS",
            r"`mvlib.py: marker_object_points(L)`: corners (-L/2, L/2, 0) ... in the order TL, TR, BR, BL; `relative_pose()`, `inv_T()`, `reproj_rms()`",
            r"`mvtools.py: result_to_row()`, `pose_to_cols()`: pose -> `x_mm, y_mm, z_mm` + rotation vector `rx, ry, rz` [rad]",
            r"`mvtools.py: recording_times(path, n)`: time stamps from `<name>.csv`, else index / fps", r"`t3_pose.py: draw_result()`: outlines, axes, text bar"],
     opts=["asked: recording [newest `.avi` / `.mp4` in data\\recordings or training\\real]; camera file [data\\calibration\\camera.npz]; "
           "camera still [y]; pose of M from classical / network [classical] (network file [training\\models\\resnet18_v1.pt]); output folder "
           "[output\\pose]; annotated video [y]", "constant `AMBIGUITY_RATIO` = 1.5 (mvtools)", "hidden option `--max-frames`",
           "with the network source the output name gets the suffix `_net`"],
     io=[("IN", r"`data\recordings\<name>.avi` + `<name>.csv`", "3.3, 3.4, 4.1, 6.2, 6.3, 7.2, 7.3", REC_AVI + "; " + REC_CSV, "7-230 MB"),
         ("IN", MARKERS + "; " + CAMERA, "1.1 / 1.4; 3.2", "JSON; NPZ", "-"),
         ("IN", r"`training\models\<net>.pt` (network source only)", "5.2 / 5.4 / 5.6", "checkpoint; {tab:pt}", "42.7 MB"),
         ("OUT", r"`output\pose\<name>_poses.csv`", "3.6 (invariance), 4.3 (animation); `t7_bonus_occlusion.py` (not a README step)",
          POSES, "about 0.43 KB per frame (E1_p1: 100 rows, 43 KB)"),
         ("OUT", r"`output\pose\<name>_annotated.avi`", "4.3 (shown next to the animation when it exists)", "AVI MJPG 30 fps, size of the input "
          "(1280x720), outlines, axes of F and M, pose text", "about 73 KB per frame (invariance_1: 109 MB)")],
     notes=[r"Step 6.4 does not read these CSV files: `t6_evaluate.py` recomputes the poses with the same `process_video()`. "
            r"`output\pose\E1_checkA.md` is a hand-written summary of the five `E1_p<k>_poses.csv` (no script writes it)."])

step("3.6", "Camera-invariance report (check B)", r"python t3_check_invariance.py", "`code\\t3_check_invariance.py`",
     "Reads the poses of the invariance recording and compares how much the camera moved (spread of T_CF: position span and rotation) with "
     "how much the relative pose T_FM changed (RMS jitter and maximum deviation of the position, rotation RMS and maximum). Only frames with "
     "both markers and F not held are used. The verdict says whether invariance is shown (camera moved more than 50 mm or 10 deg).",
     funcs=[r"`t3_check_invariance.py: analyse(rows)`: `cols_to_pose` of CF and FM, `mvlib.mean_rotation` (chordal L2 mean), spans and RMS values",
            r"`t3_check_invariance.py: figure()`: camera motion and relative pose against time, same scale", r"`t3_check_invariance.py: report()`: the Markdown table and verdict",
            r"`mvlib.py: mean_rotation(Rs)`, `rot_error_deg()`"],
     opts=["asked: poses file [newest `*_poses.csv` in output\\pose], expected centre-to-centre distance [mm] (Enter = skip; 280 was given)"],
     io=[("IN", r"`output\pose\invariance_1_poses.csv`", "step 3.5 (camera still = n)", POSES, "653 KB (1499 rows)"),
         ("OUT", r"`output\pose\invariance_1_invariance.md`", "report (check B)", "Markdown: table camera motion against relative pose, mean position, "
          "centre distance, verdict", "1.0 KB"),
         ("OUT", r"`output\pose\invariance_1_invariance.png`", "report", "PNG, 2 panels (x, y, z minus mean against t)", "86 KB")],
     notes=["The existing `invariance_1_invariance.md` ends with a bullet \"Hand measurements: ...\" that the script does not write (added by hand)."])

# ------------------------------------------------------------------------------------------------ T4
step("4.1", "Guided recording of M moving", r"python t4_record_guided.py --name T4_motion_1 --seconds 30", "`code\\t4_record_guided.py` (plan t4)",
     "Records 30 s of M moved by hand while F stays still, with the moves shown on the window (left / right, closer / farther, turn, up / down, "
     "slow circle; the instruction changes by itself). R starts; it stops and saves by itself. Warnings appear when M is not seen, F is not "
     "seen or F moved by more than 40 px. No truth file is written for this plan.",
     funcs=[r"`t4_record_guided.py: main()`: plans t4 / train / bonus / motion, naming, camera loop, writer, files",
            r"`t4_record_guided.py: panel()`, `big()`: the instruction panel and large text", r"`t2_detect_live.py: Camera`; `mvlib.py: MarkerDetector`"],
     opts=["`--name` [T4_motion_1 for plan t4]", "`--seconds` [30]", "`--plan` [t4]", "keys: R start, SPACE pause / resume, S stop and save now, Q close "
           "(closing while recording keeps the frames if more than 150)", "`--camera` [0]; 1280x720"],
     io=[("IN", "camera", "laptop camera", "BGR frames 1280x720", "-"),
         ("IN", MARKERS, "1.1 / 1.4", JSON_MK, SZ_MK),
         ("OUT", r"`data\recordings\T4_motion_1.avi`", "4.2 (poses), 5.7 (network against classical), 6.6 (network animation)", REC_AVI + "; 900 frames", "62 MB"),
         ("OUT", r"`data\recordings\T4_motion_1.csv`", "4.2 / 6.6 (time stamps)", REC_CSV, "11 KB"),
         ("OUT", r"`data\recordings\T4_motion_1.json`", "documentation", REC_JSON + "; `tag` demo", "0.3 KB")])

step("4.2", "Poses of that recording", r"python t3_pose.py", "`code\\t3_pose.py` (as step 3.5)",
     "The same program as step 3.5 (see there for the functions and options), run on `T4_motion_1` with camera still = y and the classical "
     "source. Its poses drive the animation of step 4.3.",
     io=[("IN", r"`data\recordings\T4_motion_1.avi` + `.csv`", "step 4.1", REC_AVI + "; " + REC_CSV, "62 MB"),
         ("IN", MARKERS + "; " + CAMERA, "1.1 / 1.4; 3.2", "JSON; NPZ", "-"),
         ("OUT", r"`output\pose\T4_motion_1_poses.csv`", "4.3", POSES + " (900 rows)", "365 KB"),
         ("OUT", r"`output\pose\T4_motion_1_annotated.avi`", "4.3 (video shown on the left)", "AVI MJPG 1280x720, 900 frames, 30 fps", "70.7 MB")])

step("4.3", "Export the animation", r"python t4_animate.py export", "`code\\t4_animate.py` (mode export)",
     "Reads the relative poses and writes a video: on the left the (annotated) camera video, on the right a 3D view in the frame of F (axes x "
     "red, y green, z blue, trajectory of M) above the x, y, z [mm] and roll, pitch, yaw [deg] curves, with a cursor synchronised to the video. "
     "With smoothing on, the poses first go through a gated constant-velocity Kalman filter (position) and an alpha-beta filter on SO(3) "
     "(orientation); frames without a measurement are then predicted for up to 1 s.",
     funcs=[r"`t4_animate.py: load_poses_csv(path)`: rows, times `t_s`, T_FM from the `FM_` columns (`mvtools.cols_to_pose`)",
            r"`t4_animate.py: filter_poses(t, poses)`: Kalman + alpha-beta; restart after 5 gate rejections; None after 1 s without a measurement",
            r"`mvlib.py: ConstantVelocityKalman`: state [p, v], white-noise acceleration, chi-square gate 16.266 (99.9 %, 3 dof)",
            r"`mvlib.py: AlphaBetaSO3`: predict R <- exp(w dt) R; update with r = log(R_meas R^T), alpha 0.5, beta 0.15",
            r"`t4_animate.py: PosePlot`: matplotlib 3D view and curves rendered to a 720x480 image per frame",
            r"`t4_animate.py: export(rows, t, poses, video, out_base, formats, target_fps=15)`: every 2nd frame, MP4 (mp4v) and / or GIF (half size)",
            r"`t4_animate.py: hstack_to_height()`: video and plot side by side"],
     opts=["asked: output folder [output\\animation]; smoothing [y]; poses file [newest `*_poses.csv` in output\\pose]; video [the "
           "`_annotated.avi` if it exists, else the recording; - for none]; format mp4 / gif / both [mp4] (the README outputs need both)",
           "filter: `sigma_a` 0.15 m/s2, `sigma_m` 0.004 m, `max_coast_s` 1.0"],
     io=[("IN", r"`output\pose\T4_motion_1_poses.csv`", "step 4.2", POSES, "365 KB"),
         ("IN", r"`output\pose\T4_motion_1_annotated.avi`", "step 4.2", "AVI MJPG 1280x720", "70.7 MB"),
         ("OUT", r"`output\animation\T4_motion_1_animation.mp4`", "deliverable (T4); `make_public_media.py` makes the camera-free copies in `output\\public` (not a README step)",
          "MP4 (mp4v) 1572x480, 450 frames, 15.6 fps", "7.0 MB"),
         ("OUT", r"`output\animation\T4_motion_1_animation.gif`", "deliverable", "GIF, half size, same frames", "31 MB"),
         ("OUT", r"`output\animation\T4_motion_1_animation_filtered_poses.csv`", "none (record of the smoothed poses)",
          "CSV `frame, t_s, predicted, FM_x_mm, FM_y_mm, FM_z_mm, FM_rx, FM_ry, FM_rz` (900 rows)", "113 KB")])

step("4.4", "Live animation", r"python t4_animate.py live", "`code\\t4_animate.py` (mode live)",
     "The same display as 4.3 but from the camera, frame by frame: the annotated camera picture on the left and the 3D view and curves of "
     "the last 300 frames (about 10 s) on the right, smoothed with the same filter. S saves a screenshot; nothing else is written.",
     funcs=[r"`t4_animate.py: live(index, size, cfg, K, dist, camera_still, with_filter, out_dir)`: camera loop; plot refreshed every 3 frames",
            r"`mvtools.py: MarkerPoseEstimator`; `t3_pose.py: draw_result()`; `t4_animate.py: filter_poses()`, `PosePlot`"],
     opts=["asked: output folder [output\\animation], smoothing [y], camera file [camera.npz], camera still [y], camera index [0]",
           "keys: S screenshot, Q / Esc stop"],
     io=[("IN", "camera", "laptop camera", "frames of the size in `camera.npz` (1280x720)", "-"),
         ("IN", MARKERS + "; " + CAMERA, "1.1 / 1.4; 3.2", "JSON; NPZ", "-"),
         ("OUT", r"`output\animation\live_<HHMMSS>.png` (key S)", "report only (contains the camera image)", "PNG, camera + plot, 480 px high", "12 files exist")])

# ------------------------------------------------------------------------------------------------ T5
step("5.1", "Synthetic data set", r"python t5_make_data.py", "`code\\t5_make_data.py` (+ `synthdata.py`)",
     "Renders the synthetic images the network is first trained on, with exact labels R, t = T_CM. The camera is the calibrated camera "
     "scaled to the network input (640x360: K_net = K / 2); lens distortion is not simulated (real frames are undistorted before the network). "
     "Every image: M at a random pose (distance log-uniform, tilt up to the maximum about a random axis, any in-plane rotation, centre in view), "
     "a random background with clutter, F as a distractor in 70 % of the images, a grey patch over a corner of M in 15 %, random gain, "
     "offset, blur and noise. The three splits use the seeds seed, seed + 1, seed + 2.",
     funcs=[r"`t5_make_data.py: make_dataset(out_dir, name, cfg, K_cam, size_cam, n_train, n_val, n_test, net_w, z_range, tilt_max, seed, ...)`: "
            r"all three splits and `meta.json`",
            r"`synthdata.py: scaled_camera(K, size_cam, size_net)`: K scaled to the input width",
            r"`synthdata.py: generate_set(out_dir, split, n, seed, cfg, par)`: parallel chunks of 100 images (`ProcessPoolExecutor`, CPU count - 1 "
            r"workers) written into a memory-mapped `X_<split>.npy`",
            r"`synthdata.py: SyntheticRenderer`: `sample_pose()`, `_background()`, `_draw()` (homography of the marker patch, 2x supersampling), `render()`",
            r"`t5_make_data.py: examples_figure()`: 8 test images with the label drawn as a green outline"],
     opts=["asked: camera file [camera.npz; a PROVISIONAL fx = 900 px camera if missing]; name [synth_v1]; folder [training\\synthetic]; "
           "images train / val / test [10000 / 1000 / 1000]; input width [640]; nearest / farthest distance [0.3 / 1.2 m]; maximum tilt [60 deg]; seed [0]",
           "fixed in code: `p_distractor` 0.7, `p_occlude` 0.15, supersampling 2, quiet zone 0.25 L"],
     io=[("IN", CAMERA, "3.2", NPZ_CAM, SZ_CAM),
         ("IN", MARKERS, "1.1 / 1.4 (ids, dictionary, L)", JSON_MK, SZ_MK),
         ("OUT", r"`training\synthetic\synth_v1\X_train.npy`, `X_val.npy`, `X_test.npy`", "5.2 (train / val); 5.6 (synthetic half of every batch); "
          "6.4 (E5 uses the test split)", ".npy uint8, shapes (10000, 360, 640), (1000, 360, 640), (1000, 360, 640): grey images", "2.1 GB + 0.22 + 0.22 GB"),
         ("OUT", r"`training\synthetic\synth_v1\R_<split>.npy`, `t_<split>.npy`", "the same", ".npy float32 (N, 3, 3) rotation of M in the camera; "
          "(N, 3) position of M in the camera [m]", "0.35 MB / 0.12 MB (train)"),
         ("OUT", r"`training\synthetic\synth_v1\meta.json`", "5.2 (`t5_train.train` reads `name, net_size, K_cam, K_net, size_cam, L_m, ...`); 5.6 "
          "(copied into the real sets)", "JSON; {tab:meta}", "0.8 KB"),
         ("OUT", r"`output\network\synth_v1\examples.png`", "visual check of the labels", "PNG, 8 tiles", "0.6 MB")],
     notes=[nv("5.1", "the existing `synth_v1\\meta.json` records `z_range` [0.25, 1.0] m, while the current defaults of the questions are 0.3 and "
               "1.2 m; the set was therefore made with other answers or an earlier version of the defaults. Its generation time is 44.6 s.")])

step("5.2", "Train the whole-image network (stage 1)", r"python t5_train.py", "`code\\t5_train.py` (+ `netpose.py`, `t5_dashboard.py`)",
     "Trains a ResNet-18 (ImageNet weights; the first convolution summed over its colour channels to take one grey channel) that maps the "
     "whole 640x360 image to 9 numbers: a 6-D rotation (two 3-vectors, Gram-Schmidt) and (x/Z, y/Z, log Z). The loss is w_rot times the chordal "
     "rotation loss (squared Frobenius norm of R_hat - R) plus w_trans times the L1 loss of (x/Z, y/Z, log Z). After every epoch the validation "
     "split gives the median and 95th percentile of the rotation [deg] and translation [mm] errors; a checkpoint after every epoch lets an "
     "interrupted run resume. README answers: data set `synth_v1`, name `resnet18_v2`, 40 epochs; the loss-weight study `resnet18_v3` uses w_rot = 2.",
     funcs=[r"`t5_train.py: train(data_dir, model_name, epochs, batch, lr, w_rot, w_trans, seed, pretrained, ...)`: the training loop (AdamW, weight decay "
            r"1e-4, OneCycle schedule with 10 % warm-up, bfloat16 autocast and channels-last on CUDA, data kept on the GPU when under 2 GB)",
            r"`netpose.py: build_model(pretrained, in_ch)`: ResNet-18, new `conv1` (in_ch channels), `fc` 512 -> 9 with a bias giving an identity-like "
            r"rotation and Z = 0.6 m", r"`netpose.py: make_input()`, `preprocess()`: (x / 255 - 0.45) / 0.225, one channel",
            r"`netpose.py: six_d_to_R()`, `decode()`, `targets()`: output <-> R, t", r"`t5_train.py: losses()`: chordal + L1",
            r"`t5_train.py: evaluate()`: median / p95 of the validation errors", r"`t5_train.py: curves()`: `training_curves.png`",
            r"`t5_dashboard.py: TrainDashboard`, `example_tiles()`: live window (losses, validation errors, 4 validation images); Q stops after the step"],
     opts=["asked: data set [the last one in training\\synthetic]; model name [resnet18_v1, or crop_resnet18_v1 for a crop set]; epochs [15; 40 for a crop "
           "set]; batch [64]; learning rate [5e-4]; weights w_rot [1], w_trans [1]; seed [0]; pretrained [y]; live window [y]",
           "augmentation: gain 0.85-1.15 and offset +-10 grey levels always; noise and blur only for crop sets (`strong_aug`); gradient clipping 5 for crop sets"],
     io=[("IN", r"`training\synthetic\synth_v1\X_train.npy`, `R_train.npy`, `t_train.npy`, `_val` files, `meta.json`", "step 5.1",
          "see 5.1 and {tab:sets}", "2.6 GB"),
         ("IN", "ImageNet ResNet-18 weights", "torchvision download (cache)", "`ResNet18_Weights.IMAGENET1K_V1`", "about 45 MB"),
         ("OUT", r"`training\models\resnet18_v2.pt`", "5.3 (named as stage 1 in `crop_v1\\meta.json`), 5.4 (stage 1 of the crop network at inference), "
          "5.6 (start of the fine-tuning)", "PyTorch checkpoint `state_dict` (122 tensors; `conv1` (64, 1, 7, 7), `fc` (9, 512)) + `config`; {tab:pt}", "42.7 MB"),
         ("OUT", r"`training\models\resnet18_v2_checkpoint.pt` (during training)", "resume of an interrupted run; deleted at the end",
          "`model, opt, sched, rng, hist, step, epoch, minutes, best, settings`", "-"),
         ("OUT", r"`output\network\resnet18_v2\train_log.json`", "report (T5 training), `output\\network\\T5_runs.md` (hand-written)",
          "JSON `config` + `log` (one record per epoch); {tab:trainlog}", "14 KB"),
         ("OUT", r"`output\network\resnet18_v2\training_curves.png`, `dashboard.png`", "report", "PNG", "86 KB; 490 KB")],
     notes=["Result in the checkpoint (`config.val`, epoch 40): validation rotation median 7.3 deg, translation median 13.5 mm. The older checkpoints "
            "(`resnet18_v1/v2/v3.pt`, `crop_resnet18_v1.pt`) have fewer `config` keys than the current code writes (no `kind`, `best_epoch`, ...); "
            "`NetPose` treats a missing `kind` as `full`."])

step("5.3", "Crop data set", r"python t5_make_crops.py", "`code\\t5_make_crops.py` (+ `synthdata.py`)",
     "Renders the training set of the second stage: square windows around M, 224x224 px. Each sample puts M at a random pose in the full "
     "1280x720 calibrated camera, chooses the window as stage 1 would (side = 2.0 times the apparent side of M, at least 48 px) and perturbs its "
     "centre and size like stage-1 errors (jitter 0.12 of the marker side), renders the scene directly at full resolution inside the window and "
     "resizes it to 224 px. Stage 1 itself is not run here: its name is only checked and stored in `meta.json`. The window (u0, v0, side) is "
     "stored with each crop, because the network also gets the camera coordinates of every crop pixel.",
     funcs=[r"`t5_make_crops.py: make_crop_dataset(...)`: three splits (seeds seed, +1, +2) and `meta.json` with `kind` = crop",
            r"`synthdata.py: generate_crop_set(out_dir, split, n, seed, cfg, par, S)`, `_crop_chunk()`: one crop per sample, window from "
            r"`netpose.CROP_FACTOR` = 2.0 and `CROP_MIN` = 48", r"`t5_make_crops.py: examples_figure()`: 8 crops with the label outline"],
     opts=["asked: camera file; stage-1 network [resnet18_v2.pt, must exist in training\\models]; name [crop_v1]; folder [training\\synthetic]; "
           "crops train / val / test [30000 / 2000 / 2000]; crop size [224]; distances [0.25 / 1.0 m]; maximum tilt [60 deg]; stage-1 error [0.12]; seed [0]"],
     io=[("IN", CAMERA, "3.2", NPZ_CAM, SZ_CAM),
         ("IN", MARKERS, "1.1 / 1.4", JSON_MK, SZ_MK),
         ("IN", r"`training\models\resnet18_v2.pt` (existence only)", "step 5.2", "checkpoint (not loaded)", "42.7 MB"),
         ("OUT", r"`training\synthetic\crop_v1\X_<split>.npy`", "5.4; 5.6 (synthetic half of the crop batches)", ".npy uint8 (30000, 224, 224), (2000, 224, 224), (2000, 224, 224)", "1.4 GB + 2 x 96 MB"),
         ("OUT", r"`training\synthetic\crop_v1\R_<split>.npy`, `t_<split>.npy`, `box_<split>.npy`", "5.4; 5.6",
          ".npy float32 (N, 3, 3), (N, 3) [m], box (N, 3) = u0, v0, side in full-camera pixels", "about 1.7 MB (train)"),
         ("OUT", r"`training\synthetic\crop_v1\meta.json`", "5.4 (`kind` crop -> 3-channel network; `stage1` stored in the checkpoint); 5.6",
          "JSON; {tab:meta}", "0.9 KB"),
         ("OUT", r"`output\network\crop_v1\examples.png`", "visual check", "PNG, 8 tiles", "1.5 MB")])

step("5.4", "Train the crop network (stage 2)", r"python t5_train.py", "`code\\t5_train.py` (as step 5.2, crop data set)",
     "The same program as 5.2; because `crop_v1\\meta.json` has `kind` = crop it builds a 3-channel network (grey + the normalised camera "
     "coordinates x/Z and y/Z of every crop pixel, `netpose.coord_maps`), uses the stronger augmentation (noise and blur) and gradient clipping 5. "
     "README answers: data set `crop_v1`, name `crop_resnet18_v1`, 80 epochs, the rest as 5.2. At inference `NetPose` first runs the stage-1 "
     "network named in the checkpoint to place the window.",
     funcs=[r"`netpose.py: coord_maps(boxes, K, S)`, `make_input(x, boxes, K)`: the two coordinate channels",
            r"`netpose.py: NetPose`: `kind` crop -> loads `config['stage1']` from the same folder; `predict_batch()` runs stage 1, cuts the window with "
            r"`crop_box()` from the undistorted full frame, runs stage 2; with `passes` = 2 the window is re-centred once on stage 2's own answer",
            "all functions of 5.2"],
     opts=["as 5.2; the default epochs for a crop set are 40 (README: 80)"],
     io=[("IN", r"`training\synthetic\crop_v1\` (X, R, t, box of train / val; `meta.json`)", "step 5.3", "{tab:sets}", "1.6 GB"),
         ("OUT", r"`training\models\crop_resnet18_v1.pt`", "5.6 (start of the stage-2 fine-tuning)", "checkpoint, `conv1` (64, 3, 7, 7); `config.kind` crop, "
          "`stage1` resnet18_v2.pt, `crop_size` 224; {tab:pt}", "42.7 MB"),
         ("OUT", r"`output\network\crop_resnet18_v1\train_log.json`, `training_curves.png`, `dashboard.png`", "report", "{tab:trainlog} (80 epochs)", "27 KB; 0.1 / 0.5 MB")],
     notes=["Result in the checkpoint (epoch 80): validation rotation median 1.95 deg, translation median 8.8 mm on synthetic crops; 27 min on CUDA."])

step("5.5", "Record the real training frames", r"python t4_record_guided.py --plan train --set real_v4 --seconds 42", "`code\\t4_record_guided.py` (plan train)",
     "The guided recorder of 4.1 with the training plan: seven moves that give the network variety (M close, far, at the edges, strongly tilted, "
     "rotated like a steering wheel, next to F, free). The recordings go to `training\\<set>` and are numbered automatically (`<set>_01`, `_02`, ...). "
     "This project: 5 x `real_v2` (36 s), 5 x `real_v3` (42 s) and 7 x `real_v4` (42 s, camera pitched / turned) = 17 recordings.",
     funcs=[r"`t4_record_guided.py: main()` with `MOVES_TRAIN`; tag `train` in the JSON", "see 4.1"],
     opts=["`--plan train`, `--set` [real_v2], `--seconds` [30], `--folder` [training\\<set>], `--notes`"],
     io=[("IN", "camera", "laptop camera", "BGR frames 1280x720", "-"),
         ("IN", MARKERS, "1.1 / 1.4", JSON_MK, SZ_MK),
         ("IN", r"`training\real_v3\INSTRUCTIONS.md`, `training\real_v4\INSTRUCTIONS.md`", "hand-written plans", "Markdown", "-"),
         ("OUT", r"`training\real_v4\real_v4_01.avi` ... `real_v4_07.avi` (and `real_v2_*`, `real_v3_*`)", "5.6 (labelled, then training / validation)",
          REC_AVI + "; about 1080-1260 frames each", "88-114 MB each; 416 / 510 / 729 MB per set"),
         ("OUT", r"`training\real_v4\real_v4_<NN>.csv`, `.json`", "5.6 (only the video is read); documentation", REC_CSV + "; " + REC_JSON + ", `tag` train", "about 15 KB; 0.4-0.7 KB")])

step("5.6", "Fine-tune both networks on real + synthetic frames",
     r"python t5_finetune.py --sets real_v2 real_v3 real_v4 --suffix _ft3 --val real_v2_05 real_v3_05 real_v4_03 real_v4_07 --yes", "`code\\t5_finetune.py` (+ `t5_train.py`)",
     "Step 1 labels every frame of the 17 training recordings with the classical pipeline (T_CM, kept only if the reprojection error of M is at "
     "most 1 px); labels are saved per recording and reused. Step 2 builds two real data sets in the synthetic format, split by whole "
     "recordings (the four `--val` recordings are validation only): the undistorted frame resized to 640x360 for stage 1, and a 224 px window "
     "around M, shifted at random (jitter 0.25), for stage 2. Step 3 fine-tunes `resnet18_v2` (25 epochs) and `crop_resnet18_v1` (50 epochs) with "
     "half real and half synthetic images in every batch, learning rate 1e-4, batch 128, seed 0, 40 % hard-case augmentation, camera-rotation "
     "augmentation up to 8 deg for stage 1, keeping the epoch with the best validation score (rotation / 5 deg + translation / 10 mm).",
     funcs=[r"`t5_finetune.py: label_recordings()`, `label_recording()`: `MarkerPoseEstimator(camera_still=False)` per frame, keep T_CM if `rms_M` <= 1 px",
            r"`t5_finetune.py: build_sets(labels, K, dist, size, out_root, val_name, L, S=224, jitter=0.25)`: undistort, resize, crop window, save .npy",
            r"`t5_finetune.py: PrepWindow`: progress window of steps 1-2",
            r"`t5_train.py: train(..., init_from=<base>.pt, mix_with=<synthetic set>, keep_best=True, hard_aug=0.4, rot_aug=8 / 0, pretrained=False)`",
            r"`t5_train.py: rotate_camera()`: warps the image by K Ra K^-1 and turns the labels (stage 1)",
            r"`t5_train.py: hard_cases()`: partial cover, motion blur, low resolution, dim light, label unchanged"],
     opts=["`--sets` [real_v2]", "`--suffix` [_ft]", "`--val` [the last recording]", "`--yes`: every question answered with its default and overwrite allowed",
           "`--relabel`, `--rebuild`, `--restart`: redo labels / data sets / training", "defaults of the questions: stage 1 resnet18_v2, stage 2 crop_resnet18_v1, "
           "epochs 25 / 50, learning rate 1e-4, batch 128, live windows y", "constants: `MAX_RMS_PX` 1.0, `ROT_AUG_DEG` 8, `HARD_AUG` 0.4, `CROP_JITTER` 0.25, "
           "`SYNTHETIC` = synth_v1 (full), crop_v1 (crop)"],
     io=[("IN", r"`training\real_v2\`, `real_v3\`, `real_v4\` `*.avi` (17)", "step 5.5", REC_AVI, "1.66 GB"),
         ("IN", MARKERS + "; " + CAMERA, "1.1 / 1.4; 3.2", "JSON; NPZ (K and dist: labels and undistortion)", "-"),
         ("IN", r"`training\models\resnet18_v2.pt`, `crop_resnet18_v1.pt`", "5.2, 5.4", "checkpoints", "42.7 MB each"),
         ("IN", r"`training\synthetic\synth_v1\`, `crop_v1\` (train split, `meta.json`)", "5.1, 5.3", "{tab:sets}", "4.2 GB"),
         ("OUT", r"`training\real_v<k>\labels\<recording>_labels.npz`", "this step on the next run (reused if newer than the video and made with the same "
          "`MAX_RMS_PX`)", "NPZ `frames` int64 (n,), `R` float32 (n, 3, 3), `t` float32 (n, 3) [m]; newer files also `max_rms_px` float64", "about 45-50 KB"),
         ("OUT", r"`training\synthetic\real_v2_real_v3_real_v4_full\`", "stage-1 fine-tuning in this step", "X_train (11562, 360, 640) uint8, X_val (3725, 360, 640), "
          "R, t; `meta.json`, `build_info.json`", "3.3 GB"),
         ("OUT", r"`training\synthetic\real_v2_real_v3_real_v4_crop\`", "stage-2 fine-tuning in this step", "X_train (11562, 224, 224) uint8, X_val (3725, ...), "
          "R, t, box; `meta.json`, `build_info.json`", "0.73 GB"),
         ("OUT", r"`training\models\resnet18_v2_ft3.pt`", "stage 1 of the final network (`stage1` of `crop_resnet18_v1_ft3`); 5.7 (whole-image _ft3)",
          "checkpoint, `kind` full, best epoch 22 of 25; {tab:pt}", "42.7 MB"),
         ("OUT", r"`training\models\crop_resnet18_v1_ft3.pt`", "THE final network: 5.7, 5.8, 6.4, 6.5, 6.6", "checkpoint, `kind` crop, `stage1` "
          "resnet18_v2_ft3.pt, best epoch 48 of 50; {tab:pt}", "42.7 MB"),
         ("OUT", r"`output\network\resnet18_v2_ft3\`, `crop_resnet18_v1_ft3\`", "report", "`train_log.json` ({tab:trainlog}), `training_curves.png`, "
          "`dashboard.png`, `examples\\epoch_<NNN>.png`", "about 12 MB (examples)")],
     notes=["`build_info.json` holds `labels` (number of labelled frames per recording), `validation`, `crop_jitter`, `max_rms_px`, `marker_side_m`; when it "
            "matches the current run the sets are reused. The real sets' `meta.json` is the synthetic set's `meta.json` with `name`, `real`, `from_recordings`, "
            "`validation_recording` (and for crops `stage1`, `crop_jitter_real`) replaced: its `n_train`, `n_val`, `n_test`, `params` and `generation_seconds` "
            "still describe the synthetic set, not the real one. The rounds `_ft` and `_ft2` were made with the same script and fewer recordings."])

step("5.7", "Test the networks on real recordings", r"python t5_eval_real.py", "`code\\t5_eval_real.py`",
     "Runs the classical pipeline and four network variants (two-stage `_ft2` with 2 passes, whole-image `resnet18_v2_ft3`, two-stage `_ft3` with "
     "1 and 2 passes) on the five E1 recordings, each against its ruler truth: median translation and rotation errors, output rate, jitter, "
     "x / y / z bias, time per frame. On `T4_motion_1` (no truth) it prints how far each network is from the classical pose. The script has no "
     "options: the variants are the dictionary `nets` at the top.",
     funcs=[r"`t3_pose.py: process_video(video, cfg, K, dist, True, net)`: poses per frame for each method",
            r"`mvtools.py: load_truth()`, `truth_pose()`, `pose_errors()`, `cols_to_pose()`", r"`netpose.py: NetPose(path, K, dist, passes)`"],
     io=[("IN", r"`data\recordings\E1_p1` ... `E1_p5` (`.avi`, `.csv`, `.truth.csv`)", "step 3.3", "see {tab:rec}", "about 38 MB"),
         ("IN", r"`data\recordings\T4_motion_1.avi` + `.csv`", "step 4.1", REC_AVI, "62 MB"),
         ("IN", r"`training\models\crop_resnet18_v1_ft2.pt`, `crop_resnet18_v1_ft3.pt`, `resnet18_v2_ft3.pt` (+ the stage-1 files they name)", "5.6 (and the earlier round _ft2)",
          "checkpoints", "42.7 MB each"),
         ("IN", MARKERS + "; " + CAMERA, "1.1 / 1.4; 3.2", "JSON; NPZ", "-"),
         ("OUT", r"`output\network\real_E1_comparison_ft3.json`", "report (T5 real-data results), `output\\network\\T5_runs.md` (hand-written)",
          "JSON method -> recording -> `et` (median mm), `er` (median deg), `out` (%), `jit` (mm), `d` (x, y, z bias, mm), `ms`", "6.2 KB"),
         ("OUT", "terminal: per method and recording; pooled over 500 frames; T4_motion_1 differences", "the user", "text", "-")],
     notes=[nv("5.7", "the T4_motion_1 comparison is only printed, not saved; its numbers cannot be checked from a file.")])

step("5.8", "Test on degraded frames", r"python t5_test_hard_cases.py", "`code\\t5_test_hard_cases.py`",
     "Takes every second frame of `E1_p1` ... `E1_p5` in which the classical pipeline finds M, and degrades it in software: part of M covered "
     "(15 %, 30 %), horizontal motion blur (15, 31 px), lower resolution (/4, /6), dim light (x 0.25, x 0.10). The reference is the classical "
     "pose of the clean frame; a pose is usable within 20 mm and 10 deg of it. It reports how often the classical pipeline gives a pose / a usable "
     "pose and the usable share and median errors of the two-stage networks `_ft2` and `_ft3`.",
     funcs=[r"`t5_test_hard_cases.py: cover()`, `mblur()`, `lowres()`, `dim()`: the degradations (`rng` seed 0); `CASES`: the 9 cases",
            r"`t5_test_hard_cases.py: errs()`; `mvtools.MarkerPoseEstimator(camera_still=False)`; `netpose.NetPose.predict_batch()` in batches of 32"],
     io=[("IN", r"`data\recordings\E1_p1` ... `E1_p5.avi`", "step 3.3", REC_AVI, "about 38 MB"),
         ("IN", r"`training\models\crop_resnet18_v1_ft2.pt`, `crop_resnet18_v1_ft3.pt` (+ stage 1)", "5.6", "checkpoints (2 passes)", "42.7 MB each"),
         ("IN", MARKERS + "; " + CAMERA, "1.1 / 1.4; 3.2", "JSON; NPZ", "-"),
         ("OUT", r"`output\network\hard_cases_E1.json`", "report (usability)", "JSON case -> `classical_output`, `classical_usable` (%), `_ft2` / `_ft3` -> "
          "`usable` (%), `et_med` (mm), `er_med` (deg)", "2.6 KB"),
         ("OUT", r"`output\network\figures\test_degraded_cover.jpg`, `_motion.jpg`, `_resolution.jpg`, `_light.jpg`", "report figures (camera picture)",
          "JPEG, 680x420 cut of frame 10", "28-78 KB")],
     notes=[r"The folder `output\network\figures` must exist (`cv2.imwrite` does not create it). The other files there (`augmentation_*.png`, "
            r"`real_v4_first_frames.jpg`) are not written by this script.",
            nv("5.8", "the number of frames used is printed, not stored in the JSON.")])

# ------------------------------------------------------------------------------------------------ T6
step("6.1", "Live check of a placement", r"python t6_record.py --check", "`code\\t6_record.py` (check mode)",
     "The window of 3.3 without recording: what the classical pipeline measures now (pose of M in F, median of the last 30 frames), how F and "
     "M face the camera, the distance of F and how steady the reading is. Used to set up a placement before its recording.",
     funcs=["as 3.3: `MarkerPoseEstimator.process()`, `rpy_deg()`, `Panel.render()`"],
     opts=["`--check`; `--camera` [0]"],
     io=[("IN", "camera", "laptop camera", "BGR frames 1280x720", "-"),
         ("IN", MARKERS + "; " + CAMERA, "1.1 / 1.4; 3.2", "JSON; NPZ", "-"),
         ("OUT", r"`output\evaluation\t6_live.jpg`", "none (snapshot every 2 s)", "JPEG of the window (camera + table)", "315 KB"),
         ("OUT", r"`output\evaluation\t6_live.json`", "none", "JSON `camera` [x, y, z mm, roll, pitch, yaw deg], `tilt_F_M_dist` [F about x, F about y, M about x, "
          "M about y in deg, F depth in cm], `steady_mm`, `phase`", "130 B")])

step("6.2", "Record the static test recordings (E1, E2, E4)", r"python t6_record.py --name E1_turn20 --tag E1_static --truth -115 -20 90 0 20 0 --distance 61 --tilt 20",
     "`code\\t6_record.py` (13 commands of the README table; this entry covers all of them)",
     "The program of step 3.3, run once for each row of the README table; the commands differ only in their arguments: E1 with M turned "
     "20 / 40 deg (`E1_turn20`, `E1_turn40`); E2 at 40, 60, 75 cm with tilt 0 and 40 deg (`E2_d<cm>_t<deg>`); E4 dim light, left edge of M covered, "
     "clutter, shaken table and wobbled camera (`E4_blur`, `E4_blur_2` with `--frames 150`). `--distance` and `--tilt` go into the JSON and are "
     "the row keys of the E2 table of 6.4; the tag decides in which experiment a recording is pooled.",
     funcs=["as 3.3"],
     opts=["as 3.3; `--frames 150` for `E4_blur` and `E4_blur_2`"],
     io=[("IN", "camera; `--truth`, `--distance`, `--tilt`", "laptop camera; hand-measured", "-", "-"),
         ("IN", MARKERS + "; " + CAMERA, "1.1 / 1.4; 3.2", "JSON; NPZ", "-"),
         ("OUT", r"`data\recordings\<name>.avi` (13 recordings)", "6.4 (E1, E2, E4 rows), 6.5 (`E4_cover`, `E1_turn40`)", REC_AVI + "; 100 or 150 frames", "6.7-12.3 MB each"),
         ("OUT", r"`data\recordings\<name>.csv`", "6.4, 6.5", REC_CSV, "1-2 KB"),
         ("OUT", r"`data\recordings\<name>.json`", "6.4: `tag`, `notes` (E4 condition), `distance_cm`, `tilt_deg`", REC_JSON + " + `camera_check`", "0.4-0.7 KB"),
         ("OUT", r"`data\recordings\<name>.truth.csv`", "6.4, 6.5", TRUTH_1, "about 65 B")],
     notes=[r"How each placement was set up is described in `output\evaluation\T6_notes.md` (hand-written)."])

step("6.3", "Record the known motion (E3)",
     r"python t4_record_guided.py --plan motion --name E3_motion_01 --seconds 20 --pass-s 6 --truth-start -140 -20 90 --truth-end -200 -20 90",
     "`code\\t4_record_guided.py` (plan motion)",
     "M slides 60 mm along a ruler and back, following an outline drawn on the screen that moves at constant speed (60 mm in `--pass-s` = 6 s, "
     "1 cm/s): 3 s rest with a countdown, forward, 2 s rest, back, 2 s rest, repeated, with beeps at every phase. The truth file is written from "
     "this schedule: a row at every corner of the schedule with the position interpolated between the two measured stops (linear in between). "
     "Run three times (`E3_motion_01`, `_02`, `_03`).",
     funcs=[r"`t4_record_guided.py: motion_schedule(t, T, R0=3, rest=2)`: phase and position 0..1 at time t",
            r"`t4_record_guided.py: motion_breakpoints(t_end, T)`: the corner times of the schedule -> rows of the truth file",
            r"`t4_record_guided.py: main()`: the moving outline (scaled from M's size in pixels and `--travel` = |end - start|)"],
     opts=["`--plan motion` (requires `--truth-start X Y Z` and `--truth-end X Y Z` [mm])", "`--truth-angles` [0 0 0 deg]", "`--pass-s` [2.7 s]",
           "`--seconds` [30]", "`--name` [next free E3_motion_<NN>]"],
     io=[("IN", "camera; `--truth-start`, `--truth-end`", "laptop camera; hand-measured stops", "-", "-"),
         ("IN", MARKERS, "1.1 / 1.4", JSON_MK, SZ_MK),
         ("OUT", r"`data\recordings\E3_motion_0<k>.avi`", "6.4 (E3), 6.5 (`E3_motion_01`)", REC_AVI + "; 600 frames (20 s)", "50 MB"),
         ("OUT", r"`data\recordings\E3_motion_0<k>.csv`, `.json`", "6.4", REC_CSV + "; " + REC_JSON + ", `tag` E3_motion", "7 KB; 0.7 KB"),
         ("OUT", r"`data\recordings\E3_motion_0<k>.truth.csv`", "6.4 (`truth_pose` interpolates in time), 6.5",
          "CSV as {tab:rec}, several rows (14 for 20 s): `t_s` at the schedule corners, x from -140 to -200 mm, y -20, z 90, angles 0", "0.5 KB")],
     notes=["The `notes` of these JSON files end with the T4 sentence (\"T4: F still, M moved by hand ...\"): the code appends the plan's default note to "
            "the `--notes` text; it does not affect any computation."])

step("6.4", "Run the comparison (E1-E6)", r"python t6_evaluate.py --network crop_resnet18_v1_ft3 --yes", "`code\\t6_evaluate.py`",
     "Runs both pipelines on every recording in `data\\recordings` that has a `.truth.csv`, on the same frames with the same truth: classical "
     "(F and M by ArUco + IPPE + LM) and network (M from the two-stage network with 2 passes, F classical). Errors of T_FM per frame: e_t = "
     "distance of the positions [mm], e_R = angle of R_true^T R_est [deg]; a frame without a pose counts as no output. The recordings are "
     "pooled by their tag (E1_static, E3_motion, E4_robust; E2 per recording); E5 runs both methods on the held-out synthetic test split "
     "(camera frame, exact truth) and E6 measures the cost. With `--yes` every question takes its default.",
     funcs=[r"`t6_evaluate.py: run_recordings()`: `t3_pose.process_video` for each method, `errors_from_rows()`, `jitter_mm()`, `bias_mm()`",
            r"`t6_evaluate.py: synthetic_eval(data_dir, net, cfg)`: E5 on `X_test` with `K_net` and zero distortion; network with `undistort=False`",
            r"`t6_evaluate.py: cost_table()`: ms per frame, GPU and CPU latency (`NetPose.time_per_frame_ms`), lines of code, parameters, file size, training time",
            r"`t6_evaluate.py: worst_frames()`: the 5 worst frames per method and recording (3 saved as images)",
            r"`t6_evaluate.py: motion_plots()`: e_t(t), e_R(t) of the E3 recordings", r"`t6_evaluate.py: build_report()`: the tables; `fmt()`: median (p95)",
            r"`mvtools.py: load_truth()`, `truth_pose()`, `pose_errors()`, `summarize()` (n, median, p95, RMS, max)"],
     opts=["`--network` [resnet18_v1] (README: crop_resnet18_v1_ft3)", "`--yes`", "defaults of the questions: recordings data\\recordings; camera "
           "camera.npz; network training\\models\\<network>.pt (- = classical only); synthetic set = the last `training\\synthetic\\<set>\\meta.json` in "
           "sorted order (here synth_v1); output output\\evaluation; camera still y"],
     io=[("IN", r"`data\recordings\<name>.avi`, `.csv`, `.json`, `.truth.csv` (22 recordings with truth)", "3.3, 3.4, 6.2, 6.3", "{tab:rec}", "about 0.4 GB"),
         ("IN", r"`training\models\crop_resnet18_v1_ft3.pt` + `resnet18_v2_ft3.pt`", "5.6", "checkpoints", "42.7 MB each"),
         ("IN", r"`training\synthetic\synth_v1\X_test.npy`, `R_test.npy`, `t_test.npy`, `meta.json`", "5.1", "(1000, 360, 640) uint8 + labels", "0.22 GB"),
         ("IN", MARKERS + "; " + CAMERA, "1.1 / 1.4; 3.2", "JSON; NPZ", "-"),
         ("OUT", r"`output\evaluation\comparison_table.md` / `.csv`", "report (the assignment's table)", "columns `Method, E1 e_t [mm], E1 e_R [deg], E1 jitter [mm], "
          "E3 e_t [mm], E3 e_R [deg], E4 output [%], E6 [fps]`; cells median (p95)", "0.4 KB; 0.2 KB"),
         ("OUT", r"`output\evaluation\report.md`", "report", "Markdown: the table, then E1 bias / jitter, E2, E3, E4, E5, E6 sections", "6.2 KB"),
         ("OUT", r"`output\evaluation\summary.json`", "report, paper", "JSON `table`, `cost` (14 keys), `synthetic` (method -> `output_pct`, `et`, `er` summaries)", "1.8 KB"),
         ("OUT", r"`output\evaluation\worst_frames.csv`", "report", "CSV `recording, method, frame, e_t_mm, e_R_deg`", "7 KB (215 rows)"),
         ("OUT", r"`output\evaluation\worst\<method>_<recording>_frame<i>.png`", "report (camera frames)", "PNG 1280x720 with the error written on it", "129 files, 116 MB"),
         ("OUT", r"`output\evaluation\E3_<recording>.png`", "report", "PNG, e_t and e_R against t, both methods", "about 95 KB each")],
     notes=["The rows of `rec_20261115_175752` (tag other, see 3.4) are computed and appear in `worst_frames.csv`, but the tag is in no pool, so it is not "
            "in the table. `cost.loc_classical` counts the non-comment lines of `mvlib.py` + `t3_pose.py`, `loc_network` those of `netpose.py`, `synthdata.py`, "
            "`t5_make_data.py`, `t5_train.py`. The fps values change from run to run."])

step("6.5", "Both methods side by side (video)", r"python t6_side_by_side.py E3_motion_01 E4_cover E1_turn40", "`code\\t6_side_by_side.py`",
     "For each named recording, the same frames twice (every 2nd frame): on the left the classical result (outlines, axes of F and M), on the "
     "right the network (axes of F classical, axes and orange outline of M from the network). Under each picture the pose of M in F and, when a "
     "truth file exists, the errors against it. The picture is cropped to the area around the markers.",
     funcs=[r"`t6_side_by_side.py: side_by_side(name, cfg, K, dist, net, out_dir, step=2)`: classical estimator, network in batches of 32, crop from the "
            r"1st-99th percentile of the corner positions, panels 560 px wide",
            r"`t6_side_by_side.py: read_all()`, `outline()`; `mvtools.load_truth()`, `truth_pose()`, `pose_errors()`"],
     opts=["positional: recording names", "`--network` [crop_resnet18_v1_ft3]", "`--animation`: step 6.6 instead"],
     io=[("IN", r"`data\recordings\<name>.avi`, `.csv`, `.truth.csv`", "6.2, 6.3", "{tab:rec}", "-"),
         ("IN", r"`training\models\crop_resnet18_v1_ft3.pt` (+ stage 1)", "5.6", "checkpoint", "42.7 MB"),
         ("IN", MARKERS + "; " + CAMERA, "1.1 / 1.4; 3.2", "JSON; NPZ", "-"),
         ("OUT", r"`output\evaluation\side_by_side_<name>.mp4`", "report / presentation", "MP4 (mp4v) 15 fps, 1132 px wide, height from the crop "
          "(E3_motion_01 1132x424, 300 frames; E4_cover 1132x444, 50 frames)", "0.6-4.2 MB")])

step("6.6", "T4 animation with the network's poses", r"python t6_side_by_side.py T4_motion_1 --animation", "`code\\t6_side_by_side.py` (option --animation)",
     "Runs the network pipeline over `T4_motion_1` (F classical, M network) and writes the T4 display of 4.3 with these poses, without the filter. "
     "In the camera part only boxes around the two markers are kept (the rest is dark grey), with the outline of F in green and of M in orange.",
     funcs=[r"`t6_side_by_side.py: network_animation(name, cfg, K, dist, net, out_dir)`: `t3_pose.process_video(..., net)`, then `t4_animate.export(..., "
            r"with_filter=False, frame_fn=draw)`"],
     opts=["`--animation`, `--network` [crop_resnet18_v1_ft3]"],
     io=[("IN", r"`data\recordings\T4_motion_1.avi`, `.csv`", "4.1", REC_AVI, "62 MB"),
         ("IN", r"`training\models\crop_resnet18_v1_ft3.pt` (+ stage 1)", "5.6", "checkpoint", "42.7 MB"),
         ("IN", MARKERS + "; " + CAMERA, "1.1 / 1.4; 3.2", "JSON; NPZ", "-"),
         ("OUT", r"`output\animation\T4_motion_1_network_animation.mp4`", "report / presentation (T4 with the network)", "MP4 1572x480, 450 frames, 15.6 fps", "5.1 MB")],
     notes=["No poses CSV is written in this step: the network poses exist only inside the run."])

# ------------------------------------------------------------------------------------------------ Bonus
step("7.1", "Live check of the occlusion set-up", r"python t7_setup_guided.py --centre 325 --travel 228 --obstacle 180 263", "`code\\t7_setup_guided.py`",
     "Shows the camera with F and M and a checklist with live numbers (green = ok, red = fix it): both markers seen, room to the picture's edges, "
     "camera distance, centre-to-centre distance against the ruler value, same depth, same height, boards parallel. On the picture it draws the "
     "path of M's centre (`--travel` mm toward F), where M will end (dashed square) and the band the obstacle has to cover.",
     funcs=[r"`t7_setup_guided.py: main()`: `MarkerPoseEstimator(camera_still=False)`, the checks and drawings", r"`t7_setup_guided.py: project()`: points in the camera -> pixels"],
     opts=["`--centre` [377 mm]", "`--travel` [280 mm]", "`--obstacle A B` [155 235 mm, ruler readings from M's left corner at the start]", "`--camera` [0]",
           "keys: S save picture + numbers, Q close"],
     io=[("IN", "camera", "laptop camera", "BGR frames 1280x720", "-"),
         ("IN", MARKERS + "; " + CAMERA, "1.1 / 1.4; 3.2", "JSON; NPZ", "-"),
         ("OUT", r"`output\bonus\setup_live.jpg`, `setup_live.json` (every 2 s)", "none (snapshot for a helper)", "JPEG window; JSON `centre_mm, depth_mm, "
          "height_mm, turn_deg, camera_to_F_cm, room_px, all_ok, red`", "299 KB; 304 B"),
         ("OUT", r"`output\bonus\setup_<HHMMSS>.png` (key S)", "documentation", "PNG window (camera picture)", "-")],
     notes=["The existing `setup_live.json` compares with \"ruler 377\", the default `--centre`: it was last written by a run without the README arguments."])

step("7.2", "Record passes without the obstacle", r"python t4_record_guided.py --plan bonus --name E7_nocover_slow_2 --seconds 60", "`code\\t4_record_guided.py` (plan bonus)",
     "The guided recorder with the bonus plan: a repeated cycle rest (3 s, countdown 3-2-1) - GO (a pacing bar and an outline that moves from "
     "M's start toward F at constant speed in `--pass-s` seconds) - rest 2 s at the end - bring M back (3 s), with beeps. Several passes per "
     "recording; M is never hidden here, so its true pose is measured in every frame. This project: `E7_nocover_slow_2` and `E7_nocover_fast`.",
     funcs=[r"`t4_record_guided.py: main()`, plan bonus: cycle length 3 + `pass_s` + 2 + 3 s; ghost outline moved by (F - M) x `travel` / `centre`"],
     opts=["`--plan bonus`, `--seconds` [30], `--pass-s` [2.7 s], `--travel` [273 mm], `--centre` [377 mm], `--name` [next free E7_pass_<NN>]"],
     io=[("IN", "camera", "laptop camera", "BGR frames 1280x720", "-"),
         ("IN", MARKERS, "1.1 / 1.4", JSON_MK, SZ_MK),
         ("OUT", r"`data\recordings\E7_nocover_slow_2.avi`", "7.4, 7.5", REC_AVI + "; about 1800 frames (60 s)", "about 0.15 GB"),
         ("OUT", r"`data\recordings\E7_nocover_slow_2.csv`, `.json`", "7.4-7.5 (time stamps); documentation", REC_CSV + "; " + REC_JSON + ", `tag` E7_occlusion", "-")],
     notes=["No truth file is written for the bonus plan: 7.4-7.6 take the truth from the classical measurement itself."])

step("7.3", "Record passes behind the obstacle", r"python t4_record_guided.py --plan bonus --name E7_box_slow --seconds 75 --pass-s 2.5 --travel 228 --centre 325",
     "`code\\t4_record_guided.py` (plan bonus; same program as 7.2, other arguments)",
     "The same as 7.2 with the obstacle in front of the middle of the path and the speed set by `--pass-s` (seconds per pass); `--travel` and "
     "`--centre` scale the moving outline to the second layout. This project: `E7_box_slow` (2.5 s), `E7_box_fast` (1.7 s), `E7_box_5cms` and "
     "`E7_box_5cms_2` (4.6 s, 95 s long), and with the first layout `E7_cover_slow_3`, `E7_cover_slow_4`, `E7_cover_fast`.",
     funcs=["as 7.2"],
     opts=["as 7.2"],
     io=[("IN", "camera", "laptop camera", "BGR frames 1280x720", "-"),
         ("IN", MARKERS, "1.1 / 1.4", JSON_MK, SZ_MK),
         ("OUT", r"`data\recordings\E7_box_slow.avi` (and the others)", "7.4, 7.6, 7.7", REC_AVI + "; E7_box_slow 2251 frames (75 s)", "180 MB (E7_box_5cms_2: 229 MB)"),
         ("OUT", r"`data\recordings\E7_box_slow.csv`, `.json`", "7.4-7.7 (time stamps); documentation (layout in `notes`)", REC_CSV + "; " + REC_JSON + ", `tag` E7_occlusion",
          "29 KB; 0.8 KB")])

step("7.4", "Measure the passes of a recording", r"python t7_passes.py E7_box_slow", "`code\\t7_passes.py`",
     "Tracks M with the classical pipeline and finds the passes: the main direction of motion (first singular vector of the positions), the "
     "position s along it smoothed over 0.2 s, rests where the speed stays under 15 mm/s for at least 0.5 s, and a pass between two rests that "
     "moves toward F by at least `--min-travel`. For each pass: start and end time, centre-to-centre distance to F at both rests, travel, "
     "speed over the middle 80 % (line fit) and its spread, distance off the straight line, and the time M was not measured.",
     funcs=[r"`t7_passes.py: track(video, cfg, K, dist)`: t and M's position in F [mm] per frame (NaN where missing)",
            r"`t7_passes.py: find_passes(t, P, min_travel=150, rest_speed=15, rest_s=0.5)`: returns the axis, s(t) and the passes (also used by 7.5 and 7.6)",
            r"`t7_passes.py: main()`: the per-pass numbers, the CSV files and the figure"],
     opts=["positional: recording name", "`--min-travel` [150 mm]"],
     io=[("IN", r"`data\recordings\E7_box_slow.avi` + `.csv`", "7.3", REC_AVI, "180 MB"),
         ("IN", MARKERS + "; " + CAMERA, "1.1 / 1.4; 3.2", "JSON; NPZ", "-"),
         ("OUT", r"`output\bonus\<name>_passes.csv`", "documentation (`E7_summary.md`, hand-written)", "CSV `pass, t_start_s, t_end_s, centre_start_mm, centre_end_mm, "
          "travel_mm, duration_s, speed_mm_s, speed_std_mm_s, off_line_rms_mm, off_line_max_mm, hidden_s, start_x_mm, start_y_mm, start_z_mm, end_x_mm, end_y_mm, end_z_mm`",
          "0.5-0.9 KB"),
         ("OUT", r"`output\bonus\<name>_track.csv`", "none", "CSV `t_s, centre_mm, along_mm` (one row per frame; empty where not measured)", "24-40 KB"),
         ("OUT", r"`output\bonus\<name>_passes.png`", "documentation", "PNG: centre distance against time, passes shaded", "50-67 KB")],
     notes=[nv("7.4", "the outputs of the README example (`E7_box_slow_passes.csv` / `_track.csv` / `.png`) are not in `output\\bonus`; the formats were "
               "checked on `E7_nocover_fast_passes.csv` and `E7_nocover_fast_track.csv`.")])

step("7.5", "Predictor on the uncovered passes, hidden in software", r"python t7_virtual_occlusion.py E7_nocover_slow_2 E7_nocover_fast", "`code\\t7_virtual_occlusion.py`",
     "In the passes without obstacle the true pose is measured in every frame. A virtual obstacle removes the measurement while M's left corner "
     "is between two ruler readings (default 78-235 mm); the occlusion predictor runs without measurements there, and its output is compared "
     "frame by frame with the measured pose and with the baseline 'hold the last pose': errors at 0.25 / 0.5 / 1 / 1.5 / 2 s and at the end, the "
     "horizon where the error first passes 20 and 40 mm, the time the pose is declared lost, and the jump at re-acquisition.",
     funcs=[r"`t7_virtual_occlusion.py: poses(video, cfg, K, dist)`: t and T_FM per frame (also used by 7.6)",
            r"`t7_bonus_occlusion.py: OcclusionPredictor(dt, L, sigma_a=0.15, sigma_m=0.004, lost_factor=2, rot_gate_deg=30, rot_tau=0.1, line_s=None)`: "
            r"Kalman (position) + alpha-beta on SO(3); `step(T or None)` returns the pose, predicted / lost flags, sigma [mm], innovation, jump",
            r"`t7_passes.py: find_passes()`; `mvlib.py: ConstantVelocityKalman`, `AlphaBetaSO3`, `rot_error_deg()`"],
     opts=["positional: recording names", "`--hide A B` [78 235 mm]", "`--line`: keep the velocity only along the line of motion (`line_s` 0.6 s)",
           "`--rot-tau` [predictor default 0.1 s; -1 = no fade-out]", "`--tag`: suffix of the output files",
           "predictor: lost when 3 sigma > 2 L (154 mm); gate chi-square 16.266 and 30 deg; restart after 5 rejections"],
     io=[("IN", r"`data\recordings\E7_nocover_slow_2.avi`, `E7_nocover_fast.avi` + `.csv`", "7.2", REC_AVI, "about 0.3 GB"),
         ("IN", MARKERS + "; " + CAMERA, "1.1 / 1.4; 3.2", "JSON; NPZ", "-"),
         ("OUT", r"`output\bonus\virtual_occlusion.csv`", "report (bonus), `E7_summary.md` (hand-written)", "CSV, one row per pass, 28 columns: `recording, pass, "
          "hidden_s, speed_hidden_mm_s, speed_just_before_mm_s, visible_before_s`, `pred_mm_<t>s` / `hold_mm_<t>s` for t = 0.25, 0.5, 1, 1.5, 2, `pred_mm_end, "
          "hold_mm_end, pred_deg_end, hold_deg_end, pred_mm_max, sigma3_mm_end, lost_after_s, jump_mm`, `pred_horizon_s_20mm`, `hold_horizon_s_20mm`, `_40mm` pair",
          "1.8 KB (11 passes)"),
         ("OUT", r"`output\bonus\virtual_occlusion.md`", "report", "Markdown: medians per recording, horizon table", "1.2 KB"),
         ("OUT", r"`output\bonus\virtual_occlusion.png`", "report", "PNG: error against time hidden, one line per pass, predictor and hold", "180 KB")])

step("7.6", "Predictor behind the real obstacle",
     r"python t7_real_occlusion.py E7_box_5cms_2 E7_box_5cms E7_box_slow E7_box_fast E7_cover_slow_4 E7_cover_slow_3 E7_cover_fast --min-runup 50",
     "`code\\t7_real_occlusion.py`",
     "Behind a real obstacle the truth is known exactly only when M is seen again. For every pass the longest run of frames without M is the "
     "obstacle; the predictor runs through it and the error at reappearance (predicted pose against the first measured pose) is compared with "
     "'hold the last pose'. In between, errors are given against a straight line at constant speed (an assumption). A pass is clean when M was "
     "seen moving for at least `--min-runup` mm before it disappeared; the summary uses clean passes, split into slower and faster than 125 mm/s.",
     funcs=[r"`t7_real_occlusion.py: main()`: passes, hidden run, predictor, errors, CSV / Markdown / figure",
            r"`t7_virtual_occlusion.py: poses()`, `t7_passes.py: find_passes()`, `t7_bonus_occlusion.py: OcclusionPredictor` (innovation = error at reappearance)"],
     opts=["positional: recording names", "`--min-runup` [60 mm] (README: 50)", "`--line`, `--rot-tau` as 7.5"],
     io=[("IN", r"`data\recordings\E7_box_*`, `E7_cover_*` (7 recordings) `.avi` + `.csv`", "7.3", REC_AVI, "about 1.2 GB"),
         ("IN", MARKERS + "; " + CAMERA, "1.1 / 1.4; 3.2", "JSON; NPZ", "-"),
         ("OUT", r"`output\bonus\real_occlusion.csv`", "report (bonus), `E7_summary.md`", "CSV, one row per hidden pass, 22 columns: `recording, pass, clean, last_seen_mm, "
          "seen_again_mm, hidden_s, speed_hidden_mm_s, pred_mm, hold_mm, pred_deg, hold_deg, sigma3_mm, lost_after_s, event`, `pred_mm_<t>s` / `hold_mm_<t>s` for "
          "t = 0.25, 0.5, 1, 1.5", "7.4 KB (48 passes)"),
         ("OUT", r"`output\bonus\real_occlusion.md`", "report", "Markdown: one row per speed group (medians over clean passes)", "1 KB"),
         ("OUT", r"`output\bonus\real_occlusion.png`", "report", "PNG: error at reappearance against time hidden", "52 KB")],
     notes=[r"`output\bonus\real_occlusion_all42.csv` is not written under that name by any script in `code\` (an earlier copy; origin not verified)."])

step("7.7", "Animation with the predicted pose", r"python t7_animate.py E7_box_5cms_2 --start 37.5 --seconds 50 --gif", "`code\\t7_animate.py`",
     "Runs the classical pipeline and the predictor from the start of the recording, then animates the part from `--start` for `--seconds`: "
     "the T4 display with the predicted segment dashed and its growing uncertainty; on the camera picture a green outline where M is measured "
     "and a red box where M is predicted (none once the pose is lost). For privacy everything outside the path of M and a small box around M "
     "is cut out (dark grey).",
     funcs=[r"`t7_animate.py: measure(video, cfg, K, dist, seconds)`: t, T_FM, last T_CF and corners per frame",
            r"`t7_animate.py: work_area()`, `pixelate_outside()`: the part of the picture that stays visible",
            r"`t4_animate.py: export(rows, t, poses, video, out_base, formats, frame_fn=draw)`: rows with `predicted` column -> dashed segment and sigma; no filter",
            r"`t7_bonus_occlusion.py: OcclusionPredictor`"],
     opts=["positional: recording", "`--start` [0 s]", "`--seconds` [32; 0 = whole recording]", "`--gif`", "`--show-all` (no masking)", "`--pixelate` (instead of cutting)"],
     io=[("IN", r"`data\recordings\E7_box_5cms_2.avi` + `.csv`", "7.3", REC_AVI + "; 95 s", "229 MB"),
         ("IN", MARKERS + "; " + CAMERA, "1.1 / 1.4; 3.2", "JSON; NPZ", "-"),
         ("OUT", r"`output\bonus\E7_box_5cms_2_predicted_poses.csv`", "the export in this step; `make_public_media.py` (not a README step)",
          "CSV `frame, t_s, measured, predicted, lost, sigma_mm, FM_x_mm, FM_y_mm, FM_z_mm, FM_rx, FM_ry, FM_rz` (frames 1124 ... 2623)", "192 KB (1500 rows)"),
         ("OUT", r"`output\bonus\E7_box_5cms_2_occlusion.mp4`", "deliverable (bonus)", "MP4 1572x480, 750 frames, 15.5 fps", "6.4 MB"),
         ("OUT", r"`output\bonus\E7_box_5cms_2_occlusion.gif`", "deliverable", "GIF, half size", "20.5 MB")],
     notes=["The `*_occlusion_frames.jpg` files in `output\\bonus` are not written by this script (origin not verified)."])

step("7.8", "Live: classical pipeline + predictor", r"python t7_live.py", "`code\\t7_live.py`",
     "The predictor on the live camera with free motion: green outline = M measured, red box = where M is predicted while it is not measured "
     "(with 3 sigma in the caption), 'pose LOST' when the uncertainty is too large. On the right the 3D view and the position and angle curves "
     "of the last seconds, predicted stretches shaded. The filter time step follows the real frame time (clipped to 5-200 ms).",
     funcs=[r"`t7_live.py: set_dt(pr, dt)`: rebuilds F and Q of the Kalman filter and the alpha-beta dt for the measured frame time",
            r"`t7_live.py: main()`: camera loop, `OcclusionPredictor.step()`, drawing", r"`t4_animate.py: PosePlot`, `hstack_to_height()`"],
     opts=["`--camera` [0]", "`--window-s` [20 s shown in the plots]", "keys: S screenshot, C clear the plots, Q / Esc stop"],
     io=[("IN", "camera", "laptop camera", "BGR frames 1280x720", "-"),
         ("IN", MARKERS + "; " + CAMERA, "1.1 / 1.4; 3.2", "JSON; NPZ", "-"),
         ("OUT", r"`output\bonus\live_<HHMMSS>.png` (key S)", "report only (camera image)", "PNG window", "none exist")])

# ------------------------------------------------------------------------------------------------ Everything at once
step("8.1", "Live: classical, network and predictor together", r"python live.py",
     "`live.py` (project root; `code\\live.py` is a stub that runs it with `runpy`), which opens the window of `code\\t8_live_all.py`",
     "**The launcher** `live.py` first checks the set-up in five numbered steps and repairs what it can: (1) the project folder (`code` and `data` "
     "next to it); (2) the Python packages (NumPy, OpenCV with ArUco, Pillow, Matplotlib; if one is missing it prints the install commands and "
     "stops; PyTorch is optional); (3) the markers: if `data\\markers.json` or the two marker images are missing (or with `--regenerate-markers`) "
     "it calls `t1_generate_markers.generate` (defaults DICT_6X6_250, ids 0 / 23, 80 mm, or the values of an existing `markers.json`; an existing "
     "measured side is kept); (4) the calibration: `camera.npz`, rebuilt from `camera.json` if only that exists, else an approximate camera "
     "(1280x720, 74 deg field of view, no distortion) written to `camera_approx.npz`, and the network is then switched off; (5) the network file. "
     "Then it calls `t8_live_all.main()` with `--camera-file <file>`, `--no-network` if needed, and every other argument passed on. "
     "**The window** (`t8_live_all.py`) shows the three methods on the same picture. Per frame: the classical pipeline measures F and M (`camera_still=True`, so a missing F "
     "is replaced by its last pose); the two-stage network predicts T_CM of M (if its file exists and it is switched on); the occlusion predictor gets "
     "the classical relative pose, with the filter time step set to the measured frame time (clipped to 5-200 ms). Outlines: green = F and M classical, "
     "red = M from the network, blue = M predicted while the classical pipeline has no measurement of M (none once the pose is LOST). The panel on the "
     "right gives, for each method, x, y, z [mm] and roll, pitch, yaw [deg] of M in F, the F-M centre distance and the camera-M distance; for the network "
     "also its difference to the classical pose, for the predictor its state and 3 sigma. At the end a one-line summary is printed.",
     funcs=[r"`live.py: check_folder()`, `check_python()`, `check_markers(regenerate)`, `check_camera()` (returns the camera file and whether it is "
            r"calibrated), `check_network(name, exact)`; `main()`: the checks, then `t8_live_all.main()` with the built `sys.argv`",
            r"`t1_generate_markers.py: generate()`: only when markers are missing (see 1.1)",
            r"`mvtools.py: MarkerPoseEstimator.process(gray)`: classical T_CF, T_CM, T_FM (see 3.5)",
            r"`netpose.py: NetPose(path, K, dist).predict_one(gray)`: two-stage network, default 2 passes (see 5.4); T_FM of the network = inverse(last T_CF) T_CM",
            r"`t7_bonus_occlusion.py: OcclusionPredictor(1/30, L).step(T_FM or None)`: Kalman + alpha-beta predictor (see 7.5)",
            r"`t8_live_all.py: set_dt(pr, dt)`: rebuilds the Kalman F and Q and the alpha-beta dt for the frame time (as `t7_live.set_dt`)",
            r"`t8_live_all.py: outline()`, `draw_outline()`: projected marker square and label; `numbers()`, `pose_lines()`: the values of the panel; `Panel.render()`: the 540 px panel",
            r"`mvlib.py: inv_T()`, `rot_error_deg()`; `mvtools.py: rpy_deg()`; `t2_detect_live.py: Camera` (1280x720)"],
     opts=["options of `live.py`: `--check` (only the checks, no window), `--regenerate-markers` (write the markers and the dictionary table again); "
           "all other arguments go to `t8_live_all.py`",
           "options of `t8_live_all.py`: `--camera-file PATH` [data\\calibration\\camera.npz] (set by `live.py`), `--no-network` (set by `live.py` when the "
           "network file, PyTorch or a real calibration is missing)",
           "`--camera` [0]", "`--network` [crop_resnet18_v1_ft3] (file in training\\models; if missing, a message and the run continues without network)",
           "`--video NAME`: a recording of data\\recordings instead of the camera (time stamps from its `.csv`)", "`--frames N` [0 = all] with `--video`",
           "`--no-window` with `--video`: only the summary", "`--save-frame PATH` with `--video`: writes the display of the last frame to PATH",
           "keys: S screenshot, N network on / off, C restart the predictor, Q / Esc quit"],
     io=[("IN", "camera (index 0), or `data\\recordings\\<NAME>.avi` + `.csv` with `--video`", "laptop camera; steps 3.3-7.3", "BGR frames 1280x720; " + REC_CSV, "-"),
         ("IN", MARKERS + r", `output\markers\marker_F_id0.png`, `marker_M_id23.png`", "1.1 / 1.4 (generated here if missing)", JSON_MK, SZ_MK),
         ("IN", CAMERA + r", or `data\calibration\camera.json`", "3.2", NPZ_CAM + "; JSON copy", SZ_CAM),
         ("IN", r"`training\models\crop_resnet18_v1_ft3.pt` + the stage-1 file it names (`resnet18_v2_ft3.pt`)", "5.6", "checkpoints; {tab:pt}", "42.7 MB each"),
         ("OUT", r"`output\markers\`, `output\dictionary\`, `data\markers.json` (only if markers are missing, or `--regenerate-markers`)",
          "every later run and every step (as from 1.1)", "the files of step 1.1", "as 1.1"),
         ("OUT", r"`data\calibration\camera.npz` (only if rebuilt from `camera.json`)", "every pose step", NPZ_CAM, SZ_CAM),
         ("OUT", r"`data\calibration\camera_approx.npz` (only without any calibration)", "this window (`--camera-file`)", "NPZ like `camera.npz`: fx = fy = 849 px, "
          "cx 640, cy 360, `dist` zeros, `rms` 0", "not present"),
         ("OUT", r"`output\live\live_all_<HHMMSS>.png` (key S)", "none (report / demonstration; contains the camera image)", "PNG: camera picture + panel, (1280 + 540) x 720", "none exist yet"),
         ("OUT", "file given by `--save-frame` (optional)", "none", "image of the last display (format from the extension)", "-"),
         ("OUT", "terminal: frames, frames/s, % classical pose of M, predicted and lost frames, median network - classical difference [mm, deg]", "the user", "text", "-")],
     notes=["The frames/s shown is the processing rate of one frame (from `time.perf_counter`), not the camera rate. The folder `output\\live` is created at start. "
            "The window itself writes nothing else (no poses CSV); the launcher writes files only in the repair cases listed in the table."])

# ================================================================================================ data flow
FLOW = [
    (r"`data\markers.json`", "1.1, edited 1.4 (8.1 if missing)", "every later step (`mvtools.load_markers_cfg`); 1.3"),
    (r"`data\calibration\board.json`", "1.2, edited 1.4", "3.1, 3.2"),
    (r"`output\markers\`, `output\calibration\chessboard_<...>.png`", "1.1, 1.2", "1.3"),
    (r"`data\calibration\images\view_<NNN>.png`", "3.1", "3.2"),
    (r"`data\calibration\camera.npz`", "3.2", "2.2-2.4 (if present), 3.3-8.1 (`mvtools.load_camera`); `K_cam` in 5.1, 5.3, checkpoints"),
    (r"`data\recordings\E1_p<k>.*`", "3.3", "3.5, 5.7, 5.8, 6.4"),
    (r"`data\recordings\invariance_1.*`", "3.4", "3.5 -> 3.6"),
    (r"`data\recordings\T4_motion_1.*`", "4.1", "4.2 -> 4.3; 5.7; 6.6"),
    (r"`output\pose\<name>_poses.csv`, `_annotated.avi`", "3.5, 4.2", "3.6, 4.3"),
    (r"`training\synthetic\synth_v1\`", "5.1", "5.2, 5.6 (mixed), 6.4 (E5)"),
    (r"`training\models\resnet18_v2.pt`", "5.2", "5.3 (name), 5.4 (stage 1), 5.6"),
    (r"`training\synthetic\crop_v1\`", "5.3", "5.4, 5.6 (mixed)"),
    (r"`training\models\crop_resnet18_v1.pt`", "5.4", "5.6"),
    (r"`training\real_v2..v4\<rec>.avi`", "5.5", "5.6"),
    (r"`training\real_v<k>\labels\`, `training\synthetic\real_<...>_full / _crop`", "5.6", "5.6 (reused on the next run)"),
    (r"`training\models\resnet18_v2_ft3.pt`, `crop_resnet18_v1_ft3.pt`", "5.6", "5.7, 5.8, 6.4, 6.5, 6.6, 8.1 (live)"),
    (r"`data\recordings\E1_turn<..>, E2_<..>, E4_<..>.*`", "6.2", "6.4, 6.5"),
    (r"`data\recordings\E3_motion_0<k>.*`", "6.3", "6.4, 6.5"),
    (r"`data\recordings\E7_nocover_<..>.*`", "7.2", "7.4, 7.5"),
    (r"`data\recordings\E7_box_<..>, E7_cover_<..>.*`", "7.3", "7.4, 7.6, 7.7"),
    (r"`output\evaluation\`, `output\network\`, `output\bonus\` results", "5.7, 5.8, 6.4-6.6, 7.4-7.7", "the report and the paper (not read by code)"),
]

PARTS = [("Setup", "0."), ("T1 - Markers", "1."), ("T2 - Detection", "2."), ("T3 - Calibration and poses", "3."), ("T4 - Animation", "4."),
         ("T5 - Network", "5."), ("T6 - Comparison", "6."), ("Bonus - prediction behind an obstacle (T7)", "7."),
         ("Everything at once (live)", "8.")]

IO_COLS = [("", 0.32, "c"), ("File or device", 1.8, "l"), ("Produced by (IN) / used later by (OUT)", 1.35, "l"), ("Format and contents", 2.2, "l"), ("Size", 0.6, "l")]


def write_step(P, s):
    P.h(3, f"Step {s['num']} - {s['title']}", numbered=False)
    P.p(f"**Command:** `{s['cmd']}`", after=1, keep=True, align="l")
    P.p(f"**Script:** {s['script']}", after=2, align="l")
    P.p(f"**What it does.** {s['what']}")
    if s["funcs"]:
        P.p("**Main functions.**", after=1, keep=True)
        P.bullets(s["funcs"], size=9)
    if s["opts"]:
        P.p("**Options and defaults.** " + "; ".join(s["opts"]) + ".", size=9.5)
    if s["io"]:
        rows = [[io, path, link, fmt, size] for io, path, link, fmt, size in s["io"]]
        P.table("io_" + s["num"].replace(".", "_"), f"Step {s['num']}: inputs (IN) and outputs (OUT).", IO_COLS, rows, size=7.5)
    for n in s["notes"]:
        P.p(n, size=9.5)


def write(P):
    # ------------------------------------------------------------------------------------------ title
    P.p("**Machine Vision - Assignment 2: Marker Pose Tracking**", size=15, align="c", after=2)
    P.p("Code documentation, step by step along README.md", size=12, align="c", after=2)
    P.p("Classical pipeline (ArUco, IPPE + Levenberg-Marquardt, relative pose) and learned pipeline (two-stage ResNet-18), evaluation E1-E6 and bonus", size=9.5,
        align="c", after=10, italic=True)

    # ------------------------------------------------------------------------------------------ 1 overview
    P.h(1, "Overview", "overview")
    P.p("The project measures the pose of a mobile ArUco marker M in the frame of a fixed marker F with a laptop camera, in two ways. The **classical "
        "pipeline** detects both markers (OpenCV ArUco, sub-pixel corners), estimates the pose of each marker in the camera with `SOLVEPNP_IPPE_SQUARE` "
        "followed by Levenberg-Marquardt refinement, and computes T_FM = inverse(T_CF) T_CM. The **learned pipeline** keeps F classical and gets M "
        "from a two-stage ResNet-18: stage 1 finds M in the whole image, stage 2 regresses the exact pose from a 224 px window around it; both were "
        "trained on synthetic images and fine-tuned on real frames labelled by the classical pipeline. Recordings with hand-measured truth compare "
        "both (E1-E6), and the bonus predicts the pose of M while an obstacle hides it (Kalman filter + alpha-beta filter on SO(3)). "
        "This document follows the 50 numbered steps of `README.md` ({sec:steps}); the formats shared by several steps are in {sec:formats}.")
    P.p("**Folders.** `code\\` holds every script; `code\\config.py` is the one place where the folders are defined (environment variable, else an "
        "`_OVERRIDE` line, else a folder next to `code`). `data\\` holds the inputs: `markers.json`, `calibration\\` (`board.json`, `images\\`, "
        "`camera.npz`) and `recordings\\` (test recordings and their truth). `training\\` holds everything for the network: `synthetic\\` (data sets), "
        "`real_v2\\`, `real_v3\\`, `real_v4\\` (real training recordings) and `models\\` (checkpoints). `output\\` holds what the scripts generate "
        "(markers, dictionary, print, detection, calibration, pose, animation, network, evaluation, bonus, public). `paper\\` builds the report "
        "(`docx_lib.py`, `figcheck.py` are used read-only here).")
    P.p("**Shared code.** `mvlib.py` (lecture toolkit: SE(3) helpers, `marker_pose`, `MarkerDetector`, `ConstantVelocityKalman`, `AlphaBetaSO3`, "
        "`mean_rotation`), `mvtools.py` (questions with defaults, `markers.json` / `camera.npz` loading, `MarkerPoseEstimator`, pose <-> CSV, truth "
        "and error metrics), `netpose.py` (network definition and `NetPose` inference), `synthdata.py` (synthetic renderer), `t5_train.py` (training "
        "loop, also called by `t5_finetune.py`), `t7_bonus_occlusion.py` (`OcclusionPredictor`, used by 7.5-7.8).")
    P.p("**Coordinate conventions** (headers of `mvlib.py` and `mvtools.py`). Camera frame C: x right, y down, z forward out of the lens; pixels "
        "(u, v) from the top-left corner, u right, v down. Marker frame: origin at the centre of the black square, x right, y up as the marker is read, "
        "z out of the printed side toward the viewer; the four corners are ordered TL, TR, BR, BL (as `detectMarkers` returns them and IPPE_SQUARE "
        "needs them). T_AB is the 4x4 matrix that maps coordinates given in B into A (P_A = T_AB P_B); T_FM is the pose of M in F. Metres "
        "inside the code; millimetres and degrees in files and tables, rotations in CSV files as rotation vectors [rad]. Roll-pitch-yaw is ZYX: "
        "R = Rz(yaw) Ry(pitch) Rx(roll); in the truth files x + = M right of F, y + = M higher, z + = M nearer the camera, pitch = turn about the vertical axis.")
    P.fig(FIG, "flow", "Data flow between the steps (step numbers in brackets). Arrows: which files a later step reads. Not drawn: every step "
          "reads `data\\markers.json` and every pose step reads `camera.npz`.", width=5.6)
    P.table("flow", "Which step writes each file and which later steps read it.",
            [("File(s)", 2.55, "l"), ("Written by step", 1.15, "l"), ("Read by step(s)", 2.57, "l")], FLOW, size=7.5)

    # ------------------------------------------------------------------------------------------ 2 steps
    P.h(1, "The steps of README.md", "steps")
    P.p("One entry per numbered step of `README.md`, in its order and with its numbers. Each entry gives the command exactly as in the README, the "
        "script, what it does, the main functions (as `file.py: function()`), the options with their defaults ([value] = default answer or option "
        "value), and a table of the inputs (IN: who produced them) and outputs (OUT: which later step reads them and for what). Sizes are those of "
        "the files in the project; `<name>` stands for a recording name, `<split>` for train / val / test. Camera images are named but never shown.")
    for title, prefix in PARTS:
        P.h(2, title)
        for s in S:
            if s["num"].startswith(prefix):
                write_step(P, s)

    # ------------------------------------------------------------------------------------------ 3 formats
    P.h(1, "Reference: file formats shared by several steps", "formats")
    P.p("Checked on the real files with read-only code (`docs\\code_documentation_build\\inspect_files.py`): CSV headers, JSON keys, "
        "`numpy.load(..., mmap_mode=\"r\")` shapes and dtypes, `torch.load(..., map_location=\"cpu\")` keys, and the size, frame count, rate and codec of "
        "the videos with OpenCV.")

    P.h(2, "Marker and board configuration", "fmt_markers")
    P.table("markers", r"`data\markers.json` (written by 1.1, edited in 1.4; read by every later step through `mvtools.load_markers_cfg`).",
            [("Key", 1.35, "l"), ("Type", 0.6, "l"), ("Value here", 1.2, "l"), ("Meaning / use", 3.12, "l")],
            [["`dictionary`", "string", "DICT_6X6_250", "OpenCV predefined dictionary; `getattr(cv2.aruco, name)` -> `dict_id`"],
             ["`id_fixed`", "int", "0", "id of F"], ["`id_moving`", "int", "23", "id of M"],
             ["`nominal_side_mm`", "float", "80.0", "side of the black square as generated"],
             ["`measured_side_mm`", "float or null", "77.0", "calliper value of the printed square; if set, L = 0.077 m is used for every pose"],
             ["`quiet_zone_cells`", "int", "2", "white border in marker cells"], ["`note`", "string", "-", "reminder text"],
             ["(added in memory)", "-", "-", "`load_markers_cfg` adds `dict_id`, `L` [m] and `side_is_measured`"]], size=8)
    P.table("board", r"`data\calibration\board.json` (written by 1.2, edited in 1.4; read by 3.1 and 3.2).",
            [("Key", 1.6, "l"), ("Type", 0.6, "l"), ("Value here", 0.9, "l"), ("Meaning / use", 3.17, "l")],
            [["`cols`, `rows`", "int", "9, 6", "inner corners (the board has 10 x 7 squares)"],
             ["`square_mm`", "float", "25.0", "nominal square"], ["`measured_square_mm`", "float or null", "24.0",
              "printed square; `load_board` uses it as `square_used_mm` for the 3D corner grid (scale of every translation)"],
             ["`note`", "string", "-", "reminder text"]], size=8)

    P.h(2, "Camera calibration", "fmt_camera")
    P.table("camera", r"`data\calibration\camera.npz` (written by 3.2 with `mvtools.save_camera`; read by `mvtools.load_camera`) and its copy `camera.json`.",
            [("Key", 0.9, "l"), ("Type / shape", 1.2, "l"), ("Value here", 2.2, "l"), ("Meaning", 1.97, "l")],
            [["`K`", "float64 (3, 3)", "fx 846.59, fy 845.25, cx 641.12, cy 361.78", "intrinsic matrix [px]"],
             ["`dist`", "float64 (5,)", "0.0922, -0.1348, 0, 0, 0", "k1 k2 p1 p2 k3 (model k1k2: p1 = p2 = k3 = 0)"],
             ["`width`, `height`", "int64 scalars", "1280, 720", "image size the calibration is valid for"],
             ["`rms`", "float64 scalar", "1.184", "RMS reprojection error [px]"],
             ["`camera.json` only", "JSON", "`K, dist, width, height, rms_px, model, n_views, square_mm`", "human-readable copy (model k1k2, 20 views, 24 mm); not read by the code"]],
            size=8, note="Size: `camera.npz` 1.3 KB, `camera.json` 393 B.")

    P.h(2, "A recording", "fmt_rec")
    P.p("Every recorder (`t3_record.py`, `t6_record.py`, `t4_record_guided.py`) writes the same set of files with the same base name; test recordings "
        "are in `data\\recordings`, training recordings in `training\\real_v<k>`. One recording is one unit of the train / test split.")
    P.table("rec", "The files of a recording (example `data\\recordings\\E1_p1`).",
            [("File", 0.85, "l"), ("Format and contents", 4.0, "l"), ("Example size", 1.42, "l")],
            [["`<name>.avi`", "MJPG, 1280x720, 3-channel BGR, container rate 30 fps (the true rate is in the `.csv`); read with `cv2.VideoCapture`. "
              "E1_p1: 100 frames.", "7.5 MB (100 frames); 50 MB (E3, 600); 229 MB (E7_box_5cms_2)"],
             ["`<name>.csv`", "header `frame,t_s`; one row per frame: index, seconds since recording start (4 decimals). `mvtools.recording_times` uses it "
              "(fallback: index / fps).", "1.1 KB per 100 frames"],
             ["`<name>.json`", "keys `name, tag, notes, distance_cm, tilt_deg, resolution, frames, duration_s, fps, dictionary, created`; `t6_record.py` adds "
              "`camera_check` (median live reading: x, y, z, roll, pitch, yaw). Tags: E1_static, E2_dist_tilt, E3_motion, E4_robust, E7_occlusion, train, demo, other. "
              "Some existing files carry an extra key `recorded_by`.", "0.3-0.8 KB"],
             ["`<name>.truth.csv`", "header `t_s,x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg`: hand-measured pose of M in F. One row = static pose for every "
              "frame (E1, E2, E4); several rows = known motion, linearly interpolated in time by `mvtools.truth_pose` (E3). Not written for T4, training "
              "and E7 recordings.", "65 B (1 row); 0.5 KB (E3, 14 rows)"]], size=8)

    P.h(2, "The pose CSV of t3_pose.py", "fmt_poses")
    P.table("poses", r"Columns of `output\pose\<name>_poses.csv` (`mvtools.result_to_row` + roll / pitch / yaw added by `t3_pose.process_video`); one row per frame.",
            [("Column(s)", 1.7, "l"), ("Type / unit", 1.0, "l"), ("Meaning", 3.57, "l")],
            [["`frame`", "int", "frame index in the video"], ["`t_s`", "float [s]", "time stamp from `<name>.csv`"],
             ["`F_found`, `M_found`", "0 / 1", "marker detected in this frame (`M_found` = 1 always with the network source)"],
             ["`F_held`", "0 / 1", "F not detected, the last T_CF reused (camera still = y)"],
             ["`amb_F`, `amb_M`", "0 / 1", "planar ambiguity: second / best IPPE reprojection error below 1.5; the candidate nearest the previous rotation was used"],
             ["`rms_F_px`, `rms_M_px`", "float [px]", "RMS reprojection error of the chosen pose; empty / nan when not measured"],
             ["`M_source`", "text", "classical or network"],
             ["`FM_x_mm`, `FM_y_mm`, `FM_z_mm`", "float [mm]", "position of M in F (T_FM translation); nan when there is no relative pose"],
             ["`FM_rx`, `FM_ry`, `FM_rz`", "float [rad]", "rotation of M in F as a rotation vector (axis x angle, `cv2.Rodrigues`)"],
             ["`CM_...` (6 columns)", "mm, rad", "pose of M in the camera, T_CM"], ["`CF_...` (6 columns)", "mm, rad", "pose of F in the camera, T_CF"],
             ["`roll_deg`, `pitch_deg`, `yaw_deg`", "float [deg]", "ZYX angles of T_FM (3 decimals); empty when there is no relative pose"]],
            size=8, note="31 columns. Read back with `mvtools.cols_to_pose(row, \"FM\")`. Size about 0.43 KB per frame.")

    P.h(2, "Synthetic and real data sets", "fmt_sets")
    P.table("sets", r"Files of a data set folder `training\synthetic\<set>\` (written by 5.1, 5.3, 5.6; read by `synthdata.load_set` / `load_boxes`).",
            [("File", 1.15, "l"), ("dtype", 0.55, "l"), ("Shape", 1.55, "l"), ("Contents", 3.02, "l")],
            [["`X_<split>.npy`", "uint8", "(N, 360, 640) whole image; (N, 224, 224) crop", "grey images (whole image: camera scaled to 640x360, no distortion "
              "in synthetic sets, undistorted real frames); crops: window resized to 224"],
             ["`R_<split>.npy`", "float32", "(N, 3, 3)", "rotation of M in the camera (label R of T_CM)"],
             ["`t_<split>.npy`", "float32", "(N, 3)", "position of M in the camera [m] (full-resolution camera frame)"],
             ["`box_<split>.npy`", "float32", "(N, 3), crop sets only", "window u0, v0, side in pixels of the 1280x720 camera"],
             ["`meta.json`", "JSON", "-", "see {tab:meta}"], ["`build_info.json`", "JSON", "real sets only", "`labels, validation, crop_jitter, max_rms_px, marker_side_m`"]],
            size=8, note="Splits: synthetic sets train / val / test; real sets train / val (whole recordings). Sizes: synth_v1 10000 / 1000 / 1000 (2.6 GB); crop_v1 "
                         "30000 / 2000 / 2000 (1.6 GB); real_v2_real_v3_real_v4_full / _crop 11562 / 3725 (3.3 GB / 0.73 GB). `synth_full_test` (1000 test images at "
                         "1280x720, seed 100) is not made by a README step.")
    P.table("meta", r"Keys of `meta.json` of a data set.",
            [("Key", 1.6, "l"), ("synth_v1", 1.45, "l"), ("crop_v1", 1.45, "l"), ("Use", 1.77, "l")],
            [["`name`", "synth_v1", "crop_v1", "`data_set` in the checkpoint; resume check"], ["`kind`", "(absent = full)", "crop", "3-channel network, boxes"],
             ["`dictionary`, `id_moving`, `id_fixed`", "DICT_6X6_250, 23, 0", "same", "copied to the checkpoint"],
             ["`L_m`", "0.077", "0.077", "marker side [m]; crop window, hard cases"],
             ["`K_cam`, `size_cam`", "calibrated K, [1280, 720]", "same", "camera the network is valid for"],
             ["`K_net`, `net_size`", "K / 2, [640, 360]", "K_cam, [224, 224]", "camera of the network input"],
             ["`crop_size`, `stage1`", "-", "224, resnet18_v2.pt", "crop size; stage-1 file name"],
             ["`n_train`, `n_val`, `n_test`, `seed`", "10000, 1000, 1000, 0", "30000, 2000, 2000, 0", "set sizes, seed"],
             ["`provisional_camera`", "false", "false", "true if made before the calibration"],
             ["`params`", "z_range [0.25, 1.0], tilt 60, p_distractor 0.7, p_occlude 0.15", "the same + jitter 0.12", "rendering parameters"],
             ["`generation_seconds`", "44.6", "75.9", "time to render"],
             ["real sets only", "`real`, `from_recordings`, `validation_recording`", "+ `crop_jitter_real`", "added by 5.6 (see note of step 5.6)"]], size=7.5)

    P.h(2, "Network checkpoint and training log", "fmt_pt")
    P.table("pt", r"A checkpoint `training\models\<net>.pt` (`torch.save` of a dict; read by `netpose.NetPose` and as `init_from` in 5.6).",
            [("Entry", 1.3, "l"), ("Contents", 4.97, "l")],
            [["`state_dict`", "122 tensors of the ResNet-18 (CPU): `conv1.weight` (64, 1, 7, 7) for whole-image networks, (64, 3, 7, 7) for crop networks; "
              "`fc.weight` (9, 512), `fc.bias` (9,). 11.17 M parameters (full) / 11.18 M (crop). File 42.7 MB."],
             ["`config`", "dict written by `t5_train.train`: `best_epoch, arch, net_size, K_cam, size_cam, K_net, L_m, dictionary, id_moving, provisional_camera, "
              "data_set, epochs, batch, lr, w_rot, w_trans, seed, pretrained, n_train, n_params, train_minutes, device, val` (the epoch record used), "
              "`stopped_early, epochs_done, kind, crop_size, stage1, strong_aug, clip, rot_aug_deg, hard_aug, fine_tuned_from, mixed_with, real_recordings, "
              "validation_recording`. Older checkpoints (`resnet18_v1/v2/v3`, `crop_resnet18_v1`, `_ft`) have a subset of these keys."],
             ["checkpoint during training", "`<net>_checkpoint.pt`: `model, opt, sched, rng, hist, step, epoch, minutes, best, settings`; deleted when training ends"]],
            size=8)
    P.table("models", "The checkpoints in `training\\models` (from their `config`).",
            [("File", 1.55, "l"), ("Step", 0.8, "l"), ("kind / data set", 1.45, "l"), ("Epochs (best)", 0.75, "l"), ("Validation median rot / trans", 1.72, "l")],
            [["`resnet18_v1.pt`", "5.2", "full / synth_v1", "15", "23.5 deg / 26.4 mm"], ["`resnet18_v2.pt`", "5.2", "full / synth_v1", "40", "7.3 deg / 13.5 mm"],
             ["`resnet18_v3.pt`", "5.2", "full / synth_v1, w_rot 2", "40", "8.9 deg / 16.7 mm"],
             ["`crop_resnet18_v1.pt`", "5.4", "crop / crop_v1", "80", "1.95 deg / 8.8 mm"],
             ["`resnet18_v2_ft.pt`, `crop_..._ft.pt`", "5.6 (round 1)", "real_v2 sets", "15", "8.2 deg / 14.3 mm; 8.1 deg / 6.8 mm"],
             ["`resnet18_v2_ft2.pt`, `crop_..._ft2.pt`", "5.6 (round 2)", "real_v2_real_v3 sets", "25 (8; 21)", "6.6 deg / 29.4 mm; 3.8 deg / 6.5 mm"],
             ["`resnet18_v2_ft3.pt`, `crop_..._ft3.pt`", "5.6", "real_v2_real_v3_real_v4 sets", "25 (22); 50 (48)", "5.8 deg / 15.4 mm; 3.7 deg / 6.4 mm"]],
            size=7.5, note="Validation sets differ (synthetic for the first four rows, real validation recordings for the fine-tuned ones), so the numbers "
                           "are not comparable across rows.")
    P.table("trainlog", r"`output\network\<net>\train_log.json` (written at the end of `t5_train.train`).",
            [("Key", 1.1, "l"), ("Contents", 5.17, "l")],
            [["`config`", "the same dict as the checkpoint's `config`"],
             ["`log`", "list, one record per epoch: `epoch, loss, rot_loss, trans_loss` (training means), `minutes` (elapsed), `val_rot_med, val_rot_p95` [deg], "
              "`val_trans_med, val_trans_p95` [mm]"],
             ["next to it", "`training_curves.png` (loss, validation rotation and translation), `dashboard.png` (last dashboard), `examples\\epoch_<NNN>.png` "
              "(4 validation images per epoch; fine-tuning runs)"]], size=8,
            note="Sizes 6-27 KB (15-80 epochs). The fine-tuning console logs in `output\\network\\logs\\` are not written by the code.")

    P.h(2, "Main result files in output", "fmt_results")
    P.table("results", "Result files and the step that writes them.",
            [("File", 2.2, "l"), ("Step", 0.55, "l"), ("Format", 3.52, "l")],
            [["`output\\calibration\\report.json`, `per_view.csv`", "3.2", "see step 3.2"],
             ["`output\\detection\\detection_matrix.csv` / `.md`", "2.3", "one row per cell (17 columns); Markdown matrix and limits"],
             ["`output\\detection\\guided\\guided_report.md`", "2.2, 2.4", "Markdown table step / file / result"],
             ["`output\\pose\\invariance_1_invariance.md` / `.png`", "3.6", "check B table and verdict"],
             ["`output\\animation\\T4_motion_1_animation.mp4` / `.gif`, `_network_animation.mp4`", "4.3, 6.6", "MP4 1572x480 at 15.6 fps; GIF half size"],
             ["`output\\network\\<net>\\train_log.json`", "5.2, 5.4, 5.6", "{tab:trainlog}"],
             ["`output\\network\\real_E1_comparison_ft3.json`", "5.7", "method -> E1 recording -> `et, er, out, jit, d, ms`"],
             ["`output\\network\\hard_cases_E1.json`", "5.8", "case -> classical output / usable, `_ft2` / `_ft3` usable and median errors"],
             ["`output\\evaluation\\comparison_table.md` / `.csv`, `report.md`, `summary.json`, `worst_frames.csv`, `E3_<rec>.png`", "6.4", "see step 6.4"],
             ["`output\\evaluation\\side_by_side_<name>.mp4`", "6.5", "MP4 1132 px wide, 15 fps"],
             ["`output\\bonus\\<name>_passes.csv`, `_track.csv`", "7.4", "see step 7.4"],
             ["`output\\bonus\\virtual_occlusion.*`, `real_occlusion.*`", "7.5, 7.6", "CSV per pass (28 / 22 columns), Markdown summary, PNG"],
             ["`output\\bonus\\<name>_predicted_poses.csv`, `_occlusion.mp4` / `.gif`", "7.7", "CSV with `measured, predicted, lost, sigma_mm` + `FM_` pose"],
             ["`T2_summary.md`, `E1_checkA.md`, `T5_runs.md`, `T6_notes.md`, `E7_summary.md`, `measurements.csv`", "-", "hand-written summaries; no script writes them"]],
            size=7.5)

    # ------------------------------------------------------------------------------------------ 4 not verified
    P.h(1, "Points that could not be verified", "nv")
    P.p("Everything above was read from the code or checked on the files in the project, except the following points.")
    P.bullets([f"Step {k}: {t}" for k, t in NOT_VERIFIED] + [
        "Files with no producing script in `code\\` (searched by name): `output\\network\\two_stage_vs_full.json` (named in `presentation\\slide_figures.py`), "
        "`output\\network\\figures\\augmentation_*.png`, `real_v4_first_frames.jpg`, `output\\bonus\\*_occlusion_frames.jpg`, `real_occlusion_all42.csv`, "
        "`output\\evaluation\\what_to_measure_turned_M.png`, `training\\synthetic\\synth_full_test`, `output\\network\\logs\\*.log`; their origin is not documented here.",
        "No pipeline step was run for this document (no camera, generation, training or evaluation); behaviour is described from the code."], size=9.5)


if __name__ == "__main__":
    probs = dataflow.make(FIG)
    if probs:
        raise SystemExit(f"diagram has overlaps: {probs}")
    assert len(S) == 50, len(S)
    second, unresolved = build(write, OUT)
    # list items carry long code names: left-aligned, so that justification does not open wide gaps
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    d = Document(OUT)
    for par in d.paragraphs:
        if par.style.name == "List Bullet" or par.text.startswith(("Shared code.", "Options and defaults.")):
            par.alignment = WD_ALIGN_PARAGRAPH.LEFT
    d.save(OUT)
    print("written", OUT, "unresolved references:", unresolved)
