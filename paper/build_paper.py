"""build_paper.py -- writes the paper as a Word file (run from the project folder, after make_figures.py):

    python paper/build_paper.py

Output: paper/Marker_Pose_Classical_vs_Learned.docx.  Text and numbers are in this file; the per-recording tables (E1-E4) are
computed here from paper/data/test_errors.csv so they cannot drift from the figures.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from docx_lib import build  # noqa: E402
from references import REFS  # noqa: E402

FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "Marker_Pose_Classical_vs_Learned.docx")
REPO = "https://github.com/danvalentin13/machine-vision-marker-pose-classical-aruco--vs-resnet"


# ---------------------------------------------------------------------------------------------- numbers from the data
def _rows(p):
    with open(os.path.join(ROOT, p), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


E = {}
for _x in _rows("paper/data/test_errors.csv"):
    E.setdefault(_x["recording"], []).append([_fl(_x[k]) for k in ("et_classical_mm", "eR_classical_deg", "et_network_mm", "eR_network_deg",
                                                                  "xc", "yc", "zc", "xn", "yn", "zn")])
E = {k: np.array(v) for k, v in E.items()}


def stats(name, k):
    """k = 0 classical, 1 network: e_t median, e_R median, bias, jitter, output %, usable %, RMSE, max e_t, max e_R."""
    a = E[name]
    et, er, p = a[:, 2 * k], a[:, 2 * k + 1], a[:, 4 + 3 * k:7 + 3 * k]
    ok = np.isfinite(et)
    out = dict(out=100 * ok.mean(), use=100 * np.mean((et < 20) & (er < 10)))
    if not ok.any():
        return dict(out, et=None, er=None, rmse=None, etmax=None, ermax=None, p95=None, bias=None, jit=None)
    tr = _rows(f"data/recordings/{name}.truth.csv")
    out.update(et=np.nanmedian(et), er=np.nanmedian(er), rmse=np.sqrt(np.nanmean(et ** 2)), etmax=np.nanmax(et), ermax=np.nanmax(er),
               p95=np.nanpercentile(et[ok], 95))
    if len(tr) == 1:
        truth = np.array([_fl(tr[0]["x_mm"]), _fl(tr[0]["y_mm"]), _fl(tr[0]["z_mm"])])
        q = p[np.isfinite(p).all(1)]
        out.update(bias=np.linalg.norm(q.mean(0) - truth), jit=np.sqrt(np.mean(np.sum((q - q.mean(0)) ** 2, axis=1))))
    return out


def f1(v, d=1):
    return "–" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def run_rows():
    """One row per saved training run, from output/network/<run>/train_log.json."""
    import json
    spec = [("resnet18_v1", "1", "synthetic whole images (run A)"), ("resnet18_v2", "1", "synthetic whole images (run B)"),
            ("resnet18_v3", "1", "synthetic whole images (run C)"), ("crop_resnet18_v1", "2", "synthetic crops (run D)"),
            ("resnet18_v2_ft", "1", "round 1: real_v2 and synthetic"), ("crop_resnet18_v1_ft", "2", "round 1: real_v2 and synthetic"),
            ("resnet18_v2_ft2", "1", "round 2: real_v2, v3 and synthetic"), ("crop_resnet18_v1_ft2", "2", "round 2: real_v2, v3 and synthetic"),
            ("resnet18_v2_ft3", "1", "round 3: real_v2, v3, v4 and synthetic"), ("crop_resnet18_v1_ft3", "2", "round 3: real_v2, v3, v4 and synthetic")]
    out = []
    for name, stage, data in spec:
        d = json.load(open(os.path.join(ROOT, "output", "network", name, "train_log.json")))
        cfg, log = d["config"], d["log"]
        k = cfg.get("best_epoch", len(log))
        e = log[k - 1]
        lr = r"$5 \times 10^{-4}$" if abs(cfg["lr"] - 5e-4) < 1e-9 else r"$10^{-4}$"
        out.append([name, stage, data, f"{len(log)} ({k})", lr, f"{cfg['train_minutes']:.1f}", f"{e['rot_loss']:.4f} / {e['trans_loss']:.4f}",
                    f"{e['val_rot_med']:.2f} ({e['val_rot_p95']:.1f})", f"{e['val_trans_med']:.1f} ({e['val_trans_p95']:.0f})"])
    return out



# ---------------------------------------------------------------------------------------------- the paper
def write(P):
    B = P.base

    # ------------------------------------------------------------------ title block
    P.p("Marker Pose Tracking with a Laptop Camera: Classical Fiducial Geometry versus a Two-Stage Learned Regressor", size=17, align="c", bold=True, after=8)
    P.p("Dan-Valentin Sutu, Mihai-Daniel Carp, Ilie-Catalin Jureschi", size=11, align="c", after=1)
    P.p("“Gheorghe Asachi” Technical University of Iași, Romania", size=9.5, align="c", after=1, italic=True)
    P.p("Machine Vision, Assignment 2 (master’s course project). November 2026", size=9.5, align="c", after=10, italic=True)

    P.p("Abstract", size=B, align="c", bold=True, after=3)
    P.p(r"We compare two ways of measuring the six-degree-of-freedom pose of a mobile fiducial marker relative to a fixed one with a single laptop "
        r"webcam and two printed 77 mm ArUco markers: a classical pipeline (corner detection, planar pose from IPPE, Levenberg–Marquardt refinement, "
        r"a temporal rule for the planar ambiguity) and a learned one, a two-stage ResNet-18 regressor that predicts a 6-D rotation and a "
        r"depth-normalised translation, trained on 40 000 domain-randomised synthetic images and fine-tuned on 11 562 real frames labelled by the "
        r"classical pipeline. Both run on the same frames; the ground truth comes from a ruler and a protractor. On 21 test recordings the classical "
        r"pipeline reaches a median error of 7.8 mm and 3.8° in static scenes, with 0.6 mm of jitter, at 147 frames per second; its error is a "
        r"repeatable bias, consistent with the limits of our calibration and of hand measurement, not noise. Trained on synthetic images alone the "
        r"network fails on real frames (26.5 mm median, 107 mm at the 95th percentile). Three rounds of fine-tuning, with a camera-rotation and a "
        r"hard-case augmentation, bring it to 8.1 mm and 4.0°, with twice the jitter and one sixth of the speed. The network still returns a pose "
        r"where the detector returns none (a covered edge on the bench, strong blur in a simulated test), but it is then less accurate, mostly in depth, and it cannot say that no "
        r"marker is present. A constant-velocity Kalman predictor bridges occlusions: after 1.7 s behind an obstacle its position error is 50 mm, "
        r"against 174 mm for holding the last pose. We report the failures in full (a marker cut without its quiet zone that produced a false 18° "
        r"tilt, an evaluation error that flattered the first occlusion results by a factor of up to two, a calibration board that was not flat), because they "
        r"shaped the final numbers more than any design choice did.", size=B - 0.5, indent=0.35, after=4)
    P.p(r"**Index terms:** fiducial markers, ArUco, planar pose estimation, camera calibration, pose regression, synthetic-to-real transfer, "
        r"Kalman filter.", size=B - 0.5, indent=0.35, after=8)

    # ================================================================== 1 Introduction
    P.h(1, "Introduction", "intro")
    P.p(r"Square fiducial markers are the standard way to give a camera a metric reference with six degrees of freedom: four corners of known "
        r"geometry are detected, a binary code identifies the marker, and a planar perspective-n-point problem gives its pose [@garrido,@olson,@romero]. "
        r"The method is fast and its failures are easy to read, but it is all or nothing: when the code cannot be decoded (an edge is covered, the "
        r"image is blurred, the marker is seen at a steep angle) there is no pose at all. Regression networks [@kendall,@xiang] degrade more "
        r"gradually in principle, at the price of training data and of having to learn a geometry that the classical solver gets for free.")
    P.p(r"This paper reports a complete comparison of the two at the scale of a course project: one laptop webcam, two printed markers, a ruler "
        r"and a protractor. The task is the pose of a mobile marker M in the frame of a fixed marker F, $\lsup{F}{\mathbf{T}_{M}}$, which does not "
        r"depend on where the camera stands. Both pipelines are measured on the same frames, against truth measured by hand. The work follows the "
        r"six tasks of the assignment (T1 markers and set-up, T2 detection, T3 calibration and classical pose, T4 recording and animation, T5 learned "
        r"pose, T6 comparison) and its bonus task (prediction through an occlusion). The contributions are:")
    P.bullets([
        r"a classical pipeline with sub-pixel corners, IPPE [@collins] and Levenberg–Marquardt refinement, a temporal rule for the planar ambiguity "
        r"[@schweighofer], and two independent checks of its output (ruler positions; invariance to camera motion);",
        r"a two-stage regressor built on ResNet-18 [@he], with a continuous 6-D rotation output [@zhou], a depth-normalised translation and a crop "
        r"stage that receives the viewing ray of every pixel, cf. [@li_cdpn,@liu_coordconv], trained on domain-randomised images [@tobin] and "
        r"fine-tuned on real frames labelled by the classical pipeline;",
        r"an evaluation on 21 real test recordings with hand-measured truth (static positions, distance and tilt, a known motion, four kinds of "
        r"disturbance), on a held-out synthetic set, and of the cost of each method;",
        r"a predictor that carries the pose through an occlusion, evaluated on 49 passes behind a real obstacle;",
        r"an account of what went wrong and how each problem was found ({sec:failures}), which we consider the most transferable part of the work.",
    ])
    P.p(r"*Related work.* ArUco [@garrido,@romero] and AprilTag [@olson] are the usual square markers; their pose is a planar problem with a "
        r"well-known two-fold ambiguity [@schweighofer], solved in closed form by IPPE [@collins]. Learned alternatives range from absolute pose "
        r"regression, PoseNet [@kendall], to object-pose networks that decouple rotation from translation, PoseCNN [@xiang], and that zoom on the "
        r"object before regressing, CDPN [@li_cdpn]; Deep ChArUco [@hu_charuco] shows that a network can find marker corners in images too dark or "
        r"blurred for the classical detector. Training on rendered images with randomised appearance [@tobin,@tremblay] is the common way to obtain "
        r"labels. Our notation follows Corke [@corke] and the course lectures [@lectures].")

    # ================================================================== 2 Mathematical description
    P.h(1, "Problem statement and mathematical description", "math")
    P.h(2, "Frames and the relative pose", "frames")
    P.p(r"Three right-handed frames are used ({fig:setup}a): the camera {C} (x to the right, y down, z along the optical axis), the fixed marker "
        r"{F} and the mobile marker {M} (origin at the marker centre, x to the right and y up when the marker is read upright, z out of the marker "
        r"towards the viewer). The pose of a frame B in a frame A is the homogeneous matrix")
    P.eq(r"\lsup{A}{\mathbf{T}_{B}} = \begin{bmatrix} \lsup{A}{\mathbf{R}_{B}} & \lsup{A}{\mathbf{t}_{B}} \\ \mathbf{0}^{\top} & 1 \end{bmatrix} "
         r"\in \mathrm{SE}(3), \qquad \lsup{A}{\mathbf{p}} = \lsup{A}{\mathbf{T}_{B}}\,\lsup{B}{\mathbf{p}}", "pose")
    P.p(r"which maps coordinates given in B to coordinates given in A. The camera measures $\lsup{C}{\mathbf{T}_{F}}$ and "
        r"$\lsup{C}{\mathbf{T}_{M}}$; the quantity of interest is the relative pose")
    P.eq(r"\lsup{F}{\mathbf{T}_{M}} = \left(\lsup{C}{\mathbf{T}_{F}}\right)^{-1}\,\lsup{C}{\mathbf{T}_{M}}", "rel")
    P.p(r"We report its translation $(x, y, z)$ in millimetres and its rotation as roll, pitch and yaw in degrees, in the ZYX convention")
    P.eq(r"\mathbf{R} = \mathbf{R}_{z}(\mathrm{yaw})\,\mathbf{R}_{y}(\mathrm{pitch})\,\mathbf{R}_{x}(\mathrm{roll})", "rpy")
    P.p(r"For upright markers, $x > 0$ means M to the right of F as the camera sees it, $y > 0$ means M higher, $z > 0$ means M nearer the camera, "
        r"and the pitch is the turn of M about the vertical axis. Internally all rotations are matrices; angles are used only for output.")

    P.h(2, "Camera model and calibration", "camera")
    P.p(r"A point $(X, Y, Z)$ in {C} has the normalised image coordinates $x = X/Z$, $y = Y/Z$. The lens displaces them radially [@brown],")
    P.eq(r"\begin{bmatrix} x_{d} \\ y_{d} \end{bmatrix} = \left(1 + k_{1} r^{2} + k_{2} r^{4}\right) \begin{bmatrix} x \\ y \end{bmatrix}, "
         r"\qquad r^{2} = x^{2} + y^{2}", "dist")
    P.p(r"and the intrinsic matrix maps the result to pixels:")
    P.eq(r"\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \begin{bmatrix} x_{d} \\ y_{d} \\ 1 \end{bmatrix}, \qquad "
         r"\mathbf{K} = \begin{bmatrix} f_{x} & 0 & c_{x} \\ 0 & f_{y} & c_{y} \\ 0 & 0 & 1 \end{bmatrix}", "K")
    P.p(r"We write $\mathbf{u} = \pi(\mathbf{K}, k_{1}, k_{2}, \mathbf{p})$ for this projection. Calibration [@zhang] estimates $\mathbf{K}$, "
        r"$k_{1}$, $k_{2}$ and the pose $\mathbf{T}_{i}$ of a chessboard in each of $N$ views by minimising the reprojection error of its $n$ corners "
        r"$\mathbf{P}_{j}$:")
    P.eq(r"\underset{\mathbf{K},\,k_{1},\,k_{2},\,\{\mathbf{T}_{i}\}}{\min}\; \sum_{i=1}^{N}{\sum_{j=1}^{n}{\norm{\mathbf{u}_{ij} - "
         r"\pi\left(\mathbf{K}, k_{1}, k_{2}, \mathbf{T}_{i}\mathbf{P}_{j}\right)}^{2}}}", "calib")
    P.p(r"The square root of the mean of these squared distances is the RMS reprojection error, reported for all views together and per view. "
        r"Tangential distortion and the third radial coefficient are fixed to zero: it is the simplest model that fits this lens.")

    P.h(2, "Pose of a planar square marker", "pnp")
    P.p(r"In its own frame the corners of a marker of side $L$ are $\mathbf{P}_{j} = (\pm L/2, \pm L/2, 0)$. Their images $\mathbf{u}_{j}$ "
        r"determine the pose only up to the two-fold ambiguity of a planar target [@schweighofer]: when perspective is weak, the marker tilted by "
        r"$+\theta$ and by $-\theta$ project almost identically. IPPE [@collins] returns both candidates in closed form from the homography between "
        r"marker and image. Each is refined with the Levenberg–Marquardt method [@levenberg,@marquardt]:")
    P.eq(r"\left(\hat{\mathbf{R}}, \hat{\mathbf{t}}\right) = \underset{\mathbf{R},\,\mathbf{t}}{\mathrm{arg min}}\; "
         r"\sum_{j=1}^{4}{\norm{\mathbf{u}_{j} - \pi\left(\mathbf{K}, k_{1}, k_{2}, \mathbf{R}\mathbf{P}_{j} + \mathbf{t}\right)}^{2}}", "refine")
    P.p(r"Let $\varepsilon_{1} \le \varepsilon_{2}$ be the RMS reprojection errors of the two refined candidates. The pose is called ambiguous when")
    P.eq(r"\rho = \frac{\varepsilon_{2}}{\varepsilon_{1}} < 1.5", "amb")
    P.p(r"and in that case the candidate whose rotation is closest to the one accepted in the previous frame is kept, instead of the one with the "
        r"smaller error. Otherwise the first candidate is taken.")

    P.h(2, "Error measures", "errors")
    P.p(r"Against a true pose $(\mathbf{R}, \mathbf{t})$ the position and rotation errors of an estimate are the Euclidean distance and the "
        r"geodesic angle [@huynh]:")
    P.eq(r"e_{t} = \norm{\hat{\mathbf{t}} - \mathbf{t}}, \qquad e_{R} = \arccos\left(\frac{\mathrm{tr}\left(\hat{\mathbf{R}}^{\top}\mathbf{R}\right) - 1}{2}\right)", "err")
    P.p(r"For a static recording of $K$ frames with positions $\mathbf{p}_{k}$ and mean $\bar{\mathbf{p}}$ we separate what averaging can remove "
        r"from what it cannot:")
    P.eq(r"b = \norm{\bar{\mathbf{p}} - \mathbf{p}_{\mathrm{truth}}}\;\;(\mathrm{bias}), \qquad "
         r"j = \sqrt{\frac{1}{K}\sum_{k=1}^{K}{\norm{\mathbf{p}_{k} - \bar{\mathbf{p}}}^{2}}}\;\;(\mathrm{jitter})", "bias")
    P.p(r"The output rate is the share of frames in which a method returns a pose. A frame is called usable when $e_{t} < 20$ mm and "
        r"$e_{R} < 10°$. Summaries are medians, with the 95th percentile in brackets.")

    P.h(2, "Size of a marker cell in the image", "cell")
    P.p(r"A 6 × 6 marker is 8 cells wide with its black border. At distance $d$ and tilt $\theta$ one cell covers, across the direction of the tilt,")
    P.eq(r"c = \frac{f\,L\cos\theta}{8\,d}\;\;[\mathrm{px}]", "cell")
    P.p(r"The rule of thumb from the lectures [@lectures] is that decoding fails near 2 px per cell; {sec:t2} tests it.")

    P.h(2, "Learned regressor: output and loss", "netmath")
    P.p(r"The network maps an image to nine numbers. Six of them form two vectors $\mathbf{a}_{1}, \mathbf{a}_{2}$ that are orthonormalised "
        r"into a rotation matrix [@zhou]:")
    P.eq(r"\mathbf{b}_{1} = \frac{\mathbf{a}_{1}}{\norm{\mathbf{a}_{1}}}, \qquad \mathbf{b}_{2} = \frac{\mathbf{a}_{2} - "
         r"\left(\mathbf{b}_{1}^{\top}\mathbf{a}_{2}\right)\mathbf{b}_{1}}{\norm{\mathbf{a}_{2} - \left(\mathbf{b}_{1}^{\top}\mathbf{a}_{2}\right)\mathbf{b}_{1}}}, "
         r"\qquad \hat{\mathbf{R}} = \begin{bmatrix} \mathbf{b}_{1} & \mathbf{b}_{2} & \mathbf{b}_{1}\times\mathbf{b}_{2} \end{bmatrix}", "sixd")
    P.p(r"This representation is continuous, unlike Euler angles or quaternions, which is what makes it easy to regress. The other three numbers "
        r"encode the translation as a direction in the image and the logarithm of the depth,")
    P.eq(r"\boldsymbol{\tau} = \left(\frac{X}{Z},\;\frac{Y}{Z},\;\log Z\right), \qquad \hat{\mathbf{t}} = \hat{Z}\left(\hat{\tau}_{1},\;\hat{\tau}_{2},\;1\right)^{\top}, "
         r"\qquad \hat{Z} = e^{\hat{\tau}_{3}}", "tau")
    P.p(r"so that the same relative depth error costs the same near and far. The loss is the weighted sum of a chordal rotation term and an "
        r"absolute translation term, averaged over the batch:")
    P.eq(r"\mathcal{L} = w_{R}\,\norm{\hat{\mathbf{R}} - \mathbf{R}}_{F}^{2} + w_{t}\,\norm{\hat{\boldsymbol{\tau}} - \boldsymbol{\tau}}_{1}", "loss")
    P.p(r"The chordal term is a function of the angle $\theta$ between the two rotations only,")
    P.eq(r"\norm{\hat{\mathbf{R}} - \mathbf{R}}_{F}^{2} = 4\left(1 - \cos\theta\right)", "chord")
    P.p(r"which lets a loss value be read as an angle: 0.0145, for instance, is 4.9°.")
    P.p(r"*Crop stage.* A first network sees the whole frame and gives a coarse position $(X, Y, Z)$ of M. A square window of side")
    P.eq(r"s = \frac{2\,f\,L}{Z}\;\;[\mathrm{px}], \qquad s \ge 48", "crop")
    P.p(r"centred on the projected marker centre (twice the apparent side of the marker) is cut from the undistorted full-resolution frame and "
        r"resampled to 224 × 224 px for a second network. A crop no longer tells where in the image it came from, and the same crop content means "
        r"a different pose at the centre and at the edge of the picture. Each crop pixel $(u, v)$ therefore carries two more channels, the "
        r"viewing ray $((u - c_{x})/f_{x},\;(v - c_{y})/f_{y})$. At run time the window is re-centred once on the second network’s own answer "
        r"(a second pass).")
    P.p(r"*Camera-rotation augmentation.* If the camera turns by $\mathbf{R}_{a}$ about its centre, the image changes by a homography that needs "
        r"no depth, and the label changes with it:")
    P.eq(r"\mathbf{H} = \mathbf{K}\,\mathbf{R}_{a}\,\mathbf{K}^{-1}, \qquad \mathbf{R} \leftarrow \mathbf{R}_{a}\mathbf{R}, \qquad "
         r"\mathbf{t} \leftarrow \mathbf{R}_{a}\mathbf{t}", "homog")
    P.p(r"Warping a training image by $\mathbf{H}$ thus gives an exact new training pair, as if the laptop screen had been tilted.")

    P.h(2, "Prediction through an occlusion", "predmath")
    P.p(r"The position of M in {F} is tracked with a constant-velocity Kalman filter [@kalman,@barshalom]. With the state "
        r"$\mathbf{x} = (\mathbf{p}, \mathbf{v})$ and the frame time $\Delta t$,")
    P.eq(r"\mathbf{x}_{k+1} = \mathbf{F}\mathbf{x}_{k} + \mathbf{w}_{k}, \qquad \mathbf{F} = \begin{bmatrix} \mathbf{I} & \Delta t\,\mathbf{I} \\ "
         r"\mathbf{0} & \mathbf{I} \end{bmatrix}, \qquad \mathbf{Q} = \sigma_{a}^{2} \begin{bmatrix} \Delta t^{4}/4\,\mathbf{I} & "
         r"\Delta t^{3}/2\,\mathbf{I} \\ \Delta t^{3}/2\,\mathbf{I} & \Delta t^{2}\,\mathbf{I} \end{bmatrix}", "kf")
    P.p(r"where $\mathbf{Q}$ is the covariance of a white-noise acceleration of standard deviation $\sigma_{a}$. The measurement is the position, "
        r"$\mathbf{z} = \mathbf{H}\mathbf{x} + \mathbf{n}$ with $\mathbf{H} = [\mathbf{I}\;\;\mathbf{0}]$ and noise $\sigma_{m}$. While M is "
        r"hidden only the prediction runs and the covariance grows, $\mathbf{P} \leftarrow \mathbf{F}\mathbf{P}\mathbf{F}^{\top} + \mathbf{Q}$. "
        r"A measurement is fused only if its innovation passes a chi-square gate (99.9 %, three degrees of freedom):")
    P.eq(r"\boldsymbol{\nu}^{\top}\mathbf{S}^{-1}\boldsymbol{\nu} \le 16.27, \qquad \boldsymbol{\nu} = \mathbf{z} - \mathbf{H}\hat{\mathbf{x}}, "
         r"\qquad \mathbf{S} = \mathbf{H}\mathbf{P}\mathbf{H}^{\top} + \sigma_{m}^{2}\mathbf{I}", "gate")
    P.p(r"The rotation is tracked by an alpha-beta filter on SO(3) with the angular velocity $\boldsymbol{\omega}$ expressed in {F}. Prediction "
        r"and update are")
    P.eq(r"\mathbf{R}^{-} = \exp\left(\left[\boldsymbol{\omega}\right]_{\times}\Delta t\right)\mathbf{R}, \qquad "
         r"\mathbf{r} = \log\left(\mathbf{R}_{\mathrm{meas}}\left(\mathbf{R}^{-}\right)^{\top}\right)^{\vee}", "so3a")
    P.eq(r"\mathbf{R} \leftarrow \exp\left(\alpha\left[\mathbf{r}\right]_{\times}\right)\mathbf{R}^{-}, \qquad "
         r"\boldsymbol{\omega} \leftarrow \boldsymbol{\omega} + \frac{\beta}{\Delta t}\,\mathbf{r}", "so3b")
    P.p(r"While M is hidden the angular velocity is faded out, $\boldsymbol{\omega} \leftarrow \boldsymbol{\omega}\,e^{-\Delta t/\tau_{\omega}}$, "
        r"and the pose is declared lost when the position uncertainty exceeds twice the marker side:")
    P.eq(r"3\sqrt{\mathrm{tr}\left(\mathbf{P}_{pp}\right)} > 2L", "lost")
    P.p(r"After a lost pose the filter restarts from the next measurement. The baseline is to hold the last measured pose.")

    # ================================================================== 3 Materials and methods
    P.h(1, "Materials and methods", "methods")
    P.h(2, "Materials", "materials")
    P.p(r"{tab:materials} lists everything that was used. Nothing in the set-up is laboratory equipment; this is deliberate in the assignment and "
        r"it bounds the accuracy of the ground truth to about a millimetre and a degree at best.")
    P.table("materials", "Materials, instruments and software.", [("Item", 1.1, "l"), ("Description", 3.42, "l"), ("Used for", 1.75, "l")], [
        ["Computer", "laptop, Windows 11, NVIDIA GeForce RTX 5080 Laptop GPU (16 GB)", "all processing, training"],
        ["Camera", "built-in laptop webcam, 1280 × 720 px, about 30 frames per second, horizontal field of view 74° (from the calibration)", "all recordings"],
        ["Markers", "two ArUco markers, dictionary DICT_6X6_250, ID 0 (fixed, F) and ID 23 (mobile, M), printed on A4 paper; black square 77 mm "
                    "(80 mm intended); quiet zone of 2 cells in the design", "pose targets"],
        ["Supports", "rigid black board with binder clips and adhesive tape for F (and for M in the static tests); a small box as a sliding support "
                     "for M; a white backing sheet behind M (added during T6)", "holding the markers flat and upright"],
        ["Chessboard", "9 × 6 inner corners, printed on A4 paper, square 24 mm (25 mm intended), held on a clipboard", "camera calibration"],
        ["Ruler", "30 cm, 1 mm divisions", "marker size; gaps and positions (truth); path of M"],
        ["Tape measure", "1 mm divisions", "distance from the lens to the markers"],
        ["Protractor", "1° divisions", "turn of M; angles of the laptop screen"],
        ["Obstacles", "a tin box and a light-bulb box (7 cm wide), standing between the camera and the ruler", "occlusion experiment"],
        ["Disturbances", "paper strip (cover), room lamp off (dim light), desk objects (tape measure, roll of tape, highlighter, glue stick, jar, "
                         "paper star) placed around the markers (clutter)", "robustness tests (E4)"],
        ["Software", "Python 3.12; OpenCV 5.0 [@bradski,@opencv] (ArUco, calibration, PnP); NumPy; PyTorch 2.11 with CUDA 12.8 [@paszke]; "
                     "torchvision ResNet-18 pretrained on ImageNet [@he,@deng]; Matplotlib", "implementation"],
        ["Starter code", "the lecture toolkit `mvlib.py` [@lectures], reused unchanged (Appendix C)", "geometry helpers, detector wrapper, filters"],
        ["AI assistant", "Claude (Anthropic), see the statement at the end", "code, analysis scripts, drafting"],
    ], size=8)
    P.fig(os.path.join(FIG, "fig_setup.png"), "setup",
          r"Set-up. (a) Top view with the three frames, the two measured poses and the relative pose (dashed); $s$, $n$ and $\psi$ are the "
          r"quantities measured by ruler and protractor for the ground truth: position along F’s board, distance in front of it, turn about the "
          r"vertical axis. The y axes point out of the page for the markers and into it for the camera. (b) The bench of the occlusion experiment "
          r"as the camera sees it. (c) The two printed markers, cut out of camera frames at 41 cm.")

    P.h(2, "Markers, bench and ground truth (T1)", "t1m")
    P.p(r"The markers were generated with OpenCV from DICT_6X6_250 [@garrido] with a quiet zone of two cells, printed, and fixed to a rigid board. "
        r"The printed black square was measured with the ruler; the measured value, not the nominal one, is stored in the configuration file that "
        r"every program reads. For the ground truth, F’s board defines the directions of {F}: positions of M are measured along the board ($s$), in "
        r"front of it ($n$) and in height, with the ruler, and the turn $\psi$ of M about the vertical axis with the protractor ({fig:setup}a). "
        r"Each test recording has a truth file with these values; for the motion test it holds the time schedule of the path.")

    P.h(2, "Calibration procedure", "calibm")
    P.p(r"Twenty views of the chessboard were captured with a guided tool that asks for a distance, a region of the image and a tilt for each "
        r"view, so that the corners cover the picture and the board is seen under different angles. Corners are found with the sector-based "
        r"detector of OpenCV at sub-pixel accuracy and the model of {sec:camera} is fitted [@zhang]. Two checks go beyond the overall RMS error: "
        r"the error of every view, and a held-out test in which every fifth view is left out of the fit, its board pose is estimated with the "
        r"camera fitted on the other 16, and its reprojection error is measured.")

    P.h(2, "Classical pipeline (T2, T3)", "classm")
    P.p(r"Per frame ({fig:pipeline}, top): (1) the ArUco detector finds candidate squares, decodes them and refines the four corners to sub-pixel "
        r"accuracy; (2) for F and for M, IPPE gives the two planar candidates and each is refined, {eq:refine}; (3) the ambiguity rule, "
        r"{eq:amb}, chooses between them; (4) {eq:rel} gives the relative pose. If F is not detected in a frame and the camera is known to be "
        r"still, the last pose of F is held; if M is not detected there is no output. The geometric helpers, the detector wrapper and the two "
        r"filters come from the lecture toolkit and are reused unchanged; Appendix C explains each of them.")
    P.fig(os.path.join(FIG, "fig_pipeline.png"), "pipeline",
          r"The two pipelines. Both use the same frames and the same camera model. In the learned pipeline only the pose of M comes from the "
          r"network; the pose of F is always the classical one, so that both methods are compared on the same quantity, $\lsup{F}{\mathbf{T}_{M}}$.")

    P.h(2, "Recording and visualisation (T4)", "t4m")
    P.p(r"A guided recorder writes, for every recording, the video, the time stamp of each frame, a description of the set-up and the truth "
        r"file. It shows the instruction for the current step on the screen; for the motion test it draws an outline that moves on the known "
        r"schedule and that the hand follows with M. The animation tool shows the camera image next to a 3-D view with both frames and the "
        r"trajectory of M, and the curves of $x, y, z$ and roll, pitch, yaw against time, synchronised with the video; it runs live or exports a "
        r"file. A second tool shows a table of truth, current camera reading and their difference during the set-up of a test, which is how "
        r"several of the problems of {sec:failures} were caught before recording.")

    P.h(2, "Learned pipeline (T5)", "netm")
    P.p(r"*Architecture.* Both stages are ResNet-18 networks [@he] pretrained on ImageNet [@deng]. The input is grey, so the weights of the first "
        r"convolution are summed over the colour channels; the final layer is replaced by a linear layer with the nine outputs of {sec:netmath}, "
        r"initialised to the identity rotation and a depth of 0.6 m ({fig:arch}). Stage 1 receives the undistorted frame resized to 640 × 360 px (11.17 million "
        r"parameters). Stage 2 receives the 224 × 224 px crop, {eq:crop}, and the two ray channels, whose input weights start at zero (11.18 "
        r"million parameters). A network is valid only for the camera and the marker size it was trained with.")
    P.fig(os.path.join(FIG, "fig_arch.png"), "arch",
          r"The network of both stages: ResNet-18 with a nine-number head. Top: the blocks, with the stem, the four layers of two basic blocks each, "
          r"global average pooling and the linear head. Middle: the size of the output of every block for stage 1 (the whole frame, 360 × 640 px, grey) "
          r"and for stage 2 (a crop of 224 × 224 px, grey plus the two ray channels). Bottom: a basic block. BN is batch normalisation.")
    P.p(r"*Synthetic data.* A renderer draws M with a random pose on a random background, and in 70 % of the images also F as a distractor "
        r"({fig:data}a, b). The depth is log-uniform between 0.25 and 1.0 m, the tilt is up to 60° about a random in-plane axis, the in-plane "
        r"rotation is free and the marker may be anywhere in view. The background is a smooth random texture with up to six rectangles and lines; "
        r"gain (0.6 to 1.25), offset (±20 grey levels), Gaussian blur (0.3 to 1.2 px) and noise (1 to 8 grey levels) are random, and 15 % of the "
        r"images have a patch over part of M [@tobin]. The camera is the calibrated one without lens distortion, because real frames are "
        r"undistorted before they reach the network. Three sets were rendered: 10 000 / 1 000 / 1 000 whole images (training / "
        r"validation / test) for stage 1; 30 000 / 2 000 / 2 000 crops for stage 2, cut with the centre and scale of the window perturbed by 12 % "
        r"of the marker size to imitate the errors of stage 1; and 1 000 new full-resolution images for an end-to-end test (seed 0 for the first two sets, seed 100 for the third).")
    P.p(r"*Training.* AdamW [@loshchilov] with weight decay $10^{-4}$, a one-cycle learning rate [@smith] with a peak of $5 \times 10^{-4}$ after a "
        r"warm-up of 10 % of the steps, batch 64, mixed precision. Stage 1 is augmented with random gain and offset; stage 2 in addition with "
        r"noise and blur, and its gradient norm is clipped at 5. The loss weights were chosen by comparing $w_{R} = w_{t} = 1$ with $w_{R} = 2$ "
        r"({sec:t5}).")
    P.p(r"*Real frames and fine-tuning.* Seventeen recordings of about 40 s (20 450 frames) were made in which F stands still and M is moved by "
        r"hand: close, far, to the edges of the picture, strongly tilted, rotated in its plane, right next to F. Their labels are the classical "
        r"poses $\lsup{C}{\mathbf{T}_{M}}$, kept only where the reprojection error is at most 1 px: 15 287 labelled frames. They are split by "
        r"recording, never by frame, because neighbouring frames of a video are almost identical: 13 recordings (11 562 frames) for training and "
        r"4 whole recordings (3 725 frames) for validation. Fine-tuning starts from the synthetic networks with a peak learning rate of $10^{-4}$; "
        r"every batch is half real and half synthetic, and the network of the best validation epoch is kept. It was done in three rounds as the "
        r"recordings accumulated ({tab:rounds}); each round answered a failure of the previous one ({sec:sim2real}).")
    P.fig(os.path.join(FIG, "fig_data.png"), "data",
          r"Training data. (a) Synthetic whole images for stage 1: M (ID 23) with random pose, background, blur and noise; F (ID 0) appears as "
          r"a distractor. (b) Synthetic crops for stage 2. (c) A real crop of M and the five degradations of the software hard-case test "
          r"({sec:sim2real}); the same kinds of degradation, with random strength, form the hard-case augmentation of round 3.")

    P.h(2, "Evaluation protocol (T6)", "evalm")
    P.p(r"The test set consists of 21 recordings with a truth file (Appendix A), none of which was used for training: E1, static accuracy at "
        r"five positions and two angles, 100 frames each; E2, three camera distances (about 40, 61 and 75 cm by tape measure; 41, 61 and 77 cm by the camera’s own estimate, used as labels below) with M facing the camera and turned by "
        r"40°; E3, a known motion, three recordings of 20 s in which M slides 60 mm along a ruler and back at 1 cm/s; E4, five recordings with a "
        r"disturbance (dim light, clutter, mild and strong shaking of the camera, an edge of M covered). E5 uses held-out synthetic images "
        r"with exact truth in the camera frame (the 640 × 360 px test split read by the evaluation program, and the 1 000 full-resolution images). E6 is the cost: speed on the same laptop, size of the model, training time and amount of code. "
        r"The network evaluated everywhere is the one of round 3, with two passes of the crop.")
    P.p(r"One point of protocol must be stated plainly. No test frame was ever trained on. However, the choice between the fine-tuning rounds, "
        r"the second pass of the crop and the hard-case augmentation were judged on the five position recordings of E1 (P1 to P5) and on the T4 "
        r"recording, so these recordings influenced design choices. The two angle recordings of E1 and all of E2, E3 and E4 were recorded after "
        r"the network was final; results are given for them separately where it matters.")

    P.h(2, "Occlusion experiment (bonus)", "bonusm")
    P.p(r"F stands at the right of a ruler; M slides along the ruler towards F, upright and facing the camera, and passes behind an obstacle that "
        r"stands between the camera and the ruler ({fig:setup}b). The recorder paces the hand with a countdown, beeps and an outline that moves "
        r"at the target speed. The classical pipeline provides the measurements; frames in which M is not detected are the hidden ones. The "
        r"predictor of {sec:predmath} runs with $\sigma_{a} = 0.15$ m/s², $\sigma_{m} = 4$ mm, $\alpha = 0.5$, $\beta = 0.15$, a rotation gate of "
        r"30° and $\tau_{\omega} = 0.1$ s.")
    P.p(r"Two evaluations are used because a hand does not give a known constant speed. In the first, uncovered passes are hidden in software "
        r"over the stretch that the obstacle would cover, so the measured pose is the truth in every hidden frame. In the second, with the real "
        r"obstacle, the exact truth exists only at the moment of reappearance: the predicted pose is compared with the first measured pose after "
        r"the obstacle. The hidden stretch of a pass is the longest run of frames without a measurement.")

    # ================================================================== 4 Results
    P.h(1, "Results", "results")
    P.h(2, "T1: markers and set-up", "t1")
    P.p(r"Dictionary DICT_6X6_250; fixed marker F: ID 0; mobile marker M: ID 23. The printed black squares measure 77 mm instead of the intended "
        r"80 mm, and the chessboard squares 24 mm instead of 25 mm: the printer scaled every page to 96 %. The check of the task (printed side "
        r"within 1 mm of the intended side) is therefore not met. The measured sizes are used everywhere; had the nominal 80 mm been used, every "
        r"distance in this paper would be 3.9 % too long (13 mm at the largest marker spacing).")

    P.h(2, "T2: detection", "t2")
    P.p(r"Detection with sub-pixel refinement and drawing runs at 29 to 31 frames per second, which is the frame rate of the camera. In the live "
        r"demonstration both IDs were read correctly in all four steps (both markers still; M moving; M tilted by 45°; M at 95 cm). In the "
        r"recording of the moving step (33.8 s, 1 014 frames, M between 31 and 66 cm) M was detected in 100 % and F in 94 % of the frames, and no "
        r"foreign ID was ever reported.")
    P.p(r"{fig:t2}a gives the detection limits: 10 frames per setting, a setting passes with at least 9 detections. Up to 75 cm the marker is "
        r"detected at every tilt up to 80° and never at 85°; at 1 m the limit is 75° (5 of 10 frames at 80°). The largest distance was not "
        r"reached: the bench allows 1 m, where a cell still covers 8.4 px (measured; {eq:cell} gives 8.1 px). With the 2 px rule the expected limit is about 4.2 m. The "
        r"tilt limit is the same rule in disguise: taking the cell size measured at 0° times the cosine of the tilt, 1.0 to 1.6 px per cell remain at the "
        r"settings that fail, and 1.9 to 5.2 px at the last settings that pass. The one exception is 30 cm at 85°, which fails with 2.6 px per cell: at very steep angles the "
        r"perspective and the narrow quiet zone stop the detector before the pixel count does. The distance and tilt computed from the pose "
        r"agree with tape measure and protractor ({fig:t2}b, c) within 2.9 cm (mean +0.3 cm) and, for tilts of 30° and more, within 4° (mean "
        r"−0.6°). At 0° the pose reports tilts of 2° to 8°: the tilt of a marker that faces the camera is poorly determined, the first sign of "
        r"the ambiguity of {sec:pnp}.")
    P.fig(os.path.join(FIG, "fig_t2.png"), "t2",
          r"Detection limits of the classical detector for M. (a) Detected frames out of 10 for each distance and tilt, and the cell size across "
          r"the tilt in pixels (the size measured at 0° times the cosine of the tilt). (b, c) Distance and tilt computed from the marker pose against the hand measurement, for the 28 "
          r"settings with a detection.")

    P.h(2, "Calibration", "calib")
    P.p(r"{tab:calib} and {fig:calib} summarise the calibration. The RMS reprojection error is 1.18 px. This is high for a chessboard "
        r"calibration, and the reason is visible in {fig:calib}e: rectified to the board plane, the grid lines of a view are curved, so the "
        r"printed sheet was not flat on its clipboard. The error per view ranges from 0.32 to 2.44 px (median 0.80 px), with two views at "
        r"2.44 px. The held-out test does not contradict the model: the four views left out of the fit have a mean error of 0.58 px. These four are, however, among the better views of the full fit as well (0.48 to 0.65 px), so the test says little about the views that fit badly. The "
        r"residuals grow from 0.8 px near the principal point to 1.8 px in the outer ring ({fig:calib}f), and the focal length moves between "
        r"833 and 860 px (3 %) when single views are left out. A scale uncertainty of the order of ±1.5 % in the measured distances has to be "
        r"expected from this calibration. It was accepted as it is and not repeated; {sec:t3} and {sec:discussion} come back to its effect.")
    P.table("calib", "Camera calibration (model of {sec:camera}).", [("Quantity", 1.72, "l"), ("Value", 1.38, "l"), ("Quantity", 1.82, "l"), ("Value", 1.35, "l")], [
        ["Board", "9 × 6 inner corners, 24.0 mm", "Image size", "1280 × 720 px"],
        ["Views / corners", "20 / 1 080", "Distortion model", "radial, $k_{1}$ and $k_{2}$"],
        [r"$f_{x}$, $f_{y}$", "846.59, 845.25 px", r"$c_{x}$, $c_{y}$", "641.12, 361.78 px"],
        [r"$k_{1}$, $k_{2}$", "0.0922, −0.1348", "Horizontal field of view", "74.2°"],
        ["RMS reprojection error", "1.184 px", "Per view: min / median / max", "0.32 / 0.80 / 2.44 px"],
        ["Held-out views (4): mean RMS", "0.576 px", "RMS of the 16 fitting views", "1.293 px"],
        ["Board distance in the views", "28 to 74 cm", "Board tilt in the views", "8° to 38°"],
        [r"$f_{x}$ with one view left out", "833 to 860 px (3 %)", "Residual RMS, centre / outer ring", "0.82 / 1.81 px"],
    ], size=8.5)
    P.fig(os.path.join(FIG, "fig_calib.png"), "calib",
          r"Calibration. (a) Positions of the 1 080 detected corners in the image, coloured by the error of their view. (b) Reprojection "
          r"residuals of all corners. (c) Error per view; the four held-out views are shown with the error they have when left out of the fit; "
          r"dashed: overall RMS. (d) One view with the detected corners; everything outside the board is cut away. (e) The same view rectified "
          r"to the board plane: the lines bend, the printed sheet is not flat. (f) Radial shift of the fitted model and residual RMS against "
          r"the distance from the principal point.")

    P.h(2, "T3: classical pose and its two checks", "t3")
    P.p(r"Check A compares the relative pose with ruler truth at five positions of M beside F, gaps of 50 to 250 mm between the black squares, "
        r"both boards against one straight edge, 100 frames each ({tab:checka}, {fig:t3}a). Over all 500 frames the position error has a median "
        r"of 8.8 mm (95th percentile 13.6 mm, largest 14.1 mm) and the rotation error a median of 3.96° (largest 6.3°). The jitter is 0.2 to "
        r"0.5 mm. The error is therefore a bias, not noise: averaging frames does not remove it. Its largest component is the depth along F’s "
        r"normal, +3 to +9 mm, although the two printed surfaces are at the same depth. Depth is the weakest direction of a planar pose seen "
        r"almost frontally. At the largest spacing (P5, markers near the edges of the image) the spacing itself is measured 11 mm, or 3.4 %, too "
        r"short, where lens distortion and the calibration error matter most.")
    P.p(r"Check B tests {eq:rel} directly: the markers stand still and the laptop is moved by hand ({fig:t3}b). Over 1 499 frames the camera "
        r"moved by 126 mm and turned by up to 11.0°, while the relative position stayed within 11.4 mm of its mean (RMS 2.7 mm) and the relative "
        r"rotation within 3.8° of its chordal mean [@moakher] (RMS 0.94°). The distance between the centres was measured as 276.9 mm (standard deviation 0.4 mm) against 277 mm "
        r"by ruler (edges 200 mm apart plus one side).")
    P.table("checka", "Check A: classical relative pose against ruler truth at five positions (100 frames each).",
            [("Position", 0.7, "l"), ("Gap [mm]", 0.7, "c"), ("Truth x [mm]", 0.85, "c"), ("Camera − truth, x / y / z [mm]", 1.6, "c"),
             (r"$e_{t}$ median (max) [mm]", 1.0, "c"), (r"$e_{R}$ median [deg]", 0.72, "c"), ("Jitter [mm]", 0.7, "c")], [
        ["P4", "50", "−127", "−1.7 / +1.5 / +3.0", "3.8 (4.1)", "1.95", "0.18"],
        ["P3", "100", "−177", "−2.0 / +5.4 / +9.0", "10.7 (11.7)", "3.96", "0.41"],
        ["P2", "150", "−227", "+0.5 / +1.2 / +5.9", "6.0 (7.5)", "6.02", "0.46"],
        ["P1", "200", "−277", "−3.5 / +1.7 / +7.9", "8.8 (9.4)", "5.25", "0.35"],
        ["P5", "250", "−327", "+11.1 / +1.9 / +7.3", "13.5 (14.1)", "3.40", "0.49"],
    ], size=8.5)
    P.fig(os.path.join(FIG, "fig_t3.png"), "t3",
          r"The two checks of the classical pose. (a) Check A: difference between camera and ruler in the three directions of {F}. (b) Check B: "
          r"displacement of the camera (norm of the change of $\lsup{C}{\mathbf{t}_{F}}$) and deviation of the relative position "
          r"$\lsup{F}{\mathbf{t}_{M}}$ from its mean while the laptop is moved by hand.")

    P.h(2, "T4: recording and animation", "t4")
    P.p(r"The demonstration recording lasts 30 s (900 frames): M is moved by hand to the sides, closer and farther, turned, lifted and moved in a "
        r"circle. {fig:t4} shows what the animation displays, for both methods on the same frames. The classical pipeline gives a pose in 88 % "
        r"of the frames; the gaps are frames without a detection of M, where it moves fast or is partly outside the picture (the cause was not logged). Where both give a pose, the "
        r"position from the network differs from the classical one by 4.3 mm in the median (95th percentile 8.4 mm) and the rotation by 2.2°. "
        r"This is the check of T5: the network runs in the T3 and T4 pipeline on a sequence it was not trained on. The exported animations (the "
        r"camera image beside the 3-D view and the curves) are part of the repository in a version without the camera image.")

    P.h(2, "T5: training on synthetic images", "t5")
    P.p(r"{tab:synth} and {fig:synth} give the four training runs on synthetic data (all ten saved runs are listed in {tab:runs}, Appendix B). Run A (15 epochs) was under-trained: its rotation loss was "
        r"still falling and its validation error was 23.5°. Run B, the same network for 40 epochs, reached 7.3° and 13.5 mm. Run C doubled the "
        r"rotation weight; this made both errors slightly worse (8.9°, 16.7 mm), so $w_{R} = w_{t} = 1$ was kept. With one seed per run, "
        r"differences of 1° to 2° may partly be run-to-run variation. The final training rotation loss of run B, 0.0145, is 4.9° by "
        r"{eq:chord}, below its validation error of 7.3°: the network fits its training images better than new ones, and the 95th percentile of "
        r"33° marks the hard cases (small, far, steep or covered markers).")
    P.fig(os.path.join(FIG, "fig_t4.png"), "t4",
          r"The T4 recording as shown by the animation tool. (a) Path of M in the frame of F (blue: classical, orange: network) with the axes "
          r"of F. (b) Translation and roll, pitch, yaw of M in F against time. The network curves are drawn only where the classical pipeline "
          r"also has a pose; gaps are frames without a detection of M.")
    P.p(r"The crop stage (run D) changes the picture: on its validation crops the rotation error is 1.95° and the position error 8.8 mm. "
        r"End to end, on 1 000 new full-resolution synthetic images, the two-stage network has a rotation error of 1.7° against 7.6° for stage 1 "
        r"alone, and 83 % usable poses against 51 % ({tab:e2e}). The gain is largest for far markers, which stage 1 sees with about 65 px and "
        r"stage 2 with about 112 px. The position error still grows with distance, because depth comes from the apparent size.")
    P.table("synth", "Training runs on synthetic data. Validation errors are medians with the 95th percentile in brackets, on the synthetic "
                     "validation set of the run (1 000 whole images or 2 000 crops).",
            [("Run", 1.55, "l"), ("Epochs", 0.55, "c"), (r"$w_{R}$ / $w_{t}$", 0.6, "c"), ("Time [min]", 0.62, "c"), ("Final training loss, rotation / translation", 1.25, "c"),
             (r"Validation $e_{R}$ [deg]", 0.85, "c"), (r"Validation $e_{t}$ [mm]", 0.85, "c")], [
        ["A: whole image", "15", "1 / 1", "7.5", "0.409 / 0.109", "23.5 (61)", "26.4 (134)"],
        ["B: whole image (chosen as stage 1)", "40", "1 / 1", "20.7", "0.0145 / 0.0487", "7.3 (33)", "13.5 (75)"],
        ["C: whole image", "40", "2 / 1", "20.3", "0.0160 / 0.0628", "8.9 (36)", "16.7 (95)"],
        ["D: crop (stage 2)", "80", "1 / 1", "27.1", "0.0010 / 0.0171", "1.95 (6.3)", "8.8 (31)"],
    ], size=8.5)
    P.table("e2e", "End-to-end test on 1 000 new full-resolution synthetic images (exact truth). Usable: $e_{t} < 20$ mm and $e_{R} < 10°$. "
                   "The last three columns give the medians $e_{t}$ [mm] / $e_{R}$ [deg] by distance.",
            [("Network", 1.25, "l"), (r"$e_{R}$ [deg]", 0.82, "c"), (r"$e_{t}$ [mm]", 0.82, "c"), ("Usable", 0.68, "c"), ("0.25 to 0.4 m", 0.9, "c"),
             ("0.4 to 0.6 m", 0.9, "c"), ("0.6 to 1.0 m", 0.9, "c")], [
        ["Stage 1 alone (run B)", "7.6 (27.4)", "14.3 (87)", "50.5 %", "7.8 / 5.4", "12.8 / 6.9", "27.1 / 10.8"],
        ["Two-stage (B and D)", "1.7 (6.9)", "9.7 (41)", "83.2 %", "5.1 / 1.5", "9.3 / 1.6", "16.7 / 2.0"],
    ], size=8.5)
    P.fig(os.path.join(FIG, "fig_synth_train.png"), "synth",
          r"Training on synthetic data: the two terms of the training loss, {eq:loss}, and the validation errors for every epoch of the four runs "
          r"of {tab:synth}. In (c) and (d) solid lines are medians and dotted lines 95th percentiles.")

    P.h(2, "From synthetic to real frames", "sim2real")
    P.p(r"The synthetic networks do not work on real frames. On the E1 position recordings the two-stage network trained on synthetic images "
        r"alone has a median position error of 26.5 mm with a 95th percentile of 107 mm and a jitter as large as its error; on the T4 recording "
        r"its rotation differs from the classical one by 142° in the median ({tab:steps}). The rendered images lack what the real ones have: a "
        r"hand, a room, uneven light, the print and the paper. Three rounds of fine-tuning on real frames closed the gap ({tab:rounds}, "
        r"{fig:finetune}, {fig:rounds}).")
    P.p(r"Round 1 (3 164 real frames) brought the median to 10.0 mm, but left a tail: a 95th percentile of 124 mm, almost all from one "
        r"recording, P4, in which M stands right next to F (34 mm median for the two-stage network, 103 mm for stage 1 alone). There the estimate of stage 1 jumped between positions on some frames; M and F stand side by side in that recording. The "
        r"training recordings had almost never shown M that close to F.")
    P.p(r"Round 2 added five recordings with exactly this case and with F at other distances, and three changes. First, the camera-rotation "
        r"augmentation, {eq:homog}, for stage 1 (pitch and yaw up to 8°, roll up to 10°): the validation recording of this round was the only "
        r"one with the laptop screen pitched, and without the augmentation stage 1 placed M about 82 px too low on it, as if it had learned "
        r"where things are in the room rather than in the image (13.9° and 92 mm; with the augmentation 6.6° and 29 mm, {fig:rounds}b). "
        r"Second, larger window offsets for the real crops (25 % of the marker size instead of 12 %). Third, the second pass of the crop at run "
        r"time, which alone took the jitter from 8.4 to 2.1 mm. The tail disappeared (95th percentile 17.8 mm) and P4 fell to 2.8 mm.")
    P.p(r"Round 3 added seven recordings with the camera pitched and turned at 44 to 71 cm, used four whole recordings for validation, and "
        r"introduced the hard-case augmentation: in 40 % of the training images part of the marker is covered, or the image is blurred by "
        r"motion, reduced in resolution, or dimmed, while the label stays valid. On E1 P1 to P5 the result is 6.7 mm and 3.6° with a jitter of "
        r"1.3 mm, against 8.8 mm, 4.0° and 0.4 mm for the classical pipeline. The network is here slightly closer to the ruler than the method "
        r"that labelled its training data, presumably because it averages over many slightly different classical depth errors; these are, however, "
        r"the recordings that informed the design ({sec:evalm}), and {sec:t6} shows that the advantage does not carry over to the recordings made "
        r"afterwards.")
    P.p(r"The effect of the hard-case augmentation was measured on 250 frames of E1 degraded in software ({fig:data}c, {fig:rounds}c, "
        r"Appendix B). With 30 % of M covered the classical detector gives nothing, round 2 gives a usable pose in 51 % of the frames and round "
        r"3 in 91 %; with a motion blur of 31 px the figures are 0 %, 14 % and 100 %. Frames at one tenth of the brightness defeat all three. "
        r"Two caveats apply: the degradations are simulated and of the same kind as the augmentation, and the reference there is the classical "
        r"pose of the clean frame, not the ruler.")
    P.p(r"{fig:finetune} shows why stage 2 is needed. The validation position error of stage 1 jumps from epoch to epoch between 10 and 95 mm "
        r"(14 to 29 mm at the epochs that were kept), while stage 2 settles at 6 to 7 mm. On E1, stage 1 of round 3 alone still fails on P4 (71 mm); the crop stage "
        r"and its second pass recover it.")
    P.table("rounds", "The three rounds of fine-tuning. Validation errors ($e_{R}$ in degrees / $e_{t}$ in mm) are medians on whole recordings never trained on, at the epoch that "
                      "was kept; the validation set differs from round to round, so the rounds are compared on the test recordings ({tab:steps}).",
            [("Round", 0.45, "c"), ("Real frames, training / validation", 0.9, "c"), ("Validation recordings", 0.95, "l"), ("New in this round", 1.55, "l"),
             (r"Stage 1: epochs (kept): $e_{R}$ / $e_{t}$", 1.0, "c"), (r"Stage 2: epochs (kept): $e_{R}$ / $e_{t}$", 1.0, "c"), ("Time [min]", 0.42, "c")], [
        ["1", "3 164 / 589", "1 (camera upright)", "real frames, half of every batch", "15 (15): 8.2 / 14.3", "15 (15): 8.1 / 6.8", "5.5"],
        ["2", "7 358 / 930", "1 (screen pitched)", "M next to F; camera-rotation augmentation; crop offsets 25 %; second pass; best epoch kept",
         "25 (8): 6.6 / 29.4", "25 (21): 3.8 / 6.5", "32.9"],
        ["3", "11 562 / 3 725", "4 (upright, pitched, pitched and turned, turned)", "camera pitched and turned; hard-case augmentation in 40 % of the images",
         "25 (22): 5.8 / 15.4", "50 (48): 3.7 / 6.4", "48.9"],
    ], size=8)
    P.table("steps", "Real test frames after each step of the work: E1 positions P1 to P5 (500 frames, ruler truth) and the T4 recording "
                     "(difference to the classical pose). Median (95th percentile).",
            [("Method", 2.05, "l"), (r"E1 $e_{t}$ [mm]", 0.92, "c"), (r"E1 $e_{R}$ [deg]", 0.92, "c"), ("Jitter [mm]", 0.6, "c"), (r"P4 $e_{t}$ [mm]", 0.6, "c"),
             ("T4 difference, median", 1.18, "c")], [
        ["Classical", "8.8 (13.6)", "3.96 (6.1)", "0.38", "3.8", "–"],
        ["Stage 1, synthetic only", "59.8 (190.9)", "11.88 (32.2)", "21.3", "–", "157 mm, 25.0°"],
        ["Two-stage, synthetic only", "26.5 (106.5)", "9.46 (37.6)", "26.5", "–", "196 mm, 142.5°"],
        ["Two-stage, round 1, one pass", "10.0 (124.3)", "5.69 (36.7)", "16.3", "34.0", "7.2 mm, 2.7°"],
        ["Two-stage, round 2, one pass", "12.0 (19.5)", "4.87 (7.5)", "8.4", "8.1", "4.3 mm, 2.1°"],
        ["Two-stage, round 2, two passes", "9.2 (17.8)", "3.88 (6.6)", "2.1", "2.8", "4.3 mm, 2.1°"],
        ["Stage 1 alone, round 3", "12.6 (88.0)", "6.90 (9.9)", "7.0", "70.8", "7.0 mm, 7.0°"],
        ["Two-stage, round 3, one pass", "8.2 (12.8)", "3.74 (6.1)", "2.0", "3.5", "3.7 mm, 2.6°"],
        ["**Two-stage, round 3, two passes (final)**", "**6.7 (12.5)**", "**3.56 (6.3)**", "**1.26**", "**3.1**", "**4.3 mm, 2.2°**"],
    ], size=8.5, rules=(0, 2))
    P.fig(os.path.join(FIG, "fig_finetune.png"), "finetune",
          r"Fine-tuning on real frames: training loss and validation errors (medians) per epoch for the three rounds and both stages. Stars mark "
          r"the epoch that was kept. The training loss of rounds 2 and 3 is higher than that of round 1 because of the augmentations; the "
          r"validation sets differ between rounds (round 2: one recording with the screen pitched, the hardest for stage 1).")
    P.fig(os.path.join(FIG, "fig_rounds.png"), "rounds",
          r"What each change bought. (a) The two-stage network on the real E1 positions after training on synthetic data only and after each "
          r"round of fine-tuning, with the classical pipeline for comparison (logarithmic scale). (b) Stage 1 of round 2 on the validation "
          r"recording with the pitched screen, without and with the camera-rotation augmentation. (c) Usable poses on 250 E1 frames degraded in "
          r"software.")

    P.h(2, "T6: comparison on the test recordings", "t6")
    P.h(3, "E1, static accuracy", "e1")
    e1 = ["E1_p4", "E1_p3", "E1_p2", "E1_p1", "E1_p5", "E1_turn20", "E1_turn40"]
    e2 = ["E2_d40_t0", "E2_d40_t40", "E2_d60_t0", "E2_d60_t40", "E2_d75_t0", "E2_d75_t40"]
    lab = {"E1_p4": ("P4, gap 50 mm", "34", "0"), "E1_p3": ("P3, gap 100 mm", "≈35", "0"), "E1_p2": ("P2, gap 150 mm", "≈35", "0"),
           "E1_p1": ("P1, gap 200 mm", "≈35", "0"), "E1_p5": ("P5, gap 250 mm", "37", "0"), "E1_turn20": ("turned 20°", "61", "20"),
           "E1_turn40": ("turned 40°", "61", "40"), "E2_d40_t0": ("E2, 41 cm", "41", "0"), "E2_d40_t40": ("E2, 41 cm", "41", "40"),
           "E2_d60_t0": ("E2, 61 cm", "61", "0"), "E2_d60_t40": ("E2, 61 cm", "61", "40"), "E2_d75_t0": ("E2, 77 cm", "77", "0"),
           "E2_d75_t40": ("E2, 77 cm", "77", "40")}
    trows = []
    for n in e1 + e2:
        c, w = stats(n, 0), stats(n, 1)
        trows.append([("E1, " if n.startswith("E1") else "") + lab[n][0], lab[n][1], lab[n][2], f1(c["et"]), f1(c["er"], 2), f1(c["bias"]), f1(c["jit"], 2),
                      f1(w["et"]), f1(w["er"], 2), f1(w["bias"]), f1(w["jit"], 2)])
    P.p(r"{tab:static} and {fig:e12}a give the static results per recording. Pooled over the seven E1 recordings (700 frames) the classical "
        r"pipeline has a position error of 7.8 mm (13.6 mm) and a rotation error of 3.84° (6.04°); the network 8.1 mm (13.0 mm) and 4.04° "
        r"(6.48°). The mean jitter is 0.55 mm against 1.29 mm. The two halves of E1 behave differently. On the five positions that informed "
        r"the design the network is closer to the ruler than the classical pipeline at every position. On the two angle recordings made "
        r"afterwards it is not: 11.8 mm (13.8 mm) and 5.46° against 6.2 mm (8.4 mm) and 2.80° for the classical pipeline. On all eight static "
        r"recordings made after the network was final (E1 angles and E2) the figures are 11.5 mm (25.6 mm) and 4.47° for the network against "
        r"8.0 mm (9.0 mm) and 3.97°. The honest summary is that the network is level with the classical pipeline in the configuration it was "
        r"developed on and 3 to 4 mm worse elsewhere.")
    P.table("static", "Static test recordings, 100 frames each: median errors, bias and jitter, {eq:bias}, of both methods. P1 to P5 informed "
                      "design choices; all others were recorded after the network was final. The camera distance of P1 to P3 was not measured (between 34 and 37 cm).",
            [("Recording", 1.07, "l"), ("Camera [cm]", 0.52, "c"), ("Turn of M [deg]", 0.48, "c"),
             (r"Classical: $e_{t}$ [mm]", 0.55, "c"), (r"$e_{R}$ [deg]", 0.5, "c"), ("bias [mm]", 0.5, "c"), ("jitter [mm]", 0.5, "c"),
             (r"Network: $e_{t}$ [mm]", 0.55, "c"), (r"$e_{R}$ [deg]", 0.5, "c"), ("bias [mm]", 0.5, "c"), ("jitter [mm]", 0.5, "c")],
            trows, size=8, rules=(4, 6))
    P.fig(os.path.join(FIG, "fig_e1_e2.png"), "e12",
          r"Static accuracy. (a) E1: position error at the five positions and the two angles (bars: median; whiskers: 5th to 95th percentile). "
          r"(b, c) E2: position and rotation error against the camera distance, with M facing the camera and turned by 40°.")

    P.h(3, "E2, distance and tilt", "e2")
    P.p(r"The classical error does not depend on distance or turn in the tested range: 7.5 to 8.7 mm and 3.6° to 4.4° at every setting "
        r"({fig:e12}b, c). A constant error of this size at every distance points to the truth and to the calibration rather than to the "
        r"method (the depth bias of check A is of the same size). The network matches it when M faces the camera at 41 and 61 cm (7.9 and "
        r"6.4 mm) and falls behind when M is far or turned, most when it is both: 25.1 mm and 10.1° at 77 cm and 40°, where no frame is "
        r"usable. The error is almost entirely depth (−24.6 mm along F’s normal), the direction a regressor must infer from the apparent size "
        r"of a marker that covers about 85 px. Neither method stopped giving poses in this range. There is no recording at 60°, and 77 cm was "
        r"the largest distance the room allowed: the points where each method fails were not reached in E2, and we rely on {sec:t2} for the "
        r"classical detector.")

    P.h(3, "E3, known motion", "e3")
    m3 = []
    for n in ("E3_motion_01", "E3_motion_02", "E3_motion_03"):
        for k, name in ((0, "classical"), (1, "network")):
            s = stats(n, k)
            m3.append([n.replace("E3_motion_0", "E3, recording ") if k == 0 else "", name, f1(s["rmse"]), f1(s["et"]), f1(s["etmax"]), f1(s["er"], 2), f1(s["ermax"]),
                       f1(s["use"])])
    P.p(r"In the three motion recordings both methods follow the path with median errors of 5.5 to 7.4 mm ({tab:e3}, {fig:e3}); pooled, 6.6 mm (12.4 mm) for the "
        r"classical pipeline and 6.3 mm (13.7 mm) for the network, with rotation errors of 5.82° and 5.21°. Two effects that are not errors of either method are "
        r"visible. First, the truth is the schedule of the on-screen outline, and the hand does not follow it exactly: on the way back M is behind the outline "
        r"by 0.6 to 1.0 s in the median and by up to 2.0 s, which is 20 mm at 1 cm/s, in the first recording ({fig:e3}a). The error peak of about 20 mm in "
        r"{fig:e3}b is this lag; both methods show it at the same moment. While M rests the median errors are 5.3 to 6.6 mm, while it moves 5.7 to 8.0 mm, so "
        r"the E3 figures are upper bounds for both methods. An earlier estimate in the project notes, a lag of at most 0.5 s, was too optimistic. Second, the "
        r"classical pipeline has isolated outliers of 59 mm, in 1, 4 and 6 frames of the three recordings. In these frames the detector returns a quadrilateral "
        r"about 9 % smaller than in the neighbouring frames, with a reprojection error of 0.1 to 0.7 px (0.9 px is the median of all frames), which places M 49 "
        r"to 58 mm farther from the camera; the rotation is not affected, so this is not the planar ambiguity, and the rule, {eq:amb}, does not catch it. We did "
        r"not establish the cause. A gate on the innovation, {eq:gate}, would reject these frames. The network has no such outliers (largest error 14 to 20 mm).")
    P.table("e3", "E3: errors on the known motion (600 or 601 frames per recording).",
            [("Recording", 1.2, "l"), ("Method", 0.85, "l"), ("RMSE [mm]", 0.7, "c"), (r"$e_{t}$ median [mm]", 0.8, "c"), (r"$e_{t}$ max [mm]", 0.7, "c"),
             (r"$e_{R}$ median [deg]", 0.8, "c"), (r"$e_{R}$ max [deg]", 0.7, "c"), ("Usable [%]", 0.52, "c")], m3, size=8.5, rules=(1, 3))
    P.fig(os.path.join(FIG, "fig_e3.png"), "e3",
          r"E3, first recording. (a) Position of M along the ruler: the schedule of the guide outline (truth) and the two estimates; the hand "
          r"follows the outline late on the way back. (b) Position error; the single-frame peak of the classical pipeline is a frame in which "
          r"M’s outline is read about 9 % too small. (c) Rotation error.")

    P.h(3, "E4, robustness", "e4")
    cond = {"E4_dim": "dim light", "E4_clutter": "clutter around the markers", "E4_blur": "camera shaken mildly", "E4_blur_2": "camera shaken strongly",
            "E4_cover": "left edge of M covered (20 %)"}
    r4 = []
    for n, c_ in cond.items():
        c, w = stats(n, 0), stats(n, 1)
        r4.append([c_, f1(c["out"], 0), f1(c["use"], 0), f1(c["et"]), f1(c["er"], 2), f1(w["out"], 0), f1(w["use"], 0), f1(w["et"]), f1(w["er"], 2)])
    P.p(r"{tab:e4} and {fig:e4} give the output rate, the share of usable frames and the errors under five disturbances, with M beside F at "
        r"61 cm. Dim light, clutter and mild shaking do not affect the classical pipeline (3 to 4 mm). Strong shaking of the camera blurs "
        r"about a third of the frames; the detector still finds the markers in all of them and 93 % of the frames are usable. With a paper "
        r"strip over the left edge of M (about 20 % of the black square) the classical pipeline gives no pose at all: the detector needs the "
        r"complete border.")
    P.p(r"The network gives a pose in every frame of every recording. This must be read with care: it has no detection output, so its output "
        r"rate is 100 % by construction, and only the usable rate says something. That rate is 98 % to 99 % under dim light, clutter and mild "
        r"shaking and 81 % under strong shaking, with position errors of 12 to 17 mm where the classical pipeline has 3 to 4 mm. With the "
        r"edge covered the network places the outline correctly in the image ({fig:e4}a) and its rotation error is 2.7°, the lowest of the "
        r"five recordings, but its position is 32.6 mm off, 32 mm of it in depth, so no frame is within the 20 mm bound. Averaged over the "
        r"five recordings the output rates are 80 % and 100 % and the usable rates 79 % and 76 %. A covered marker is thus the one case in "
        r"which the network gives information the classical pipeline does not, and even there it is coarse.")
    P.table("e4", "E4: output rate, usable frames and median errors under five disturbances (100 to 150 frames each). The output rate of "
                  "the network is 100 % by construction.",
            [("Disturbance", 1.75, "l"), ("Classical: output [%]", 0.6, "c"), ("usable [%]", 0.55, "c"), (r"$e_{t}$ [mm]", 0.55, "c"), (r"$e_{R}$ [deg]", 0.55, "c"),
             ("Network: output [%]", 0.6, "c"), ("usable [%]", 0.55, "c"), (r"$e_{t}$ [mm]", 0.55, "c"), (r"$e_{R}$ [deg]", 0.55, "c")], r4, size=8.5)
    P.fig(os.path.join(FIG, "fig_e4.png"), "e4",
          r"E4. (a) M under three disturbances with the outline given by each method; with the edge covered there is no classical outline. "
          r"(b) Share of frames with a pose (light bars) and with a usable pose (dark bars). (c) Median position error.")

    P.h(3, "E5, held-out synthetic images", "e5")
    P.p(r"{tab:e5} gives both methods on held-out synthetic images with exact truth in the camera frame. On the test split of the stage-1 set, which has half "
        r"the camera resolution, the classical pipeline gives a pose in 87 % of the images, with 3.8 mm (25.2 mm) and 0.33° (3.91°), and the network 11.8 mm "
        r"(81.2 mm) and 1.96° (6.80°). This is the figure of the evaluation program and of {tab:summary}; it understates both methods, and the network more, "
        r"because its crop is then cut from the half-size image. On the 1 000 images of full camera resolution the classical pipeline gives a pose in 90 % of "
        r"the images, with 1.6 mm (11.8 mm) and 0.11° (1.33°), and the network 6.8 mm (34.3 mm) and 1.85° (5.78°). The images without a classical pose are "
        r"mostly those with a patch over M (15 % of each set), besides very small and very steep markers. The comparison with the bench is instructive. With an "
        r"exact camera model and exact truth the classical code reaches 1.6 mm and 0.1° over 0.25 to 1 m and up to 60° of tilt; on the bench it has 8 mm and 4°. "
        r"The difference must come from what the rendered images do not contain: the calibration error, the print and its mount, and the hand-measured truth. "
        r"Which of them dominates, these data do not say. The network is about as accurate on real frames near its training conditions (8 mm) as on the rendered "
        r"images (7 mm).")
    e5 = json.load(open(os.path.join(HERE, "data", "e5_full.json")))
    r5 = []
    for key, name in (("synth_v1", "640 × 360 px (test split of the stage-1 set)"), ("synth_full_test", "1280 × 720 px (full camera resolution)")):
        for m in ("classical", "network"):
            d = e5[key][m]
            r5.append([name if m == "classical" else "", m, f"{d['output_pct']:.0f}", f"{d['et_median']:.1f} ({d['et_p95']:.1f})", f"{d['er_median']:.2f} ({d['er_p95']:.2f})",
                       f"{d['usable_pct']:.0f}"])
    P.table("e5", "E5: held-out synthetic images with exact truth, 1 000 images per set; the network is the final one (round 3, two passes).",
            [("Image set", 2.3, "l"), ("Method", 0.8, "l"), ("Output [%]", 0.65, "c"), (r"$e_{t}$ [mm]", 0.95, "c"), (r"$e_{R}$ [deg]", 0.95, "c"), ("Usable [%]", 0.62, "c")],
            r5, size=8.5, rules=(1,))

    P.h(3, "E6, cost", "e6")
    P.p(r"{tab:cost} compares the cost. The classical pipeline needs 6.8 ms per frame on the processor (147 frames per second in the evaluation run; "
        r"5.4 to 6.8 ms in other runs). The learned pipeline needs 39.8 ms (25 frames per second) with the graphics processor, of which 32 ms are "
        r"the networks (stage 1 once, stage 2 twice because of the second pass) and the rest the classical pose of F; on the processor alone the networks take "
        r"44 ms. The network also needs what the classical pipeline does not: 40 000 rendered images, 17 labelled recordings, about two hours "
        r"of training in total and two files of 45 MB that are valid for one camera and one marker size.")
    P.table("cost", "E6: cost of the two pipelines on the same laptop.", [("Quantity", 2.6, "l"), ("Classical", 1.6, "l"), ("Learned (two-stage, two passes)", 2.07, "l")], [
        ["Time per frame, whole pipeline", "6.8 ms (processor)", "39.8 ms (graphics processor)"],
        ["Frames per second", "147", "25"],
        ["Networks alone, per frame", "–", "32.2 ms (graphics processor), 44.3 ms (processor)"],
        ["Parameters", "none learned", "11.17 + 11.18 million"],
        ["Model files", "–", "2 × 44.8 MB"],
        ["Training data", "–", "40 000 synthetic images and 11 562 labelled real frames"],
        ["Training time", "–", "47.8 min on synthetic data; 5.5 + 32.9 + 48.9 min for the three rounds"],
        ["Code, core of the method (lines)", "266", "770"],
        ["Needs per camera and marker size", "one calibration", "one calibration, new data, new training"],
    ], size=8.5)

    P.h(3, "Summary and worst frames", "summary")
    P.p(r"{tab:summary} is the comparison table of the assignment and {fig:cdf} shows the distributions behind it. For a marker that is "
        r"completely visible the classical pipeline is about as accurate as the network in E1 and E3 and clearly more accurate in E2, about twice as steady and six times "
        r"faster. {fig:cdf}c makes the nature of the errors explicit: for both methods the bias is at least twice the jitter in every static "
        r"recording except the strongly shaken camera, so neither would gain much from averaging frames; the network has more of both "
        r"(medians of 11.6 mm bias and 2.1 mm jitter against 7.7 mm and 1.0 mm).")
    P.p(r"The worst frames of the two methods are of different kinds. For the classical pipeline they are the single-frame outliers of E3 "
        r"(59 mm), blurred frames of the strongly shaken camera (up to 32 mm) and, among undisturbed frames, the position with the markers at the edges of the image (P5, 14 mm). For the network they are the "
        r"covered marker (up to 41 mm) and the far, turned marker (about 30 mm and 12°). The classical failures are rare and large or absent "
        r"altogether (no pose); those of the network are systematic and tied to depth.")
    P.table("summary", "Comparison of the two pipelines on the test set. Median (95th percentile). E1, E2: static; E3: motion; E4: mean over "
                       "the five robustness recordings; E5: held-out synthetic images of 640 × 360 px; E6: speed.",
            [("Method", 0.62, "l"), (r"E1 $e_{t}$ [mm]", 0.66, "c"), (r"E1 $e_{R}$ [deg]", 0.7, "c"), ("E1 jitter [mm]", 0.45, "c"), (r"E2 $e_{t}$ [mm]", 0.7, "c"),
             (r"E3 $e_{t}$ [mm]", 0.66, "c"), (r"E3 $e_{R}$ [deg]", 0.7, "c"), ("E4 output / usable [%]", 0.62, "c"), (r"E5 $e_{t}$ [mm] / output [%]", 0.74, "c"),
             ("E6 [fps]", 0.42, "c")], [
        ["Classical", "7.8 (13.6)", "3.84 (6.04)", "0.55", "8.1 (9.1)", "6.6 (12.4)", "5.82 (9.54)", "80 / 79", "3.8 / 87", "147"],
        ["Network", "8.1 (13.0)", "4.04 (6.48)", "1.29", "11.0 (26.1)", "6.3 (13.7)", "5.21 (7.11)", "100 / 76", "11.8 / 100", "25"],
    ], size=8)
    P.fig(os.path.join(FIG, "fig_cdf.png"), "cdf",
          r"Error distributions on the test set. (a, b) Cumulative distribution of the position and rotation error over all frames of each "
          r"experiment (frames without a pose are left out: the classical curve of E4 has no frame of the covered marker). (c) Bias against "
          r"jitter for every static recording of E1, E2 and E4; points above the dashed line have more bias than jitter.")

    P.h(2, "Bonus: prediction through an occlusion", "bonus")
    P.p(r"*Software-hidden passes* ({tab:virtual}, {fig:bonus}b). With exact truth in every hidden frame, the predictor has 7.6 mm of error "
        r"after 0.25 s, 15 mm after 0.5 s and 25 mm after 1 s at about 7 cm/s, against 22, 42 and 82 mm for holding the last pose. At about "
        r"17 cm/s the figures are 23, 56 and 94 mm against 52, 91 and 165 mm. The time until the error passes 20 mm is 0.85 s against 0.26 s "
        r"at the slow speed and 0.22 s against 0.06 s at the fast one. With $\sigma_{a} = 0.15$ m/s² the true error stays inside the 3σ bound "
        r"of the filter in 87 % of the hidden frames (smaller values are over-confident: 70 % to 82 %), and the pose is declared lost after "
        r"about 1.4 s. At re-acquisition the output jumps by the prediction error, 39 mm and 83 mm in the median: the gate prevents a wrong "
        r"lock, it does not remove the jump.")
    P.p(r"*Real obstacle* ({tab:real}, {fig:bonus}a, c). In 49 passes behind a box the predictor is closer than the held pose at the moment of "
        r"reappearance in 44. At 8 to 12 cm/s, hidden for 1.7 s, its error is 50 mm against 174 mm; at 13 to 23 cm/s, hidden for 1.2 s, 96 mm "
        r"against 182 mm; in the 33 passes hidden for 0.5 to 2 s, the range the task asks for, 64 mm against 178 mm. The slowest passes (4 to "
        r"6 cm/s) are hidden for about 3 s, longer than the task asks; the pose is declared lost before M reappears in all of them, and the "
        r"error is 56 mm against 158 mm. The limit is the hand, not the filter: within a pass the speed changes by tens of percent, and M is "
        r"often still accelerating when it disappears, so the velocity the filter has learned is too low and the prediction lags.")
    P.p(r"The rotation is not predicted better than it is held: 20° to 31° against 18° to 22° at reappearance. The first frames after the "
        r"obstacle are noisy, a marker that faces the camera is ambiguous in tilt, and these recordings were made before the white-margin "
        r"problem of {sec:failures} was found. On the fast uncovered passes, extrapolating the angular velocity while hidden gave 42° of rotation error against about 11° with the "
        r"fade-out, which is why it is faded out. Two choices were made on these same recordings and are disclosed: the fade-out time constant was chosen on the uncovered "
        r"passes, and an option that keeps only the velocity along the line of motion was tested (better in the two slow sets, 55 to 40 mm and 111 to 87 mm; worse "
        r"at the other speeds) and not adopted.")
    P.table("virtual", "Occlusion in software on uncovered passes: position error of the predictor / of holding the last pose, medians over "
                       "the passes, against the time hidden. Horizon: hidden time at which the error first passes the bound.",
            [("Speed", 0.6, "l"), ("Passes", 0.5, "c"), ("at 0.25 s [mm]", 0.75, "c"), ("at 0.5 s [mm]", 0.75, "c"), ("at 1 s [mm]", 0.75, "c"),
             ("Horizon 20 mm [s]", 0.82, "c"), ("Horizon 40 mm [s]", 0.9, "c"), ("Jump at return [mm]", 0.6, "c"), ("Lost after [s]", 0.6, "c")], [
        ["7 cm/s", "4", "7.6 / 22", "15 / 42", "25 / 82", "0.85 / 0.26", "not reached / 0.49", "39", "1.35"],
        ["17 cm/s", "7", "23 / 52", "56 / 91", "94 / 165", "0.22 / 0.06", "0.45 / 0.22", "83", "1.05"],
    ], size=8.5,
            note="At 17 cm/s only 3 of the 7 passes are still hidden at 1 s, and the pose was declared lost in 2 of the 7 (in all 4 at 7 cm/s).")
    P.table("real", "Real obstacle, 49 passes: errors at the moment M is measured again (exact truth), predictor / hold, medians per speed "
                    "group. The errors at 0.5 s and 1 s assume a straight path at constant speed between the last and the next measurement.",
            [("Speed while hidden", 0.85, "l"), ("Passes", 0.45, "c"), ("Hidden [s]", 0.8, "c"), (r"$e_{t}$ at return [mm]", 0.75, "c"), ("Predictor better", 0.62, "c"),
             (r"$e_{t}$ at 0.5 s [mm]", 0.7, "c"), (r"$e_{t}$ at 1 s [mm]", 0.7, "c"), (r"$e_{R}$ at return [deg]", 0.7, "c"), ("Lost before return", 0.7, "c")], [
        ["4 to 6 cm/s", "15", "3.1 (2.7 to 4.3)", "56 / 158", "12 of 15", "16 / 25", "28 / 50", "31 / 22", "15 of 15"],
        ["8 to 12 cm/s", "14", "1.7 (1.3 to 2.1)", "50 / 174", "14 of 14", "18 / 60", "32 / 113", "23 / 20", "11 of 14"],
        ["13 to 23 cm/s", "20", "1.2 (0.8 to 1.3)", "96 / 182", "18 of 20", "44 / 84", "76 / 161", "20 / 18", "8 of 20"],
    ], size=8.5,
            note="At 1 s only 10 of the 20 fastest passes are still hidden; the other columns use all passes of a group.")
    P.fig(os.path.join(FIG, "fig_bonus.png"), "bonus",
          r"Prediction through an occlusion. (a) Four moments of one pass at about 5 cm/s: M measured, its predicted pose 0.4 s and 1.2 s "
          r"after it disappeared behind the box, and M measured again; everything outside the path is cut away. (b) Uncovered passes hidden in "
          r"software: position error against the time hidden; the predictor curves end where the pose is declared lost. (c) Real obstacle: "
          r"error of the predictor against the error of holding the last pose at reappearance; points below the dashed line are passes in "
          r"which the predictor is better.")

    # ================================================================== 5 Failures
    P.h(1, "Failures and what they taught us", "failures")
    P.p(r"{tab:failures} lists what went wrong, in the order of the tasks. Three of the entries changed reported numbers and deserve more than a "
        r"line.")
    P.p(r"*The white margin.* While setting up T6 the live table showed M leaning by 18° and standing 2 cm nearer than the ruler said, with a "
        r"jitter of 3 to 4 mm, although both boards were flat. A clip, the tape and the upper edge of the sheet were blamed in turn, wrongly. "
        r"The cause was found by enlarging the camera image ({fig:margin}): M’s sheet had been cut to about 1 mm of white at its left and "
        r"right sides, and on the black board the detector’s corners slipped outward, so that the square was measured 9 % too wide. A white "
        r"sheet behind M restored the quiet zone: the ratio of width to height went from 1.09 to 1.01, the lean from 18° to 2° and the jitter "
        r"to 0.7 mm. The angle recordings of E1 and all of E2, E3 and E4 were made after this fix. The five position recordings of E1 and the occlusion "
        r"recordings were made before it: their rotation errors may contain this effect, and it is part of the noisy rotations of the occlusion passes. The marker design had a quiet zone of two cells; it was lost with a pair of scissors.")
    P.p(r"*The first occlusion results.* The first evaluation with the real obstacle reported 35 to 41 mm for the predictor, a fifth of the "
        r"baseline. The script took the hidden stretch of a pass as the frames from the first to the last missing measurement, so a single "
        r"dropped frame before or after the obstacle moved its ends: some passes were classified as lost at the start, and some errors at "
        r"reappearance were measured after the filter had already been updated and were too small. With the longest run of missing frames the "
        r"errors are 50 to 96 mm, a factor of 2 to 3.5 instead of 5. The earlier numbers were withdrawn.")
    P.p(r"*The synthetic network on real frames.* The first network looked finished on its own validation set (1.7° end to end) and was "
        r"unusable on the camera. Each round of real data then exposed a specific gap in the training data rather than a general weakness: M "
        r"next to F, a tilted camera, a covered or blurred marker. Each was closed by recording that case or by an augmentation that produces "
        r"it exactly. The synthetic validation error was never a guide to any of this.")
    P.fig(os.path.join(FIG, "fig_margin.png"), "margin",
          r"The white-margin failure. (a) M as mounted for the first T6 attempts, with about 1 mm of white at its sides on a black board, and "
          r"F with its full quiet zone (camera crops). (b to d) Width-to-height ratio of the detected square of M, false lean of M and jitter "
          r"before and after a white sheet was placed behind M.")
    P.table("failures", "Failures, how they were noticed, their cause and what was done.",
            [("What went wrong", 1.45, "l"), ("How it showed", 1.52, "l"), ("Cause", 1.55, "l"), ("What was done", 1.75, "l")], [
        ["T1: printed markers 77 mm, not 80 mm", "ruler measurement; T1 check not met", "printer scaled the page to 96 %", "measured sizes stored and used everywhere"],
        ["Calibration RMS 1.18 px; two views at 2.44 px", "per-view errors; curved lines in the rectified view", "printed chessboard not flat on its clipboard",
         "quantified (held-out 0.58 px, $f_{x}$ ±1.5 %), not repeated; a rigid board is needed"],
        ["T5: first network at 23.5°", "rotation loss still falling at the end", "15 epochs are too few", "40 epochs"],
        ["T5: doubling the rotation weight", "both validation errors worse", "not established (one seed per run)", "equal weights kept"],
        ["E5 evaluation gave poor numbers", "errors did not match the validation", "distortion-free synthetic images were undistorted", "undistortion made an option"],
        ["Synthetic-only network unusable on real frames", "26.5 mm (107 mm) on E1; 142° on T4", "domain gap: hand, room, light, print", "fine-tuning on real frames, half of every batch"],
        ["Round 1: one position fails (34 mm; stage 1 alone 103 mm)", "per-recording errors", "M next to F was absent from training; the estimate of stage 1 jumps there",
         "recordings of that case; second pass of the crop"],
        ["Tilted laptop: M placed 82 px too low", "validation on the one pitched recording (92 mm)", "stage 1 used the room as a reference",
         "camera-rotation augmentation; recordings with the camera pitched and turned"],
        ["Round 3 training stalled in epoch 3", "no progress; 15.9 of 16.3 GB of graphics memory used", "images on the graphics card spilled into shared memory",
         "large sets kept in main memory; restart from the saved epoch"],
        ["Validation crashed after an edit", "error message on a tensor that requires gradients", "a decorator ended up on the wrong function when code was inserted",
         "moved back; self-test run after edits"],
        ["F mounted a quarter turn rotated", "live table: 91° between F and M", "F re-clipped between sessions", "F turned back; the truth is given in F’s own directions"],
        ["False lean of M by 18°, 2 cm too near", "live table against ruler truth", "no white margin at M’s sides on a black board", "white backing sheet ({fig:margin})"],
        ["E3: two takes discarded", "track against the guide", "guide too fast to follow; start shifted by 26 mm", "1 cm/s; start measured again"],
        ["E4: first shake recording had no blur", "classical detector at 100 %", "shaking too gentle", "second recording with the screen wobbled; blur made by the camera, not by M"],
        ["Bonus: first results too good (35 to 41 mm)", "passes “lost at the start” that were not", "hidden stretch taken from first to last missing frame",
         "longest run of missing frames; numbers withdrawn"],
        ["Bonus: rotation extrapolated to 42° (fast uncovered passes)", "error against the next measurement", "angular velocity estimated from hand wobble", "angular velocity faded out while hidden"],
        ["Bonus: slow passes lost before reappearing", "hidden for 3 to 6 s", "slower than the task’s 0.5 to 2 s", "kept as examples of a lost pose; faster sets recorded"],
    ], size=7.5)

    # ================================================================== 6 Discussion
    P.h(1, "Discussion", "discussion")
    P.p(r"*Which pipeline.* For this task, a printed marker in view of a calibrated camera, the classical pipeline is the one to use: it is more "
        r"accurate than the network on 11 of the 16 recordings made after the network was final (the network is ahead on four, by at most 2 mm, and is alone in answering on the one with a covered edge), about twice as steady, six times "
        r"faster, needs no data, and when it fails it says so by giving no pose. The network earns a place only as a fallback for frames "
        r"without a detection, and there its answer should be treated as coarse: 3° in rotation but 30 mm in depth with a covered edge. A "
        r"sensible system would use the classical pose when there is one, the network when there is none, and the predictor of {sec:predmath} "
        r"with its gate to bridge short gaps and to reject isolated outliers such as those of E3. We did not build this combination.")
    P.p(r"*Where each fails.* The classical pipeline fails when part of the border is covered (no pose with a real cover of about 20 % of the square; "
        r"a pose in 29 % of the frames with a simulated cover of 15 %), under a simulated motion blur of about 30 px, beyond 80° of tilt and, by the pixel rule, beyond about 4 m for this marker; and it "
        r"has rare single-frame outliers. The network fails gradually: far and turned markers (25 mm at 77 cm and 40°), anything that changes "
        r"the apparent size, very dark frames, and any situation outside its training recordings, as each round of fine-tuning showed. It also "
        r"cannot report that M is absent, and it inherits the bias of the classical poses that labelled its training frames.")
    P.p(r"*Where the 8 mm come from.* The classical error on the bench is a bias of about 8 mm with a jitter of 0.2 to 1.5 mm, while on rendered images of the "
        r"same resolution the same code reaches 1.6 mm and 0.1°. The difference points to the set-up rather than to the algorithm. The candidates are a "
        r"calibration with 1.18 px of error and a focal length uncertain by about ±1.5 % (4 mm over a spacing of 277 mm), the weak depth of a marker seen almost "
        r"frontally (+3 to +9 mm in check A), boards that are parallel to within 2° to 4° at best, a quiet zone that was too narrow in the early recordings, and "
        r"a ruler read to a millimetre. We did not separate them. A flat calibration board would be the first thing to change.")
    P.p(r"*Limits of this study.* One camera, one marker size and one room. The ground truth is measured by hand, and for the motion test it is "
        r"a schedule followed by hand, which the hand trails by up to 2 s. Five of the seven E1 recordings influenced design choices. E2 has no setting at 60° and none beyond "
        r"77 cm. Blur was produced by shaking the camera, not by moving M fast. Only M is estimated by the network; F is always classical. "
        r"Every training run used a single seed, and training on the graphics processor is not repeatable bit for bit. In the occlusion "
        r"experiment the truth during the occlusion is exact only in the software-hidden passes.")

    # ================================================================== 7 Conclusion
    P.h(1, "Conclusion", "conclusion")
    P.p(r"With a laptop camera and printed markers the classical pipeline measures the relative pose of two markers with a median error of about 8 mm and 4° and "
        r"a jitter of about 1 mm, at 147 frames per second. On rendered images with an exact camera the same code reaches 1.6 mm and 0.1°, so most of the bench "
        r"error does not come from the algorithm. A two-stage network reaches the same level only after three rounds of real data, each of which repaired a "
        r"specific failure, and stays about twice as noisy and six times slower; its one advantage, a pose for a partly covered marker, comes with a depth error "
        r"of 30 mm. A constant-velocity predictor cuts the error of holding the last pose to between a half and a third across occlusions of 1 to 2 s and is "
        r"limited by the hand that moves the marker. The parts of this work that we would carry to another project are not in the networks: measure what was "
        r"printed, calibrate on a flat board, keep the quiet zone, look at single recordings rather than pooled medians, and check an evaluation script as "
        r"carefully as the method it evaluates.")

    # ------------------------------------------------------------------ end matter
    P.h(1, "Author contributions", numbered=False)
    P.p(r"D.-V. S.: set-up and all recordings, measurements, experiments, analysis, writing. M.-D. C.: [contribution to be completed by the "
        r"team]. I.-C. J.: [contribution to be completed by the team]. All authors can explain every part of the code and of this report.")
    P.h(1, "Use of AI tools", numbered=False)
    P.p(r"An AI assistant (Claude, Anthropic) was used for writing and debugging the code, for the analysis and evaluation scripts, for the "
        r"literature search, and for drafting and typesetting this report and its figures. The authors designed and carried out all "
        r"experiments, took all measurements, reviewed the code, the results and the text, and take full responsibility for the published data "
        r"and conclusions. Other reused material: the lecture toolkit (Appendix C), OpenCV, PyTorch and a ResNet-18 pretrained on ImageNet.")
    P.h(1, "Code and data availability", numbered=False)
    P.p(f"The code, the result files behind every number and figure of this paper and a README with one command per step are at {REPO}. "
        r"The camera recordings are not public, because they show the room and the person moving the marker; the test recordings with their "
        r"truth files are available to the examiners. Trained networks and rendered data sets can be regenerated with the commands of the README "
        r"(seed 0).", align="j")

    P.h(1, "References", numbered=False)
    P.references(REFS)

    # ================================================================== Appendices
    P.h(1, "Appendix A. Test recordings", numbered=False)
    P.p(r"{tab:tests} lists the 21 test recordings with their ground truth. Positions are those of the centre of M in the frame of F "
        r"({sec:frames}); the pitch is the turn of M about the vertical axis. The camera distance is measured to F.")
    P.table("tests", "Test recordings (1280 × 720 px, about 30 frames per second).",
            [("Recording", 1.05, "l"), ("Frames", 0.5, "c"), ("Truth x, y, z [mm]", 1.05, "c"), ("Pitch [deg]", 0.5, "c"), ("Camera [cm]", 0.55, "c"), ("Set-up", 2.62, "l")], [
        ["E1_p1 … E1_p5", "5 × 100", "−277, −227, −177, −127, −327; 0 (P4: 3); 0", "0", "34 to 37", "M beside F on one straight edge; gaps 200, 150, 100, 50, 250 mm; boards turned 20° to 30° from the camera"],
        ["E1_turn20, E1_turn40", "2 × 100", "−115, −20, 90", "20, 40", "61", "M on its own support in front of F, turned about the vertical axis"],
        ["E2_d40_t0, _t40", "2 × 100", "−115, −20, 90", "0, 40", "41", "as above; only the laptop is moved"],
        ["E2_d60_t0, _t40", "2 × 100", "−115, −20, 90", "0, 40", "61", "as above"],
        ["E2_d75_t0, _t40", "2 × 100", "−115, −20, 90", "0, 40", "77", "as above; the largest distance the room allows"],
        ["E3_motion_01 … _03", "600, 601, 601", "−140 to −200, −20, 90", "0", "61", "M slides 60 mm along a ruler and back, 1 cm/s, rests of 2 s; truth = schedule of the guide outline"],
        ["E4_dim", "100", "−135, 3.5, 0", "0", "61", "room light off"],
        ["E4_clutter", "100", "−135, 3.5, 0", "0", "61", "six desk objects in front of and between the markers, none covering them"],
        ["E4_blur", "150", "−135, 3.5, 0", "0", "61", "table and laptop shaken gently (no visible blur)"],
        ["E4_blur_2", "150", "−135, 3.5, 0", "0", "61", "laptop screen wobbled quickly by hand (about a third of the frames blurred)"],
        ["E4_cover", "100", "−135, 3.5, 0", "0", "61", "left vertical edge of M covered by a paper strip, about 20 % of the square"],
    ], size=8)
    P.p(r"Further recordings: T4_motion_1 (900 frames, free motion of M), invariance_1 (1 499 frames, check B), and twelve occlusion recordings "
        r"of 36 to 95 s: three without an obstacle, five with a tin box and four with a light-bulb box.", size=B - 0.5)

    P.h(1, "Appendix B. Training data, settings and the hard-case test", numbered=False)
    P.table("train", "The 17 real training recordings (F still, M moved by hand; labels from the classical pipeline with a reprojection error of at most 1 px).",
            [("Set", 0.8, "l"), ("Recordings", 0.7, "c"), ("Frames", 0.6, "c"), ("Labelled", 0.65, "c"), ("Validation recording", 1.15, "l"), ("Set-up", 2.37, "l")], [
        ["real_v2 (round 1)", "5", "5 376", "3 753", "real_v2_05 (589 frames)", "camera upright, F at 31 to 41 cm"],
        ["real_v3 (round 2)", "5", "6 284", "4 535", "real_v3_05 (930 frames)", "F at 40 to 67 cm, left, middle and right of the picture; other light; M right next to F; one recording with the screen pitched"],
        ["real_v4 (round 3)", "7", "8 790", "6 999", "real_v4_03, real_v4_07 (1 130 and 1 076 frames)", "camera pitched inward and outward by 15°, laptop turned by 8° to 15°, 44 to 71 cm"],
        ["Total", "17", "20 450", "15 287", "4 recordings, 3 725 frames", "13 recordings and 11 562 frames for training"],
    ], size=8)
    P.table("hyper", "Training settings.", [("Setting", 1.9, "l"), ("Synthetic training", 2.0, "l"), ("Fine-tuning (rounds 1 to 3)", 2.37, "l")], [
        ["Optimiser", "AdamW, weight decay $10^{-4}$", "the same"],
        ["Learning rate", r"one cycle, peak $5 \times 10^{-4}$, 10 % warm-up, cosine decay", r"one cycle, peak $10^{-4}$"],
        ["Batch", "64", "64 (round 1), 128 (rounds 2 and 3); half real, half synthetic"],
        ["Epochs, stage 1 / stage 2", "40 / 80", "15 / 15, 25 / 25, 25 / 50; best validation epoch kept from round 2"],
        ["Loss weights $w_{R}$, $w_{t}$", "1, 1", "1, 1"],
        ["Augmentation, stage 1", "gain, offset", "gain, offset; camera rotation ±8° pitch and yaw, ±10° roll (rounds 2, 3); hard cases 40 % (round 3)"],
        ["Augmentation, stage 2", "gain, offset, noise, blur; window offsets 12 %; gradient clipping 5", "the same; window offsets 25 % for real crops (rounds 2, 3); hard cases 40 % (round 3)"],
        ["Precision, seed", "mixed precision (bfloat16), seed 0", "the same"],
        ["Initial weights", "ResNet-18 pretrained on ImageNet", "the synthetic networks (runs B and D)"],
    ], size=8)
    P.table("runs", "Every saved training run, from its training log: loss and validation errors at the epoch that was kept (the last epoch "
                    "for the synthetic runs and for round 1). Validation: median (95th percentile) on the validation set of the run.",
            [("Run (network file)", 1.12, "l"), ("Stage", 0.38, "c"), ("Training data", 1.3, "l"), ("Epochs (kept)", 0.5, "c"), ("Peak learning rate", 0.55, "c"),
             ("Time [min]", 0.4, "c"), ("Training loss, rotation / translation", 0.82, "c"), (r"Validation $e_{R}$ [deg]", 0.6, "c"),
             (r"Validation $e_{t}$ [mm]", 0.6, "c")], run_rows(), size=7.5, rules=(3, 5, 7),
            note="Two further runs were started and stopped, and are not in the table: the first attempt of round 2, without the camera-rotation "
                 "augmentation (stopped when the validation error stayed near 13.9° and 92 mm; only this last reading was kept), and the first "
                 "attempt of round 3, which stalled in epoch 3 when the graphics memory was full and was restarted from the saved epoch 2.")
    P.table("hard", "Software hard-case test on 250 frames of E1 (P1 to P5). Reference: the classical pose of the clean frame; usable: within "
                    "20 mm and 10° of it. In brackets: median $e_{t}$ [mm] / $e_{R}$ [deg].",
            [("Degradation", 1.35, "l"), ("Classical: output [%]", 0.9, "c"), ("usable [%]", 0.72, "c"), ("Network, round 2: usable [%]", 1.65, "c"),
             ("Network, round 3: usable [%]", 1.65, "c")], [
        ["none", "100", "100", "100 (3.7 / 1.1)", "100 (4.7 / 1.2)"],
        ["15 % of M covered", "29", "28", "100 (5.5 / 3.1)", "100 (6.9 / 2.3)"],
        ["30 % of M covered", "0", "0", "51 (10.9 / 9.0)", "91 (5.9 / 5.1)"],
        ["motion blur 15 px", "100", "100", "100 (3.2 / 1.5)", "100 (4.7 / 1.3)"],
        ["motion blur 31 px", "0", "0", "14 (20.0 / 11.5)", "100 (8.9 / 4.1)"],
        ["resolution / 4", "100", "100", "99 (3.6 / 1.5)", "100 (3.9 / 1.0)"],
        ["resolution / 6", "100", "100", "100 (4.4 / 1.7)", "100 (4.2 / 0.9)"],
        ["brightness × 0.25", "97", "97", "100 (6.7 / 4.3)", "100 (5.0 / 1.9)"],
        ["brightness × 0.10", "0", "0", "0 (209 / 19.8)", "0 (124 / 31.2)"],
    ], size=8)

    P.h(1, "Appendix C. Reused starter code", numbered=False)
    P.p(r"The lecture toolkit `mvlib.py` [@lectures] is used unchanged. {tab:starter} explains every function and class of it that the project "
        r"calls. Everything else (tools, recorders, renderer, networks, training, evaluation, predictor logic) was written for this project.")
    P.table("starter", "Functions and classes reused from the lecture toolkit.",
            [("Name", 1.55, "l"), ("What it does", 3.12, "l"), ("Used in", 1.6, "l")], [
        ["make_T, inv_T", r"build a 4 × 4 pose from $\mathbf{R}$ and $\mathbf{t}$; invert it as $[\mathbf{R}^{\top},\,-\mathbf{R}^{\top}\mathbf{t}]$ without a general matrix inverse", "everywhere"],
        ["T_from_rvec_tvec", "turn OpenCV’s rotation vector and translation into a 4 × 4 pose (Rodrigues formula)", "classical pose"],
        ["rotvec, expm_so3", "logarithm and exponential map of SO(3): rotation matrix to rotation vector and back", "filters, renderer, tests"],
        ["rot_error_deg, trans_error", "the two error measures, {eq:err}", "evaluation"],
        ["marker_object_points", "the four corners of a square of side $L$ in the marker frame, in the order the detector returns them and IPPE requires", "pose, renderer, drawing"],
        ["reproj_rms", "RMS reprojection error of the four corners for a given pose", "ambiguity ratio, label filter"],
        ["marker_pose", "IPPE (both candidates), Levenberg–Marquardt refinement, best candidate and ambiguity ratio", "reference for our estimator, which adds the temporal rule"],
        ["relative_pose", "the relative pose, {eq:rel}", "classical and learned pipeline"],
        ["MarkerDetector", "wrapper of OpenCV’s ArUco detector with sub-pixel corner refinement; returns the corners per ID", "T2 to T6, bonus"],
        ["ConstantVelocityKalman", "the filter, {eq:kf}, with the gate, {eq:gate}", "animation filter, predictor"],
        ["AlphaBetaSO3", "the rotation filter, {eq:so3a} and {eq:so3b}", "animation filter, predictor"],
        ["mean_rotation, mean_pose", "chordal mean of rotations by singular value decomposition [@moakher]; mean of several measurements of a static pose", "check B"],
    ], size=8)

    P.h(1, "Appendix D. Reproducing the numbers", numbered=False)
    P.p(r"Every number of this paper comes from a file written by one of the project’s programs; the README lists 50 numbered steps with one "
        r"command each and a table that says which file holds which number. The main ones, run in the folder `code`: `t3_calibrate.py` "
        r"(calibration, {tab:calib}); `t2_matrix.py` (detection limits, {fig:t2}); `t3_pose.py` and `t3_check_invariance.py` (checks A and B); "
        r"`t5_make_data.py`, `t5_make_crops.py`, `t5_train.py` (synthetic data and training, {tab:synth}); `t5_finetune.py --sets real_v2 "
        r"real_v3 real_v4 --suffix _ft3` (round 3); `t5_eval_real.py` and `t5_test_hard_cases.py` ({tab:steps}, {tab:hard}); `t6_evaluate.py "
        r"--network crop_resnet18_v1_ft3` (E1 to E6); `t7_virtual_occlusion.py` and `t7_real_occlusion.py` (bonus); `t8_live_all.py` shows the three methods together on the live camera. The figures and the "
        r"per-recording tables of this paper are produced by `paper/prep_data.py`, `paper/make_figures.py` and `paper/build_paper.py`; "
        r"`make_figures.py` checks every figure for overlapping labels before it writes it.", size=B - 0.5)


if __name__ == "__main__":
    import time
    for attempt in range(8):                                            # Word may still be releasing the file after a render
        try:
            open(OUT, "ab").close()
            break
        except PermissionError:
            time.sleep(1.5)
    else:                                                               # the file is open in Word: write next to it
        OUT = OUT[:-5] + "_new.docx"
        print("the paper is open in Word; writing", os.path.basename(OUT), "instead")
    lay = os.path.join(HERE, "layout.json")                             # written by tune_layout.py: figures that slide past text blocks
    defer = json.load(open(lay)) if os.path.exists(lay) else {}
    paper, unresolved = build(write, OUT, defer=defer)
    print("written:", OUT)
    print("figures", paper.n["fig"], "| tables", paper.n["tab"], "| equations", paper.n["eq"], "| references", len(paper.cites))
    if unresolved:
        print("UNRESOLVED cross-references:", unresolved)
    unused = [k for k in REFS if k not in paper.cites]
    if unused:
        print("references not cited:", unused)
