"""make_figures.py -- every figure of the paper, drawn from the saved result files (run from the project folder):

    python paper/prep_data.py      (once: per-frame errors, calibration residuals, camera cut-outs)
    python paper/make_figures.py   [figure keys...]

Writes paper/figures/fig_*.png at the size they have in the paper (text width 6.27 in, 300 dpi).
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figcheck  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "paper", "figures")
PH = os.path.join(OUT, "photos")
DAT = os.path.join(ROOT, "paper", "data")
RES = os.path.join(ROOT, "output")
W = 6.27                                              # text width [in]
CL, NET, PRED, HOLD, RED, GREY = "#1F3A5F", "#D9731A", "#2E7D32", "#8C8C8C", "#B03A2E", "#404040"
XC, YC, ZC = "#C0392B", "#2E8B3D", "#2C6FB7"          # axis colours x y z
TS = 8.5                                              # panel title size

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": TS, "axes.labelsize": 8, "legend.fontsize": 6.9, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": True, "legend.facecolor": "white", "legend.edgecolor": "none", "legend.framealpha": 1.0, "legend.borderpad": 0.25, "legend.borderaxespad": 0.45, "grid.linewidth": 0.4, "grid.alpha": 0.35, "savefig.dpi": 300, "figure.dpi": 100, "axes.titlepad": 4,
    "axes.labelpad": 2.5, "axes.axisbelow": True,
})


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


PROBLEMS = {}
PROBLEMS_BOX = {}


def save(fig, name, data=True):
    """Checks the figure for overlapping elements (figcheck.py), then writes it."""
    PROBLEMS[name] = figcheck.check(fig, name, data=data)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


WHITE = dict(fc="white", ec="none", pad=0.5)      # ground for a note that sits on grid lines


def full(fig, left=0.078, right=0.995, **kw):
    """The axes use the whole text width, so the saved image is about 6.3 in wide and the lettering has its nominal size on the page."""
    fig.subplots_adjust(left=left, right=right, **kw)


def ttl(ax, s, x=0.0):
    """Panel title, left-aligned with the axes, never wider than the panel (keep s short)."""
    ax.set_title(s, fontsize=TS, fontweight="bold", loc="left", x=x, pad=4)


def show(ax, path, label=None):
    im = plt.imread(path)
    ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1, interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.4); s.set_color("#444444")
    if label:
        ax.set_xlabel(label, fontsize=6.5, labelpad=2)


def arrow(ax, p, q, col="k", lw=0.9, style="-|>", ls="-", ms=7, z=3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, lw=lw, color=col, ls=ls, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def train_log(name):
    return json.load(open(os.path.join(RES, "network", name, "train_log.json")))


def test_errors():
    out = {}
    for x in rows(os.path.join(DAT, "test_errors.csv")):
        out.setdefault(x["recording"], []).append([fnum(x[k]) for k in ("et_classical_mm", "eR_classical_deg", "et_network_mm", "eR_network_deg",
                                                                        "xc", "yc", "zc", "xn", "yn", "zn")])
    return {k: np.array(v) for k, v in out.items()}


# ======================================================================================= Fig. set-up and frames
def fig_setup():
    fig = plt.figure(figsize=(W, 2.3))
    fig.text(0.0, 0.955, "(a) Frames and ruler measures (top view)", fontsize=TS, fontweight="bold")
    fig.text(0.425, 0.955, "(b) Occlusion bench (camera view)", fontsize=TS, fontweight="bold")
    fig.text(0.845, 0.955, "(c) Camera crops", fontsize=TS, fontweight="bold")
    # ---- (a)
    ax = fig.add_axes([0.0, 0.0, 0.40, 0.92]); ax.set_xlim(0, 10); ax.set_ylim(0, 8.45); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((0.6, 7.0), 9.0, 0.2, fc="#2b2b2b", ec="none"))
    ax.text(0.6, 7.33, "board", fontsize=6.8, color=GREY)
    fx, fy = 7.0, 6.82
    ax.add_patch(Rectangle((fx - 1.0, fy), 2.0, 0.18, fc="white", ec="k", lw=0.7))
    mx, my, psi = 2.8, 4.6, np.radians(25)
    d = np.array([np.cos(psi), -np.sin(psi)])
    a, b = np.array([mx, my]) - d, np.array([mx, my]) + d
    ax.plot([a[0], b[0]], [a[1], b[1]], color="k", lw=2.8, solid_capstyle="butt", zorder=2)
    ax.plot([a[0], b[0]], [a[1], b[1]], color="white", lw=1.5, solid_capstyle="butt", zorder=2)
    for (ox, oy), ang in (((fx, fy), 0.0), ((mx, my), -psi)):
        ex, ez = np.array([np.cos(ang), np.sin(ang)]), np.array([np.sin(ang), -np.cos(ang)])
        arrow(ax, (ox, oy), (ox + 1.2 * ex[0], oy + 1.2 * ex[1]), XC, 1.1)
        arrow(ax, (ox, oy), (ox + 1.2 * ez[0], oy + 1.2 * ez[1]), ZC, 1.1)
        ax.plot(ox, oy, "o", ms=4.4, mfc="white", mec=YC, mew=1.0, zorder=5); ax.plot(ox, oy, ".", ms=1.8, color=YC, zorder=6)
    ax.text(fx + 1.32, fy, "$x$", color=XC, fontsize=7, va="center"); ax.text(fx + 0.14, fy - 1.25, "$z$", color=ZC, fontsize=7, va="center")
    ax.text(mx + 1.2, my - 0.82, "$x$", color=XC, fontsize=7, va="center", ha="center"); ax.text(mx - 0.78, my - 1.2, "$z$", color=ZC, fontsize=7, va="center", ha="center")
    ax.text(fx + 0.17, 7.36, "{F}  fixed, id 0", fontsize=6.8, va="bottom")
    ax.text(0.15, 5.85, "{M}  mobile,\nid 23", fontsize=6.8, va="center", linespacing=1.1)
    cx, cy = 5.3, 1.11                                                                       # camera origin
    ax.add_patch(Rectangle((cx - 0.55, cy - 0.66), 1.1, 0.42, fc="#d9d9d9", ec="k", lw=0.7))
    ax.add_patch(Polygon([[cx - 0.2, cy - 0.24], [cx + 0.2, cy - 0.24], [cx + 0.32, cy], [cx - 0.32, cy]], fc="#9a9a9a", ec="k", lw=0.6))
    arrow(ax, (cx, cy), (cx + 1.1, cy), XC, 1.1); arrow(ax, (cx, cy), (cx, cy + 1.1), ZC, 1.1)
    ax.plot(cx, cy, "o", ms=4.4, mfc="white", mec=YC, mew=1.0, zorder=5); ax.plot(cx, cy, "x", ms=2.6, color=YC, zorder=6, mew=0.8)
    ax.text(cx + 1.22, cy, "$x$", color=XC, fontsize=7, va="center"); ax.text(cx + 0.14, cy + 1.12, "$z$", color=ZC, fontsize=7, va="center")
    ax.text(cx, 0.08, "{C}  laptop camera", fontsize=6.8, ha="center")
    arrow(ax, (cx + 0.45, cy + 1.25), (fx - 0.12, fy - 1.42), "k", 0.8, rad=0.1)
    arrow(ax, (cx - 0.4, cy + 1.25), (mx + 0.3, my - 0.75), "k", 0.8, rad=-0.1)
    arrow(ax, (fx - 0.3, fy - 0.2), (mx + 0.55, my + 0.15), RED, 1.2, ls=(0, (4, 2)), rad=0.14)
    ax.text(6.75, 3.55, r"${}^{C}\mathbf{T}_{F}$", fontsize=8.5); ax.text(3.45, 2.1, r"${}^{C}\mathbf{T}_{M}$", fontsize=8.5)
    ax.text(4.75, 5.0, r"${}^{F}\mathbf{T}_{M}$", fontsize=9, color=RED)
    ax.annotate("", (mx, 8.02), (fx, 8.02), arrowprops=dict(arrowstyle="<->", lw=0.6, color=GREY, shrinkA=0, shrinkB=0))
    ax.text((mx + fx) / 2, 8.1, "$s$", fontsize=7, ha="center", color=GREY, va="bottom")
    ax.plot([mx, mx], [7.2, 8.1], color=GREY, lw=0.4, ls=":"); ax.plot([fx, fx], [7.2, 8.1], color=GREY, lw=0.4, ls=":")
    ax.annotate("", (mx, my + 0.12), (mx, 7.0), arrowprops=dict(arrowstyle="<->", lw=0.6, color=GREY, shrinkA=0, shrinkB=0))
    ax.text(mx + 0.16, 6.0, "$n$", fontsize=7, color=GREY, va="center")
    ax.add_patch(Arc((mx, my), 3.3, 3.3, theta1=-25, theta2=0, lw=0.6, color=GREY)); ax.plot([mx, mx + 1.95], [my, my], color=GREY, lw=0.4, ls=":")
    ax.text(mx + 1.86, my - 0.36, r"$\psi$", fontsize=7.5, color=GREY, va="center")
    # ---- (b)
    ax = fig.add_axes([0.425, 0.0, 0.40, 0.92]); ax.set_xlim(0, 10); ax.set_ylim(0, 8.45); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((0.3, 1.9), 9.4, 0.3, fc="#efe3a8", ec="k", lw=0.5))
    for i in range(48):
        ax.plot([0.4 + i * 0.196] * 2, [2.2, 2.2 - (0.17 if i % 5 == 0 else 0.09)], color="k", lw=0.3)
    ax.text(0.3, 1.45, "ruler, mm divisions", fontsize=6.8, color=GREY)

    def marker(x, y, lab):
        ax.add_patch(Rectangle((x - 1.05, y - 1.05), 2.1, 2.1, fc="white", ec="k", lw=0.7))
        ax.add_patch(Rectangle((x - 0.78, y - 0.78), 1.56, 1.56, fc="#222222", ec="k", lw=0.6))
        ax.text(x, y, lab, color="white", fontsize=8, ha="center", va="center", fontweight="bold")
    marker(8.4, 3.55, "F"); marker(1.5, 3.55, "M")
    ax.add_patch(Rectangle((4.0, 2.2), 1.9, 4.2, fc="#c9c9c9", ec="k", lw=0.7, zorder=4))
    ax.text(4.95, 6.6, "obstacle (box)", fontsize=6.8, ha="center", va="bottom")
    ax.add_patch(Rectangle((3.9, 2.5), 2.1, 2.1, fc="none", ec=RED, lw=1.1, zorder=6)); ax.text(4.95, 4.78, "predicted", color=RED, fontsize=6.8, ha="center", va="bottom", zorder=7)
    arrow(ax, (2.7, 3.55), (3.72, 3.55), "k", 1.0); ax.text(3.2, 3.75, "$v$", fontsize=8, ha="center", va="bottom")
    ax.text(1.5, 4.85, "M slides\ntoward F", fontsize=6.9, ha="center", va="bottom", color=GREY, linespacing=1.1)
    ax.text(8.4, 4.85, "F stays\nvisible", fontsize=6.9, ha="center", va="bottom", color=GREY, linespacing=1.1)
    ax.annotate("", (1.5, 0.98), (8.4, 0.98), arrowprops=dict(arrowstyle="<->", lw=0.6, color=GREY, shrinkA=0, shrinkB=0))
    ax.text(4.95, 0.2, "325–377 mm centre to centre at the start", fontsize=6.8, ha="center", color=GREY)
    # ---- (c)
    a1 = fig.add_axes([0.86, 0.535, 0.125, 0.125 * W / 2.3]); show(a1, os.path.join(PH, "view_F_raw.png"), "F, ID 0")
    a2 = fig.add_axes([0.86, 0.065, 0.125, 0.125 * W / 2.3]); show(a2, os.path.join(PH, "view_E2_d40_t0_raw.png"), "M, ID 23")
    save(fig, "fig_setup")


# ======================================================================================= Fig. the two pipelines
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(W, 2.2)); ax.set_xlim(0, 100); ax.set_ylim(0, 35); ax.axis("off")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    FS = 6.1

    def box(x, y, w, h, text, fc, fs=FS):
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="k", lw=0.65, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.18)

    def ar(p, q, col="k", rad=0.0):
        arrow(ax, p, q, col, 0.8, ms=6, rad=rad)
    cb, nb = "#dfe7f1", "#fbe6d3"
    h = 8.6
    box(0.4, 13.2, 9.2, h, "camera\nframe\n1280 × 720", "#ffffff")
    yc, yn = 24.0, 2.4
    ax.text(13.0, yc + h + 1.0, "CLASSICAL  (both markers)", color=CL, fontsize=7, fontweight="bold")
    ax.text(13.0, yn + h + 1.0, "LEARNED  (marker M only)", color=NET, fontsize=7, fontweight="bold")
    xs, ws = [13.0, 31.5, 48.0, 64.5], [15.5, 13.5, 13.5, 14.5]
    tx = ["ArUco detection\nand sub-pixel\ncorners", "IPPE:\ntwo planar\nsolutions", "LM refinement\n(reprojection\nerror)", "ambiguity rule:\nclosest to the\nprevious frame"]
    for x, w_, t in zip(xs, ws, tx):
        box(x, yc, w_, h, t, cb)
    for i in range(3):
        ar((xs[i] + ws[i] + 0.5, yc + h / 2), (xs[i + 1] - 0.5, yc + h / 2), CL)
    ar((5.0, 22.2), (12.5, yc + h / 2), CL, rad=-0.35)
    xn, wn = [13.0, 27.5, 46.0, 64.0], [11.5, 15.5, 15.0, 15.0]
    tn = ["undistort,\nresize to\n640 × 360", "stage 1: ResNet-18\non the whole image\n(coarse position)", "crop 224 × 224,\n2 × marker side,\n+ 2 ray channels", "stage 2: ResNet-18\n6-D rotation,\n(x/Z, y/Z, log Z)"]
    for x, w_, t in zip(xn, wn, tn):
        box(x, yn, w_, h, t, nb)
    for i in range(3):
        ar((xn[i] + wn[i] + 0.5, yn + h / 2), (xn[i + 1] - 0.5, yn + h / 2), NET)
    ar((5.0, 12.8), (12.5, yn + h / 2), NET, rad=0.35)
    ar((xn[3] + 7.5, yn + h + 0.6), (xn[2] + 7.5, yn + h + 0.6), NET, rad=0.5)
    ax.text(62.3, yn + h + 6.3, "second pass: re-centre the crop", fontsize=6.2, color=NET, ha="center")
    box(84.0, 13.2, 15.6, h, "relative pose\n" + r"${}^{F}\mathbf{T}_{M}=({}^{C}\mathbf{T}_{F})^{-1}\,{}^{C}\mathbf{T}_{M}$", "#ffffff", fs=6.0)
    ar((xs[3] + ws[3] + 0.5, yc + h / 2), (91.8, 13.2 + h + 0.5), CL, rad=-0.3)
    ar((xn[3] + wn[3] + 0.5, yn + h / 2), (91.8, 12.7), NET, rad=0.3)
    ax.text(89.5, yc + h / 2 + 2.2, r"${}^{C}\mathbf{T}_{F},\ {}^{C}\mathbf{T}_{M}$", fontsize=7, color=CL, ha="center")
    ax.text(89.5, yn + h / 2 - 2.6, r"${}^{C}\mathbf{T}_{M}$", fontsize=7, color=NET, ha="center")
    ax.text(40.0, 20.9, r"${}^{C}\mathbf{T}_{F}$ always comes from the classical branch", fontsize=6.6, color="#333333", ha="center", style="italic")
    save(fig, "fig_pipeline")


# ======================================================================================= Fig. calibration
def fig_calib():
    c = rows(os.path.join(DAT, "calib_corners.csv")); v = rows(os.path.join(DAT, "calib_views.csv"))
    view = np.array([int(x["view"]) for x in c]); uv = np.array([[fnum(x["u"]), fnum(x["v"])] for x in c]); dd = np.array([[fnum(x["du"]), fnum(x["dv"])] for x in c])
    rms = np.array([fnum(x["rms_px"]) for x in v]); ho = np.array([x["held_out"] == "1" for x in v]); hor = np.array([fnum(x["held_out_rms_px"]) for x in v])
    tilt = np.array([fnum(x["tilt_deg"]) for x in v])
    fig = plt.figure(figsize=(W, 3.5))
    y1, y2 = 0.955, 0.455                                              # title rows
    # (a) where the corners were seen
    a = fig.add_axes([0.065, 0.615, 0.255, 0.29])
    sc = a.scatter(uv[:, 0], uv[:, 1], s=2.0, c=rms[view - 1], cmap="viridis_r", vmin=0.3, vmax=2.5, linewidths=0)
    a.add_patch(Rectangle((0, 0), 1280, 720, fc="none", ec="k", lw=0.6)); a.set_xlim(-25, 1305); a.set_ylim(745, -25); a.set_aspect("equal")
    a.set_xticks([0, 640, 1280]); a.set_yticks([0, 360, 720]); a.set_xlabel("u [px]"); a.set_ylabel("v [px]")
    for sp in a.spines.values():
        sp.set_visible(False)
    cax = fig.add_axes([0.335, 0.655, 0.011, 0.21])
    cb = fig.colorbar(sc, cax=cax); cb.set_label("view RMS [px]", fontsize=6.3, labelpad=2); cb.ax.tick_params(labelsize=6, length=2)
    fig.text(0.0, y1, "(a) Corner coverage, 20 views", fontsize=TS, fontweight="bold")
    # (b) residual cloud
    b = fig.add_axes([0.485, 0.6, 0.178, 0.178 * W / 3.5])
    b.scatter(dd[:, 0], dd[:, 1], s=1.5, color=CL, alpha=0.45, linewidths=0)
    b.add_patch(Circle((0, 0), 1.184, fc="none", ec=RED, lw=0.8, ls="--"))
    b.set_xlim(-5, 5); b.set_ylim(-5, 5); b.set_aspect("equal"); b.grid(True); b.set_xlabel(r"$\Delta u$ [px]"); b.set_ylabel(r"$\Delta v$ [px]")
    b.set_xticks([-4, -2, 0, 2, 4]); b.set_yticks([-4, -2, 0, 2, 4])
    fig.text(0.425, y1, "(b) Residuals (circle: RMS)", fontsize=TS, fontweight="bold")
    # (c) per view
    d = fig.add_axes([0.775, 0.6, 0.225, 0.32])
    x = np.arange(1, 21)
    d.bar(x[~ho], rms[~ho], 0.7, color=CL, label="fitted view"); d.bar(x[ho], rms[ho], 0.7, color="#9ab0c9", label="held-out view")
    d.scatter(x[ho], hor[ho], marker="_", s=45, color=NET, lw=1.3, zorder=4, label="error when held out")
    d.axhline(1.184, color=RED, lw=0.7, ls="--")
    d.set_xticks([1, 5, 10, 15, 20]); d.set_xlabel("view"); d.set_ylabel("reprojection RMS [px]"); d.set_ylim(0, 4.3); d.set_yticks([0, 1, 2, 3, 4]); d.grid(True, axis="y")
    d.legend(loc="upper left", ncol=1, borderaxespad=0.45, labelspacing=0.12, handlelength=1.0, fontsize=6.2)
    fig.text(0.715, y1, "(c) Error per view", fontsize=TS, fontweight="bold")
    # (d, e) one view: the board only, and rectified
    e1 = fig.add_axes([0.0, 0.065, 0.315, 0.36]); show(e1, os.path.join(PH, "calib_view.png"), "detected corners (view 9, tilt 38 deg)")
    fig.text(0.0, y2, "(d) One board view (cut out)", fontsize=TS, fontweight="bold")
    e2 = fig.add_axes([0.355, 0.065, 0.315, 0.36]); show(e2, os.path.join(PH, "calib_rect.png"), "curved lines: the sheet is not flat")
    fig.text(0.355, y2, "(e) The same view, rectified", fontsize=TS, fontweight="bold")
    # (f) the fitted distortion and the residual size against the radius
    f = fig.add_axes([0.775, 0.115, 0.225, 0.3])
    fx_, cx_, cy_, k1, k2 = 846.593, 641.122, 361.777, 0.0922014, -0.1347621
    r = np.linspace(0, 734, 200); rn = r / fx_
    f.plot(r, fx_ * rn * (k1 * rn ** 2 + k2 * rn ** 4), color=CL, lw=1.1, label="radial shift of the model")
    rad = np.hypot(uv[:, 0] - cx_, uv[:, 1] - cy_); mag = np.hypot(dd[:, 0], dd[:, 1])
    edges = np.linspace(0, 700, 8); mid = 0.5 * (edges[1:] + edges[:-1])
    rb = [np.sqrt(np.mean(mag[(rad >= lo) & (rad < hi)] ** 2)) if ((rad >= lo) & (rad < hi)).sum() > 20 else np.nan for lo, hi in zip(edges[:-1], edges[1:])]
    f.plot(mid, rb, color=RED, marker="o", ms=2.6, lw=0.9, label="residual RMS per ring")
    f.axhline(0, color="k", lw=0.4); f.set_xlabel("distance from the principal point [px]"); f.set_ylabel("shift, residual [px]"); f.grid(True); f.set_xlim(0, 734); f.set_ylim(-9.5, 9.5)
    f.set_xticks([0, 200, 400, 600])
    f.legend(loc="lower left", borderaxespad=0.45, labelspacing=0.12, handlelength=1.4, fontsize=6.2)
    fig.text(0.715, y2, "(f) Distortion model", fontsize=TS, fontweight="bold")
    print("     calib: corners %d, residual rms %.3f px, by ring %s, tilt range %.0f-%.0f deg" % (len(uv), np.sqrt(np.mean(mag ** 2)), np.round(rb, 2), tilt.min(), tilt.max()))
    save(fig, "fig_calib")


# ======================================================================================= Fig. detection limits
def fig_t2():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(W, 1.95), gridspec_kw=dict(width_ratios=[1.75, 0.72, 0.72], wspace=0.5))
    full(fig)
    r = rows(os.path.join(RES, "detection", "detection_matrix.csv"))
    D, T = [30, 50, 75, 100], [0, 30, 45, 60, 70, 75, 80, 85]
    M = np.full((len(D), len(T)), np.nan); P = np.full_like(M, np.nan)
    for x in r:
        i, j = D.index(int(float(x["target_distance_cm"]))), T.index(int(float(x["target_tilt_deg"])))
        M[i, j] = fnum(x["detected"]); P[i, j] = fnum(x["px_per_cell"])
    p0 = P[:, 0]
    cm = matplotlib.colors.LinearSegmentedColormap.from_list("d", ["#cf6a5d", "#f0d98c", "#8fc08d"])
    a.imshow(np.ma.masked_invalid(M), cmap=cm, vmin=0, vmax=10, aspect="auto")
    for i in range(len(D)):
        for j in range(len(T)):
            if np.isfinite(M[i, j]):
                a.text(j, i - 0.14, f"{int(M[i, j])}/10", ha="center", va="center", fontsize=6.6, fontweight="bold")
                a.text(j, i + 0.26, f"{p0[i] * np.cos(np.radians(T[j])):.1f}", ha="center", va="center", fontsize=6.2, color="#111111")
            else:
                a.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc="#e6e6e6", ec="none")); a.text(j, i, "not\ntested", ha="center", va="center", fontsize=6.3, color="#222222")
    a.set_xticks(range(len(T))); a.set_xticklabels([f"{t}" for t in T]); a.set_yticks(range(len(D))); a.set_yticklabels([f"{d_} cm" for d_ in D])
    a.set_xlabel("tilt about the vertical axis [deg]"); a.set_ylabel("distance to the lens")
    a.set_xticks(np.arange(-0.5, len(T)), minor=True); a.set_yticks(np.arange(-0.5, len(D)), minor=True)
    a.grid(which="minor", color="w", lw=1.0, alpha=1); a.tick_params(which="minor", length=0)
    for s in a.spines.values():
        s.set_visible(False)
    ttl(a, "(a) Detected frames of 10 (below: px per cell)")
    td, ed = np.array([fnum(x["target_distance_cm"]) for x in r]), np.array([fnum(x["est_distance_cm"]) for x in r])
    tt, et = np.array([fnum(x["target_tilt_deg"]) for x in r]), np.array([fnum(x["est_tilt_deg"]) for x in r])
    ok = np.isfinite(ed)
    b.plot([20, 112], [20, 112], color="k", lw=0.5, ls="--"); b.scatter(td[ok], ed[ok], s=7, color=CL, alpha=0.7, linewidths=0, zorder=3)
    b.set_xlim(20, 112); b.set_ylim(20, 112); b.set_xticks([30, 50, 75, 100]); b.set_yticks([30, 50, 75, 100]); b.grid(True)
    b.set_xlabel("tape measure [cm]"); b.set_ylabel("from the pose [cm]"); ttl(b, "(b) Distance", x=-0.42)
    ok = np.isfinite(et)
    c.plot([0, 96], [0, 96], color="k", lw=0.5, ls="--"); c.scatter(tt[ok], et[ok], s=7, color=CL, alpha=0.7, linewidths=0, zorder=3)
    c.set_xlim(-4, 96); c.set_ylim(-4, 96); c.set_xticks([0, 30, 60, 90]); c.set_yticks([0, 30, 60, 90]); c.grid(True)
    c.set_xlabel("protractor [deg]"); c.set_ylabel("from the pose [deg]"); ttl(c, "(c) Tilt", x=-0.42)
    print("     T2 cross-check: distance max |diff| %.1f cm (mean %+.1f), tilt max |diff| %.1f deg (mean %+.1f)" % (
        np.nanmax(np.abs(ed - td)), np.nanmean(ed - td), np.nanmax(np.abs(et - tt)), np.nanmean(et - tt)))
    save(fig, "fig_t2")


# ======================================================================================= Fig. T3 checks
def fig_t3():
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 1.9), gridspec_kw=dict(width_ratios=[1, 1.3], wspace=0.3))
    full(fig)
    gap = [50, 100, 150, 200, 250]
    bias = np.array([[-1.7, 1.5, 3.0], [-2.0, 5.4, 9.0], [0.5, 1.2, 5.9], [-3.5, 1.7, 7.9], [11.1, 1.9, 7.3]])
    x = np.arange(5)
    for k, (c, lab) in enumerate(zip((XC, YC, ZC), ("x, along the markers", "y, height", "z, depth"))):
        a.bar(x + (k - 1) * 0.26, bias[:, k], 0.25, color=c, label=lab)
    a.axhline(0, color="k", lw=0.5); a.set_xticks(x); a.set_xticklabels([f"{g}" for g in gap]); a.set_xlabel("gap between the black squares [mm]")
    a.set_ylabel("camera $-$ ruler [mm]"); a.grid(True, axis="y"); a.set_ylim(-5.5, 19)
    a.legend(loc="upper left", ncol=3, handlelength=0.8, borderaxespad=0.45, columnspacing=0.7, handletextpad=0.3, fontsize=6.2)
    ttl(a, "(a) Check A: bias at five positions", x=-0.14)
    r = rows(os.path.join(RES, "pose", "invariance_1_poses.csv"))
    t = np.array([fnum(q["t_s"]) for q in r])
    CF = np.array([[fnum(q[f"CF_{c}_mm"]) for c in "xyz"] for q in r]); FM = np.array([[fnum(q[f"FM_{c}_mm"]) for c in "xyz"] for q in r])
    ok = np.isfinite(CF).all(1) & np.isfinite(FM).all(1)
    cam = np.linalg.norm(CF - np.nanmedian(CF[:30], axis=0), axis=1); rel = np.linalg.norm(FM - np.nanmean(FM[ok], axis=0), axis=1)
    b.plot(t[ok], cam[ok], color=GREY, lw=1.0, label="camera displacement"); b.plot(t[ok], rel[ok], color=CL, lw=0.8, label="deviation of the relative position")
    b.set_xlabel("time [s]"); b.set_ylabel("displacement [mm]"); b.grid(True); b.set_xlim(0, t[ok].max()); b.set_ylim(0, 170)
    b.legend(loc="upper left", borderaxespad=0.45, labelspacing=0.15, ncol=2, columnspacing=0.8, handlelength=1.2, fontsize=6.2)
    ttl(b, "(b) Check B: the camera moves, the relative pose stays", x=-0.1)
    save(fig, "fig_t3")


# ======================================================================================= Fig. T4 trajectory
def fig_t4():
    a = np.loadtxt(os.path.join(DAT, "T4_motion_1_tracks.csv"), delimiter=",", skiprows=1)
    t, c, n = a[:, 0], a[:, 1:7], a[:, 7:13].copy()
    n[~np.isfinite(c[:, 0])] = np.nan                                 # the network always answers: shown only where M was detected
    fig = plt.figure(figsize=(W, 2.1))
    ax = fig.add_axes([0.0, 0.06, 0.33, 0.84], projection="3d")
    ax.plot(c[:, 0], c[:, 2], c[:, 1], color=CL, lw=0.7); ax.plot(n[:, 0], n[:, 2], n[:, 1], color=NET, lw=0.5, alpha=0.85)
    for v_, col in (((70, 0, 0), XC), ((0, 0, 70), YC), ((0, 70, 0), ZC)):
        ax.plot([0, v_[0]], [0, v_[1]], [0, v_[2]], color=col, lw=1.6)
    ax.text(0, 0, 82, "F", fontsize=7, ha="center")
    ax.set_xlabel("x [mm]", labelpad=-6, fontsize=6.8); ax.set_ylabel("z [mm]", labelpad=-6, fontsize=6.8); ax.set_zlabel("y [mm]", labelpad=-8, fontsize=6.8)
    ax.tick_params(labelsize=6.0, pad=-4); ax.view_init(20, -64); ax.set_box_aspect((1.6, 1, 1))
    ax.set_xticks([-400, -200, 0]); ax.set_yticks([0, 100]); ax.set_zticks([0, 50, 100])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.03); axis._axinfo["grid"]["linewidth"] = 0.3
    fig.text(0.0, 0.94, "(a) Path of M in the frame of F", fontsize=TS, fontweight="bold")
    a1, a2 = fig.add_axes([0.43, 0.55, 0.57, 0.33]), fig.add_axes([0.43, 0.13, 0.57, 0.33])
    for k, (col, lab) in enumerate(zip((XC, YC, ZC), ("x", "y", "z"))):
        a1.plot(t, c[:, k], color=col, lw=0.9, label=lab); a1.plot(t, n[:, k], color=col, lw=0.6, ls=(0, (2, 1.2)))
    for k, (col, lab) in enumerate(zip((XC, YC, ZC), ("roll", "pitch", "yaw"))):
        a2.plot(t, c[:, 3 + k], color=col, lw=0.9, label=lab); a2.plot(t, n[:, 3 + k], color=col, lw=0.6, ls=(0, (2, 1.2)))
    a1.set_ylabel("position [mm]"); a2.set_ylabel("angle [deg]"); a2.set_xlabel("time [s]"); a1.set_ylim(-470, 260); a2.set_ylim(-17, 68)
    a1.set_xticklabels([]); a1.grid(True); a2.grid(True); a1.set_xlim(0, t.max()); a2.set_xlim(0, t.max())
    a1.legend(ncol=3, loc="upper left", borderaxespad=0.45, columnspacing=0.8, handlelength=1.2)
    a2.legend(ncol=3, loc="upper left", borderaxespad=0.45, columnspacing=0.8, handlelength=1.2)
    fig.text(0.37, 0.94, "(b) Translation and roll-pitch-yaw (solid: classical, dashed: network)", fontsize=TS, fontweight="bold")
    save(fig, "fig_t4")


# ======================================================================================= Fig. training data
def fig_data():
    fig = plt.figure(figsize=(W, 2.2))
    fig.text(0.0, 0.955, "(a) Synthetic whole images, 640 × 360 (stage 1)", fontsize=TS, fontweight="bold")
    for j, k in enumerate((0, 1, 3)):
        ax = fig.add_axes([j * 0.337, 0.50, 0.326, 0.42]); show(ax, os.path.join(PH, f"synth_full_{k}.png"))
    fig.text(0.0, 0.40, "(b) Synthetic crops (stage 2)", fontsize=TS, fontweight="bold")
    fig.text(0.337, 0.40, "(c) Real crop of M and the degradations of the hard-case test", fontsize=TS, fontweight="bold")
    names = [("synth_crop_0", ""), ("synth_crop_1", ""), ("synth_crop_4", ""), ("hard_clean", "clean"), ("hard_cover30", "30 % covered"),
             ("hard_blur31", "blur 31 px"), ("hard_res6", "resolution / 6"), ("hard_light025", "light × 0.25"), ("hard_light010", "light × 0.10")]
    w = 0.104
    for j, (n, t) in enumerate(names):
        ax = fig.add_axes([j * 0.112, 0.07, w, w * W / 2.2]); show(ax, os.path.join(PH, n + ".png"), t or None)
    save(fig, "fig_data")


# ======================================================================================= Fig. synthetic training curves
def fig_synth_train():
    fig, axs = plt.subplots(1, 4, figsize=(W, 1.75), gridspec_kw=dict(wspace=0.62))
    full(fig)
    runs = [("resnet18_v1", "whole image, 15 epochs", "#9ab0c9", "-"), ("resnet18_v2", "whole image, 40 epochs", CL, "-"),
            ("resnet18_v3", r"whole image, 40 epochs, $w_R$ = 2", "#5b8a3a", "-"), ("crop_resnet18_v1", "crop stage, 80 epochs", NET, "-")]
    for name, lab, col, ls in runs:
        log = train_log(name)["log"]; ep = [x["epoch"] for x in log]
        axs[0].plot(ep, [x["rot_loss"] for x in log], color=col, ls=ls, label=lab)
        axs[1].plot(ep, [x["trans_loss"] for x in log], color=col, ls=ls)
        axs[2].plot(ep, [x["val_rot_med"] for x in log], color=col, ls=ls); axs[2].plot(ep, [x["val_rot_p95"] for x in log], color=col, ls=(0, (1, 1.2)), lw=0.7)
        axs[3].plot(ep, [x["val_trans_med"] for x in log], color=col, ls=ls); axs[3].plot(ep, [x["val_trans_p95"] for x in log], color=col, ls=(0, (1, 1.2)), lw=0.7)
    for ax, yl, tt in zip(axs, ("rotation loss", "translation loss", "$e_R$ [deg]", "$e_t$ [mm]"),
                          ("(a) Train: rotation", "(b) Train: position", "(c) Valid.: rotation", "(d) Valid.: position")):
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel(yl); ax.grid(True); ttl(ax, tt, x=-0.36); ax.set_xlim(0, 82); ax.minorticks_off()
        ax.set_xticks([0, 20, 40, 60, 80])
    axs[0].set_yticks([0.001, 0.01, 0.1, 1]); axs[0].set_yticklabels(["0.001", "0.01", "0.1", "1"]); axs[0].set_ylim(0.0008, 12)
    axs[1].set_yticks([0.02, 0.05, 0.1, 0.2, 0.5]); axs[1].set_yticklabels(["0.02", "0.05", "0.1", "0.2", "0.5"]); axs[1].set_ylim(0.015, 0.8)
    axs[2].set_yticks([2, 5, 10, 20, 50, 100]); axs[2].set_yticklabels(["2", "5", "10", "20", "50", "100"]); axs[2].set_ylim(1.5, 240)
    axs[3].set_yticks([5, 10, 20, 50, 100, 200, 500]); axs[3].set_yticklabels(["5", "10", "20", "50", "100", "200", "500"]); axs[3].set_ylim(4.5, 1500)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.27), columnspacing=1.2, handlelength=1.6, fontsize=6.5)
    save(fig, "fig_synth_train")


# ======================================================================================= Fig. fine-tuning curves
def fig_finetune():
    fig, axs = plt.subplots(2, 3, figsize=(W, 3.05), gridspec_kw=dict(wspace=0.4, hspace=0.5))
    full(fig)
    rounds = [("_ft", "round 1: 3 164 frames, 15 epochs", "#9aa5b1"), ("_ft2", "round 2: 7 358 frames, 25 epochs", "#e3a46c"), ("_ft3", "round 3: 11 562 frames, 25 / 50 epochs", NET)]
    for i, (stage, st_name) in enumerate((("resnet18_v2", "Stage 1"), ("crop_resnet18_v1", "Stage 2"))):
        for suf, lab, col in rounds:
            d = train_log(stage + suf); log = d["log"]; ep = np.array([x["epoch"] for x in log])
            best = d["config"].get("best_epoch", len(log))
            for j, key in enumerate(("loss", "val_rot_med", "val_trans_med")):
                y = np.array([x[key] for x in log])
                axs[i, j].plot(ep, y, color=col, lw=1.0, label=lab)
                if j > 0:
                    axs[i, j].plot(best, y[best - 1], marker="*", ms=6, color=col, mec="k", mew=0.4, zorder=5)
        for j, (yl, tt) in enumerate((("training loss", "training loss"), ("validation $e_R$ [deg]", "validation rotation"), ("validation $e_t$ [mm]", "validation position"))):
            ax = axs[i, j]; ax.set_xlabel("epoch"); ax.set_ylabel(yl); ax.grid(True, which="both"); ax.set_yscale("log")
            ttl(ax, f"({'abcdef'[3 * i + j]}) {st_name}: {tt}", x=-0.24)
    axs[0, 0].set_yticks([0.05, 0.1, 0.2, 0.5]); axs[0, 0].set_yticklabels(["0.05", "0.1", "0.2", "0.5"]); axs[0, 0].set_ylim(0.04, 0.8)
    axs[1, 0].set_yticks([0.02, 0.05, 0.1, 0.2]); axs[1, 0].set_yticklabels(["0.02", "0.05", "0.1", "0.2"]); axs[1, 0].set_ylim(0.017, 0.22)
    axs[0, 1].set_yticks([5, 7, 10, 15]); axs[0, 1].set_yticklabels(["5", "7", "10", "15"]); axs[0, 1].set_ylim(4.5, 16)
    axs[1, 1].set_yticks([3, 4, 6, 8, 12]); axs[1, 1].set_yticklabels(["3", "4", "6", "8", "12"]); axs[1, 1].set_ylim(3, 13)
    axs[0, 2].set_yticks([10, 20, 50, 100]); axs[0, 2].set_yticklabels(["10", "20", "50", "100"]); axs[0, 2].set_ylim(9, 130)
    axs[1, 2].set_yticks([5, 7, 10, 15, 20]); axs[1, 2].set_yticklabels(["5", "7", "10", "15", "20"]); axs[1, 2].set_ylim(5, 22)
    for ax in axs.ravel():
        ax.minorticks_off()
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), columnspacing=1.2, handlelength=1.6, fontsize=6.5)
    save(fig, "fig_finetune")


# ======================================================================================= Fig. what each change bought
def fig_rounds():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(W, 1.95), gridspec_kw=dict(width_ratios=[1.05, 0.72, 1.6], wspace=0.5))
    full(fig)
    lab = ["synthetic only", "round 1", "round 2", "round 3", "classical"]
    med, p95, jit = [26.5, 10.0, 9.2, 6.7, 8.8], [106.5, 124.3, 17.8, 12.5, 13.6], [26.5, 16.3, 2.12, 1.26, 0.38]
    x = np.arange(5)
    a.bar(x, med, 0.62, color=[NET] * 4 + [CL], zorder=2, label="median")
    a.scatter(x, p95, marker="_", s=90, color="k", lw=1.1, zorder=3, label="95th percentile")
    a.scatter(x, jit, marker="D", s=11, color="white", edgecolor="k", lw=0.7, zorder=4, label="jitter")
    a.set_yscale("log"); a.set_ylim(0.25, 900); a.set_yticks([0.3, 1, 3, 10, 30, 100]); a.set_yticklabels(["0.3", "1", "3", "10", "30", "100"]); a.minorticks_off()
    a.set_xticks(x); a.set_xticklabels(lab, fontsize=6.6, rotation=38, ha="right", rotation_mode="anchor"); a.set_ylabel("$e_t$ on E1 P1–P5 [mm]"); a.grid(True, axis="y")
    from matplotlib.lines import Line2D
    a.legend([Line2D([], [], color="k", marker="_", ls="", ms=9, mew=1.1), Line2D([], [], color="k", marker="D", mfc="white", ls="", ms=3.5, mew=0.7)],
             ["95th percentile", "jitter"], loc="upper right", borderaxespad=0.45, labelspacing=0.15, handletextpad=0.2, fontsize=6.2, ncol=2, columnspacing=0.6)
    ttl(a, "(a) Real test frames", x=-0.3)
    # (b) camera-rotation augmentation: validation on the recording with the tilted screen (stage 1)
    xb = np.array([0, 1])
    b.bar(xb - 0.2, [13.9, 6.6], 0.38, color="#f0b98a", label="rotation [deg]"); b.bar(xb + 0.2, [92, 29], 0.38, color=NET, label="position [mm]")
    for xx, v_ in zip(list(xb - 0.2) + list(xb + 0.2), [13.9, 6.6, 92, 29]):
        b.text(xx, v_ * 1.14, f"{v_:g}", ha="center", fontsize=6.3, bbox=WHITE)
    b.set_yscale("log"); b.set_ylim(3, 600); b.set_yticks([3, 10, 30, 100]); b.set_yticklabels(["3", "10", "30", "100"]); b.minorticks_off()
    b.set_xticks(xb); b.set_xticklabels(["without", "with"], fontsize=6.5); b.set_ylabel("stage-1 validation error"); b.grid(True, axis="y")
    b.legend(loc="upper right", borderaxespad=0.45, labelspacing=0.15, handlelength=0.9, handletextpad=0.3, fontsize=6.2)
    ttl(b, "(b) Rotation augm.", x=-0.42)
    hh = json.load(open(os.path.join(RES, "network", "hard_cases_E1.json")))
    keys = list(hh.keys())
    short = {"clean": "clean", "cover 15 % of M": "cover 15 %", "cover 30 % of M": "cover 30 %", "motion blur 15 px": "blur 15 px",
             "motion blur 31 px": "blur 31 px", "resolution / 4": "resolution / 4", "resolution / 6": "resolution / 6", "light x 0.25": "light x 0.25",
             "light x 0.10": "light x 0.10"}
    x = np.arange(len(keys))
    c.bar(x - 0.27, [hh[k]["classical_usable"] for k in keys], 0.26, color=CL, label="classical")
    c.bar(x, [hh[k]["_ft2"]["usable"] for k in keys], 0.26, color="#f0b98a", label="round 2")
    c.bar(x + 0.27, [hh[k]["_ft3"]["usable"] for k in keys], 0.26, color=NET, label="round 3 (hard cases)")
    c.set_xticks(x); c.set_xticklabels([short[k].replace(" x ", " × ") for k in keys], fontsize=6.3, rotation=38, ha="right", rotation_mode="anchor"); c.set_ylabel("usable poses [%]"); c.set_ylim(0, 132); c.set_yticks([0, 25, 50, 75, 100])
    c.grid(True, axis="y"); c.legend(loc="upper center", ncol=3, borderaxespad=0.45, columnspacing=0.7, handlelength=0.9, handletextpad=0.3, fontsize=6.2)
    ttl(c, "(c) Frames degraded in software", x=-0.14)
    save(fig, "fig_rounds")


# ======================================================================================= Fig. E1 and E2
def fig_e1_e2():
    E = test_errors()
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(W, 1.95), gridspec_kw=dict(width_ratios=[1.6, 1, 1], wspace=0.4))
    full(fig)
    names = ["E1_p4", "E1_p3", "E1_p2", "E1_p1", "E1_p5", "E1_turn20", "E1_turn40"]
    lab = ["P4\n50", "P3\n100", "P2\n150", "P1\n200", "P5\n250", "turn\n20°", "turn\n40°"]
    x = np.arange(len(names))
    for k, (col, off, lb) in enumerate(((CL, -0.2, "classical"), (NET, 0.2, "network"))):
        med = np.array([np.nanmedian(E[n][:, 2 * k]) for n in names])
        lo = med - np.array([np.nanpercentile(E[n][:, 2 * k], 5) for n in names]); hi = np.array([np.nanpercentile(E[n][:, 2 * k], 95) for n in names]) - med
        a.bar(x + off, med, 0.38, color=col, label=lb, yerr=[lo, hi], error_kw=dict(lw=0.6, capsize=1.5, capthick=0.6, ecolor="k"))
    a.axvline(4.5, color=GREY, lw=0.5, ls=":"); a.text(4.62, 21.6, "recorded after\nthe network\nwas final", fontsize=6.2, color=GREY, va="top", ha="left", linespacing=1.1, bbox=WHITE)
    a.set_xticks(x); a.set_xticklabels(lab, fontsize=6.2); a.set_ylabel("$e_t$ [mm]  (median, 5–95 %)"); a.set_ylim(0, 22); a.grid(True, axis="y")
    a.legend(loc="upper left", ncol=2, borderaxespad=0.45, columnspacing=0.8, handlelength=1.0)
    ttl(a, "(a) E1: positions (gap in mm) and angles", x=-0.14)
    d = [40.9, 61.4, 76.7]
    for ax, col_i, yl in ((b, 0, "$e_t$ median [mm]"), (c, 1, "$e_R$ median [deg]")):
        for k, (col, lb) in enumerate(((CL, "classical"), (NET, "network"))):
            for tl, ls, mk in ((0, "-", "o"), (40, (0, (3, 1.5)), "s")):
                v = [np.nanmedian(E[f"E2_d{dd}_t{tl}"][:, 2 * k + col_i]) for dd in (40, 60, 75)]
                ax.plot(d, v, color=col, ls=ls, marker=mk, ms=3.2, mfc=col if tl == 0 else "white", lw=1.0, label=f"{lb}, {tl}°")
        ax.set_xlabel("camera to F [cm]"); ax.set_ylabel(yl); ax.grid(True); ax.set_xticks([41, 61, 77]); ax.set_xlim(36, 82)
    b.set_ylim(0, 34); c.set_ylim(0, 15.5)
    b.legend(loc="upper left", borderaxespad=0.45, labelspacing=0.12, handlelength=1.8, fontsize=6.3, ncol=1)
    ttl(b, "(b) E2: position", x=-0.3); ttl(c, "(c) E2: rotation", x=-0.3)
    save(fig, "fig_e1_e2")


# ======================================================================================= Fig. error distributions, bias and jitter
def fig_cdf():
    E = test_errors()
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(W, 1.95), gridspec_kw=dict(wspace=0.4))
    full(fig)
    groups = [("E1", "-"), ("E2", (0, (4, 1.5))), ("E3", (0, (1, 1.2))), ("E4", (0, (4, 1.2, 1, 1.2)))]
    for ax, col_i, xl, xmax in ((a, 0, "$e_t$ [mm]", 40), (b, 1, "$e_R$ [deg]", 14)):
        for k, col in enumerate((CL, NET)):
            for g, ls in groups:
                v = np.concatenate([E[n][:, 2 * k + col_i] for n in E if n.startswith(g)])
                v = np.sort(v[np.isfinite(v)])
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, ls=ls, lw=0.9, label=f"{'classical' if k == 0 else 'network'}, {g}")
        ax.set_xlim(0, xmax); ax.set_ylim(0, 1.02); ax.set_xlabel(xl); ax.set_ylabel("fraction of frames"); ax.grid(True)
    a.legend(loc="lower right", borderaxespad=0.45, labelspacing=0.1, handlelength=2.1, fontsize=6.2, ncol=1)
    ttl(a, "(a) Position error, all frames", x=-0.24); ttl(b, "(b) Rotation error, all frames", x=-0.24)
    for k, (col, lb, mk) in enumerate(((CL, "classical", "o"), (NET, "network", "s"))):
        bias, jit = [], []
        for n in E:
            if n.startswith("E3"):
                continue
            p = E[n][:, 4 + 3 * k:7 + 3 * k]; p = p[np.isfinite(p).all(1)]
            if len(p) < 10:
                continue
            tr = rows(os.path.join(ROOT, "data", "recordings", n + ".truth.csv"))[0]
            truth = np.array([fnum(tr["x_mm"]), fnum(tr["y_mm"]), fnum(tr["z_mm"])])
            bias.append(np.linalg.norm(p.mean(0) - truth)); jit.append(np.sqrt(np.mean(np.sum((p - p.mean(0)) ** 2, axis=1))))
        c.scatter(jit, bias, s=11, color=col, marker=mk, label=lb, alpha=0.85, linewidths=0, zorder=3)
        print("     %s: bias median %.1f mm, jitter median %.2f mm (static recordings: %d)" % (lb, np.median(bias), np.median(jit), len(bias)))
    c.plot([0.1, 40], [0.1, 40], color="k", lw=0.5, ls="--")
    c.set_xscale("log"); c.set_yscale("log"); c.set_xlim(0.1, 40); c.set_ylim(1, 90); c.set_xlabel("jitter [mm]"); c.set_ylabel("bias [mm]"); c.grid(True, which="both")
    c.set_xticks([0.1, 1, 10]); c.set_xticklabels(["0.1", "1", "10"]); c.set_yticks([1, 3, 10, 30]); c.set_yticklabels(["1", "3", "10", "30"])
    c.legend(loc="upper left", borderaxespad=0.45, labelspacing=0.15, handletextpad=0.2, ncol=2, columnspacing=0.6)
    ttl(c, "(c) Bias against jitter, static", x=-0.24)
    save(fig, "fig_cdf")


# ======================================================================================= Fig. E3
def fig_e3():
    a = np.loadtxt(os.path.join(DAT, "E3_motion_01_errors.csv"), delimiter=",", skiprows=1)
    t = a[:, 0]
    fig, (p, q, r) = plt.subplots(1, 3, figsize=(W, 1.85), gridspec_kw=dict(wspace=0.42))
    full(fig)
    p.plot(t, a[:, 1], color="k", lw=1.3, label="truth (guide)"); p.plot(t, a[:, 2], color=CL, lw=0.8, label="classical"); p.plot(t, a[:, 3], color=NET, lw=0.8, label="network")
    p.set_ylabel("x of M in F [mm]"); p.set_ylim(-212, -98); p.legend(loc="upper center", ncol=1, borderaxespad=0.45, labelspacing=0.12, handlelength=1.4, handletextpad=0.4, fontsize=6.2)
    q.plot(t, a[:, 4], color=CL, lw=0.7, label="classical"); q.plot(t, a[:, 6], color=NET, lw=0.7, label="network"); q.set_ylabel("$e_t$ [mm]"); q.set_ylim(0, 68)
    r.plot(t, a[:, 5], color=CL, lw=0.7, label="classical"); r.plot(t, a[:, 7], color=NET, lw=0.7, label="network"); r.set_ylabel("$e_R$ [deg]"); r.set_ylim(0, 19)
    k = int(np.nanargmax(a[:, 4]))
    q.annotate("one frame: outline\nof M read 9 %\ntoo small", (t[k] + 0.15, a[k, 4] - 2), (t[k] + 2.6, 38), fontsize=6.2, color=GREY, va="center", linespacing=1.1, bbox=WHITE,
               arrowprops=dict(arrowstyle="-", lw=0.4, color=GREY, shrinkA=1, shrinkB=1))
    for ax, s in ((p, "(a) Position along the ruler"), (q, "(b) Position error"), (r, "(c) Rotation error")):
        ax.set_xlabel("time [s]"); ax.grid(True); ax.set_xlim(0, 20); ttl(ax, s, x=-0.24)
    for ax in (q, r):
        ax.legend(loc="upper right", ncol=2, borderaxespad=0.45, columnspacing=0.6, handlelength=1.0, handletextpad=0.3, fontsize=6.2)
    save(fig, "fig_e3")


# ======================================================================================= Fig. E4
def fig_e4():
    E = test_errors()
    fig = plt.figure(figsize=(W, 1.9))
    fig.text(0.0, 0.94, "(a) Classical (green) and network (orange) outline", fontsize=TS, fontweight="bold")
    wi = 0.142
    for j, (n, t) in enumerate((("E4_dim", "dim light"), ("E4_cover", "edge of M covered"), ("E4_blur_2", "camera shaken"))):
        ax = fig.add_axes([j * 0.152, 0.33, wi, wi * W / 1.9]); show(ax, os.path.join(PH, f"view_{n}_both.png"), t)
    names, lab = ["E4_dim", "E4_clutter", "E4_blur", "E4_blur_2", "E4_cover"], ["dim light", "clutter", "mild shake", "strong shake", "edge covered"]
    x = np.arange(5)
    a = fig.add_axes([0.535, 0.27, 0.19, 0.59]); b = fig.add_axes([0.81, 0.27, 0.19, 0.59])
    for k, (col, off, lb) in enumerate(((CL, -0.2, "classical"), (NET, 0.2, "network"))):
        outp = [100 * np.isfinite(E[n][:, 2 * k]).mean() for n in names]
        use = [100 * np.mean((E[n][:, 2 * k] < 20) & (E[n][:, 2 * k + 1] < 10)) for n in names]
        a.bar(x + off, outp, 0.38, color=col, alpha=0.3); a.bar(x + off, use, 0.38, color=col, label=lb)
        med = [np.nanmedian(E[n][:, 2 * k]) if np.isfinite(E[n][:, 2 * k]).any() else 0 for n in names]
        b.bar(x + off, med, 0.38, color=col, label=lb)
        print("     E4", lb, "output", np.round(outp, 1), "usable", np.round(use, 1), "median", np.round(med, 1))
    b.text(4 - 0.2, 1.0, "none", fontsize=6.2, ha="center", color=CL, rotation=90, va="bottom")
    for ax in (a, b):
        ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=6.3, rotation=38, ha="right", rotation_mode="anchor"); ax.grid(True, axis="y")
    a.set_ylabel("frames [%]"); a.set_ylim(0, 132); a.set_yticks([0, 25, 50, 75, 100]); b.set_ylabel("$e_t$ median [mm]"); b.set_ylim(0, 38)
    a.legend(loc="upper center", ncol=2, borderaxespad=0.45, columnspacing=0.8, handlelength=1.0, fontsize=6.3)
    fig.text(0.47, 0.94, "(b) Pose given; usable (dark)", fontsize=TS, fontweight="bold")
    fig.text(0.775, 0.94, "(c) Position error", fontsize=TS, fontweight="bold")
    save(fig, "fig_e4")


# ======================================================================================= Fig. bonus
def fig_bonus():
    fig = plt.figure(figsize=(W, 2.7))
    fig.text(0.0, 0.955, "(a) One pass behind the box", fontsize=TS, fontweight="bold")
    caps = ("1  M measured (green)", "2  hidden for 0.4 s: predicted (red)", "3  hidden for 1.2 s: predicted (red)", "4  M measured again")
    hh, gap = 0.158, 0.064
    for i, cap in enumerate(caps):
        y = 0.905 - (i + 1) * (hh + gap) + gap - 0.005
        ax = fig.add_axes([0.0, y, 0.30, hh]); show(ax, os.path.join(PH, f"bonus_{i}.png"))
        ax.set_anchor("W")
        fig.text(0.0, y + hh + 0.006, cap, fontsize=6.6, va="bottom")
    v = rows(os.path.join(RES, "bonus", "virtual_occlusion.csv"))
    a = fig.add_axes([0.385, 0.14, 0.255, 0.74])
    T = [0.25, 0.5, 1.0, 1.5, 2.0]
    for rec, ls, mk, lb in (("E7_nocover_slow_2", "-", "o", "7 cm/s"), ("E7_nocover_fast", (0, (3, 1.5)), "s", "17 cm/s")):
        rr = [x for x in v if x["recording"] == rec]
        for kind, col in (("pred", PRED), ("hold", HOLD)):
            m = []
            for tt in T:
                vals = np.array([fnum(x[f"{kind}_mm_{tt:g}s"]) for x in rr])
                m.append(np.nanmedian(vals) if np.isfinite(vals).sum() >= max(2, len(rr) // 2) else np.nan)
            a.plot(T, m, color=col, ls=ls, marker=mk, ms=3.2, mfc=col if ls == "-" else "white", lw=1.0,
                   label=f"{'predictor' if kind == 'pred' else 'hold'}, {lb}")
    a.axhline(20, color="k", lw=0.4, ls=":"); a.text(2.05, 23, "20 mm", fontsize=6.2, ha="right", bbox=WHITE)
    a.set_xlabel("time hidden [s]"); a.set_ylabel("$e_t$ median [mm]"); a.grid(True); a.set_ylim(0, 285); a.set_xlim(0.15, 2.1); a.set_yticks([0, 50, 100, 150, 200, 250])
    a.legend(loc="upper left", borderaxespad=0.45, labelspacing=0.12, handlelength=1.9, fontsize=6.2, ncol=1)
    fig.text(0.335, 0.955, "(b) Hidden in software", fontsize=TS, fontweight="bold")
    r = rows(os.path.join(RES, "bonus", "real_occlusion.csv"))
    b = fig.add_axes([0.745, 0.14, 0.255, 0.74])
    sp = np.array([fnum(x["speed_hidden_mm_s"]) for x in r]); pr = np.array([fnum(x["pred_mm"]) for x in r]); ho = np.array([fnum(x["hold_mm"]) for x in r])
    groups = [(sp < 75, "#8fc98b", "4–6 cm/s", "v"), ((sp >= 75) & (sp < 125), PRED, "8–12 cm/s", "o"), (sp >= 125, "#173f1a", "13–23 cm/s", "s")]
    for m, col, lb, mk in groups:
        b.scatter(ho[m], pr[m], s=13, color=col, marker=mk, label=f"{lb} (n = {int(m.sum())})", zorder=3, edgecolor="none")
    b.plot([0, 320], [0, 320], color="k", lw=0.5, ls="--")
    b.set_xlim(120, 215); b.set_ylim(0, 320); b.set_xlabel("hold: $e_t$ at reappearance [mm]"); b.set_ylabel("predictor: $e_t$ at reappearance [mm]")
    b.grid(True); b.legend(loc="upper left", borderaxespad=0.45, labelspacing=0.12, handletextpad=0.1, fontsize=6.2)
    fig.text(0.695, 0.955, "(c) Real obstacle, 49 passes", fontsize=TS, fontweight="bold")
    print("     bonus: predictor better in", int(np.sum(pr < ho)), "of", len(pr), "| groups", [int(m.sum()) for m, *_ in groups],
          "| medians pred/hold", [(round(float(np.median(pr[m])), 1), round(float(np.median(ho[m])), 1)) for m, *_ in groups])
    save(fig, "fig_bonus")


# ======================================================================================= Fig. the white-margin failure
def fig_margin():
    fig = plt.figure(figsize=(W, 1.7))
    fig.text(0.0, 0.93, "(a) The cause: M has almost no white margin at its sides", fontsize=TS, fontweight="bold")
    a = fig.add_axes([0.045, 0.13, 0.16, 0.72]); show(a, os.path.join(PH, "margin_M.png"), "M, on the black board")
    for xy, xt in (((0.03, 0.5), (-0.26, 0.5)), ((0.97, 0.5), (1.26, 0.5))):
        a.annotate("", xy, xt, xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", lw=0.9, color=RED, mutation_scale=7))
    b = fig.add_axes([0.275, 0.13, 0.18, 0.72]); show(b, os.path.join(PH, "margin_F.png"), "F, full quiet zone")
    lab = ["before", "after"]
    spec = (((1.09, 1.01), "width / height of M", "(b) Shape", (0.95, 1.13), 0.555), ((18, 2), "false lean of M [deg]", "(c) Lean", (0, 22), 0.71),
            ((3.5, 0.7), "jitter [mm]", "(d) Jitter", (0, 4.3), 0.865))
    for vals, yl, tt, ylim, x0 in spec:
        ax = fig.add_axes([x0 + 0.045, 0.22, 0.09, 0.58])
        ax.bar([0, 1], vals, 0.6, color=[RED, PRED])
        for x, v_ in zip((0, 1), vals):
            ax.text(x, v_ + (ylim[1] - ylim[0]) * 0.03, "3–4" if v_ == 3.5 else f"{v_:g}", ha="center", fontsize=6.3, bbox=WHITE)
        ax.set_xticks([0, 1]); ax.set_xticklabels(lab, fontsize=6.2, linespacing=1.0); ax.set_ylabel(yl); ax.set_ylim(*ylim); ax.grid(True, axis="y"); ax.set_xlim(-0.6, 1.6)
        fig.text(x0, 0.93, tt, fontsize=TS, fontweight="bold")
    save(fig, "fig_margin")


# ======================================================================================= Fig. network architecture
def fig_arch():
    """ResNet-18 as used here: the blocks, the size of their outputs for both stages, the head and what the nine outputs mean."""
    fig, ax = plt.subplots(figsize=(W, 2.95))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.set_xlim(0, 100); ax.set_ylim(0, 47); ax.axis("off")
    FS, nb = 6.3, "#fbe6d3"
    texts = []                                                         # (text artist, box) pairs, checked below: the text must fit its box

    def box(x, y, w, h, text, fc=nb, fs=FS):
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="k", lw=0.6, zorder=2))
        texts.append((ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.15), (x, y, w, h)))

    H = 13.0
    cols = [("input", 7.0, "image", "#ffffff"), ("stem", 10.6, "conv 7 × 7, 64\nstride 2\nBN, ReLU\nmax pool 3 × 3\nstride 2", nb),
            ("layer 1", 8.8, "2 basic\nblocks\n64 channels", nb), ("layer 2", 8.8, "2 basic\nblocks\n128 channels\nstride 2", nb),
            ("layer 3", 8.8, "2 basic\nblocks\n256 channels\nstride 2", nb), ("layer 4", 8.8, "2 basic\nblocks\n512 channels\nstride 2", nb),
            ("pool", 7.2, "global\naverage\npooling", nb), ("head", 7.2, "linear\n512 to 9", nb)]
    s1 = ["1 × 360 × 640", "64 × 90 × 160", "64 × 90 × 160", "128 × 45 × 80", "256 × 23 × 40", "512 × 12 × 20", "512", "9"]
    s2 = ["3 × 224 × 224", "64 × 56 × 56", "64 × 56 × 56", "128 × 28 × 28", "256 × 14 × 14", "512 × 7 × 7", "512", "9"]
    x, yc, gap, xs, ytitle = 9.8, 33.5, 1.1, [], 33.5 + 9.6
    for name, w, text, fc in cols:
        box(x, yc - H / 2, w, H, text, fc)
        ax.text(x + w / 2, ytitle, name, ha="center", va="bottom", fontsize=6.9, fontweight="bold")
        xs.append((x, w))
        x += w + gap
    for (xa, wa), (xb, wb) in zip(xs[:-1], xs[1:]):
        arrow(ax, (xa + wa + 0.1, yc), (xb - 0.1, yc), "k", 0.7, ms=5)
    xo, wo = xs[-1][0] + xs[-1][1] + 2.4, 12.0                         # the nine outputs
    box(xo, yc + 0.7, wo, 8.4, "6 numbers:\ntwo vectors,\northonormalised\nto R", "#ffffff", 6.0)
    box(xo, yc - 9.1, wo, 8.4, "3 numbers:\nx/Z, y/Z, log Z\ngive t", "#ffffff", 6.0)
    xh = xs[-1][0] + xs[-1][1]
    arrow(ax, (xh + 0.1, yc + 1.0), (xo - 0.1, yc + 4.9), "k", 0.7, ms=5)
    arrow(ax, (xh + 0.1, yc - 1.0), (xo - 0.1, yc - 4.9), "k", 0.7, ms=5)
    ax.text(xo + wo / 2, ytitle, "outputs", ha="center", va="bottom", fontsize=6.9, fontweight="bold")
    # size of the output of every block, for both stages
    ax.plot([0.3, 99.7], [22.4, 22.4], color="#888888", lw=0.4); ax.plot([0.3, 99.7], [12.2, 12.2], color="#888888", lw=0.4)
    ax.text(0.3, 20.5, "size of the output of each block (channels × height × width)", fontsize=6.3, style="italic", va="center")
    for y, lab, sizes in ((17.3, "stage 1", s1), (14.2, "stage 2", s2)):
        ax.text(0.3, y, lab, fontsize=6.6, va="center", fontweight="bold")
        for (xa, wa), t in zip(xs, sizes):
            ax.text(xa + wa / 2, y, t, fontsize=6.1, ha="center", va="center")
    # a basic block
    ax.text(0.3, 4.6, "basic block", fontsize=6.9, fontweight="bold", va="center")
    names, w, x0, yb, hb = ["conv 3 × 3", "BN", "ReLU", "conv 3 × 3", "BN", "add", "ReLU"], [9.8, 5.2, 6.4, 9.8, 5.2, 5.4, 6.4], 18.0, 2.0, 5.2
    pos, x = [], x0
    for n_, w_ in zip(names, w):
        box(x, yb, w_, hb, n_, "#ffffff" if n_ != "add" else "#e8e8e8", 6.2)
        pos.append((x, w_))
        x += w_ + 2.3
    ax.text(x0 - 3.6, yb + hb / 2, "in", fontsize=6.4, ha="right", va="center")
    arrow(ax, (x0 - 3.2, yb + hb / 2), (x0 - 0.1, yb + hb / 2), "k", 0.7, ms=5)
    for (xa, wa), (xb_, wb_) in zip(pos[:-1], pos[1:]):
        arrow(ax, (xa + wa + 0.1, yb + hb / 2), (xb_ - 0.1, yb + hb / 2), "k", 0.7, ms=5)
    xe = pos[-1][0] + pos[-1][1]
    arrow(ax, (xe + 0.1, yb + hb / 2), (xe + 3.2, yb + hb / 2), "k", 0.7, ms=5)
    ax.text(xe + 3.7, yb + hb / 2, "out", fontsize=6.4, va="center")
    xa_ = pos[5][0] + pos[5][1] / 2                                    # the skip connection: the input goes straight to the addition
    ax.plot([x0 - 1.7, x0 - 1.7, xa_], [yb + hb / 2, yb + hb + 2.0, yb + hb + 2.0], color="k", lw=0.7, zorder=1)
    arrow(ax, (xa_, yb + hb + 2.0), (xa_, yb + hb + 0.1), "k", 0.7, ms=5)
    ax.text((x0 - 1.7 + xa_) / 2, yb + hb + 2.5, "skip connection (a 1 × 1 convolution with stride 2 where the size changes)", fontsize=6.0, ha="center", va="bottom")
    fig.canvas.draw()                                                  # every text must lie inside its box, with a margin
    r, inv, bad = fig.canvas.get_renderer(), ax.transData.inverted(), []
    for t_, (bx, by, bw, bh) in texts:
        e = t_.get_window_extent(r)
        (x0_, y0_), (x1_, y1_) = inv.transform((e.x0, e.y0)), inv.transform((e.x1, e.y1))
        if x0_ < bx + 0.25 or x1_ > bx + bw - 0.25 or y0_ < by + 0.25 or y1_ > by + bh - 0.25:
            bad.append(t_.get_text().replace(chr(10), " | "))
    PROBLEMS_BOX["fig_arch"] = bad
    if bad:
        print("      TEXT OUTSIDE ITS BOX:", bad)
    save(fig, "fig_arch", data=False)


FIGS = dict(setup=fig_setup, pipeline=fig_pipeline, arch=fig_arch, calib=fig_calib, t2=fig_t2, t3=fig_t3, t4=fig_t4, data=fig_data, synth=fig_synth_train,
            finetune=fig_finetune, rounds=fig_rounds, e12=fig_e1_e2, cdf=fig_cdf, e3=fig_e3, e4=fig_e4, bonus=fig_bonus, margin=fig_margin)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for k in (sys.argv[1:] or FIGS):
        FIGS[k]()
    bad = {k: v for k, v in list(PROBLEMS.items()) + list(PROBLEMS_BOX.items()) if v}
    print("done ->", OUT, "| overlap check:", "ALL CLEAN" if not bad else f"{len(bad)} figure(s) with overlaps: {', '.join(bad)}")
