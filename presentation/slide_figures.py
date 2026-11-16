"""slide_figures.py -- every figure and equation strip of the presentation, drawn for the slide.

    python presentation/slide_figures.py [keys...]

One typeface (Calibri, also for the mathematics), text of 13 pt and more at slide size, blue = classical,
orange = network, black and grey for everything else.  Data loaders come from a COPY of the paper's figure script
(presentation/mf_paper.py); data from presentation/data; camera pictures only from presentation/figures/photos
(marker-only cut-outs).  Every figure is 12.3 in wide (the content width of a slide) and is written as the whole canvas,
so its left edge and its lettering have the same position and size on every slide.
Checks before a figure is written: figcheck.check (no overlapping elements), nothing outside the canvas, text inside
its box in the diagrams.  The smallest font of every figure goes to figures/_meta.json for the build script.
"""
import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Polygon, Rectangle
from matplotlib.text import Text

import figcheck
import mf_paper as mf

OUT = os.path.join(HERE, "figures")
mf.OUT, mf.PH, mf.DAT = OUT, os.path.join(OUT, "photos"), os.path.join(HERE, "data")
PH, DAT, RES, ROOT = mf.PH, mf.DAT, mf.RES, mf.ROOT
CL, NET = "#1F3A5F", "#D9731A"                       # classical, network
CL2, CL3 = "#7C93B2", "#C9D3E0"                      # lighter blues (classical quantities)
NET2, NET3, NET4 = "#EBA76A", "#F6D5B8", "#8A4A12"   # lighter / darker oranges (network variants)
INK, MID, PALE, GREY = "#1A1A1A", "#6E6E6E", "#B5B5B5", "#404040"
rows, fnum = mf.rows, mf.fnum
WS = 12.3                       # width of a slide figure [in]
TS, FS, SM = 15, 14, 13         # panel title, labels, smallest text [pt]
U = 0.123                       # unit of the diagrams [in]

plt.rcParams.update({
    "font.family": "Calibri", "mathtext.fontset": "custom", "mathtext.rm": "Calibri", "mathtext.it": "Calibri:italic", "mathtext.bf": "Calibri:bold",
    "mathtext.fallback": "stixsans",
    "font.size": FS, "axes.titlesize": SM, "axes.labelsize": FS, "legend.fontsize": SM, "xtick.labelsize": SM, "ytick.labelsize": SM,
    "axes.linewidth": 0.9, "lines.linewidth": 1.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 4,
    "ytick.major.size": 4, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": True, "legend.facecolor": "white", "legend.edgecolor": "none", "legend.framealpha": 1.0, "legend.borderpad": 0.2,
    "legend.borderaxespad": 0.4, "legend.labelspacing": 0.2, "legend.handletextpad": 0.5,
    "grid.linewidth": 0.6, "grid.alpha": 0.35, "savefig.dpi": 200, "figure.dpi": 100, "axes.labelpad": 4,
})
PROBLEMS, META = {}, {}
META_FILE = os.path.join(OUT, "_meta.json")


def new(H):
    return plt.figure(figsize=(WS, H))


def axi(fig, l, b, w, h, **kw):
    """Axes placed in inches."""
    H = fig.get_size_inches()[1]
    return fig.add_axes([l / WS, b / H, w / WS, h / H], **kw)


def head(fig, x_in, s):
    """Panel title: same size and the same distance from the top in every figure."""
    H = fig.get_size_inches()[1]
    fig.text(x_in / WS, (H - 0.30) / H, s, fontsize=TS, fontweight="bold", va="bottom", ha="left")


def toplegend(fig, handles, labels, x_in, ncol, y_off=0.42, **kw):
    """A legend row under the panel titles, centred at x_in."""
    H = fig.get_size_inches()[1]
    kw.setdefault("columnspacing", 1.3); kw.setdefault("handlelength", 1.5)
    return fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(x_in / WS, (H - y_off) / H), ncol=ncol, borderaxespad=0.0, **kw)


def save(fig, name, data=True, boxes=()):
    """Checks (overlaps, canvas, boxes), records the smallest font, writes the whole canvas."""
    prob = list(figcheck.check(fig, name, data=data))
    r = fig.canvas.get_renderer()
    H = fig.get_size_inches()[1]
    tb = fig.get_tightbbox(r)
    if tb.x0 < -0.02 or tb.y0 < -0.02 or tb.x1 > WS + 0.02 or tb.y1 > H + 0.02:
        prob.append(f"content outside the canvas: x {tb.x0:.2f}..{tb.x1:.2f} of {WS}, y {tb.y0:.2f}..{tb.y1:.2f} of {H}")
    for t_, ax, (bx, by, bw, bh) in boxes:                          # text of a diagram box must lie inside the box
        e = t_.get_window_extent(r)
        (x0, y0), (x1, y1) = ax.transData.inverted().transform([(e.x0, e.y0), (e.x1, e.y1)])
        if x0 < bx + 0.3 or x1 > bx + bw - 0.3 or y0 < by + 0.2 or y1 > by + bh - 0.2:
            prob.append("text outside its box: " + t_.get_text().replace(chr(10), " | "))
    sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    META[name + ".png"] = [200, round(min(sizes), 1)]
    PROBLEMS[name] = prob
    for p_ in prob:
        if not p_.startswith(("text on", "legend on", "text in")):
            print("      -", p_)
    fig.savefig(os.path.join(OUT, name + ".png"))
    plt.close(fig)


def show(ax, path, label=None, recolour=False):
    im = plt.imread(path)[..., :3] if plt.imread(path).ndim == 3 else plt.imread(path)
    if recolour and im.ndim == 3:                                    # outline drawn green in the cut-out -> blue (classical)
        hsv = rgb_to_hsv(np.clip(im, 0, 1))
        m = (hsv[..., 0] > 0.22) & (hsv[..., 0] < 0.45) & (hsv[..., 1] > 0.35) & (hsv[..., 2] > 0.35)
        hsv[..., 0] = np.where(m, 0.60, hsv[..., 0]); hsv[..., 1] = np.where(m, np.maximum(hsv[..., 1], 0.75), hsv[..., 1])
        im = hsv_to_rgb(hsv)
    ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=1, interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.6); s.set_color("#444444")
    if label:
        ax.set_xlabel(label, fontsize=SM, labelpad=4)


def arrow(ax, p, q, col="k", lw=1.4, style="-|>", ls="-", ms=13, z=3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, lw=lw, color=col, ls=ls, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def rot(v):
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(v, float) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def angle_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def mean_rot(Rs):
    Um, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    return Um @ np.diag([1, 1, np.sign(np.linalg.det(Um @ Vt))]) @ Vt


def truth_xyz(name):
    tr = rows(os.path.join(ROOT, "data", "recordings", name + ".truth.csv"))[0]
    return np.array([fnum(tr["x_mm"]), fnum(tr["y_mm"]), fnum(tr["z_mm"])])


def logticks(ax, ticks, lim):
    ax.set_yscale("log"); ax.set_yticks(ticks); ax.set_yticklabels([f"{k:g}" for k in ticks]); ax.set_ylim(*lim); ax.minorticks_off()


# ======================================================================================= T1  set-up
def sl_setup():
    fig = new(4.2)
    head(fig, 0.0, "(a) Frames and ruler measures (top view)"); head(fig, 5.2, "(b) Occlusion bench (camera view)"); head(fig, 10.45, "(c) Camera crops")
    ax = axi(fig, 0.0, 0.0, 4.9, 3.6); ax.set_xlim(0, 10); ax.set_ylim(0, 8.45); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((0.6, 7.0), 9.0, 0.2, fc="#2b2b2b", ec="none"))
    ax.text(0.6, 7.36, "board", fontsize=SM, color=GREY)
    fx, fy = 7.0, 6.82
    ax.add_patch(Rectangle((fx - 1.0, fy), 2.0, 0.18, fc="white", ec="k", lw=1.2))
    mx, my, psi = 2.8, 4.6, np.radians(25)
    d = np.array([np.cos(psi), -np.sin(psi)])
    a, b = np.array([mx, my]) - d, np.array([mx, my]) + d
    ax.plot([a[0], b[0]], [a[1], b[1]], color="k", lw=5.2, solid_capstyle="butt", zorder=2)
    ax.plot([a[0], b[0]], [a[1], b[1]], color="white", lw=2.8, solid_capstyle="butt", zorder=2)
    for (ox, oy), ang in (((fx, fy), 0.0), ((mx, my), -psi)):
        ex, ez = np.array([np.cos(ang), np.sin(ang)]), np.array([np.sin(ang), -np.cos(ang)])
        arrow(ax, (ox, oy), (ox + 1.2 * ex[0], oy + 1.2 * ex[1]), GREY, 2.0)
        arrow(ax, (ox, oy), (ox + 1.2 * ez[0], oy + 1.2 * ez[1]), GREY, 2.0)
        ax.plot(ox, oy, "o", ms=8, mfc="white", mec=GREY, mew=1.6, zorder=5); ax.plot(ox, oy, ".", ms=3.5, color=GREY, zorder=6)
    ax.text(fx + 1.34, fy, "$x$", color=GREY, fontsize=FS, va="center"); ax.text(fx + 0.16, fy - 1.25, "$z$", color=GREY, fontsize=FS, va="center")
    ax.text(mx + 1.2, my - 0.86, "$x$", color=GREY, fontsize=FS, va="center", ha="center"); ax.text(mx - 0.8, my - 1.2, "$z$", color=GREY, fontsize=FS, va="center", ha="center")
    ax.text(fx + 0.17, 7.36, "{F}  fixed, ID 0", fontsize=SM, va="bottom")
    ax.text(0.1, 5.85, "{M}  mobile,\nID 23", fontsize=SM, va="center", linespacing=1.1)
    cx, cy = 5.3, 1.11
    ax.add_patch(Rectangle((cx - 0.55, cy - 0.60), 1.1, 0.36, fc="#d9d9d9", ec="k", lw=1.2))
    ax.add_patch(Polygon([[cx - 0.2, cy - 0.24], [cx + 0.2, cy - 0.24], [cx + 0.32, cy], [cx - 0.32, cy]], fc="#9a9a9a", ec="k", lw=1.0))
    arrow(ax, (cx, cy), (cx + 1.1, cy), GREY, 2.0); arrow(ax, (cx, cy), (cx, cy + 1.1), GREY, 2.0)
    ax.plot(cx, cy, "o", ms=8, mfc="white", mec=GREY, mew=1.6, zorder=5); ax.plot(cx, cy, "x", ms=4.5, color=GREY, zorder=6, mew=1.3)
    ax.text(cx + 1.24, cy, "$x$", color=GREY, fontsize=FS, va="center"); ax.text(cx + 0.16, cy + 1.12, "$z$", color=GREY, fontsize=FS, va="center")
    ax.text(cx, 0.14, "{C}  laptop camera", fontsize=SM, ha="center")
    arrow(ax, (cx + 0.45, cy + 1.25), (fx - 0.12, fy - 1.42), MID, 1.4, rad=0.1)
    arrow(ax, (cx - 0.4, cy + 1.25), (mx + 0.3, my - 0.75), MID, 1.4, rad=-0.1)
    arrow(ax, (fx - 0.3, fy - 0.2), (mx + 0.55, my + 0.15), "k", 2.4, ls=(0, (4, 2)), rad=0.14)
    ax.text(6.75, 3.55, r"${}^{C}\mathbf{T}_{F}$", fontsize=17, color=GREY); ax.text(3.5, 2.05, r"${}^{C}\mathbf{T}_{M}$", fontsize=17, color=GREY)
    ax.text(4.75, 4.95, r"${}^{F}\mathbf{T}_{M}$", fontsize=18, color="k")
    ax.annotate("", (mx, 8.02), (fx, 8.02), arrowprops=dict(arrowstyle="<->", lw=1.1, color=GREY, shrinkA=0, shrinkB=0))
    ax.text((mx + fx) / 2, 8.1, "$s$", fontsize=FS, ha="center", color=GREY, va="bottom")
    ax.plot([mx, mx], [7.2, 8.1], color=GREY, lw=0.8, ls=":"); ax.plot([fx, fx], [7.2, 8.1], color=GREY, lw=0.8, ls=":")
    ax.annotate("", (mx, my + 0.12), (mx, 7.0), arrowprops=dict(arrowstyle="<->", lw=1.1, color=GREY, shrinkA=0, shrinkB=0))
    ax.text(mx + 0.18, 6.0, "$n$", fontsize=FS, color=GREY, va="center")
    ax.add_patch(Arc((mx, my), 3.3, 3.3, theta1=-25, theta2=0, lw=1.1, color=GREY)); ax.plot([mx, mx + 1.95], [my, my], color=GREY, lw=0.8, ls=":")
    ax.text(mx + 1.9, my - 0.38, r"$\psi$", fontsize=FS, color=GREY, va="center")
    # ---- (b)
    ax = axi(fig, 5.2, 0.0, 4.9, 3.6); ax.set_xlim(0, 10); ax.set_ylim(0, 8.45); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((0.3, 1.9), 9.4, 0.3, fc="#ececec", ec="k", lw=0.9))
    for i in range(48):
        ax.plot([0.4 + i * 0.196] * 2, [2.2, 2.2 - (0.17 if i % 5 == 0 else 0.09)], color="k", lw=0.6)
    ax.text(0.3, 1.38, "ruler, mm divisions", fontsize=SM, color=GREY)

    def marker(x, y, lab):
        ax.add_patch(Rectangle((x - 1.05, y - 1.05), 2.1, 2.1, fc="white", ec="k", lw=1.2))
        ax.add_patch(Rectangle((x - 0.78, y - 0.78), 1.56, 1.56, fc="#222222", ec="k", lw=1.0))
        ax.text(x, y, lab, color="white", fontsize=16, ha="center", va="center", fontweight="bold")
    marker(8.4, 3.55, "F"); marker(1.5, 3.55, "M")
    ax.add_patch(Rectangle((4.0, 2.2), 1.9, 4.2, fc="#c9c9c9", ec="k", lw=1.2, zorder=4))
    ax.text(4.95, 6.62, "obstacle (box)", fontsize=SM, ha="center", va="bottom")
    ax.add_patch(Rectangle((3.9, 2.5), 2.1, 2.1, fc="none", ec="k", lw=1.8, ls=(0, (4, 2)), zorder=6))
    ax.text(4.95, 4.8, "predicted", color="k", fontsize=SM, ha="center", va="bottom", zorder=7)
    arrow(ax, (2.7, 3.55), (3.72, 3.55), "k", 1.8); ax.text(3.2, 3.78, "$v$", fontsize=FS, ha="center", va="bottom")
    ax.text(1.5, 4.85, "M slides\ntoward F", fontsize=SM, ha="center", va="bottom", color=GREY, linespacing=1.1)
    ax.text(8.4, 4.85, "F stays\nvisible", fontsize=SM, ha="center", va="bottom", color=GREY, linespacing=1.1)
    ax.annotate("", (1.5, 0.98), (8.4, 0.98), arrowprops=dict(arrowstyle="<->", lw=1.1, color=GREY, shrinkA=0, shrinkB=0))
    ax.text(4.95, 0.12, "325–377 mm centre to centre at the start", fontsize=SM, ha="center", color=GREY)
    # ---- (c)
    a1 = axi(fig, 10.6, 2.2, 1.45, 1.45); show(a1, os.path.join(PH, "view_F_raw.png"), "F, ID 0")
    a2 = axi(fig, 10.6, 0.38, 1.45, 1.45); show(a2, os.path.join(PH, "view_E2_d40_t0_raw.png"), "M, ID 23")
    save(fig, "sl_setup")


# ======================================================================================= T2  detection limits
def sl_t2():
    fig = new(3.8)
    a = axi(fig, 1.0, 0.72, 5.3, 2.5)
    r = rows(os.path.join(RES, "detection", "detection_matrix.csv"))
    Dd, T = [30, 50, 75, 100], [0, 30, 45, 60, 70, 75, 80, 85]
    M = np.full((len(Dd), len(T)), np.nan); P = np.full_like(M, np.nan)
    for x in r:
        i, j = Dd.index(int(float(x["target_distance_cm"]))), T.index(int(float(x["target_tilt_deg"])))
        M[i, j] = fnum(x["detected"]); P[i, j] = fnum(x["px_per_cell"])
    p0 = P[:, 0]
    cm = matplotlib.colors.LinearSegmentedColormap.from_list("d", [CL, "#9FB3CC", "#E6ECF3"])
    a.imshow(np.ma.masked_invalid(M), cmap=cm, vmin=0, vmax=10, aspect="auto")
    for i in range(len(Dd)):
        for j in range(len(T)):
            if np.isfinite(M[i, j]):
                ink = "white" if M[i, j] < 3 else "k"
                a.text(j, i - 0.19, f"{int(M[i, j])}/10", ha="center", va="center", fontsize=SM, fontweight="bold", color=ink)
                a.text(j, i + 0.24, f"{p0[i] * np.cos(np.radians(T[j])):.1f}", ha="center", va="center", fontsize=SM, color=ink)
            else:
                a.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc="#eeeeee", ec="none")); a.text(j, i, "n/a", ha="center", va="center", fontsize=SM, color=GREY)
    a.set_xticks(range(len(T))); a.set_xticklabels([f"{t}" for t in T]); a.set_yticks(range(len(Dd))); a.set_yticklabels([f"{d_} cm" for d_ in Dd])
    a.set_xlabel("tilt about the vertical axis [deg]"); a.set_ylabel("distance")
    a.set_xticks(np.arange(-0.5, len(T)), minor=True); a.set_yticks(np.arange(-0.5, len(Dd)), minor=True)
    a.grid(which="minor", color="w", lw=1.8, alpha=1); a.tick_params(which="minor", length=0)
    for s in a.spines.values():
        s.set_visible(False)
    head(fig, 0.0, "(a) Detected frames of 10; below: pixels per marker cell")
    td, ed = np.array([fnum(x["target_distance_cm"]) for x in r]), np.array([fnum(x["est_distance_cm"]) for x in r])
    tt, et = np.array([fnum(x["target_tilt_deg"]) for x in r]), np.array([fnum(x["est_tilt_deg"]) for x in r])
    b = axi(fig, 7.5, 0.72, 1.9, 2.5); c = axi(fig, 10.2, 0.72, 1.9, 2.5)
    ok = np.isfinite(ed)
    b.plot([20, 110], [20, 110], color="k", lw=0.9, ls="--"); b.scatter(td[ok], ed[ok], s=30, color=CL, alpha=0.7, linewidths=0, zorder=3)
    b.set_xlim(20, 110); b.set_ylim(20, 110); b.set_xticks([30, 50, 75, 100]); b.set_yticks([30, 50, 75, 100]); b.grid(True)
    b.set_xlabel("tape measure [cm]"); b.set_ylabel("from the pose [cm]")
    ok = np.isfinite(et)
    c.plot([0, 90], [0, 90], color="k", lw=0.9, ls="--"); c.scatter(tt[ok], et[ok], s=30, color=CL, alpha=0.7, linewidths=0, zorder=3)
    c.set_xlim(-4, 90); c.set_ylim(-4, 90); c.set_xticks([0, 30, 60, 90]); c.set_yticks([0, 30, 60, 90]); c.grid(True)
    c.set_xlabel("protractor [deg]"); c.set_ylabel("from the pose [deg]")
    head(fig, 6.75, "(b) Distance check"); head(fig, 9.45, "(c) Tilt check")
    save(fig, "sl_t2")


# ======================================================================================= T3  the two pipelines (diagram)
def sl_pipeline():
    H = 3.7
    fig = new(H)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, H / U); ax.axis("off")
    boxes = []

    def box(x, y, w, h, text, fc, fs=SM):
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="k", lw=0.9, zorder=2))
        boxes.append((ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.15), ax, (x, y, w, h)))

    def ar(p, q, col="k", rad=0.0):
        arrow(ax, p, q, col, 1.4, ms=12, rad=rad)
    cb, nb, h = "#dfe7f1", "#fbe6d3", 7.2
    yc, yn = 20.4, 1.6
    box(0.4, 11.0, 10.4, h, "camera\nframe\n1280 × 720", "#ffffff")
    ax.text(13.0, yc + h + 0.6, "CLASSICAL  (both markers)", color=CL, fontsize=FS, fontweight="bold", va="bottom")
    ax.text(13.0, yn + h + 0.6, "LEARNED  (marker M only)", color=NET, fontsize=FS, fontweight="bold", va="bottom")
    xs, ws = [13.0, 31.0, 47.0, 63.0], [15.5, 13.5, 13.5, 15.0]
    tx = ["ArUco detection\nand sub-pixel\ncorners", "IPPE:\ntwo planar\nsolutions", "LM refinement\n(reprojection\nerror)", "ambiguity rule:\nclosest to the\nprevious frame"]
    for x, w_, t in zip(xs, ws, tx):
        box(x, yc, w_, h, t, cb)
    for i in range(3):
        ar((xs[i] + ws[i] + 0.4, yc + h / 2), (xs[i + 1] - 0.4, yc + h / 2), CL)
    ar((5.6, 18.6), (12.5, yc + h / 2), CL, rad=-0.3)
    xn, wn = [13.0, 27.0, 46.0, 63.5], [11.5, 16.5, 15.0, 15.5]
    tn = ["undistort,\nresize to\n640 × 360", "stage 1: ResNet-18\non the whole image\n(coarse position)", "crop 224 × 224,\n2 × marker side,\n+ 2 ray channels", "stage 2: ResNet-18\n6-D rotation,\n(x/Z, y/Z, log Z)"]
    for x, w_, t in zip(xn, wn, tn):
        box(x, yn, w_, h, t, nb)
    for i in range(3):
        ar((xn[i] + wn[i] + 0.4, yn + h / 2), (xn[i + 1] - 0.4, yn + h / 2), NET)
    ar((5.6, 10.6), (12.5, yn + h / 2), NET, rad=0.3)
    ar((xn[3] + 8.0, yn + h + 0.5), (xn[2] + 7.5, yn + h + 0.5), NET, rad=0.45)
    ax.text(62.0, yn + h + 5.6, "second pass: re-centre the crop", fontsize=SM, color=NET, ha="center", va="center")
    box(82.0, 11.0, 17.6, h, "relative pose\n" + r"${}^{F}\mathbf{T}_{M}=({}^{C}\mathbf{T}_{F})^{-1}\,{}^{C}\mathbf{T}_{M}$", "#ffffff")
    ar((xs[3] + ws[3] + 0.4, yc + h / 2), (90.8, 11.0 + h + 0.4), CL, rad=-0.3)
    ar((xn[3] + wn[3] + 0.4, yn + h / 2), (90.8, 10.6), NET, rad=0.3)
    ax.text(90.5, yc + h / 2 + 2.6, r"${}^{C}\mathbf{T}_{F},\ {}^{C}\mathbf{T}_{M}$", fontsize=FS, color=CL, ha="center", va="center")
    ax.text(90.5, yn + h / 2 - 2.6, r"${}^{C}\mathbf{T}_{M}$", fontsize=FS, color=NET, ha="center", va="center")
    ax.text(31.0, 17.0, r"${}^{C}\mathbf{T}_{F}$ always comes from the classical branch", fontsize=SM, color=GREY, ha="center", va="center", style="italic")
    save(fig, "sl_pipeline", data=False, boxes=boxes)


# ======================================================================================= T3  calibration
def sl_calib():
    c = rows(os.path.join(DAT, "calib_corners.csv")); v = rows(os.path.join(DAT, "calib_views.csv"))
    uv = np.array([[fnum(x["u"]), fnum(x["v"])] for x in c]); dd = np.array([[fnum(x["du"]), fnum(x["dv"])] for x in c])
    rms = np.array([fnum(x["rms_px"]) for x in v]); ho = np.array([x["held_out"] == "1" for x in v]); hor = np.array([fnum(x["held_out_rms_px"]) for x in v])
    fig = new(3.9)
    a = axi(fig, 0.75, 0.72, 2.75, 2.6)
    x = np.arange(1, 21)
    a.bar(x[~ho], rms[~ho], 0.7, color=CL, label="fitted view"); a.bar(x[ho], rms[ho], 0.7, color=CL2, label="held-out view")
    a.scatter(x[ho], hor[ho], marker="_", s=130, color="k", lw=2.4, zorder=4, label="error when held out")
    a.axhline(1.184, color="#444444", lw=1.0, ls="--")
    a.set_xticks([1, 5, 10, 15, 20]); a.set_xlabel("view"); a.set_ylabel("reprojection error [px]"); a.set_ylim(0, 4.9); a.set_yticks([0, 1, 2, 3, 4]); a.grid(True, axis="y")
    a.legend(loc="upper left", handlelength=1.0)
    head(fig, 0.0, "(a) Error per view (dashed: 1.18 px)")
    b = axi(fig, 4.5, 0.72, 2.6, 2.6)
    b.scatter(dd[:, 0], dd[:, 1], s=5, color=CL, alpha=0.45, linewidths=0)
    b.add_patch(Circle((0, 0), 1.184, fc="none", ec="k", lw=1.3, ls="--"))
    b.set_xlim(-5, 5); b.set_ylim(-5, 5); b.set_aspect("equal"); b.grid(True); b.set_xlabel(r"residual $\Delta u$ [px]"); b.set_ylabel(r"residual $\Delta v$ [px]")
    b.set_xticks([-4, -2, 0, 2, 4]); b.set_yticks([-4, -2, 0, 2, 4])
    head(fig, 3.8, "(b) Residuals, 1 080 corners")
    e = axi(fig, 7.5, 0.95, 2.4, 2.2); show(e, os.path.join(PH, "calib_rect.png"), "curved lines: the sheet is not flat")
    head(fig, 7.5, "(c) One view, rectified")
    f = axi(fig, 10.85, 0.72, 1.4, 2.1)
    fx_, cx_, cy_, k1, k2 = 846.593, 641.122, 361.777, 0.0922014, -0.1347621
    r = np.linspace(0, 734, 200); rn = r / fx_
    l1, = f.plot(r, fx_ * rn * (k1 * rn ** 2 + k2 * rn ** 4), color=CL)
    rad = np.hypot(uv[:, 0] - cx_, uv[:, 1] - cy_); mag = np.hypot(dd[:, 0], dd[:, 1])
    edges = np.linspace(0, 700, 8); mid = 0.5 * (edges[1:] + edges[:-1])
    rb = [np.sqrt(np.mean(mag[(rad >= lo) & (rad < hi)] ** 2)) if ((rad >= lo) & (rad < hi)).sum() > 20 else np.nan for lo, hi in zip(edges[:-1], edges[1:])]
    l2, = f.plot(mid, rb, color=MID, marker="o", ms=5, lw=1.4)
    f.axhline(0, color="k", lw=0.7); f.set_xlabel("radius [px]"); f.set_ylabel("shift [px]"); f.grid(True); f.set_xlim(0, 734); f.set_ylim(-6, 9.5); f.set_xticks([0, 300, 600])
    head(fig, 10.1, "(d) Radial model")
    fig.legend([l1, l2], ["model shift", "residual RMS"], loc="upper left", bbox_to_anchor=(10.1 / WS, (3.9 - 0.42) / 3.9), ncol=1, borderaxespad=0.0, handlelength=1.3)
    save(fig, "sl_calib")


# ======================================================================================= T3  the two pose checks
def sl_t3():
    fig = new(3.6)
    a = axi(fig, 0.8, 0.72, 3.2, 2.3)
    gap = [50, 100, 150, 200, 250]
    bias = np.array([[-1.7, 1.5, 3.0], [-2.0, 5.4, 9.0], [0.5, 1.2, 5.9], [-3.5, 1.7, 7.9], [11.1, 1.9, 7.3]])
    x = np.arange(5)
    for k, (c, lab) in enumerate(zip((CL, CL2, CL3), ("x (along the markers)", "y (height)", "z (depth)"))):
        a.bar(x + (k - 1) * 0.26, bias[:, k], 0.25, color=c, label=lab)
    a.axhline(0, color="k", lw=0.8); a.set_xticks(x); a.set_xticklabels([f"{g}" for g in gap]); a.set_xlabel("gap between the black squares [mm]")
    a.set_ylabel("camera − ruler [mm]"); a.grid(True, axis="y"); a.set_ylim(-5.5, 27); a.set_yticks([-5, 0, 5, 10, 15])
    a.legend(loc="upper left", handlelength=0.8)
    head(fig, 0.0, "(a) Check A: bias at five ruler positions")
    rr = rows(os.path.join(RES, "pose", "invariance_1_poses.csv"))
    t = np.array([fnum(q["t_s"]) for q in rr])
    CF = np.array([[fnum(q[f"CF_{k}_mm"]) for k in "xyz"] for q in rr]); FM = np.array([[fnum(q[f"FM_{k}_mm"]) for k in "xyz"] for q in rr])
    CFr = np.array([[fnum(q[f"CF_r{k}"]) for k in "xyz"] for q in rr]); FMr = np.array([[fnum(q[f"FM_r{k}"]) for k in "xyz"] for q in rr])
    ok = np.isfinite(CF).all(1) & np.isfinite(FM).all(1)
    cam = np.linalg.norm(CF - np.nanmedian(CF[:30], axis=0), axis=1); rel = np.linalg.norm(FM - np.nanmean(FM[ok], axis=0), axis=1)
    b = axi(fig, 5.05, 0.72, 3.0, 2.3)
    b.plot(t[ok], cam[ok], color=MID, lw=1.8, label="camera"); b.plot(t[ok], rel[ok], color=CL, lw=1.3, label="relative pose")
    b.set_xlabel("time [s]"); b.set_ylabel("position change [mm]"); b.grid(True); b.set_xlim(0, t[ok].max()); b.set_ylim(0, 175)
    b.legend(loc="upper left", handlelength=1.2, ncol=2, columnspacing=1.0)
    head(fig, 4.3, "(b) Check B: position")
    Rc = [rot(v) for v in CFr[ok]]; Rm = [rot(v) for v in FMr[ok]]
    Rcm, Rmean = mean_rot(Rc), mean_rot(Rm)
    ca = np.array([angle_deg(Rcm, R) for R in Rc]); ra = np.array([angle_deg(Rmean, R) for R in Rm])
    c = axi(fig, 9.2, 0.72, 3.0, 2.3)
    c.plot(t[ok], ca, color=MID, lw=1.8, label="camera"); c.plot(t[ok], ra, color=CL, lw=1.3, label="relative pose")
    c.set_xlabel("time [s]"); c.set_ylabel("rotation change [deg]"); c.grid(True); c.set_xlim(0, t[ok].max()); c.set_ylim(0, 16); c.set_yticks([0, 5, 10, 15])
    c.legend(loc="upper left", handlelength=1.2, ncol=2, columnspacing=1.0)
    head(fig, 8.45, "(c) Check B: rotation")
    save(fig, "sl_t3")


# ======================================================================================= T4  the recorded free motion
def sl_t4():
    a = np.loadtxt(os.path.join(DAT, "T4_motion_1_tracks.csv"), delimiter=",", skiprows=1)
    t, c, n = a[:, 0], a[:, 1:7], a[:, 7:13].copy()
    n[~np.isfinite(c[:, 0])] = np.nan
    H = 3.75
    fig = new(H)
    ax = fig.add_axes([0.0, 0.02, 0.30, 0.84], projection="3d")
    ax.plot(c[:, 0], c[:, 2], c[:, 1], color=CL, lw=1.5); ax.plot(n[:, 0], n[:, 2], n[:, 1], color=NET, lw=1.1, alpha=0.9)
    for v_ in ((70, 0, 0), (0, 0, 70), (0, 70, 0)):
        ax.plot([0, v_[0]], [0, v_[1]], [0, v_[2]], color="k", lw=2.4)
    ax.text(0, 0, 88, "F", fontsize=FS, ha="center")
    ax.set_xlabel("x [mm]", labelpad=5, fontsize=SM); ax.set_ylabel("z [mm]", labelpad=3, fontsize=SM); ax.set_zlabel("y [mm]", labelpad=3, fontsize=SM)
    ax.tick_params(labelsize=SM, pad=-1); ax.view_init(20, -64); ax.set_box_aspect((1.6, 1, 1))
    ax.set_xticks([-400, -200, 0]); ax.set_yticks([0, 100]); ax.set_zticks([0, 50, 100])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.03); axis._axinfo["grid"]["linewidth"] = 0.5
    head(fig, 0.0, "(a) Path of M in the frame of F")
    names = ("x [mm]", "y [mm]", "z [mm]", "roll [deg]", "pitch [deg]", "yaw [deg]")
    lims = ((-450, -100), (-30, 170), (-30, 190), (-15, 15), (-15, 15), (-15, 60))
    ticks = ((-400, -200), (0, 100), (0, 100), (-10, 0, 10), (-10, 0, 10), (0, 25, 50))
    for k in range(6):
        col, row = k % 3, k // 3
        p = axi(fig, 4.75 + col * 2.6, 2.05 - row * 1.43, 2.05, 1.02)
        p.plot(t, c[:, k], color=CL, lw=1.7); p.plot(t, n[:, k], color=NET, lw=1.1)
        p.set_xlim(0, t.max()); p.set_ylim(*lims[k]); p.set_yticks(ticks[k]); p.grid(True); p.set_xticks([0, 10, 20, 30])
        p.set_title(names[k], loc="left", fontsize=SM, pad=3)
        if row == 0:
            p.set_xticklabels([])
        else:
            p.set_xlabel("time [s]")
    head(fig, 4.0, "(b) Pose of M in F against time (blue: classical, orange: network)")
    save(fig, "sl_t4")


# ======================================================================================= T5  the two-stage pipeline (diagram)
def sl_network():
    H = 3.3
    fig = new(H)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, H / U); ax.axis("off")
    nb, boxes = "#fbe6d3", []

    def box(x, w, text, fc):
        ax.add_patch(Rectangle((x, 9.0), w, 8.0, fc=fc, ec="k", lw=0.9, zorder=2))
        boxes.append((ax.text(x + w / 2, 13.0, text, ha="center", va="center", fontsize=SM, zorder=3, linespacing=1.15), ax, (x, 9.0, w, 8.0)))
    xs = [0.3, 14.3, 28.3, 42.9, 56.9, 72.9, 86.4]
    ws = [11.4, 11.4, 12.0, 11.4, 13.4, 11.0, 13.3]
    tx = ["camera frame\n1280 × 720\ngrey", "undistort,\nresize to\n640 × 360", "stage 1\nResNet-18\nwhole image", "coarse pose:\nwhere M is,\nhow large",
          "crop 224 × 224\n2 × marker side\n+ 2 ray channels", "stage 2\nResNet-18\ncrop", "pose of M:\n6-D rotation,\nx/Z, y/Z, log Z"]
    fc = ["#ffffff", nb, nb, "#ffffff", nb, nb, "#ffffff"]
    for x, w, t, f in zip(xs, ws, tx, fc):
        box(x, w, t, f)
    for i in range(6):
        arrow(ax, (xs[i] + ws[i] + 0.4, 13.0), (xs[i + 1] - 0.4, 13.0), NET, 1.4, ms=12)
    arrow(ax, (xs[6] + 6.7, 17.5), (xs[4] + 6.7, 17.5), NET, 1.4, ms=12, rad=0.42)
    ax.text(xs[5] + 5.5, 25.3, "second pass: the crop is re-centred on this answer", fontsize=SM, color=NET, ha="center", va="center")
    for i, s in ((2, "11.17 M parameters"), (4, "from the full-resolution frame"), (5, "11.18 M parameters")):
        ax.text(xs[i] + ws[i] / 2 + (1.5 if i == 5 else -1.0 if i == 4 else 0), 7.0, s, fontsize=SM, color=GREY, ha="center", va="center")
    ax.text(0.3, 2.3, "Stage 1 runs once per frame, stage 2 twice.  The fixed marker F is always measured by the classical pipeline.",
            fontsize=SM, color=INK, ha="left", va="center")
    save(fig, "sl_network", data=False, boxes=boxes)


# ======================================================================================= T5  the architecture of one stage (ResNet-18)
def sl_arch():
    H = 4.45
    fig = new(H)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, H / U); ax.axis("off")
    nb, boxes = "#fbe6d3", []

    def box(x, y, w, h, text, fc=nb):
        ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="k", lw=0.9, zorder=2))
        boxes.append((ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=SM, zorder=3, linespacing=1.12), ax, (x, y, w, h)))
    Hb, yb = 11.4, 22.3
    cols = [("input", 7.4, "image", "#ffffff"), ("stem", 11.0, "conv 7 × 7, 64\nstride 2\nBN, ReLU\nmax pool 3 × 3\nstride 2", nb),
            ("layer 1", 9.6, "2 basic\nblocks\n64 channels", nb), ("layer 2", 9.6, "2 basic\nblocks\n128 channels\nstride 2", nb),
            ("layer 3", 9.6, "2 basic\nblocks\n256 channels\nstride 2", nb), ("layer 4", 9.6, "2 basic\nblocks\n512 channels\nstride 2", nb),
            ("pool", 7.4, "global\naverage\npooling", nb), ("head", 7.4, "linear\n512 to 9", nb)]
    s1 = ["1 × 360 × 640", "64 × 90 × 160", "64 × 90 × 160", "128 × 45 × 80", "256 × 23 × 40", "512 × 12 × 20", "512", "9"]
    s2 = ["3 × 224 × 224", "64 × 56 × 56", "64 × 56 × 56", "128 × 28 × 28", "256 × 14 × 14", "512 × 7 × 7", "512", "9"]
    x, gap, xs = 8.6, 0.85, []
    for name, w, text, fc in cols:
        box(x, yb, w, Hb, text, fc)
        ax.text(x + w / 2, yb + Hb + 0.5, name, ha="center", va="bottom", fontsize=FS, fontweight="bold")
        xs.append((x, w))
        x += w + gap
    yc = yb + Hb / 2
    for (xa, wa), (xb_, wb_) in zip(xs[:-1], xs[1:]):
        arrow(ax, (xa + wa + 0.05, yc), (xb_ - 0.05, yc), "k", 1.1, ms=8)
    xo, wo = xs[-1][0] + xs[-1][1] + 2.2, 100 - 0.3 - (xs[-1][0] + xs[-1][1] + 2.2)
    box(xo, yc + 0.3, wo, 5.4, "6 numbers:\ntwo vectors,\northonormal: R", "#ffffff")
    box(xo, yc - 5.7, wo, 5.4, "3 numbers:\nx/Z, y/Z, log Z\ngive t", "#ffffff")
    xh = xs[-1][0] + xs[-1][1]
    arrow(ax, (xh + 0.1, yc + 0.8), (xo - 0.1, yc + 3.0), "k", 1.1, ms=9); arrow(ax, (xh + 0.1, yc - 0.8), (xo - 0.1, yc - 3.0), "k", 1.1, ms=9)
    ax.text(xo + wo / 2, yb + Hb + 0.5, "outputs", ha="center", va="bottom", fontsize=FS, fontweight="bold")
    # size of the output of every block, both stages
    ax.plot([0.3, 99.7], [21.2, 21.2], color="#888888", lw=0.7); ax.plot([0.3, 99.7], [12.9, 12.9], color="#888888", lw=0.7)
    ax.text(0.3, 19.6, "size of the output of each block (channels × height × width)", fontsize=SM, style="italic", va="center")
    for y, lab, sizes in ((16.9, "stage 1", s1), (14.5, "stage 2", s2)):
        ax.text(0.3, y, lab, fontsize=FS, va="center", fontweight="bold")
        for (xa, wa), t in zip(xs, sizes):
            ax.text(xa + wa / 2, y, t, fontsize=SM, ha="center", va="center")
    # a basic block
    ax.text(0.3, 4.3, "basic block", fontsize=FS, fontweight="bold", va="center")
    names, w, x0, y0, hb = ["conv 3 × 3", "BN", "ReLU", "conv 3 × 3", "BN", "add", "ReLU"], [10.0, 5.4, 6.8, 10.0, 5.4, 5.8, 6.8], 19.0, 2.2, 4.2
    pos, x = [], x0
    for n_, w_ in zip(names, w):
        box(x, y0, w_, hb, n_, "#ffffff" if n_ != "add" else "#e4e4e4")
        pos.append((x, w_))
        x += w_ + 2.3
    ym = y0 + hb / 2
    ax.text(x0 - 3.9, ym, "in", fontsize=SM, ha="right", va="center")
    arrow(ax, (x0 - 3.4, ym), (x0 - 0.1, ym), "k", 1.1, ms=9)
    for (xa, wa), (xb_, wb_) in zip(pos[:-1], pos[1:]):
        arrow(ax, (xa + wa + 0.1, ym), (xb_ - 0.1, ym), "k", 1.1, ms=9)
    xe = pos[-1][0] + pos[-1][1]
    arrow(ax, (xe + 0.1, ym), (xe + 3.4, ym), "k", 1.1, ms=9)
    ax.text(xe + 3.9, ym, "out", fontsize=SM, va="center")
    xa_ = pos[5][0] + pos[5][1] / 2
    ax.plot([x0 - 1.8, x0 - 1.8, xa_], [ym, y0 + hb + 1.9, y0 + hb + 1.9], color="k", lw=1.1, zorder=1)
    arrow(ax, (xa_, y0 + hb + 1.9), (xa_, y0 + hb + 0.1), "k", 1.1, ms=9)
    ax.text((x0 - 1.8 + xa_) / 2, y0 + hb + 2.4, "skip connection (a 1 × 1 convolution with stride 2 where the size changes)", fontsize=SM, ha="center", va="bottom")
    save(fig, "sl_arch", data=False, boxes=boxes)


# ======================================================================================= T5  training data
def sl_data():
    fig = new(4.0)
    head(fig, 0.0, "(a) Synthetic whole images, 640 × 360 (stage 1)")
    w = 2.95
    for j in range(4):
        ax = axi(fig, j * (w + 0.167), 1.9, w, w * 9 / 16); show(ax, os.path.join(PH, f"synth_full_{j}.png"))
    Hh = 4.0
    fig.text(0.0, (1.58) / Hh, "(b) Synthetic crops (stage 2)", fontsize=TS, fontweight="bold", va="bottom")
    fig.text(4.155 / WS, (1.58) / Hh, "(c) A real crop of M and the degradations of the hard-case test", fontsize=TS, fontweight="bold", va="bottom")
    names = [("synth_crop_0", ""), ("synth_crop_1", ""), ("synth_crop_4", ""), ("hard_clean", "clean"), ("hard_cover30", "30 % covered"),
             ("hard_blur31", "blur 31 px"), ("hard_res6", "resolution / 6"), ("hard_light025", "light × 0.25"), ("hard_light010", "light × 0.10")]
    s = 1.2
    for j, (n, t) in enumerate(names):
        ax = axi(fig, j * (s + 0.185), 0.3, s, s); show(ax, os.path.join(PH, n + ".png"), t or None)
    save(fig, "sl_data")


# ======================================================================================= T5  training on synthetic data
def grid4(fig, j, bottom=1.08, h=2.1):
    return axi(fig, 0.9 + j * 3.1, bottom, 2.0, h)


def sl_synth():
    fig = new(3.75)
    runs = [("resnet18_v1", "A: whole image, 15 epochs", NET3, "-"), ("resnet18_v2", "B: whole image, 40 epochs", NET, "-"),
            ("resnet18_v3", "C: as B, rotation weight 2", NET, (0, (3, 1.6))), ("crop_resnet18_v1", "D: crop stage, 80 epochs", NET4, "-")]
    spec = [("rot_loss", None, "rotation loss", "(a) Training: rotation", [0.001, 0.01, 0.1, 1], (0.0008, 6)),
            ("trans_loss", None, "position loss", "(b) Training: position", [0.02, 0.05, 0.1, 0.2, 0.5], (0.015, 0.8)),
            ("val_rot_med", "val_rot_p95", "rotation error [deg]", "(c) Validation: rotation", [2, 5, 10, 20, 50, 100], (1.5, 160)),
            ("val_trans_med", "val_trans_p95", "position error [mm]", "(d) Validation: position", [10, 20, 50, 100, 200, 500], (7, 700))]
    for j, (key, key95, yl, tt, ticks, ylim) in enumerate(spec):
        ax = grid4(fig, j)
        for name, lab, col, ls in runs:
            log = mf.train_log(name)["log"]; ep = [x["epoch"] for x in log]
            ax.plot(ep, [x[key] for x in log], color=col, ls=ls, lw=1.8)
            if key95:
                ax.plot(ep, [x[key95] for x in log], color=col, lw=1.0, ls=(0, (1, 1.3)))
        logticks(ax, ticks, ylim)
        ax.set_xlabel("epoch"); ax.set_ylabel(yl); ax.grid(True); ax.set_xlim(0, 82); ax.set_xticks([0, 20, 40, 60, 80])
        head(fig, j * 3.1, tt)
    h = [Line2D([], [], color=c, lw=2.0, ls=ls) for _, _, c, ls in runs] + [Line2D([], [], color="k", lw=1.0, ls=(0, (1, 1.3)))]
    fig.legend(h, [l for _, l, _, _ in runs] + ["95th percentile"], loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.02), columnspacing=1.2, handlelength=1.5,
               borderaxespad=0.0)
    save(fig, "sl_synth")


# ======================================================================================= T5  both stages on new synthetic images
def sl_twostage():
    d = json.load(open(os.path.join(RES, "network", "two_stage_vs_full.json")))
    s1, s2 = d["resnet18_v2"], d["crop_resnet18_v1"]
    fig = new(3.6)
    a = axi(fig, 0.8, 0.95, 2.6, 2.1)
    a.bar([0, 1], [s1["usable"], s2["usable"]], 0.6, color=[NET2, NET])
    for x, v_ in zip((0, 1), (s1["usable"], s2["usable"])):
        a.text(x, v_ + 2, f"{v_:.1f} %", ha="center", va="bottom", fontsize=SM)
    a.set_xticks([0, 1]); a.set_xticklabels(["stage 1\nalone", "two-stage"], linespacing=1.0); a.set_ylabel("usable poses [%]"); a.set_ylim(0, 100)
    a.set_xlim(-0.6, 1.6); a.grid(True, axis="y")
    head(fig, 0.0, "(a) Usable: within 20 mm and 10°")
    bins = ["0.25–0.4", "0.4–0.6", "0.6–1.0"]
    keys = ["0.25-0.4", "0.4-0.6", "0.6-1.0"]
    x = np.arange(3)
    for k, (yl, tt, x0, ymax) in enumerate((("position error [mm]", "(c) Position error by distance", 8.35, 36), ("rotation error [deg]", "(b) Rotation error by distance", 4.2, 14.5))):
        ax = axi(fig, x0 + 0.75, 0.72, 3.1, 2.33)
        v1, v2 = [s1["by_dist"][b][k] for b in keys], [s2["by_dist"][b][k] for b in keys]
        ax.bar(x - 0.2, v1, 0.38, color=NET2, label="stage 1 alone"); ax.bar(x + 0.2, v2, 0.38, color=NET, label="two-stage")
        for xx, v_ in zip(list(x - 0.2) + list(x + 0.2), v1 + v2):
            ax.text(xx, v_ + ymax * 0.02, f"{v_:.1f}", ha="center", va="bottom", fontsize=SM)
        ax.set_xticks(x); ax.set_xticklabels([b + " m" for b in bins]); ax.set_ylabel(yl); ax.set_ylim(0, ymax); ax.grid(True, axis="y")
        ax.set_xlabel("distance of the marker")
        ax.legend(loc="upper left", handlelength=1.0)
        head(fig, x0, tt)
    save(fig, "sl_twostage")


# ======================================================================================= T5  fine-tuning curves
def sl_finetune():
    fig = new(3.75)
    rounds = [("_ft", "round 1: 3 164 real frames", NET3), ("_ft2", "round 2: 7 358 real frames", NET2), ("_ft3", "round 3: 11 562 real frames", NET)]
    spec = [("resnet18_v2", "val_rot_med", "rotation error [deg]", "(a) Stage 1: rotation", [5, 7, 10, 15], (4.5, 16)),
            ("resnet18_v2", "val_trans_med", "position error [mm]", "(b) Stage 1: position", [10, 20, 50, 100], (9, 130)),
            ("crop_resnet18_v1", "val_rot_med", "rotation error [deg]", "(c) Stage 2: rotation", [3, 4, 6, 8, 12], (3, 13)),
            ("crop_resnet18_v1", "val_trans_med", "position error [mm]", "(d) Stage 2: position", [5, 7, 10, 15, 20], (5, 22))]
    for j, (stage, key, yl, tt, ticks, ylim) in enumerate(spec):
        ax = grid4(fig, j)
        for suf, lab, col in rounds:
            d = mf.train_log(stage + suf); log = d["log"]; ep = np.array([x["epoch"] for x in log]); y = np.array([x[key] for x in log])
            best = d["config"].get("best_epoch", len(log))
            ax.plot(ep, y, color=col, lw=1.8); ax.plot(best, y[best - 1], marker="*", ms=13, color=col, mec="k", mew=0.7, zorder=5)
        logticks(ax, ticks, ylim)
        ax.set_xlabel("epoch"); ax.set_ylabel(yl); ax.grid(True)
        head(fig, j * 3.1, tt)
    h = [Line2D([], [], color=c, lw=2.0) for _, _, c in rounds] + [Line2D([], [], color="w", marker="*", mfc="#999999", mec="k", ms=13, ls="")]
    fig.legend(h, [l for _, l, _ in rounds] + ["epoch kept"], loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.02), columnspacing=1.6, handlelength=1.6,
               borderaxespad=0.0)
    save(fig, "sl_finetune")


# ======================================================================================= T5  what each change bought
def sl_rounds():
    fig = new(3.8)
    a = axi(fig, 0.85, 0.95, 3.0, 1.95)
    lab = ["synth.\nonly", "round\n1", "round\n2", "round\n3", "classi-\ncal"]
    med, p95, jit = [26.5, 10.0, 9.2, 6.7, 8.8], [106.5, 124.3, 17.8, 12.5, 13.6], [26.5, 16.3, 2.12, 1.26, 0.38]
    x = np.arange(5)
    a.bar(x, med, 0.62, color=[NET] * 4 + [CL], zorder=2)
    a.scatter(x, p95, marker="_", s=300, color="k", lw=2.2, zorder=3)
    a.scatter(x, jit, marker="D", s=40, color="white", edgecolor="k", lw=1.2, zorder=4)
    logticks(a, [0.3, 1, 3, 10, 30, 100], (0.25, 200))
    a.set_xticks(x); a.set_xticklabels(lab, linespacing=1.0); a.set_ylabel("position error [mm]"); a.grid(True, axis="y")
    toplegend(fig, [Line2D([], [], color="k", marker="_", ls="", ms=15, mew=2.2), Line2D([], [], color="k", marker="D", mfc="white", ls="", ms=6.5, mew=1.2)],
              ["95th percentile", "jitter"], 2.35, 2, handletextpad=0.2)
    head(fig, 0.0, "(a) Network on the E1 positions (bars: median)")
    b = axi(fig, 5.6, 0.95, 1.9, 1.95)
    xb = np.array([0, 1])
    b.bar(xb - 0.2, [13.9, 92], 0.38, color=NET3, label="without"); b.bar(xb + 0.2, [6.6, 29], 0.38, color=NET, label="with")
    for xx, v_ in zip(list(xb - 0.2) + list(xb + 0.2), [13.9, 92, 6.6, 29]):
        b.text(xx, v_ * 1.1, f"{v_:g}", ha="center", va="bottom", fontsize=SM)
    logticks(b, [3, 10, 30, 100], (3, 200))
    b.set_xticks(xb); b.set_xticklabels(["rotation\n[deg]", "position\n[mm]"], linespacing=1.0); b.set_ylabel("validation error"); b.grid(True, axis="y")
    toplegend(fig, *b.get_legend_handles_labels(), 6.4, 2, handletextpad=0.4)
    head(fig, 4.85, "(b) Rotation augmentation")
    hh = json.load(open(os.path.join(RES, "network", "hard_cases_E1.json")))
    keys = list(hh.keys())
    short = {"clean": "clean", "cover 15 % of M": "cover 15 %", "cover 30 % of M": "cover 30 %", "motion blur 15 px": "blur 15 px",
             "motion blur 31 px": "blur 31 px", "resolution / 4": "resolution / 4", "resolution / 6": "resolution / 6", "light x 0.25": "light × 0.25",
             "light x 0.10": "light × 0.10"}
    c = axi(fig, 8.75, 1.32, 3.5, 1.58)
    x = np.arange(len(keys))
    c.bar(x - 0.27, [hh[k]["classical_usable"] for k in keys], 0.26, color=CL, label="classical")
    c.bar(x, [hh[k]["_ft2"]["usable"] for k in keys], 0.26, color=NET3, label="round 2")
    c.bar(x + 0.27, [hh[k]["_ft3"]["usable"] for k in keys], 0.26, color=NET, label="round 3")
    c.set_xticks(x); c.set_xticklabels([short[k] for k in keys], rotation=40, ha="right", rotation_mode="anchor"); c.set_ylabel("usable poses [%]")
    c.set_ylim(0, 100); c.set_yticks([0, 25, 50, 75, 100]); c.grid(True, axis="y")
    toplegend(fig, *c.get_legend_handles_labels(), 10.3, 3, handletextpad=0.4)
    head(fig, 8.0, "(c) Usable poses on degraded frames")
    save(fig, "sl_rounds")


# ======================================================================================= T6  E1
def sl_e1():
    E = mf.test_errors()
    names = ["E1_p4", "E1_p3", "E1_p2", "E1_p1", "E1_p5", "E1_turn20", "E1_turn40"]
    lab = ["P4\n50", "P3\n100", "P2\n150", "P1\n200", "P5\n250", "turn\n20°", "turn\n40°"]
    x = np.arange(len(names))
    fig = new(3.6)
    spec = (("position error [mm]", "(a) Position error (median, 5–95 %)", 0, 0.0, 21), ("rotation error [deg]", "(b) Rotation error (median, 5–95 %)", 1, 4.15, 9.5))
    for yl, tt, ci, x0, ymax in spec:
        ax = axi(fig, x0 + 0.8, 0.95, 3.2, 2.1)
        for k, (col, off) in enumerate(((CL, -0.2), (NET, 0.2))):
            med = np.array([np.nanmedian(E[n][:, 2 * k + ci]) for n in names])
            lo = med - np.array([np.nanpercentile(E[n][:, 2 * k + ci], 5) for n in names]); hi = np.array([np.nanpercentile(E[n][:, 2 * k + ci], 95) for n in names]) - med
            ax.bar(x + off, med, 0.38, color=col, yerr=[lo, hi], error_kw=dict(lw=1.1, capsize=2.5, capthick=1.1, ecolor="k"))
        ax.axvline(4.5, color=GREY, lw=0.9, ls=":")
        ax.set_xticks(x); ax.set_xticklabels(lab, linespacing=1.0); ax.set_ylabel(yl); ax.set_ylim(0, ymax); ax.grid(True, axis="y")
        head(fig, x0, tt)
    c = axi(fig, 9.1, 0.95, 3.15, 2.1)
    for k, (col, off, lb) in enumerate(((CL, -0.2, "classical"), (NET, 0.2, "network"))):
        jit = []
        for n in names:
            p = E[n][:, 4 + 3 * k:7 + 3 * k]; p = p[np.isfinite(p).all(1)]
            jit.append(np.sqrt(np.mean(np.sum((p - p.mean(0)) ** 2, axis=1))))
        c.bar(x + off, jit, 0.38, color=col, label=lb)
    c.axvline(4.5, color=GREY, lw=0.9, ls=":")
    c.set_xticks(x); c.set_xticklabels(lab, linespacing=1.0); c.set_ylabel("jitter [mm]"); c.set_ylim(0, 2.9); c.set_yticks([0, 0.5, 1, 1.5, 2]); c.grid(True, axis="y")
    c.legend(loc="upper left", ncol=2, columnspacing=1.0, handlelength=1.0)
    head(fig, 8.3, "(c) Jitter")
    save(fig, "sl_e1")


# ======================================================================================= T6  E2
def sl_e2():
    E = mf.test_errors()
    d = [40.9, 61.4, 76.7]
    fig = new(3.75)
    axs = [axi(fig, 0.8 + j * 4.15, 0.72, 3.2, 2.15) for j in range(3)]
    for ax, ci in ((axs[0], 0), (axs[1], 1)):
        for k, (col, lb) in enumerate(((CL, "classical"), (NET, "network"))):
            for tl, ls, mk in ((0, "-", "o"), (40, (0, (3, 1.5)), "s")):
                v = [np.nanmedian(E[f"E2_d{dd}_t{tl}"][:, 2 * k + ci]) for dd in (40, 60, 75)]
                ax.plot(d, v, color=col, ls=ls, marker=mk, ms=7, mfc=col if tl == 0 else "white", mew=1.5, label=f"{lb}, {'facing' if tl == 0 else 'turned 40°'}")
    for k, col in enumerate((CL, NET)):
        for tl, ls, mk in ((0, "-", "o"), (40, (0, (3, 1.5)), "s")):
            v = []
            for dd in (40, 60, 75):
                n = f"E2_d{dd}_t{tl}"
                p = E[n][:, 4 + 3 * k:7 + 3 * k]; p = p[np.isfinite(p).all(1)]
                v.append((p.mean(0) - truth_xyz(n))[2])
            axs[2].plot(d, v, color=col, ls=ls, marker=mk, ms=7, mfc=col if tl == 0 else "white", mew=1.5)
    axs[2].axhline(0, color="k", lw=0.8)
    for ax, yl in zip(axs, ("position error [mm]", "rotation error [deg]", "depth error [mm]")):
        ax.set_xlabel("camera to F [cm]"); ax.set_ylabel(yl); ax.grid(True); ax.set_xticks([41, 61, 77]); ax.set_xlim(36, 82)
    axs[0].set_ylim(0, 30); axs[1].set_ylim(0, 12); axs[2].set_ylim(-28, 12)
    toplegend(fig, *axs[0].get_legend_handles_labels(), 6.15, 4, columnspacing=1.8, handlelength=2.0) if False else fig.legend(
        *axs[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, (3.75 - 0.42) / 3.75), ncol=4, columnspacing=1.8, handlelength=2.0, borderaxespad=0.0)
    head(fig, 0.0, "(a) Position error"); head(fig, 4.15, "(b) Rotation error"); head(fig, 8.3, "(c) Depth error (along F's normal)")
    save(fig, "sl_e2")


# ======================================================================================= T6  E3
def sl_e3():
    a = np.loadtxt(os.path.join(DAT, "E3_motion_01_errors.csv"), delimiter=",", skiprows=1)
    t = a[:, 0]
    fig = new(3.9)
    p, q, r = (axi(fig, 0.9 + j * 4.15, 0.72, 3.0, 2.55) for j in range(3))
    p.plot(t, a[:, 1], color="k", lw=2.4, label="truth (guide)"); p.plot(t, a[:, 2], color=CL, lw=1.5, label="classical"); p.plot(t, a[:, 3], color=NET, lw=1.5, label="network")
    p.set_ylabel("x of M in F [mm]"); p.set_ylim(-212, -84); p.set_yticks([-200, -180, -160, -140]); p.legend(loc="upper center", ncol=1, handlelength=1.5)
    q.plot(t, a[:, 4], color=CL, lw=1.2, label="classical"); q.plot(t, a[:, 6], color=NET, lw=1.2, label="network"); q.set_ylabel("position error [mm]"); q.set_ylim(0, 74)
    r.plot(t, a[:, 5], color=CL, lw=1.2, label="classical"); r.plot(t, a[:, 7], color=NET, lw=1.2, label="network"); r.set_ylabel("rotation error [deg]"); r.set_ylim(0, 21)
    k = int(np.nanargmax(a[:, 4]))
    q.annotate("one frame: outline\nof M read 9 %\ntoo small", (t[k] + 0.2, a[k, 4] - 3), (t[k] + 3.0, 40), fontsize=SM, color=GREY, va="center", linespacing=1.05,
               arrowprops=dict(arrowstyle="-", lw=0.8, color=GREY, shrinkA=2, shrinkB=1))
    for ax in (p, q, r):
        ax.set_xlabel("time [s]"); ax.grid(True); ax.set_xlim(0, 20); ax.set_xticks([0, 5, 10, 15, 20])
    for ax in (q, r):
        ax.legend(loc="upper right", ncol=2, columnspacing=1.0, handlelength=1.2)
    head(fig, 0.0, "(a) Position along the ruler"); head(fig, 4.15, "(b) Position error"); head(fig, 8.3, "(c) Rotation error")
    save(fig, "sl_e3")


# ======================================================================================= T6  E4
def sl_e4():
    E = mf.test_errors()
    fig = new(3.6)
    for j, (n, t) in enumerate((("E4_dim", "dim light"), ("E4_cover", "edge of M covered"), ("E4_blur_2", "camera shaken"))):
        ax = axi(fig, j * 1.72, 1.05, 1.55, 1.55); show(ax, os.path.join(PH, f"view_{n}_both.png"), t, recolour=True)
    head(fig, 0.0, "(a) Outlines: classical blue, network orange")
    names, lab = ["E4_dim", "E4_clutter", "E4_blur", "E4_blur_2", "E4_cover"], ["dim light", "clutter", "mild shake", "strong shake", "edge covered"]
    x = np.arange(5)
    u = axi(fig, 6.35, 1.32, 2.45, 1.4); w = axi(fig, 9.8, 1.32, 2.45, 1.4)
    for k, (col, off) in enumerate(((CL, -0.2), (NET, 0.2))):
        outp = [100 * np.isfinite(E[n][:, 2 * k]).mean() for n in names]
        use = [100 * np.mean((E[n][:, 2 * k] < 20) & (E[n][:, 2 * k + 1] < 10)) for n in names]
        u.bar(x + off, outp, 0.38, color=col, alpha=0.3); u.bar(x + off, use, 0.38, color=col)
        med = [np.nanmedian(E[n][:, 2 * k]) if np.isfinite(E[n][:, 2 * k]).any() else 0 for n in names]
        w.bar(x + off, med, 0.38, color=col)
    for ax in (u, w):
        ax.set_xticks(x); ax.set_xticklabels(lab, rotation=40, ha="right", rotation_mode="anchor"); ax.grid(True, axis="y")
    u.set_ylabel("frames [%]"); u.set_ylim(0, 100); u.set_yticks([0, 50, 100]); w.set_ylabel("position error [mm]"); w.set_ylim(0, 36); w.set_yticks([0, 10, 20, 30])
    toplegend(fig, [Rectangle((0, 0), 1, 1, fc=CL), Rectangle((0, 0), 1, 1, fc=NET)], ["classical", "network"], 8.9, 2, handlelength=1.0)
    head(fig, 5.55, "(b) Usable (dark), given (light)"); head(fig, 9.0, "(c) Median position error")
    save(fig, "sl_e4")


# ======================================================================================= T6  distributions, bias against jitter
def sl_cdf():
    E = mf.test_errors()
    fig = new(4.0)
    a, b, c = (axi(fig, 0.85 + j * 4.15, 0.72, 3.15, 2.65) for j in range(3))
    groups = [("E1", "-"), ("E2", (0, (4, 1.5))), ("E3", (0, (1, 1.2))), ("E4", (0, (4, 1.2, 1, 1.2)))]
    for ax, col_i, xl, xmax in ((a, 0, "position error [mm]", 40), (b, 1, "rotation error [deg]", 14)):
        for k, col in enumerate((CL, NET)):
            for g, ls in groups:
                v = np.concatenate([E[n][:, 2 * k + col_i] for n in E if n.startswith(g)])
                v = np.sort(v[np.isfinite(v)])
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, ls=ls, lw=1.5)
        ax.set_xlim(0, xmax); ax.set_ylim(0, 1.02); ax.set_xlabel(xl); ax.set_ylabel("fraction of frames"); ax.grid(True)
    h = [Line2D([], [], color=CL, lw=2), Line2D([], [], color=NET, lw=2)] + [Line2D([], [], color="k", lw=1.5, ls=ls) for _, ls in groups]
    a.legend(h, ["classical", "network", "E1", "E2", "E3", "E4"], loc="lower right", handlelength=2.2)
    for k, (col, lb, mk, fill) in enumerate(((CL, "classical", "o", CL), (NET, "network", "s", "white"))):
        bias, jit = [], []
        for n in E:
            if n.startswith("E3"):
                continue
            pp = E[n][:, 4 + 3 * k:7 + 3 * k]; pp = pp[np.isfinite(pp).all(1)]
            if len(pp) < 10:
                continue
            bias.append(np.linalg.norm(pp.mean(0) - truth_xyz(n))); jit.append(np.sqrt(np.mean(np.sum((pp - pp.mean(0)) ** 2, axis=1))))
        c.scatter(jit, bias, s=42, facecolor=fill, edgecolor=col, marker=mk, label=lb, linewidths=1.6, zorder=3)
    c.plot([0.1, 40], [0.1, 40], color="k", lw=0.9, ls="--")
    c.set_xscale("log"); c.set_yscale("log"); c.set_xlim(0.1, 40); c.set_ylim(1, 100); c.set_xlabel("jitter [mm]"); c.set_ylabel("bias [mm]"); c.grid(True)
    c.set_xticks([0.1, 1, 10]); c.set_xticklabels(["0.1", "1", "10"]); c.set_yticks([1, 3, 10, 30]); c.set_yticklabels(["1", "3", "10", "30"]); c.minorticks_off()
    c.legend(loc="upper left", ncol=2, columnspacing=1.0, handletextpad=0.2)
    head(fig, 0.0, "(a) Position error, all frames"); head(fig, 4.15, "(b) Rotation error, all frames"); head(fig, 8.3, "(c) Static: bias (dashed: bias = jitter)")
    save(fig, "sl_cdf")


# ======================================================================================= Bonus  the predictor and its exact-truth test
def sl_bonus_a():
    H = 3.5
    fig = new(H)
    cap = ["measured", "hidden 0.4 s: predicted", "hidden 1.2 s: predicted", "measured again"]
    for k in range(4):
        ax = axi(fig, (k % 2) * 2.5, 1.9 - (k // 2) * 1.22, 2.35, 0.75); show(ax, os.path.join(PH, f"bonus_{k}.png"), cap[k])
    head(fig, 0.0, "(a) One pass (green: measured, red: predicted)")
    v = rows(os.path.join(RES, "bonus", "virtual_occlusion.csv"))
    a = axi(fig, 6.1, 0.72, 2.75, 1.78)
    T = [0.25, 0.5, 1.0, 1.5, 2.0]
    recs = (("E7_nocover_slow_2", "-", "o", "7 cm/s"), ("E7_nocover_fast", (0, (3, 1.5)), "s", "17 cm/s"))
    for rec, ls, mk, lb in recs:
        rr = [x for x in v if x["recording"] == rec]
        for kind, col in (("pred", INK), ("hold", PALE)):
            m = []
            for tt in T:
                vals = np.array([fnum(x[f"{kind}_mm_{tt:g}s"]) for x in rr])
                m.append(np.nanmedian(vals) if np.isfinite(vals).sum() >= max(2, len(rr) // 2) else np.nan)
            a.plot(T, m, color=col, ls=ls, marker=mk, ms=7, mfc=col if ls == "-" else "white", mew=1.5)
    a.set_xlabel("time hidden [s]"); a.set_ylabel("position error [mm]"); a.grid(True); a.set_ylim(0, 180); a.set_xlim(0.15, 2.1); a.set_yticks([0, 50, 100, 150])
    h = [Line2D([], [], color=INK, lw=2), Line2D([], [], color=PALE, lw=2), Line2D([], [], color=MID, lw=1.6, marker="o", ms=7),
         Line2D([], [], color=MID, lw=1.6, ls=(0, (3, 1.5)), marker="s", ms=7, mfc="white", mew=1.5)]
    fig.legend(h, ["predictor", "hold the last pose", "7 cm/s", "17 cm/s"], loc="upper center", bbox_to_anchor=(8.75 / WS, (H - 0.42) / H), ncol=4, borderaxespad=0.0,
               columnspacing=1.4, handlelength=1.8)
    head(fig, 5.3, "(b) Hidden in software: error")
    b = axi(fig, 10.0, 0.72, 2.25, 1.78)
    x = np.arange(2)
    for k, (kind, col) in enumerate((("pred", INK), ("hold", PALE))):
        vals = [np.nanmedian([fnum(q[f"{kind}_horizon_s_20mm"]) for q in v if q["recording"] == rec]) for rec, *_ in recs]
        b.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=col)
        for xx, v_ in zip(x + (k - 0.5) * 0.38, vals):
            b.text(xx, v_ + 0.02, f"{v_:.2f}", ha="center", va="bottom", fontsize=SM)
    b.set_xticks(x); b.set_xticklabels(["7 cm/s", "17 cm/s"]); b.set_ylabel("time under 20 mm [s]"); b.set_ylim(0, 1.05); b.set_yticks([0, 0.5, 1.0]); b.grid(True, axis="y")
    head(fig, 9.3, "(c) Time under 20 mm")
    save(fig, "sl_bonus_a")


# ======================================================================================= Bonus  the real obstacle
def sl_bonus_b():
    r = rows(os.path.join(RES, "bonus", "real_occlusion.csv"))
    sp = np.array([fnum(x["speed_hidden_mm_s"]) for x in r]); pr = np.array([fnum(x["pred_mm"]) for x in r]); ho = np.array([fnum(x["hold_mm"]) for x in r])
    pdg = np.array([fnum(x["pred_deg"]) for x in r]); hdg = np.array([fnum(x["hold_deg"]) for x in r]); hid = np.array([fnum(x["hidden_s"]) for x in r])
    groups = [(sp < 75, PALE, "4–6 cm/s", "v"), ((sp >= 75) & (sp < 125), MID, "8–12 cm/s", "o"), (sp >= 125, INK, "13–23 cm/s", "s")]
    print("     bonus groups: speeds", [(round(float(sp[m].min()) / 10, 1), round(float(sp[m].max()) / 10, 1)) for m, *_ in groups], "n", [int(m.sum()) for m, *_ in groups])
    fig = new(3.6)
    a = axi(fig, 0.85, 0.72, 3.2, 2.33)
    for m, col, lb, mk in groups:
        a.scatter(ho[m], pr[m], s=42, color=col, marker=mk, label=f"{lb} (n = {int(m.sum())})", zorder=3, edgecolor="none")
    a.plot([0, 440], [0, 440], color="k", lw=0.9, ls="--")
    a.set_xlim(120, 215); a.set_ylim(0, 440); a.set_xlabel("error of holding [mm]"); a.set_ylabel("error of the predictor [mm]"); a.set_yticks([0, 100, 200, 300])
    a.grid(True); a.legend(loc="upper left", handletextpad=0.1)
    head(fig, 0.0, "(a) 49 passes (dashed: equal error)")
    x = np.arange(3)
    lab = [f"{lb}\n{Decimal(str(round(float(np.median(hid[m])), 3))).quantize(Decimal('0.1'), ROUND_HALF_UP)} s" for m, _, lb, _ in groups]
    for j, (pv, hv, yl, tt, x0, ymax) in enumerate(((pr, ho, "position error [mm]", "(b) Position at reappearance", 4.3, 250),
                                                    (pdg, hdg, "rotation error [deg]", "(c) Rotation at reappearance", 8.45, 44))):
        ax = axi(fig, x0 + 0.75, 0.95, 3.0, 2.1)
        vp, vh = [np.nanmedian(pv[m]) for m, *_ in groups], [np.nanmedian(hv[m]) for m, *_ in groups]
        ax.bar(x - 0.2, vp, 0.38, color=INK, label="predictor"); ax.bar(x + 0.2, vh, 0.38, color=PALE, label="hold")
        for xx, v_ in zip(list(x - 0.2) + list(x + 0.2), vp + vh):
            ax.text(xx, v_ + ymax * 0.02, f"{v_:.0f}", ha="center", va="bottom", fontsize=SM)
        ax.set_xticks(x); ax.set_xticklabels(lab, linespacing=1.0); ax.set_ylabel(yl); ax.set_ylim(0, ymax); ax.grid(True, axis="y")
        ax.legend(loc="upper left", handlelength=1.0, ncol=2, columnspacing=1.0)
        head(fig, x0, tt)
    save(fig, "sl_bonus_b")


# ======================================================================================= the white-margin failure
def sl_margin():
    fig = new(3.3)
    head(fig, 0.0, "(a) The cause: almost no white margin at the sides of M")
    a = axi(fig, 0.5, 0.5, 1.73, 2.25); show(a, os.path.join(PH, "margin_M.png"), "M, on the black board")
    for xy, xt in (((0.03, 0.5), (-0.25, 0.5)), ((0.97, 0.5), (1.25, 0.5))):
        a.annotate("", xy, xt, xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", lw=1.8, color="k", mutation_scale=15))
    b = axi(fig, 3.2, 0.5, 1.99, 2.25); show(b, os.path.join(PH, "margin_F.png"), "F, full quiet zone")
    spec = (((1.09, 1.01), "width / height of M", "(b) Shape", (0.95, 1.13), 6.2), ((18, 2), "false lean of M [deg]", "(c) Lean", (0, 22), 8.35),
            ((3.5, 0.7), "jitter [mm]", "(d) Jitter", (0, 4.3), 10.5))
    for vals, yl, tt, ylim, x0 in spec:
        ax = axi(fig, x0 + 0.8, 0.95, 1.0, 1.75)
        ax.bar([0, 1], vals, 0.6, color=[PALE, CL])
        for x, v_ in zip((0, 1), vals):
            ax.text(x, v_ + (ylim[1] - ylim[0]) * 0.03, "3–4" if v_ == 3.5 else f"{v_:g}", ha="center", va="bottom", fontsize=SM)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["before", "after"], rotation=30, ha="right", rotation_mode="anchor")
        ax.set_ylabel(yl); ax.set_ylim(*ylim); ax.grid(True, axis="y"); ax.set_xlim(-0.6, 1.6)
        head(fig, x0, tt)
    save(fig, "sl_margin")


# ======================================================================================= equation strips
def eq(name, parts, xs, h=0.6, fs=20):
    fig = new(h)
    for x, s in zip(xs, parts):
        fig.text(x, 0.5, s, fontsize=fs, va="center", ha="left")
    save(fig, name, data=False)


def equations():
    eq("eq_pose", [r"${}^{F}\mathbf{T}_{M}=\left({}^{C}\mathbf{T}_{F}\right)^{-1}\,{}^{C}\mathbf{T}_{M}$",
                   r"$e_t=\Vert\,\hat{\mathbf{t}}-\mathbf{t}\,\Vert$  [mm]",
                   r"$e_R=\arccos\,[\,(\mathrm{tr}\,(\mathbf{R}^{T}\hat{\mathbf{R}})-1)\,/\,2\,]$  [deg]"], [0.0, 0.36, 0.62])
    eq("eq_net", [r"output:  $(\mathbf{a}_1,\ \mathbf{a}_2;\ x/Z,\ y/Z,\ \log Z)$   (6-D rotation; position)",
                  r"loss:  $w_R\,\Vert\hat{\mathbf{R}}-\mathbf{R}\Vert_F^{2}+w_t\,\Vert\hat{\tau}-\tau\Vert_1$"], [0.0, 0.62])
    eq("eq_kalman", [r"hidden:  $\mathbf{x}_{k+1}=\mathbf{F}\,\mathbf{x}_k$,   $\mathbf{P}_{k+1}=\mathbf{F}\,\mathbf{P}_k\,\mathbf{F}^{T}+\mathbf{Q}(\sigma_a)$,   $\mathbf{x}=(\mathbf{p},\mathbf{v})$",
                     r"lost when  $3\sigma>2L$,   $\sigma^2=\mathrm{tr}\,\mathbf{P}_{pp}$"], [0.0, 0.64])


FIGS = dict(setup=sl_setup, t2=sl_t2, pipeline=sl_pipeline, calib=sl_calib, t3=sl_t3, t4=sl_t4, network=sl_network, arch=sl_arch, data=sl_data, synth=sl_synth,
            twostage=sl_twostage, finetune=sl_finetune, rounds=sl_rounds, e1=sl_e1, e2=sl_e2, e3=sl_e3, e4=sl_e4, cdf=sl_cdf, bonus_a=sl_bonus_a,
            bonus_b=sl_bonus_b, margin=sl_margin, eq=equations)

if __name__ == "__main__":
    if os.path.exists(META_FILE):
        META.update(json.load(open(META_FILE)))
    for k in (sys.argv[1:] or FIGS):
        FIGS[k]()
    json.dump(META, open(META_FILE, "w"), indent=1)
    bad = {k: v for k, v in PROBLEMS.items() if v}
    print("slide figures ->", OUT, "| checks:", "ALL CLEAN" if not bad else f"{len(bad)} with problems: {', '.join(bad)}")
