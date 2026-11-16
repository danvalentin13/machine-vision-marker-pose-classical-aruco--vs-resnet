"""dataflow.py -- the data-flow diagram of Code_Documentation.docx (matplotlib), checked with paper/figcheck.py."""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.dont_write_bytecode = True                      # nothing is written into paper/ (read-only use)
sys.path.insert(0, os.path.join(ROOT, "paper"))
import figcheck  # noqa: E402

BOX = dict(inp="#E8EEF6", data="#FFF4E0", train="#E6F4EA", out="#F2E8F5")
EDGE = "#333333"
BS = "\\"
NL = "\n"


def lines(*parts):
    return NL.join(parts)


# name: (x centre, y centre, width, height, colour, text)
NODES = {
    "hand": (10, 52, 16, 10, "inp", lines("hand-measured", "sizes and truth", "(1.4, 3.3, 6.2, 6.3)")),
    "cam": (10, 32, 16, 8, "inp", lines("laptop camera", "1280x720, ~30 fps")),
    "cfg": (32, 52, 19, 10, "data", lines("data" + BS + "markers.json,", "data" + BS + "calibration" + BS, "board.json (1.1, 1.2, 1.4)")),
    "calib": (32, 12, 19, 10, "data", lines("data" + BS + "calibration" + BS, "camera.npz / .json", "(3.1, 3.2)")),
    "rec": (55, 52, 19, 10, "data", lines("data" + BS + "recordings" + BS, ".avi .csv .json .truth.csv", "(3.3-3.4, 4.1, 6.2-6.3,", "7.2-7.3)")),
    "real": (55, 32, 19, 8, "train", lines("training" + BS + "real_v2..v4" + BS, "17 recordings (5.5)")),
    "syn": (55, 12, 19, 10, "train", lines("training" + BS + "synthetic" + BS, "synth_v1, crop_v1 (5.1, 5.3)", "real sets (5.6)")),
    "pose": (78, 52, 17, 10, "out", lines("output" + BS + "pose" + BS, "<name>_poses.csv", "(3.5, 4.2)")),
    "model": (78, 12, 17, 10, "train", lines("training" + BS + "models" + BS, "<net>.pt", "(5.2, 5.4, 5.6)")),
    "res": (96.5, 32, 12, 24, "out", lines("results in", "output" + BS, "(3.6, 4.3-4.4,", "5.7-5.8,", "6.4-6.6,", "7.4-7.8, 8.1)")),
}
EDGES = [("hand", "cfg"), ("cam", "calib"), ("cfg", "calib"), ("cam", "rec"), ("cam", "real"), ("rec", "pose"),
         ("pose", "res"), ("rec", "res"), ("calib", "syn"), ("syn", "model"), ("real", "syn"), ("model", "res"), ("model", "pose")]
CURVED = [("hand", "rec", -0.32)]                   # drawn above the markers.json box


def anchor(a, b):
    """Points on the borders of boxes a and b, on the line between their centres."""
    (xa, ya, wa, ha, *_), (xb, yb, wb, hb, *_) = NODES[a], NODES[b]

    def border(x, y, w, h, dx, dy):
        sx = (w / 2) / abs(dx) if dx else 1e9
        sy = (h / 2) / abs(dy) if dy else 1e9
        s = min(sx, sy)
        return x + dx * s, y + dy * s
    dx, dy = xb - xa, yb - ya
    return border(xa, ya, wa, ha, dx, dy), border(xb, yb, wb, hb, -dx, -dy)


def make(path):
    fig = plt.figure(figsize=(7.6, 4.3))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 103)
    ax.set_ylim(5, 66)
    ax.axis("off")
    for k, (x, y, w, h, col, text) in NODES.items():
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=BOX[col], ec=EDGE, lw=0.8))
        ax.text(x, y, text, ha="center", va="center", fontsize=6.9, family="DejaVu Sans", linespacing=1.15)
    for a, b in EDGES:
        p, q = anchor(a, b)
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=0.8, color="#555555", shrinkA=3, shrinkB=3))
    for a, b, rad in CURVED:
        (xa, ya, wa, ha, *_), (xb, yb, wb, hb, *_) = NODES[a], NODES[b]
        ax.add_patch(FancyArrowPatch((xa, ya + ha / 2), (xb, yb + hb / 2), arrowstyle="-|>", mutation_scale=9, lw=0.8, color="#555555",
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=4, shrinkB=4))
    ax.text(10, 13, lines("every step reads", "markers.json;", "every pose step", "reads camera.npz"),
            ha="center", va="center", fontsize=6.9, style="italic", linespacing=1.15)
    problems = figcheck.check(fig, "dataflow")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return problems


if __name__ == "__main__":
    print(make(os.path.join(HERE, "dataflow.png")))
