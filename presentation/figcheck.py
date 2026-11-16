"""figcheck.py -- automatic overlap check for a matplotlib figure (used by make_figures.py and the slide figures).

check(fig, name) draws the figure and reports
  1. two text labels that overlap (titles, axis labels, tick labels, legend entries, annotations, figure texts);
  2. a text that reaches into the plotting area of ANOTHER panel;
  3. a legend, or a free text inside a panel, that lies on plotted data of that panel (lines, bars, scatter points).
Returns the list of problems (empty = clean).  3D axes are skipped (their labels are projected): look at those by eye.
"""
import numpy as np
from matplotlib.text import Text

TOL = 1.0          # px of allowed touch


def _hit(a, b, tol=TOL):
    return a.x0 < b.x1 - tol and b.x0 < a.x1 - tol and a.y0 < b.y1 - tol and b.y0 < a.y1 - tol


def _poly(t, bb, tol=TOL):
    """Corners of the label's own rectangle (rotated labels: the rotated rectangle, not its upright bounding box), shrunk by tol."""
    ang = float(t.get_rotation()) % 180.0 if isinstance(t, Text) else 0.0
    cx, cy = 0.5 * (bb.x0 + bb.x1), 0.5 * (bb.y0 + bb.y1)
    if min(ang % 90.0, 90.0 - ang % 90.0) < 1.0:
        w, h, th = bb.width, bb.height, 0.0
    else:
        th = np.radians(ang)
        c, s_ = abs(np.cos(th)), abs(np.sin(th))
        den = c * c - s_ * s_
        if abs(den) < 0.05:                                             # 45 deg: the split is not unique; take the text height from the font
            h = t.get_fontsize() * t.figure.dpi / 72.0 * 1.25 * (t.get_text().count(chr(10)) + 1)
            w = max(1.0, (bb.width + bb.height) / (c + s_) - h)
        else:
            w, h = (bb.width * c - bb.height * s_) / den, (bb.height * c - bb.width * s_) / den
    w, h = max(0.5, w - 2 * tol), max(0.5, h - 2 * tol)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return np.array([cx, cy]) + (np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2.0) @ R.T


def _poly_hit(P, Q):
    """Separating-axis test for two convex polygons."""
    for A in (P, Q):
        for i in range(len(A)):
            e = A[(i + 1) % len(A)] - A[i]
            n = np.array([-e[1], e[0]])
            a, b = P @ n, Q @ n
            if a.max() <= b.min() or b.max() <= a.min():
                return False
    return True


def _inside(P, pts):
    """How many points lie inside the convex polygon P (corners in order)."""
    if not len(pts):
        return 0
    ok = np.ones(len(pts), bool)
    sgn = None
    for i in range(len(P)):
        e, d = P[(i + 1) % len(P)] - P[i], pts - P[i]
        cr = e[0] * d[:, 1] - e[1] * d[:, 0]
        if sgn is None:
            u, w_ = P[1] - P[0], P[2] - P[1]
            sgn = 1.0 if u[0] * w_[1] - u[1] * w_[0] > 0 else -1.0
        ok &= cr * sgn > 0
    return int(ok.sum())


def _is3d(ax):
    return getattr(ax, "name", "") == "3d"


def _texts(fig, r):
    """(text artist, bbox, owner axes or None) of every visible, non-empty label that is really drawn."""
    out, tick_ids = [], set()
    for ax in fig.axes:
        if _is3d(ax):
            for t in ax.findobj(Text):
                tick_ids.add(id(t))
            continue
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            eps = 1e-9 + 1e-6 * abs(hi - lo)
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                for lab in (tick.label1, tick.label2):
                    tick_ids.add(id(lab))
                    if lab.get_visible() and lab.get_text().strip() and lo - eps <= tick.get_loc() <= hi + eps and ax.axison:
                        out.append((lab, lab.get_window_extent(r), ax))
    for t in fig.findobj(Text):
        if id(t) in tick_ids or not t.get_visible() or not t.get_text().strip():
            continue
        ax = t.axes
        if ax is not None and not ax.axison and t in (ax.xaxis.label, ax.yaxis.label, ax.title):
            continue
        try:
            bb = t.get_window_extent(r)
        except Exception:
            continue
        if bb.width >= 1 and bb.height >= 1:
            out.append((t, bb, ax))
    return out


def _data_points(ax):
    """Display coordinates of what is plotted in ax: densified lines, scatter points; and the bar rectangles."""
    pts, boxes = [], []
    for ln in ax.lines:
        if not ln.get_visible():
            continue
        xy = np.column_stack([np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)])
        if len(xy) == 0:
            continue
        try:
            d = ln.get_transform().transform(xy)
        except Exception:
            continue
        d = d[np.isfinite(d).all(1)]
        if len(d) > 1 and ln.get_linestyle() not in ("None", "", " "):
            seg = np.linspace(0, 1, 12)[:, None, None]
            d = (d[None, :-1] * (1 - seg) + d[None, 1:] * seg).reshape(-1, 2)
        pts.append(d)
    for col in ax.collections:
        if not col.get_visible():
            continue
        off = np.asarray(col.get_offsets(), float)
        if off.ndim == 2 and len(off):
            try:
                pts.append(col.get_offset_transform().transform(off))
            except Exception:
                pass
    for cont in ax.containers:
        for p in getattr(cont, "patches", []):
            if p.get_visible() and abs(p.get_height()) > 0:
                boxes.append(p.get_window_extent())
    clip = ax.get_window_extent()
    pts = np.vstack(pts) if pts else np.zeros((0, 2))
    if len(pts):
        pts = pts[(pts[:, 0] >= clip.x0) & (pts[:, 0] <= clip.x1) & (pts[:, 1] >= clip.y0) & (pts[:, 1] <= clip.y1)]
    return pts, boxes


def _on_data(P, pts, boxes):
    """P: polygon of a label / legend.  Points of the data inside it, and bars it touches."""
    n = _inside(P, pts)
    nb = 0
    for b in boxes:
        Q = np.array([[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]])
        nb += bool(_poly_hit(P, Q))
    return n, nb


def check(fig, name="figure", data=True, verbose=True):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tx = _texts(fig, r)
    problems = []
    for i in range(len(tx)):
        for j in range(i + 1, len(tx)):
            if _hit(tx[i][1], tx[j][1]) and _poly_hit(_poly(tx[i][0], tx[i][1]), _poly(tx[j][0], tx[j][1])):
                problems.append(f"text on text: '{tx[i][0].get_text()[:28]}' / '{tx[j][0].get_text()[:28]}'")
    boxes_ax = [(ax, ax.get_window_extent(r)) for ax in fig.axes if not _is3d(ax) and ax.axison]
    for t, bb, owner in tx:
        for ax, ab in boxes_ax:
            if ax is not owner and _hit(bb, ab):
                problems.append(f"text in another panel: '{t.get_text()[:28]}'")
    if data:
        for ax in fig.axes:
            if _is3d(ax) or not ax.axison:
                continue
            pts, boxes = _data_points(ax)
            leg = ax.get_legend()
            if leg is not None:
                n, nb = _on_data(_poly(leg, leg.get_window_extent(r)), pts, boxes)
                if n or nb:
                    problems.append(f"legend on data ({n} points, {nb} bars) in panel '{ax.get_title(loc='left')[:24] or ax.get_ylabel()[:24]}'")
            for t in ax.texts:
                if t.get_visible() and t.get_text().strip():
                    n, nb = _on_data(_poly(t, t.get_window_extent(r)), pts, boxes)
                    if n or nb:
                        problems.append(f"text on data ({n} points, {nb} bars): '{t.get_text()[:28]}'")
    if verbose:
        print(f"   {name}: " + ("clean" if not problems else f"{len(problems)} OVERLAP(S)"))
        for p in problems:
            print("      -", p.replace("\n", " | "))
    return problems
