"""build_deck.py -- builds the whole presentation from scratch (python-pptx).

    python presentation/slide_figures.py                 (once: the figures drawn for the slides, and figures/_meta.json)
    python presentation/build_deck.py [output.pptx]      default output: Marker_Pose_Classical_vs_Learned.pptx (in this folder)

Structure: title slide, outline, one section per task of the assignment (T1 ... T6, bonus), each ending with its
conclusion, the failures, and a last slide with the overall conclusions and the references.  Every number follows the
paper (paper/render/paper_text.txt).
Style: white slides, one typeface (Calibri, also in every figure), one accent colour (dark blue) for the section label,
the title rule and the bold labels; blue = classical, orange = network in the figures.
Fixed grid: same margins, same title box, same content area, same place for the 'Conclusion Tn' line on every slide.

Checks (the deck is NOT written if one fails): overlapping boxes, a box outside the margins, a text that needs more
lines than its box holds (measured with the Calibri font files), a body line that wraps, a single word alone on the last
line of a paragraph, a line break that PowerPoint would put at a hyphen, a table cell that wraps, and a figure whose
smallest text would be shown under MIN_FIG_PT (figures/_meta.json gives each figure's smallest font).
"""
import json
import os
import re
import sys

from PIL import Image, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "Marker_Pose_Classical_vs_Learned.pptx")

# ---------------------------------------------------------------------------------------------- the grid [in]
SW, SH = 13.333, 7.5
MX = 0.5                                   # left / right margin
CW = SW - 2 * MX                           # content width
LABEL_Y, LABEL_H = 0.22, 0.24              # section label above the title
TITLE_Y, TITLE_H = 0.48, 0.94              # title box (text sits on its bottom edge)
RULE_Y = 1.47
TOP, BOTTOM = 1.6, 6.91                    # content area
CONCL_Y, CONCL_H = 5.97, 0.94              # the 'Conclusion Tn' line: same place on every slide (up to three lines)
FOOT_Y, FOOT_H = 7.04, 0.26
GAP = 0.12
FONT = "Calibri"
INK, GREY, ACCENT = RGBColor(0x1A, 0x1A, 0x1A), RGBColor(0x66, 0x66, 0x66), RGBColor(0x1F, 0x3A, 0x5F)
TITLE_PT, BODY_PT, FOOT_PT, LABEL_PT, REF_PT = 26, 18, 10, 12, 13
LINE = 1.2                                 # line pitch / font size (Calibri in PowerPoint)
PARA_GAP_PT = 5
MIN_FIG_PT = 11.5                          # smallest text of a placed figure [pt]
SHORT = "Marker pose tracking: classical geometry versus a learned regressor"
ISSUES = []
NATIVE = {k: tuple(v) for k, v in json.load(open(os.path.join(FIG, "_meta.json"))).items()}

_fonts = {}


def _font(bold):
    if bold not in _fonts:
        _fonts[bold] = ImageFont.truetype(os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "calibrib.ttf" if bold else "calibri.ttf"), 200)
    return _fonts[bold]


UNIT = re.compile(r"(\d) (mm|cm|m|px|s|%|fps|min|ms|M|MB|GB|pt)(?=[\s,.;:)/]|$)")


def bind(text):
    """A number and its unit stay on one line (no-break space)."""
    return UNIT.sub("\\1\u00a0\\2", text)


def segments(text):
    """'plain **bold** plain' -> [(text, bold), ...]"""
    return [(part, i % 2 == 1) for i, part in enumerate(text.split("**")) if part]


def width_pt(text, pt, bold=False):
    return sum(_font(b or bold).getlength(t) * pt / 200.0 for t, b in segments(text))


def wrap(text, pt, box_w_in, bold=False):
    """Greedy word wrap measured with the real font (2 % safety).  Returns (lines as word lists, problems):
    a break PowerPoint would put at a hyphen, and a single word alone on the last line."""
    limit, state = box_w_in * 72.0 * 0.98, False
    space = _font(bold).getlength(" ") * pt / 200.0
    lines, cur, cur_w, probs = [], [], 0.0, []

    def wlen(word, st):
        w, parts = 0.0, word.split("**")
        for i, part in enumerate(parts):
            if i > 0:
                st = not st
            if part:
                w += _font(st or bold).getlength(part) * pt / 200.0
        return w, st
    for word in text.split(" "):
        w, new_state = wlen(word, state)
        if not cur or cur_w + space + w <= limit:
            cur_w = w if not cur else cur_w + space + w
            cur.append(word)
        else:
            core = word.replace("**", "")
            if "-" in core[1:-1]:                                     # PowerPoint would break after the first hyphen that fits
                pre = core[:core.index("-", 1) + 1]
                if cur_w + space + width_pt(pre, pt, bold or state) <= limit:
                    probs.append(f"break at a hyphen: '{core}'")
            lines.append(cur)
            cur, cur_w = [word], w
        state = new_state
    lines.append(cur)
    if len(lines) > 1 and len(lines[-1]) == 1:
        probs.append(f"single word on the last line: '{lines[-1][0]}'")
    return lines, probs


def text_height(paras, pt, box_w_in, bold=False):
    n = sum(len(wrap(p, pt, box_w_in, bold)[0]) for p in paras)
    return (n * pt * LINE + (len(paras) - 1) * PARA_GAP_PT) / 72.0 + 0.03, n


# ---------------------------------------------------------------------------------------------- one slide
class Page:
    def __init__(self, prs, number, total):
        self.s = prs.slides.add_slide(prs.slide_layouts[6])
        self.boxes, self.number, self.total = [], number, total

    def issue(self, msg):
        ISSUES.append(f"slide {self.number}: {msg}")

    def reg(self, name, x, y, w, h):
        self.boxes.append((name, x, y, x + w, y + h))

    def check(self):
        eps = 0.005
        for name, x0, y0, x1, y1 in self.boxes:
            if not (x0 >= MX - eps and x1 <= SW - MX + eps and y0 >= 0.2 and y1 <= SH - 0.15):
                self.issue(f"'{name}' leaves the margins ({x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f})")
        for i in range(len(self.boxes)):
            for j in range(i + 1, len(self.boxes)):
                a, b = self.boxes[i], self.boxes[j]
                if a[1] < b[3] - eps and b[1] < a[3] - eps and a[2] < b[4] - eps and b[2] < a[4] - eps:
                    self.issue(f"'{a[0]}' overlaps '{b[0]}'")

    def text(self, name, paras, x, y, w, h, pt, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, max_lines=None, one_line_each=False):
        paras = [bind(p) for p in paras]
        need, lines = text_height(paras, pt, w, bold)
        if need > h + 1e-6:
            self.issue(f"text '{name}' needs {need:.2f} in, its box is {h:.2f} in")
        if max_lines is not None and lines > max_lines:
            self.issue(f"text '{name}' needs {lines} lines (allowed {max_lines})")
        for p in paras:
            ls, probs = wrap(p, pt, w, bold)
            for pr in probs:
                self.issue(f"{name}: {pr}")
            if one_line_each and len(ls) > 1:
                self.issue(f"line wraps ({width_pt(p, pt, bold) / (w * 72 * 0.98):.3f} of the width): {p}")
        tb = self.s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap, tf.auto_size, tf.vertical_anchor = True, MSO_AUTO_SIZE.NONE, anchor
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        for k, p in enumerate(paras):
            par = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            par.alignment, par.line_spacing = align, 1.0
            if k > 0:
                par.space_before = Pt(PARA_GAP_PT)
            for t, b in segments(p):
                r = par.add_run()
                r.text = t
                r.font.name, r.font.size, r.font.bold = FONT, Pt(pt), bool(b or bold)
                r.font.color.rgb = ACCENT if (b and not bold) else color
        self.reg(name, x, y, w, h)
        return need

    def rule(self, y, x0=MX, x1=SW - MX, pt=1.0, color=ACCENT):
        ln = self.s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x0), Inches(y), Inches(x1), Inches(y))
        ln.line.color.rgb, ln.line.width = color, Pt(pt)
        ln.shadow.inherit = False
        self.reg("rule", x0, y - 0.01, x1 - x0, 0.02)

    def picture(self, file, y, max_h, max_w=CW, min_pt=MIN_FIG_PT):
        path = os.path.join(FIG, file)
        wpx, hpx = Image.open(path).size
        w = min(max_w, max_h * wpx / hpx)
        h = w * hpx / wpx
        dpi, small = NATIVE[file]
        shown = small * w / (wpx / dpi)
        if shown < min_pt:
            self.issue(f"{file} would be shown with {shown:.1f} pt text (< {min_pt})")
        x = MX + (CW - w) / 2
        self.s.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
        self.reg(file, x, y, w, h)
        return h, shown

    def frame(self, title, section=None):
        if section:
            self.text("section", [section], MX, LABEL_Y, CW, LABEL_H, LABEL_PT, ACCENT, max_lines=1)
        self.text("title", [title], MX, TITLE_Y, CW, TITLE_H, TITLE_PT, INK, bold=True, anchor=MSO_ANCHOR.BOTTOM, max_lines=2)
        self.rule(RULE_Y)
        self.text("footer", [SHORT], MX, FOOT_Y, 9.0, FOOT_H, FOOT_PT, GREY)
        self.text("page", [f"{self.number} / {self.total}"], SW - MX - 1.2, FOOT_Y, 1.2, FOOT_H, FOOT_PT, GREY, align=PP_ALIGN.RIGHT)

    def conclusion(self, text):
        self.text("conclusion", [text], MX, CONCL_Y, CW, CONCL_H, BODY_PT, max_lines=3)

    def notes(self, text):
        self.s.notes_slide.notes_text_frame.text = text


# ---------------------------------------------------------------------------------------------- table without fills
def _border(tag, pt=None, rgb="1A1A1A"):
    ln = OxmlElement(tag)
    if pt is None:
        ln.set("w", "0")
        ln.append(OxmlElement("a:noFill"))
        return ln
    ln.set("w", str(int(pt * 12700))); ln.set("cap", "flat"); ln.set("cmpd", "sng"); ln.set("algn", "ctr")
    sf = OxmlElement("a:solidFill"); c = OxmlElement("a:srgbClr"); c.set("val", rgb); sf.append(c); ln.append(sf)
    d = OxmlElement("a:prstDash"); d.set("val", "solid"); ln.append(d)
    return ln


def sober_table(page, rows, col_w, y, row_h, pt):
    """A table with three horizontal rules (top, under the header, bottom), no fills, no vertical lines."""
    n, m = len(rows), len(rows[0])
    for r in rows:
        for j, cell in enumerate(r):
            if width_pt(cell, pt, bold=(r is rows[0])) > (col_w[j] - 0.2) * 72:
                page.issue(f"table cell does not fit on one line: {cell}")
    gf = page.s.shapes.add_table(n, m, Inches(MX), Inches(y), Inches(sum(col_w)), Inches(n * row_h))
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    for a in ("firstRow", "bandRow", "firstCol", "lastRow", "lastCol", "bandCol"):
        tblPr.set(a, "0")
    for el in list(tblPr):
        if el.tag.endswith("tableStyleId"):
            el.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"          # "No Style, No Grid"
    for j, w in enumerate(col_w):
        tbl.columns[j].width = Inches(w)
    for i in range(n):
        tbl.rows[i].height = Inches(row_h)
        for j in range(m):
            cell = tbl.cell(i, j)
            cell.margin_left, cell.margin_right, cell.margin_top, cell.margin_bottom = Inches(0.1), Inches(0.1), Inches(0.02), Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = False
            par = tf.paragraphs[0]
            par.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = par.add_run()
            r.text = rows[i][j]
            r.font.name, r.font.size, r.font.bold = FONT, Pt(pt), i == 0
            r.font.color.rgb = INK
            tcPr = cell._tc.get_or_add_tcPr()
            top = 1.5 if i == 0 else None
            bot = 0.75 if i == 0 else (1.5 if i == n - 1 else None)
            for k, el in enumerate((_border("a:lnL"), _border("a:lnR"), _border("a:lnT", top), _border("a:lnB", bot))):
                tcPr.insert(k, el)
            tcPr.append(OxmlElement("a:noFill"))
    page.reg("table", MX, y, sum(col_w), n * row_h)
    return n * row_h



# ---------------------------------------------------------------------------------------------- content
D, PM, X, ND = "\u00b0", "\u00b1", "\u00d7", "\u2013"
AUTHORS = "Dan-Valentin Sutu,  Mihai-Daniel Carp,  Ilie-Catalin Jureschi"
SEC = {"T1": "T1   Markers and set-up", "T2": "T2   Detection", "T3": "T3   Calibration and classical pose", "T4": "T4   Recording and animation",
       "T5": "T5   Learned pose: network, data, training, fine-tuning", "T6": "T6   Comparison on the test recordings (E1" + ND + "E6)",
       "B": "Bonus   Prediction through an occlusion", "F": "Failures, conclusions, references"}

SLIDES = [
    # ------------------------------------------------------------------------------------------ title
    dict(kind="title",
         title="Marker pose tracking with a laptop camera: classical fiducial geometry versus a two-stage learned regressor",
         lines=[AUTHORS, "\u201cGheorghe Asachi\u201d Technical University of Ia\u0219i",
                "Machine Vision, Assignment 2: marker pose tracking, classical vs learned  \u00b7  November 2026"],
         about=["One laptop camera, two printed 77 mm ArUco markers: the pose of the mobile marker in the frame of the fixed one.",
                "Two pipelines, same recordings: fiducial geometry, and a ResNet-18 regressor trained on synthetic and real frames.",
                "Ground truth from a ruler and a protractor. The talk follows tasks T1 to T6 of the assignment and the occlusion bonus."],
         code="Code and results:  github.com/danvalentin13/machine-vision-marker-pose-classical-aruco--vs-resnet",
         notes="Project presentation of the team: Dan-Valentin Sutu, Mihai-Daniel Carp, Ilie-Catalin Jureschi. We measure the pose of one printed marker "
               "relative to another with a laptop camera, in two ways, and compare them on the same recordings against a truth measured with a ruler and a "
               "protractor. Code, result tables and figures are in the repository named on the slide."),
    # ------------------------------------------------------------------------------------------ outline
    dict(kind="outline",
         title="Outline: the talk follows the tasks of the assignment, and each task ends with its conclusion from our measurements",
         lines=["**T1   Markers and set-up:**  dictionary, IDs, printed size, bench, truth convention",
                "**T2   Detection:**  live demo, frame rate, limits in distance and tilt",
                "**T3   Calibration and classical pose:**  camera model, pose from four corners, two checks",
                "**T4   Recording and animation:**  the pose of the mobile marker over time",
                "**T5   Learned pose:**  network, data, training on synthetic images, fine-tuning on real frames",
                "**T6   Comparison:**  experiments E1 to E6 on test recordings with measured truth",
                "**Bonus   Prediction through an occlusion:**  a Kalman predictor against holding the last pose",
                "**Failures, conclusions, references**"],
         notes="The structure is the task list of the assignment. Every task section ends with a line 'Conclusion Tn' that states the result with the "
               "measured numbers. The last slide gives the overall conclusions and the references."),
    # ------------------------------------------------------------------------------------------ T1
    dict(sec="T1", title="Two printed ArUco markers and one laptop camera; a ruler and a protractor give the truth in the frame of F",
         fig="sl_setup.png",
         concl="**Conclusion T1:** DICT_6X6_250, IDs 0 (fixed) and 23 (mobile). The print came out at 77 mm instead of 80 mm (page scaled to 96 %), so "
               "the 1 mm check is not met; the measured size is used everywhere (the nominal one would make every distance 3.9 % too long, 13 mm at the "
               "largest spacing).",
         notes="Fixed marker F and mobile marker M, laptop camera at 1280 by 720 and about 30 frames per second. Truth of a test recording: s along the board "
               "and n in front of it with a ruler, the turn angle psi with a protractor, expressed in the frame of F (x to the right, y up, z toward the "
               "camera; roll-pitch-yaw is ZYX). Panel b is the bench of the bonus task: M slides along a ruler behind a box. The assignment asks for a printed "
               "side within 1 mm of the intended 80 mm; ours is 77 mm, so that check is not met and we say so."),
    # ------------------------------------------------------------------------------------------ T2
    dict(sec="T2", title="Both markers are detected at the camera frame rate, up to 80" + D + " of tilt and beyond the 1 m of the bench",
         fig="sl_t2.png",
         body=["Distance and tilt from the pose agree with tape and protractor within 2.9 cm and 4" + D + " (tilts from 30" + D + ")."],
         concl="**Conclusion T2:** detection keeps up with the camera (29" + ND + "31 fps) and both IDs were read in all four live steps. M is detected up to "
               "80" + D + " of tilt up to 75 cm and 75" + D + " at 1 m; the distance limit was not reached within 1 m (8.4 px per cell measured there; "
               "about 4.2 m expected from the 2 px rule).",
         notes="29 to 31 frames per second is the camera's own frame rate: detection and drawing keep up with it. The 147 frames per second of experiment E6 "
               "is the processing speed of the classical pipeline on recorded frames, which is not limited by the camera. Panel a: ten frames per setting, a "
               "setting passes with nine detections; the small number is the size of one marker cell in pixels (measured at 0 degrees times the cosine of the "
               "tilt). At 1 m a cell measures 8.4 px (the formula gives 8.1 px). At 0 degrees the pose reports tilts of 2 to 8 degrees: the planar ambiguity."),
    # ------------------------------------------------------------------------------------------ T3
    dict(sec="T3", title="Both pipelines output the pose of M in the frame of F; the classical one measures each marker from its four corners",
         fig="sl_pipeline.png", eq="eq_pose.png",
         body=["Per marker: ArUco corners (sub-pixel), two IPPE solutions, Levenberg" + ND + "Marquardt refinement of the reprojection error.",
               "Planar ambiguity: if both solutions fit almost equally (ratio under 1.5), the one closest to the previous frame is kept."],
         notes="The relative pose is the inverse of camera-to-F times camera-to-M, so the camera position cancels. Errors: e_t is the distance between the "
               "estimated and the true position, e_R the angle of the rotation between them. If F is not seen and the camera is still, the last pose of F is "
               "held. The learned branch replaces only the pose of M; F always comes from the classical branch."),
    dict(sec="T3", title="The calibration reaches 1.18 px RMS: usable, but limited by a printed chessboard that was not flat",
         fig="sl_calib.png",
         body=["20 chessboard views (9 " + X + " 6 corners, 24 mm squares), radial k1" + ND + "k2 model: fx 846.6 px, fy 845.2 px, field of view 74" + D + ".",
               "Four held-out views re-project at 0.58 px, but they are also among the best views of the full fit.",
               "fx moves between 833 and 860 px when single views are left out: a scale uncertainty of about " + PM + "1.5 %."],
         notes="Zhang's method with OpenCV, radial model with two coefficients. Panel a: the reprojection error of every view, 0.32 to 2.44 px; the light bars "
               "are the four views held out in a second calibration, the black dashes their error when held out. Because these four fit well in the full "
               "calibration too, the held-out test says little about the bad views. Panel c: one view rectified with a homography: the grid lines bend, the "
               "paper was not flat on its clipboard. Panel d: the residuals grow from 0.8 px near the centre to 1.8 px in the outer ring."),
    dict(sec="T3", title="Against the ruler the classical pose is 4" + ND + "14 mm off, a bias rather than noise; moving the camera does not change it",
         fig="sl_t3.png",
         body=["Largest parts: depth along F's normal (+3 to +9 mm) and the widest spacing, measured 11 mm (3.4 %) too short."],
         concl="**Conclusion T3:** calibration RMS 1.18 px (0.58 px on held-out views) with a board that was not flat: scale uncertain by about " + PM
               + "1.5 %. Check A: 8.8 mm (95th percentile 13.6) and 3.96" + D + " with 0.2" + ND + "0.5 mm jitter, a bias, not noise. Check B: the camera moved "
               "126 mm and 11" + D + ", the relative pose stayed within 11.4 mm and 3.8" + D + " (RMS 2.7 mm, 0.94" + D + ").",
         notes="Check A: M at five ruler positions next to F, 100 frames each; the bars are the mean difference camera minus ruler per axis. The jitter is 0.2 "
               "to 0.5 mm, so averaging frames does not remove the error. Check B: the markers stand still and the laptop is moved; grey is the camera motion "
               "seen through F, blue the change of the relative pose, both against their mean. The distance between the centres is 276.9 mm against 277 mm by "
               "ruler."),
    # ------------------------------------------------------------------------------------------ T4
    dict(sec="T4", title="The recorder and the animation show the pose of M in the frame of F over time, live and exported",
         fig="sl_t4.png",
         body=["T4 recording: 30 s, 900 frames of free hand motion. Gaps: frames without a detection of M (cause not logged)."],
         concl="**Conclusion T4:** the recorder and the animation (camera image, 3-D view, x y z and roll pitch yaw against time) work live and exported. "
               "In the 30 s recording the classical pipeline has a pose in 88 % of the frames and the network follows it within 4.3 mm and 2.2" + D + " (median).",
         notes="This is the data behind the T4 animation, which is shown next to the camera video: the path of M in the frame of F and the six pose curves. "
               "Blue is the classical pose, orange the network; the network curve is drawn only where the detector saw M, because the network answers in every "
               "frame. The gaps are frames where M moved fast or was partly outside the picture; the cause was not logged."),
    # ------------------------------------------------------------------------------------------ T5
    dict(sec="T5", title="The learned pipeline has two stages: the first finds M in the whole image, the second regresses its pose from a crop",
         fig="sl_network.png", eq="eq_net.png",
         body=["Rotation: two 3-vectors made orthonormal (6-D), chordal loss. Position: image direction and log depth, L1 loss.",
               "Equal loss weights. The crop shows M about 112 px wide instead of 65 px, plus the viewing ray of each pixel."],
         notes="Stage 1 sees the undistorted frame at 640 by 360 and gives a coarse position. Stage 2 sees a 224 by 224 crop, twice the apparent marker side, "
               "cut from the full-resolution frame; the crop is re-centred once on the answer of stage 2, so stage 1 runs once and stage 2 twice per frame. "
               "A network is valid only for the camera and marker size it was trained with."),
    dict(sec="T5", title="Each stage is a ResNet-18 with a nine-number head: four layers of basic blocks reduce the image to 512 features",
         fig="sl_arch.png",
         body=["ImageNet-pretrained; the input is grey, so the first convolution is summed over the colour channels.",
               "Stage 1: 11.17 M parameters. Stage 2: 11.18 M, with two extra ray channels whose weights start at zero."],
         notes="Top: the blocks. The stem is a 7 by 7 convolution with 64 channels and stride 2, batch normalisation, ReLU and a 3 by 3 max pool with stride 2. "
               "Layers 1 to 4 have two basic blocks each, with 64, 128, 256 and 512 channels; layers 2 to 4 halve the size. Global average pooling gives 512 "
               "numbers and the linear head nine: six for the rotation, three for the position. Middle: the size of every output for both stages. Bottom: a "
               "basic block with its skip connection. The head starts at the identity rotation and a depth of 0.6 m."),
    dict(sec="T5", title="Training data: 40 000 synthetic images with exact labels, then 17 real recordings labelled by the classical pipeline",
         fig="sl_data.png",
         body=["Synthetic: 10 000 whole images, 30 000 crops; random pose (0.25" + ND + "1 m, tilt up to 60" + D + "), background, light, blur, noise.",
               "Real: 17 recordings, 20 450 frames, 15 287 labelled (reprojection error up to 1 px): 11 562 training, 3 725 validation.",
               "Split by recording: four whole recordings form the validation set; no test recording was used for training."],
         notes="The synthetic images are rendered with our own camera matrix (domain randomisation); in 70 % F appears as a distractor and 15 % have a patch "
               "over part of M. The real frames were recorded in three sessions with M moved by hand. Their labels are the classical poses, so the network "
               "inherits the classical bias. Panel c shows the degradations of the software hard-case test; the same kinds, with random strength, are the "
               "hard-case augmentation of round 3."),
    dict(sec="T5", title="On synthetic images the longer schedule gave the main gain; doubling the rotation weight did not help",
         fig="sl_synth.png",
         body=["A: 15 epochs, 23.5" + D + " / 26.4 mm.  B: 40 epochs, 7.3" + D + " / 13.5 mm (chosen).  C: rotation weight 2, 8.9" + D + " / 16.7 mm.",
               "D, the crop stage: 80 epochs on 30 000 crops, 1.95" + D + " / 8.8 mm on its own validation crops.",
               "AdamW, one-cycle learning rate (peak 5 " + X + " 10\u207b\u2074), batch 64, seed 0; one run each."],
         notes="Medians on the synthetic validation set; dotted lines are 95th percentiles. Run A was stopped too early: its rotation loss was still falling. "
               "Run C doubles the rotation weight and is slightly worse in both errors, so equal weights were kept; with one seed per run, 1 to 2 degrees may "
               "be run-to-run variation. The final rotation loss of B, 0.0145, corresponds to 4.9 degrees, below its validation error of 7.3: the network fits "
               "its training images better than new ones. Times: 7.5, 20.7, 20.3 and 27.1 minutes."),
    dict(sec="T5", title="On 1 000 new synthetic images the second stage removes most of the rotation error: usable poses rise from 50.5 % to 83.2 %",
         fig="sl_twostage.png",
         body=["Stage 1 alone: 7.6" + D + " and 14.3 mm (medians). Two-stage: 1.7" + D + " and 9.7 mm. Usable: within 20 mm and 10" + D + " of the truth.",
               "The position error still grows with distance (5 mm near, 17 mm beyond 0.6 m): depth comes from the apparent size."],
         notes="End-to-end test on full-resolution synthetic images that were used neither for training nor for tuning. The crop gives the second stage about "
               "twice as many pixels on the marker, which removes most of the tilt error of the whole-image network. Light orange: stage 1 alone."),
    dict(sec="T5", title="Fine-tuning on real frames took three rounds; each round added the cases on which the previous network failed",
         fig="sl_finetune.png",
         body=["Round 1 (3 164 frames): real frames. Round 2 (7 358): M next to F, camera-rotation augmentation, second crop pass.",
               "Round 3 (11 562): camera pitched and turned; hard-case augmentation (cover, blur, low resolution, dim light).",
               "Stage 1 jumps between 10 and 95 mm from epoch to epoch (14" + ND + "29 mm when kept); stage 2 settles at 6" + ND + "7 mm."],
         notes="Every batch is half real, half synthetic; learning rate 1e-4; the best validation epoch is kept. Round 1 validates on one upright recording, "
               "round 2 on the recording with the pitched screen, round 3 on four whole recordings, so the curves of different rounds are not directly "
               "comparable. Round 1 left one position failing: P4, with M right next to F (34 mm for the two-stage network, 103 mm for stage 1 alone); round 2 "
               "brought it to 2.8 mm. The first attempt of round 3 stalled when the images filled the 16 GB of the graphics card."),
    dict(sec="T5", title="Real frames close the gap: 26.5 mm with synthetic data only, 6.7 mm after the third round (E1 positions)",
         fig="sl_rounds.png",
         concl="**Conclusion T5:** on synthetic images alone the network fails on real frames (26.5 mm median, 107 mm at the 95th percentile). Three rounds of real "
               "frames bring it to 6.7 mm, 3.6" + D + " and 1.3 mm jitter on the E1 positions (which informed the design). Both stages are needed "
               "(stage 1 alone: 12.6 mm, 71 mm at one position); equal loss weights were kept.",
         notes="Panel a: the two-stage network on the five E1 positions after each step: median 26.5, 10.0, 9.2, 6.7 mm; jitter 26.5, 16.3, 2.1, 1.3 mm; the "
               "classical pipeline has 8.8 mm and 0.4 mm there. Panel b: stage 1 of round 2 on the recording with the pitched screen, without and with the "
               "camera-rotation augmentation (13.9 degrees and 92 mm against 6.6 degrees and 29 mm). Panel c: 250 E1 frames degraded in software. With 30 % "
               "covered: 0, 51 and 91 % usable; with a simulated blur of 31 px: 0, 14 and 100 %. Very dark frames defeat all three. These E1 recordings "
               "informed the design choices, so they are not an untouched test."),
    # ------------------------------------------------------------------------------------------ T6
    dict(sec="T6", title="E1, static poses: the network is level at the five positions it was tuned on and worse at the two angles recorded later",
         fig="sl_e1.png",
         body=["All seven: classical 7.8 mm and 3.84" + D + ", network 8.1 mm and 4.04" + D + " (medians); jitter 0.55 against 1.29 mm.",
               "P1" + ND + "P5 informed the design. The two angles, recorded later: classical 6.2 mm, 2.80" + D + "; network 11.8 mm, 5.46" + D + "."],
         notes="P1 to P5: M next to F at five ruler gaps, both facing the camera at about 35 cm. Turned 20 and 40 degrees: M on its own support in front of F, "
               "camera at about 61 cm; recorded after the network was final. Bars are medians, whiskers the 5 to 95 % range; the dotted line separates the two "
               "groups. On all eight static recordings made after the network was final the network has 11.5 mm against 8.0 mm."),
    dict(sec="T6", title="E2, distance and tilt: the classical error stays at 7.5" + ND + "8.7 mm, the network falls behind when M is far and turned",
         fig="sl_e2.png",
         body=["Pooled: classical 8.1 mm (95th percentile 9.1), network 11.0 mm (26.1).",
               "Network: 7.9 / 6.4 / 12.4 mm facing, 11.2 / 11.8 / 25.1 mm turned; at 77 cm and 40" + D + " the depth alone is 24.6 mm off.",
               "Distances: the camera's own estimates (about 40, 61, 75 cm by tape). No 60" + D + " recording; neither method lost M."],
         notes="Six recordings: three camera distances, M facing the camera or turned by 40 degrees. A constant classical error of about 8 mm at every distance "
               "points to the truth and the calibration rather than to the method. Panel c: the mean error along the normal of F; the network places a far, "
               "turned marker too far away, the direction a regressor must infer from the apparent size of a marker about 85 px wide."),
    dict(sec="T6", title="E3, motion along a ruler: both methods are at 6" + ND + "7 mm, and the 20 mm peak of both is the hand lagging the guide",
         fig="sl_e3.png",
         body=["Pooled: classical 6.6 mm and 5.82" + D + ", network 6.3 mm and 5.21" + D + "; while M rests, 5.3" + ND + "6.6 mm.",
               "Truth: the guide outline. On the way back the hand lags 0.6" + ND + "1.0 s (median), up to 2.0 s (20 mm) in recording 1.",
               "Classical outliers in 1, 4 and 6 frames: the outline of M is read 9 % too small, M placed 49" + ND + "58 mm farther."],
         notes="M on its box slides 60 mm along a ruler at 1 cm/s and back; the truth is the schedule of the on-screen outline that the hand follows. The error "
               "peak of about 20 mm appears in both methods at the same moment: it is the lag of the hand, so the E3 figures are upper bounds for both. In the "
               "outlier frames the detected quadrilateral is about 9 % smaller than in the neighbouring frames with a normal reprojection error; the rotation is "
               "not affected, so it is not the planar ambiguity. The cause was not established; a gate on the innovation would reject these frames."),
    dict(sec="T6", title="E4, robustness: dim light, clutter and shaking break neither method; a covered edge silences the detector and misleads the network",
         fig="sl_e4.png",
         body=["Usable (within 20 mm and 10" + D + "): classical 100 / 100 / 100 / 93 / 0 %, network 99 / 98 / 99 / 81 / 0 %.",
               "The network has no detection output: it gives a pose in every frame by construction.",
               "Edge covered: the network's rotation is 2.7" + D + ", but its position is 32.6 mm off, 32 mm of it in depth.",
               "Real strong shaking: the detector kept every frame; it fails only at the simulated 31 px blur of T5."],
         notes="Five recordings at about 61 cm, in the order of the figure: dim light, clutter around the markers, mild shaking, strong shaking of the camera, "
               "and the left edge of M covered by a paper strip (about 20 % of the square). The blur was made by shaking the camera, not by moving the marker "
               "fast. With a fully visible marker the network is 12 to 17 mm off in these recordings against 3 to 4 mm for the classical pipeline."),
    dict(sec="T6", kind="table",
         title="E5 and E6: with an exact camera and exact truth the classical code reaches 1.6 mm; it needs a sixth of the network's time per frame",
         table=[["Measure  (median)", "Classical", "Network"],
                ["E5, 640 " + X + " 360 images: pose given [%]", "87", "100"],
                ["E5, 640 " + X + " 360 images: position, rotation error", "3.8 mm, 0.33" + D, "11.8 mm, 1.96" + D],
                ["E5, 1280 " + X + " 720 images: pose given / usable [%]", "90 / 88", "100 / 85"],
                ["E5, 1280 " + X + " 720 images: position, rotation error", "1.6 mm, 0.11" + D, "6.8 mm, 1.85" + D],
                ["E6 time per frame [ms]", "6.8", "39.8 (networks: 32)"],
                ["E6 frames per second", "147", "25"],
                ["E6 parameters / model files", "none", "11.17 + 11.18 M / 2 " + X + " 44.8 MB"],
                ["E6 training", "none", "48 + 87 min"],
                ["E6 core code [lines]", "266", "770"]],
         cols=[6.0, 2.4, 3.933], row_h=0.37, pt=17,
         note="Held-out synthetic images, 1 000 per set. Network: stage 1 once, stage 2 twice per frame; training 48 min on synthetic data, 87 min fine-tuning.",
         body=["On the bench the classical code has 8 mm and 4" + D + ": most of that error is not the algorithm.",
               "Calibration, the print and its mount, and the hand-measured truth were not separated."],
         notes="E5: rendered images with exact truth in the camera frame. The evaluation program uses the 640 by 360 test split, which understates both methods "
               "and the network more, because its crop is then cut from a half-size image; at the camera's full resolution the classical code reaches 1.6 mm "
               "and 0.11 degrees. Frames without a classical pose are mostly those with a patch over M. E6, same laptop: 6.8 ms against 39.8 ms per frame for "
               "the whole pipeline; the networks take 32 ms on the graphics card and 44 ms on the processor."),
    dict(sec="T6", title="Across all test frames both methods err mostly by a bias; the network has more of both bias and jitter",
         fig="sl_cdf.png",
         body=["Static recordings, medians: classical bias 7.7 mm and jitter 1.0 mm; network bias 11.6 mm and jitter 2.1 mm.",
               "The bias is at least twice the jitter in every static recording except the strongly shaken camera."],
         notes="Left and middle: cumulative distributions of the position and rotation error over all frames of E1 to E4, per experiment; frames without a pose "
               "are left out. Right: bias against jitter for every static recording of E1, E2 and E4; the classical jitter ranges from 0.2 to 1.5 mm. More "
               "frames would not help either method; a better truth and calibration would."),
    dict(sec="T6", kind="table",
         title="Summary of E1 to E6: about equal in E1 and E3, classical clearly better in E2, twice as steady and six times faster",
         table=[["Measure  (median; 95th percentile in brackets)", "Classical", "Network"],
                ["E1 static: position error [mm]", "7.8 (13.6)", "8.1 (13.0)"],
                ["E1 static: rotation error [deg]", "3.84 (6.04)", "4.04 (6.48)"],
                ["E1 static: jitter [mm]", "0.55", "1.29"],
                ["E2 distance and tilt: position error [mm]", "8.1 (9.1)", "11.0 (26.1)"],
                ["E3 motion: position error [mm]", "6.6 (12.4)", "6.3 (13.7)"],
                ["E3 motion: rotation error [deg]", "5.82 (9.54)", "5.21 (7.11)"],
                ["E4 robustness: pose given / usable [%]", "80 / 79", "100 / 76"],
                ["E6 speed [frames per second]", "147", "25"]],
         cols=[6.9, 2.7, 2.733], row_h=0.36, pt=17,
         note="On the 16 recordings made after the network was final: classical more accurate on 11, network on 4 (by 2 mm at most).",
         concl="**Conclusion T6:** visible marker: classical is about as accurate in E1 and E3, clearly better in E2 (8.1 vs 11.0 mm), twice as "
               "steady (0.55 vs 1.29 mm) and six times faster (147 vs 25 fps). The network answers where the detector cannot (covered edge), but "
               "coarsely (32.6 mm), and degrades far and turned (25 mm at 77 cm, 40" + D + ").",
         notes="The comparison table of the assignment. E1 pools seven recordings, E2 six, E3 three, E4 five. On the sixteenth recording made after the network "
               "was final, the one with a covered edge, only the network answers. E5 is on the previous slide."),
    # ------------------------------------------------------------------------------------------ bonus
    dict(sec="B", title="A constant-velocity Kalman predictor keeps the hidden marker under 20 mm three times longer than holding its last pose",
         fig="sl_bonus_a.png", eq="eq_kalman.png",
         body=["Uncovered passes hidden in software (exact truth): 25 vs 82 mm after 1 s at 7 cm/s, 94 vs 165 mm at 17 cm/s.",
               "Rotation: alpha" + ND + "beta filter; extrapolating the turning speed gave 42" + D + ", fading it out about 11" + D + "."],
         notes="Position: constant-velocity Kalman filter with an acceleration noise of 0.15 m/s2, a measurement noise of 4 mm and a chi-square gate. While M "
               "is hidden the filter only predicts and its covariance grows; the pose is declared lost when three sigma exceeds two marker sides, after about "
               "1.4 s. The turning speed is faded out with a time constant of 0.1 s; that constant was chosen on these uncovered recordings. The 42 against "
               "11 degrees are from the fast uncovered passes."),
    dict(sec="B", title="Behind a real obstacle the predictor beats holding the last pose in 44 of 49 passes; the rotation is not predicted better",
         fig="sl_bonus_b.png",
         body=["Our first evaluation mis-detected the hidden stretch and reported 35" + ND + "41 mm; these are the corrected numbers."],
         concl="**Conclusion Bonus:** the predictor beats holding the last pose in 44 of 49 passes: 50 vs 174 mm after 1.7 s at 8" + ND + "12 cm/s, 96 vs "
               "182 mm after 1.2 s at 13" + ND + "23 cm/s, a half to a third of the error. It stays under 20 mm for 0.85 s (hold: 0.26 s) at 7 cm/s; the "
               "rotation is not predicted better than held; the pose is declared lost after about 1.4 s.",
         notes="M slides by hand along a ruler behind a box, paced by an outline on the screen; the error is measured at the moment M is seen again, where the "
               "truth is exact. Under each group: the median time hidden. At 4 to 6 cm/s M is hidden for about 3 s, longer than the task asks: 56 against "
               "158 mm. Rotation at reappearance: 20 to 31 degrees against 18 to 22 degrees; these recordings were made before the white-margin fix. The first "
               "evaluation took the hidden stretch from the first to the last missing frame; it now uses the longest run of missing frames."),
    # ------------------------------------------------------------------------------------------ failures, conclusions
    dict(sec="F", title="The largest errors came from the set-up and from our own evaluation code, not from the two algorithms",
         fig="sl_margin.png",
         body=["White margin of M cut to 1 mm: a false 18" + D + " lean and 2 cm in depth; a white backing sheet fixed it.",
               "Made before the fix: the five E1 positions and the occlusion recordings; after it: E1 angles, E2, E3, E4.",
               "Bonus evaluation: the hidden stretch was mis-detected; 35" + ND + "41 mm reported first, 50" + ND + "96 mm after the correction.",
               "Synthetic-only network: 26.5 mm on real frames; a tilted screen gave 92 mm before the rotation augmentation."],
         notes="The white margin: on the black board the corners found by the detector slipped outward, the square was measured 9 % too wide, and the pose "
               "turned that into a lean. It was found only by enlarging the camera image; the boards really were flat. Other failures are listed in the paper: "
               "training that stalled when the graphics memory filled, two E3 takes discarded, F mounted a quarter turn rotated. Lesson: look at the raw "
               "measurement before blaming the method."),
    dict(sec="F", kind="final", title="Conclusions: the classical pipeline for a visible marker; the network only as a checked fallback",
         big=["**Which pipeline:** the classical one. About as accurate in E1 and E3, better in E2, twice as steady and six times faster; it needs no training data.",
              "**Where each fails:** the detector gives no pose for a covered border; the network always answers, without a confidence, and is coarse when M "
              "is covered or far and turned (25" + ND + "33 mm).",
              "**Occlusion:** predicting the position helps for about a second; the rotation is better held than extrapolated.",
              "**What we would change first:** a flat calibration board; then a rail for an exact motion truth and a confidence output for the network."],
         refs=["[1] S. Garrido-Jurado et al., \u201cAutomatic generation and detection of highly reliable fiducial markers under occlusion,\u201d Pattern Recognit., 2014.",
               "[2] T. Collins and A. Bartoli, \u201cInfinitesimal plane-based pose estimation,\u201d Int. J. Comput. Vis., 2014.",
               "[3] Z. Zhang, \u201cA flexible new technique for camera calibration,\u201d IEEE Trans. Pattern Anal. Mach. Intell., 2000.",
               "[4] K. He et al., \u201cDeep residual learning for image recognition,\u201d CVPR, 2016.",
               "[5] Y. Zhou et al., \u201cOn the continuity of rotation representations in neural networks,\u201d CVPR, 2019.",
               "[6] J. Tobin et al., \u201cDomain randomization for transferring deep neural networks from simulation to the real world,\u201d IROS, 2017.",
               "[7] R. E. Kalman, \u201cA new approach to linear filtering and prediction problems,\u201d J. Basic Eng., 1960.",
               "[8] P. Corke, Robotics, Vision and Control, 3rd ed. Springer, 2023."],
         notes="Recommendation: use the classical pipeline whenever the marker is fully visible; add the network as a fallback for frames without a detection, "
               "but treat its answer as coarse, because it always answers. Limits: one camera, one room, a ruler truth of a few millimetres, a calibration "
               "scale uncertain by about 1.5 %, a hand-moved marker; the five E1 positions informed design choices. References: ArUco [1], IPPE [2], the "
               "calibration method [3], ResNet [4], the 6-D rotation [5], domain randomisation [6], the Kalman filter [7], the course textbook [8]."),
]


# ---------------------------------------------------------------------------------------------- build
def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)
    total, report = len(SLIDES), []
    for k, d in enumerate(SLIDES, 1):
        pg = Page(prs, k, total)
        kind, sec, shown = d.get("kind", "figure"), SEC.get(d.get("sec")), None
        if kind == "title":
            pg.text("title", [d["title"]], MX, 0.85, CW, 1.3, 30, INK, bold=True, anchor=MSO_ANCHOR.BOTTOM, max_lines=2)
            pg.rule(2.3)
            pg.text("authors", [d["lines"][0]], MX, 2.5, CW, 0.45, 22, INK, max_lines=1)
            pg.text("affiliation", [d["lines"][1]], MX, 3.07, CW, 0.38, 18, INK, max_lines=1)
            pg.text("course", [d["lines"][2]], MX, 3.53, CW, 0.36, 16, GREY, max_lines=1)
            h, _ = text_height(d["about"], BODY_PT, CW)
            pg.text("about", d["about"], MX, 4.5, CW, h + 0.02, BODY_PT, one_line_each=True)
            pg.text("code", [d["code"]], MX, 6.05, CW, 0.36, 16, GREY, max_lines=1)
        elif kind == "outline":
            pg.frame(d["title"])
            h, _ = text_height(d["lines"], 20, CW)
            pg.text("outline", d["lines"], MX, TOP + 0.25, CW, h + 0.02, 20, one_line_each=True)
        elif kind == "final":
            pg.frame(d["title"], sec)
            h, _ = text_height(d["big"], BODY_PT, CW)
            pg.text("conclusions", d["big"], MX, TOP + 0.08, CW, h + 0.02, BODY_PT)
            y = TOP + 0.08 + h + 0.02 + 0.22
            pg.text("references heading", ["References"], MX, y, CW, 0.3, 14, INK, bold=True, max_lines=1)
            y += 0.3 + 0.06
            cw, half = (CW - 0.4) / 2, (len(d["refs"]) + 1) // 2
            ends = []
            for c, part in enumerate((d["refs"][:half], d["refs"][half:])):
                hr, _ = text_height(part, REF_PT, cw)
                pg.text(f"references {c + 1}", part, MX + c * (cw + 0.4), y, cw, hr + 0.02, REF_PT, INK)
                ends.append(y + hr + 0.02)
            if max(ends) > BOTTOM + 1e-6:
                pg.issue(f"content ends at {max(ends):.2f} in")
        elif kind == "table":
            pg.frame(d["title"], sec)
            y = TOP + 0.08
            y += sober_table(pg, d["table"], d["cols"], y, d["row_h"], d["pt"]) + 0.12
            hn, _ = text_height([d["note"]], 14, CW)
            pg.text("note", [d["note"]], MX, y, CW, hn + 0.02, 14, GREY, max_lines=1)
            y += hn + 0.02
            if d.get("body"):
                hb, _ = text_height(d["body"], BODY_PT, CW)
                pg.text("body", d["body"], MX, y + 0.2, CW, hb + 0.02, BODY_PT, one_line_each=True)
                y += 0.2 + hb + 0.02
            limit = CONCL_Y - 0.1 if d.get("concl") else BOTTOM
            if y > limit + 1e-6:
                pg.issue(f"content ends at {y:.2f} in (limit {limit:.2f})")
        else:
            pg.frame(d["title"], sec)
            body = d.get("body", [])
            if len(body) > 4:
                pg.issue("more than four body lines")
            hb = text_height(body, BODY_PT, CW)[0] + 0.02 if body else 0.0
            he = 0.0
            if d.get("eq"):
                wpx, hpx = Image.open(os.path.join(FIG, d["eq"])).size
                he = CW * hpx / wpx
            limit = CONCL_Y - 0.1 if d.get("concl") else BOTTOM
            avail = limit - TOP - (GAP + he if he else 0) - (GAP + hb if hb else 0)
            h, shown = pg.picture(d["fig"], TOP, avail)
            y = TOP + h
            if he:
                pg.picture(d["eq"], y + GAP, he + 1e-6, min_pt=14)
                y += GAP + he
            if body:
                pg.text("body", body, MX, y + GAP, CW, hb, BODY_PT, one_line_each=True)
                y += GAP + hb
            if y > limit + 1e-6:
                pg.issue(f"content ends at {y:.2f} in (limit {limit:.2f})")
        if d.get("concl"):
            pg.conclusion(d["concl"])
        pg.notes(d["notes"])
        pg.check()
        what = d.get("fig") or ("table" if "table" in d else kind)
        report.append((k, (d.get("sec") or "-"), what, f"{shown:.1f} pt" if shown else "-", d["title"]))
    for k, sec, what, s, t in report:
        print(f"  {k:2d}  {sec:2s}  [{what}; figure text {s}]  {t}".encode("ascii", "replace").decode())
    if ISSUES:
        print(f"\nNOT WRITTEN: {len(ISSUES)} problem(s)")
        for i in ISSUES:
            print("  -", i.encode("ascii", "replace").decode())
        sys.exit(1)
    prs.save(OUT)
    print(f"\n{total} slides -> {OUT}")
    print("checks: every slide passed (no intersecting boxes, all inside the margins, all texts fit, no wrapped body line or table cell, "
          "no single-word last line, no break at a hyphen, figure text >= %.1f pt)" % MIN_FIG_PT)


if __name__ == "__main__":
    build()
