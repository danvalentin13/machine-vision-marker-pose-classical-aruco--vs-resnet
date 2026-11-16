"""docx_lib.py -- helpers that build the paper as a Word file with python-docx, in the house style:
A4 portrait, 1 in margins, Times New Roman, justified, line spacing 1.15, page number bottom centre, table caption above,
figure caption below, native (editable) Word equations, numbered blue citations linked to the reference list.

Text mark-up understood by Paper.p() and in captions / table cells:
    **bold**   *italic*   $latex-like maths$   [@key] or [@key1,@key2] citation(s)
    {fig:label} {tab:label} {eq:label} {sec:label}   cross-references (filled in on the second pass)
"""
import os
import re

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Inches, Mm, Pt, RGBColor

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = f'xmlns:m="{M_NS}" xmlns:w="{W_NS}"'
FONT = "Times New Roman"
BLUE = "0000FF"
TEXT_W = 6.27                                           # usable width [in]: A4 minus two 1 in margins

# ============================================================================================== LaTeX subset -> OMML
GREEK = dict(alpha="α", beta="β", gamma="γ", delta="δ", epsilon="ε", varepsilon="ε", theta="θ", lambda_="λ", mu="μ", nu="ν", pi="π", rho="ρ",
             sigma="σ", tau="τ", phi="φ", varphi="φ", psi="ψ", omega="ω", chi="χ", kappa="κ", eta="η", xi="ξ", Delta="Δ", Sigma="Σ", Omega="Ω",
             Theta="Θ", Phi="Φ")
GREEK["lambda"] = GREEK.pop("lambda_")
SYMS = dict(cdot="⋅", times="×", le="≤", leq="≤", ge="≥", geq="≥", approx="≈", neq="≠", to="→", leftarrow="←", rightarrow="→", pm="±", ldots="…",
            cdots="⋯", infty="∞", partial="∂", propto="∝", sim="∼", ll="≪", gg="≫", circ="∘", deg="°", prime="′", quad=" ", qquad="  ",
            ell="ℓ", forall="∀", mid="|", vee="∨")
SYMS["in"] = "∈"
FUNCS = {"exp", "log", "ln", "cos", "sin", "tan", "arccos", "arcsin", "arctan", "min", "max", "det", "arg"}
ACCENTS = dict(hat="̂", bar="̅", tilde="̃", dot="̇", vec="⃗")
CAL = dict(L="ℒ", N="𝒩", O="𝒪")
ZW = "​"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tokenize(s):
    toks, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            j = i + 1
            if j < len(s) and s[j].isalpha():
                while j < len(s) and s[j].isalpha():
                    j += 1
                name = s[i + 1:j]
                if name in ("text", "mathrm", "operatorname", "mathcal", "begin", "end"):
                    k = j
                    while s[k].isspace():
                        k += 1
                    assert s[k] == "{", s
                    depth, m = 0, k
                    while True:
                        depth += s[m] == "{"
                        depth -= s[m] == "}"
                        if depth == 0:
                            break
                        m += 1
                    toks.append((name.upper() if name in ("begin", "end") else "RAW", s[k + 1:m], name))
                    i = m + 1
                    continue
                toks.append(("CMD", name))
                i = j
            else:
                toks.append(("CMD", s[j]))
                i = j + 1
        elif c in "{}^_&":
            toks.append((c,))
            i += 1
        elif c.isspace():
            i += 1
        else:
            toks.append(("CH", c))
            i += 1
    return toks


class Tex:
    """Recursive-descent converter of a small LaTeX subset to Office Math (OMML)."""

    def __init__(self, s, size_pt):
        self.src, self.t, self.i, self.style, self.sz = s, tokenize(s), 0, None, int(round(size_pt * 2))

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def next(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def run(self, text, style=None):
        style = style or self.style
        mpr = {"b": '<m:rPr><m:sty m:val="b"/></m:rPr>', "p": '<m:rPr><m:sty m:val="p"/></m:rPr>', "bi": '<m:rPr><m:sty m:val="bi"/></m:rPr>'}.get(style, "")
        wb = "<w:b/>" if style in ("b", "bi") else ""
        return (f'<m:r>{mpr}<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/>{wb}<w:sz w:val="{self.sz}"/><w:szCs w:val="{self.sz}"/></w:rPr>'
                f'<m:t xml:space="preserve">{esc(text)}</m:t></m:r>')

    def seq(self, stop):
        out = []
        while self.peek() is not None and not stop(self.peek()):
            out.append(self.item())
        return "".join(out)

    def item(self):
        base = self.atom()
        sub = sup = None
        while self.peek() is not None and self.peek()[0] in ("^", "_"):
            op = self.next()[0]
            arg = self.atom()
            if op == "^":
                sup = arg
            else:
                sub = arg
        base = base or self.run(ZW)
        if sub is not None and sup is not None:
            return f"<m:sSubSup><m:e>{base}</m:e><m:sub>{sub}</m:sub><m:sup>{sup}</m:sup></m:sSubSup>"
        if sub is not None:
            return f"<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>"
        if sup is not None:
            return f"<m:sSup><m:e>{base}</m:e><m:sup>{sup}</m:sup></m:sSup>"
        return base

    def atom(self):
        tok = self.next()
        k = tok[0]
        if k == "{":
            inner = self.seq(lambda t: t[0] == "}")
            self.next()
            return inner
        if k == "CH":
            return self.run("−" if tok[1] == "-" else tok[1])
        if k == "RAW":
            if tok[2] == "mathcal":
                return self.run(CAL.get(tok[1], tok[1]), "p")
            return self.run(tok[1], "p")
        if k == "BEGIN":
            return self.env(tok[1])
        if k == "CMD":
            return self.cmd(tok[1])
        raise ValueError(f"unexpected {tok} in {self.src}")

    def delim(self):
        tok = self.next()
        if tok[0] == "CH":
            return "" if tok[1] == "." else tok[1]
        if tok[0] == "CMD":
            return {"{": "{", "}": "}", "|": "‖"}[tok[1]]
        raise ValueError(f"bad delimiter {tok} in {self.src}")

    @staticmethod
    def d(inner, a, b):
        return f'<m:d><m:dPr><m:begChr m:val="{esc(a)}"/><m:endChr m:val="{esc(b)}"/></m:dPr><m:e>{inner}</m:e></m:d>'

    def styled(self, st):
        old, self.style = self.style, st
        a = self.atom()
        self.style = old
        return a

    def cmd(self, name):
        if name in GREEK:
            return self.run(GREEK[name])
        if name in SYMS:
            return self.run(SYMS[name])
        if name == "frac":
            num, den = self.atom(), self.atom()
            return f"<m:f><m:num>{num}</m:num><m:den>{den}</m:den></m:f>"
        if name == "sqrt":
            return f'<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/><m:e>{self.atom()}</m:e></m:rad>'
        if name in ACCENTS:
            return f'<m:acc><m:accPr><m:chr m:val="{ACCENTS[name]}"/></m:accPr><m:e>{self.atom()}</m:e></m:acc>'
        if name == "mathbf":
            return self.styled("b")
        if name == "boldsymbol":
            return self.styled("bi")
        if name == "left":
            a = self.delim()
            inner = self.seq(lambda t: t == ("CMD", "right"))
            self.next()
            return self.d(inner, a, self.delim())
        if name == "norm":
            return self.d(self.atom(), "‖", "‖")
        if name == "abs":
            return self.d(self.atom(), "|", "|")
        if name in FUNCS:
            nx = self.peek()
            if nx is None or nx[0] in ("}", "&", "END", "^", "_") or nx in (("CMD", "right"), ("CMD", "\\")):
                return self.run(name, "p")
            return f"<m:func><m:fName>{self.run(name, 'p')}</m:fName><m:e>{self.item()}</m:e></m:func>"
        if name in ("sum", "prod", "int"):
            sub = sup = None
            while self.peek() is not None and self.peek()[0] in ("^", "_"):
                op = self.next()[0]
                if op == "^":
                    sup = self.atom()
                else:
                    sub = self.atom()
            body = self.item()
            ch = {"sum": "∑", "prod": "∏", "int": "∫"}[name]
            hide = ('<m:subHide m:val="1"/>' if sub is None else "") + ('<m:supHide m:val="1"/>' if sup is None else "")
            return (f'<m:nary><m:naryPr><m:chr m:val="{ch}"/><m:limLoc m:val="undOvr"/>{hide}</m:naryPr><m:sub>{sub or ""}</m:sub>'
                    f'<m:sup>{sup or ""}</m:sup><m:e>{body}</m:e></m:nary>')
        if name == "lsup":                                             # left superscript: \lsup{A}{body}
            a, body = self.atom(), self.atom()
            return f"<m:sPre><m:sub>{self.run(ZW)}</m:sub><m:sup>{a}</m:sup><m:e>{body}</m:e></m:sPre>"
        if name == "underset":
            below, base = self.atom(), self.atom()
            return f"<m:limLow><m:e>{base}</m:e><m:lim>{below}</m:lim></m:limLow>"
        if name == ",":
            return self.run(" ")
        if name in (";", ":"):
            return self.run(" ")
        if name == "!":
            return ""
        if name in ("{", "}"):
            return self.run(name)
        if name == "|":
            return self.run("‖")
        if name == "top":
            return self.run("T", "p")
        raise ValueError(f"unknown command \\{name} in {self.src}")

    def env(self, name):
        rows = [[]]
        while True:
            cell = self.seq(lambda t: t[0] == "&" or t == ("CMD", "\\") or t[0] == "END")
            tok = self.next()
            rows[-1].append(cell or self.run(ZW))
            if tok[0] == "END":
                break
            if tok[0] == "CMD":
                rows.append([])
        ncol = max(len(r) for r in rows)
        mr = "".join("<m:mr>" + "".join(f"<m:e>{c}</m:e>" for c in r + [self.run(ZW)] * (ncol - len(r))) + "</m:mr>" for r in rows)
        jc = "left" if name == "cases" else "center"
        m = (f'<m:m><m:mPr><m:mcs><m:mc><m:mcPr><m:count m:val="{ncol}"/><m:mcJc m:val="{jc}"/></m:mcPr></m:mc></m:mcs></m:mPr>{mr}</m:m>')
        dl = {"bmatrix": ("[", "]"), "pmatrix": ("(", ")"), "cases": ("{", ""), "matrix": None, "vmatrix": ("|", "|")}[name]
        return m if dl is None else self.d(m, *dl)

    def omml(self):
        body = self.seq(lambda t: False)
        return body


def omath(tex, size_pt, display=False):
    body = Tex(tex, size_pt).omml()
    if display:
        return f'<m:oMathPara {NS}><m:oMathParaPr><m:jc m:val="center"/></m:oMathParaPr><m:oMath>{body}</m:oMath></m:oMathPara>'
    return f"<m:oMath {NS}>{body}</m:oMath>"


# ============================================================================================== the document
INLINE = re.compile(r"(\$[^$]+\$|`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[@[^\]]+\]|\{(?:fig|tab|eq|sec):[A-Za-z0-9_]+\})")
UNIT = re.compile(r"(?<=[0-9]) (?=(?:%|mm|cm|px|ms|MB|GB|min|m|s|m/s|cm/s|m/s²)(?![A-Za-z]))")
THOUSANDS = re.compile(r"(?<=[0-9]) (?=[0-9]{3}(?![0-9]))")
NBSP = chr(0xA0)
UNIT_START = re.compile(r" (?:%|mm|cm|px|ms|MB|GB|min|m|s|m/s|cm/s|m/s²)(?![A-Za-z])")
CODE_FONT = "Consolas"


def nbsp(text):
    """A number and its unit stay on one line."""
    return THOUSANDS.sub(NBSP, UNIT.sub(NBSP, text))


class Paper:
    def __init__(self, known=None, base=10.0, defer=None):
        self.base = base
        self.defer = dict(defer or {})                  # figure label -> text blocks to let pass before the figure
        self.pending = []                               # [blocks left, emit function]
        self.known = known or {}                        # labels resolved in the previous pass
        self.labels = {}                                # labels defined in this pass: "fig:x" -> "3"
        self.cites, self.n = [], dict(fig=0, tab=0, eq=0)
        self.sec = [0, 0, 0]
        self.bm = 500
        self.doc = Document()
        self._setup()

    # ------------------------------------------------------------------------------------------ page and styles
    def _setup(self):
        d = self.doc
        s = d.sections[0]
        s.page_width, s.page_height = Mm(210), Mm(297)
        s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Inches(1)
        s.footer_distance = Inches(0.5)
        for cs in d.settings.element.iter(qn("w:compatSetting")):       # current Word layout, not 'Compatibility Mode'
            if cs.get(qn("w:name")) == "compatibilityMode":
                cs.set(qn("w:val"), "15")
        st = d.styles["Normal"]
        st.font.name, st.font.size = FONT, Pt(self.base)
        rpr = st.element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts")
            rpr.append(rf)
        for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rf.set(qn(a), FONT)
        pf = st.paragraph_format
        pf.alignment, pf.line_spacing, pf.space_after, pf.space_before = WD_ALIGN_PARAGRAPH.JUSTIFY, 1.15, Pt(4), Pt(0)
        pf.widow_control = True
        for name, size, before, after, italic in (("Heading 1", self.base + 2, 12, 5, False), ("Heading 2", self.base + 0.5, 9, 3, False), ("Heading 3", self.base, 7, 2, True)):
            h = d.styles[name]
            h.font.name, h.font.size, h.font.bold, h.font.italic = FONT, Pt(size), True, italic
            h.font.color.rgb = RGBColor(0, 0, 0)
            hr = h.element.get_or_add_rPr()
            f = hr.find(qn("w:rFonts"))
            if f is None:
                f = OxmlElement("w:rFonts")
                hr.append(f)
            for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                f.set(qn(a), FONT)
            for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                if f.get(qn(a)) is not None:
                    del f.attrib[qn(a)]
            hp = h.paragraph_format
            hp.alignment, hp.space_before, hp.space_after, hp.keep_with_next, hp.line_spacing = WD_ALIGN_PARAGRAPH.LEFT, Pt(before), Pt(after), True, 1.15
        # footer: page number, centred
        fp = s.footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for kind, txt in (("begin", None), (None, "PAGE"), ("end", None)):
            r = fp.add_run()
            r.font.size, r.font.name = Pt(self.base), FONT
            if kind:
                fc = OxmlElement("w:fldChar")
                fc.set(qn("w:fldCharType"), kind)
                r._r.append(fc)
            else:
                it = OxmlElement("w:instrText")
                it.set(qn("xml:space"), "preserve")
                it.text = txt
                r._r.append(it)

    # ------------------------------------------------------------------------------------------ labels, citations
    def ref(self, key):
        kind, _ = key.split(":")
        v = self.labels.get(key) or self.known.get(key) or "?"
        nb = " "
        return {"fig": f"Fig.{nb}{v}", "tab": f"Table{nb}{v}", "eq": f"Eq.{nb}({v})", "sec": f"Section{nb}{v}"}[kind]

    def cite_no(self, key):
        if key not in self.cites:
            self.cites.append(key)
        return self.cites.index(key) + 1

    # ------------------------------------------------------------------------------------------ inline content
    def _run(self, par, text, size, bold=False, italic=False):
        r = par.add_run(text)
        r.font.size, r.font.name, r.bold, r.italic = Pt(size), FONT, bold or None, italic or None
        return r

    def inline(self, par, text, size=None, bold=False, italic=False):
        size = size or self.base
        after_math = False
        for part in INLINE.split(text):
            if not part:
                continue
            if after_math and UNIT_START.match(part):
                part = NBSP + part[1:]
            after_math = part.startswith("$") and part.endswith("$") and len(part) > 2
            if part.startswith("$") and part.endswith("$") and len(part) > 2:
                par._p.append(parse_xml(omath(part[1:-1], size)))
            elif part.startswith("`") and part.endswith("`") and len(part) > 2:
                r = self._run(par, part[1:-1], size - 1, bold, italic)
                r.font.name = CODE_FONT
                rf = r._r.get_or_add_rPr().find(qn("w:rFonts"))
                for a in ("w:ascii", "w:hAnsi", "w:cs"):
                    rf.set(qn(a), CODE_FONT)
            elif part.startswith("**") and part.endswith("**") and len(part) > 4:
                self._run(par, nbsp(part[2:-2]), size, True, italic)
            elif part.startswith("*") and part.endswith("*") and len(part) > 2:
                self._run(par, nbsp(part[1:-1]), size, bold, True)
            elif part.startswith("[@"):
                keys = [k.strip().lstrip("@") for k in part[1:-1].split(",")]
                self._run(par, "(", size, bold, italic)
                for i, k in enumerate(keys):
                    if i:
                        self._run(par, ", ", size, bold, italic)
                    self._cite(par, self.cite_no(k), size)
                self._run(par, ")", size, bold, italic)
            elif part.startswith("{") and part.endswith("}") and ":" in part:
                self._run(par, self.ref(part[1:-1]), size, bold, italic)
            else:
                self._run(par, nbsp(part), size, bold, italic)

    def _cite(self, par, n, size):
        h = OxmlElement("w:hyperlink")
        h.set(qn("w:anchor"), f"ref{n}")
        h.set(qn("w:history"), "1")
        h.append(parse_xml(f'<w:r xmlns:w="{W_NS}"><w:rPr><w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:cs="{FONT}"/><w:color w:val="{BLUE}"/>'
                           f'<w:sz w:val="{int(size * 2)}"/><w:szCs w:val="{int(size * 2)}"/></w:rPr><w:t>{n}</w:t></w:r>'))
        par._p.append(h)

    # ------------------------------------------------------------------------------------------ blocks
    def p(self, text, size=None, align=None, after=None, before=None, indent=None, keep=False, italic=False, bold=False, first=None):
        par = self.doc.add_paragraph()
        if align:
            par.alignment = dict(l=WD_ALIGN_PARAGRAPH.LEFT, c=WD_ALIGN_PARAGRAPH.CENTER, r=WD_ALIGN_PARAGRAPH.RIGHT, j=WD_ALIGN_PARAGRAPH.JUSTIFY)[align]
        pf = par.paragraph_format
        if after is not None:
            pf.space_after = Pt(after)
        if before is not None:
            pf.space_before = Pt(before)
        if indent is not None:
            pf.left_indent, pf.right_indent = Inches(indent), Inches(indent)
        if first is not None:
            pf.first_line_indent = Inches(first)
        pf.keep_with_next = keep or None
        self.inline(par, text, size, bold, italic)
        self._tick()
        return par

    def h(self, level, text, label=None, numbered=True):
        if numbered:
            self.sec[level - 1] += 1
            for k in range(level, 3):
                self.sec[k] = 0
            num = ".".join(str(x) for x in self.sec[:level])
            if label:
                self.labels["sec:" + label] = num
            text = f"{num}{'.' if level == 1 else ''} {text}"
        par = self.doc.add_paragraph(style=f"Heading {level}")
        self.inline(par, text, {1: self.base + 2, 2: self.base + 0.5, 3: self.base}[level], True, level == 3)
        return par

    def bullets(self, items, size=None):
        for it in items:
            par = self.doc.add_paragraph(style="List Bullet")
            par.paragraph_format.space_after = Pt(1.5)
            self.inline(par, it, size)
        self._tick()

    def eq(self, tex, label=None):
        self.n["eq"] += 1
        if label:
            self.labels["eq:" + label] = str(self.n["eq"])
        t = self.doc.add_table(rows=1, cols=3)
        t.alignment, t.autofit = WD_TABLE_ALIGNMENT.CENTER, False
        self._table_borders(t, False)
        widths = (0.45, TEXT_W - 0.9, 0.45)
        for c, w in zip(t.rows[0].cells, widths):
            c.width = Inches(w)
            c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            self._cell_margins(c, 20, 20, 0, 0)
        mid = t.rows[0].cells[1].paragraphs[0]
        mid.alignment = WD_ALIGN_PARAGRAPH.CENTER
        mid.paragraph_format.space_after = mid.paragraph_format.space_before = Pt(1)
        mid._p.append(parse_xml(omath(tex, self.base, display=True)))
        rp = t.rows[0].cells[2].paragraphs[0]
        rp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        rp.paragraph_format.space_after = rp.paragraph_format.space_before = Pt(0)
        self._run(rp, f"({self.n['eq']})", self.base)
        lp = t.rows[0].cells[0].paragraphs[0]
        lp.paragraph_format.space_after = Pt(0)
        self._grid(t, widths)
        self._spacer(2)

    def _spacer(self, pt):
        par = self.doc.add_paragraph()
        pf = par.paragraph_format
        pf.space_after = pf.space_before = Pt(0)
        pf.line_spacing = Pt(pt)
        for r in par.runs:
            r.font.size = Pt(1)
        return par

    def _tick(self):
        """Called after a text block: deferred figures whose turn has come are placed now, in their order."""
        for item in self.pending:
            item[0] -= 1
        while self.pending and self.pending[0][0] <= 0:
            self.pending.pop(0)[1]()

    def flush(self):
        while self.pending:
            self.pending.pop(0)[1]()

    def fig(self, file, label, caption, width=TEXT_W):
        self.n["fig"] += 1
        self.labels["fig:" + label] = str(self.n["fig"])
        number, wait = self.n["fig"], self.defer.get(label, 0)
        if wait > 0 or self.pending:
            self.pending.append([max(wait, 1 if self.pending else 0), lambda: self._fig(file, number, caption, width)])
            if self.pending[0][0] <= 0:
                self._tick()
            return
        self._fig(file, number, caption, width)

    def _fig(self, file, number, caption, width):
        par = self.doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_before, par.paragraph_format.space_after, par.paragraph_format.keep_with_next = Pt(6), Pt(3), True
        from PIL import Image
        with Image.open(file) as im:
            native = im.size[0] / 300.0                                 # the figures are saved at 300 dpi
        par.add_run().add_picture(file, width=Inches(min(width, native)))
        cp = self.doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.space_after, cp.paragraph_format.line_spacing = Pt(8), 1.05
        self._run(cp, f"Figure {number}. ", self.base - 1, True)
        self.inline(cp, caption, self.base - 1)

    def table(self, label, caption, cols, rows_, size=8.0, note=None, rules=(), keep=None):
        """cols: [(header, width_in, align l/c/r)]; rows_: list of lists of marked-up strings; rules: row indices followed by a thin rule."""
        self.n["tab"] += 1
        self.labels["tab:" + label] = str(self.n["tab"])
        cp = self.doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.LEFT
        cp.paragraph_format.space_before, cp.paragraph_format.space_after, cp.paragraph_format.keep_with_next = Pt(7), Pt(3), True
        cp.paragraph_format.line_spacing = 1.05
        self._run(cp, f"Table {self.n['tab']}. ", self.base - 1, True)
        self.inline(cp, caption, self.base - 1)
        widths = [c[1] for c in cols]
        assert sum(widths) <= TEXT_W + 1e-6, (label, sum(widths))
        t = self.doc.add_table(rows=1 + len(rows_), cols=len(cols))
        t.alignment, t.autofit = WD_TABLE_ALIGNMENT.CENTER, False
        self._table_borders(t, False)
        al = dict(l=WD_ALIGN_PARAGRAPH.LEFT, c=WD_ALIGN_PARAGRAPH.CENTER, r=WD_ALIGN_PARAGRAPH.RIGHT)
        for ri, row in enumerate([[c[0] for c in cols]] + list(rows_)):
            assert len(row) == len(cols), (label, ri, row)
            tr = t.rows[ri]
            trpr = tr._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit")
            trpr.append(cs)
            if ri == 0:
                th = OxmlElement("w:tblHeader")
                trpr.append(th)
            for ci, txt in enumerate(row):
                cell = tr.cells[ci]
                cell.width = Inches(widths[ci])
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                self._cell_margins(cell, 25, 25, 50, 50)
                par = cell.paragraphs[0]
                par.alignment = al[cols[ci][2]] if ri else (WD_ALIGN_PARAGRAPH.LEFT if cols[ci][2] == "l" else WD_ALIGN_PARAGRAPH.CENTER)
                par.paragraph_format.space_after = par.paragraph_format.space_before = Pt(0)
                par.paragraph_format.line_spacing = 1.0
                if (len(rows_) <= 14 if keep is None else keep) and ri < len(rows_):
                    par.paragraph_format.keep_with_next = True         # all rows but the last: the table is not split
                self.inline(par, txt, size, bold=(ri == 0))
                b = {}
                if ri == 0:
                    b = dict(top=12, bottom=6)
                elif ri == len(rows_):
                    b = dict(bottom=12)
                elif (ri - 1) in rules:
                    b = dict(bottom=4)
                if b:
                    self._cell_border(cell, **b)
        self._grid(t, widths)
        if note:
            npar = self.doc.add_paragraph()
            npar.alignment = WD_ALIGN_PARAGRAPH.LEFT
            npar.paragraph_format.space_before, npar.paragraph_format.space_after, npar.paragraph_format.line_spacing = Pt(2), Pt(8), 1.0
            self.inline(npar, note, size)
        else:
            self._spacer(8)
        self._tick()

    # ------------------------------------------------------------------------------------------ table plumbing
    @staticmethod
    def _grid(t, widths):
        tbl = t._tbl
        tblPr = tbl.tblPr
        lay = OxmlElement("w:tblLayout")
        lay.set(qn("w:type"), "fixed")
        tblPr.append(lay)
        tw = tblPr.find(qn("w:tblW"))
        if tw is None:
            tw = OxmlElement("w:tblW")
            tblPr.append(tw)
        tw.set(qn("w:w"), str(int(round(sum(widths) * 1440))))
        tw.set(qn("w:type"), "dxa")
        grid = tbl.find(qn("w:tblGrid"))
        for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(int(round(w * 1440))))

    @staticmethod
    def _table_borders(t, on):
        tblPr = t._tbl.tblPr
        b = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            e = OxmlElement(f"w:{edge}")
            e.set(qn("w:val"), "single" if on else "nil")
            b.append(e)
        tblPr.append(b)

    @staticmethod
    def _cell_border(cell, **edges):
        tcPr = cell._tc.get_or_add_tcPr()
        b = OxmlElement("w:tcBorders")
        for edge in ("top", "bottom"):
            if edge in edges:
                e = OxmlElement(f"w:{edge}")
                e.set(qn("w:val"), "single")
                e.set(qn("w:sz"), str(edges[edge]))
                e.set(qn("w:space"), "0")
                e.set(qn("w:color"), "000000")
                b.append(e)
        tcPr.append(b)

    @staticmethod
    def _cell_margins(cell, top, bottom, left, right):
        tcPr = cell._tc.get_or_add_tcPr()
        m = OxmlElement("w:tcMar")
        for k, v in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
            e = OxmlElement(f"w:{k}")
            e.set(qn("w:w"), str(v))
            e.set(qn("w:type"), "dxa")
            m.append(e)
        tcPr.append(m)

    # ------------------------------------------------------------------------------------------ references
    def references(self, entries):
        self.flush()
        """entries: {key: marked-up IEEE text}.  Listed in the order of first citation; each entry carries a bookmark ref<n>."""
        for n, key in enumerate(self.cites, 1):
            par = self.doc.add_paragraph()
            par.alignment = WD_ALIGN_PARAGRAPH.LEFT
            pf = par.paragraph_format
            pf.left_indent, pf.first_line_indent, pf.space_after, pf.line_spacing = Inches(0.36), Inches(-0.36), Pt(2), 1.05
            self.bm += 1
            s = OxmlElement("w:bookmarkStart")
            s.set(qn("w:id"), str(self.bm))
            s.set(qn("w:name"), f"ref{n}")
            par._p.append(s)
            self._run(par, f"({n})\t", self.base - 1)
            e = OxmlElement("w:bookmarkEnd")
            e.set(qn("w:id"), str(self.bm))
            par._p.append(e)
            tabs = OxmlElement("w:tabs")
            tab = OxmlElement("w:tab")
            tab.set(qn("w:val"), "left")
            tab.set(qn("w:pos"), str(int(0.36 * 1440)))
            tabs.append(tab)
            par._p.get_or_add_pPr().append(tabs)
            self.inline(par, entries[key], self.base - 1)
        missing = [k for k in self.cites if k not in entries]
        assert not missing, missing

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def save(self, path):
        self.doc.save(path)


def build(write, path, base=10.0, defer=None):
    """Two passes: the first collects the numbers of figures, tables, equations and sections, the second writes the file."""
    first = Paper(base=base, defer=defer)
    write(first)
    first.flush()
    second = Paper(known=first.labels, base=base, defer=defer)
    write(second)
    second.flush()
    unresolved = [k for k in re.findall(r"\{((?:fig|tab|eq|sec):[A-Za-z0-9_]+)\}", open(write.__code__.co_filename, encoding="utf-8").read()) if k not in second.labels]
    second.save(path)
    return second, sorted(set(unresolved))
