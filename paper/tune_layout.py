"""tune_layout.py -- fills the pages of the paper: a figure that does not fit at the bottom of a page leaves a gap there;
this script lets such a figure slide past the next text block(s) and rebuilds, until no page (but the last) is more than
14 % empty or nothing more can be gained.  Writes paper/layout.json, which build_paper.py reads.

    python paper/tune_layout.py           (from the project folder; needs Microsoft Word and pdftoppm)
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_check as rc  # noqa: E402

LAYOUT = os.path.join(HERE, "layout.json")
PY = sys.executable
LIMIT = 0.14


def figure_labels():
    """Figure labels in the order of the P.fig(...) calls of build_paper.py."""
    src = open(os.path.join(HERE, "build_paper.py"), encoding="utf-8").read()
    return re.findall(r'P\.fig\(os\.path\.join\(FIG, "[^"]+"\), "([a-z0-9_]+)"', src)


def build_and_measure():
    subprocess.run([PY, os.path.join(HERE, "build_paper.py")], check=True, capture_output=True, text=True)
    ps = ("$w = New-Object -ComObject Word.Application; $w.Visible = $false; "
          f"try {{ $d = $w.Documents.Open('{rc.DOCX}', $false, $true); $d.SaveAs2('{rc.PDF}', 17); "
          "$out = @(); foreach ($s in $d.InlineShapes) { $out += $s.Range.Information(3) }; Write-Output ('FIGPAGES ' + ($out -join ',')); "
          "$d.Close($false) } finally { $w.Quit(); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($w); [GC]::Collect(); "
          "[GC]::WaitForPendingFinalizers(); Start-Sleep -Seconds 2 }")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=400)
    line = [x for x in r.stdout.splitlines() if x.startswith("FIGPAGES")]
    pages = [int(x) for x in line[0].split()[1].split(",")] if line else []
    fill = rc.fill(rc.pages())
    return pages, fill


def score(fill):
    return sum(max(0.0, e - LIMIT) for _, e in fill[:-1])


if __name__ == "__main__":
    labels = figure_labels()
    defer = json.load(open(LAYOUT)) if os.path.exists(LAYOUT) else {}
    json.dump(defer, open(LAYOUT, "w"), indent=1)
    pages, fill = build_and_measure()
    best, blocked = score(fill), set()
    print("start: pages", len(fill), "| gaps:", [(i, round(100 * e)) for i, e in fill[:-1] if e > LIMIT])
    for it in range(24):
        gaps = [i for i, e in fill[:-1] if e > LIMIT]
        cand = None
        for g in gaps:                                                   # the first figure on the page after a gap
            on_next = [k for k, pg in enumerate(pages) if pg == g + 1]
            if on_next and labels[on_next[0]] not in blocked and defer.get(labels[on_next[0]], 0) < 4:
                cand = labels[on_next[0]]
                break
        if cand is None:
            break
        trial = dict(defer)
        trial[cand] = trial.get(cand, 0) + 1
        json.dump(trial, open(LAYOUT, "w"), indent=1)
        p2, f2 = build_and_measure()
        s2 = score(f2)
        print(f"  {cand}: wait {trial[cand]} -> pages {len(f2)}, gaps {[(i, round(100 * e)) for i, e in f2[:-1] if e > LIMIT]}")
        if s2 < best - 1e-6 or (trial[cand] < 3 and s2 <= best + 0.02 and len(f2) <= len(fill)):
            if s2 < best - 1e-6:
                best = s2
            defer, pages, fill = trial, p2, f2
            if s2 > best + 1e-6:
                pass
        else:
            blocked.add(cand)
            json.dump(defer, open(LAYOUT, "w"), indent=1)
            pages, fill = build_and_measure()
    json.dump(defer, open(LAYOUT, "w"), indent=1)
    pages, fill = build_and_measure()
    print("layout:", defer)
    print("pages", len(fill), "| empty share [%]:", "  ".join(f"{i}:{100 * e:.0f}" for i, e in fill))
