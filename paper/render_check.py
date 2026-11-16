"""render_check.py -- renders the paper with Word itself and reports how full every page is.

    python paper/render_check.py          (from the project folder; needs Microsoft Word and pdftoppm)

Writes paper/Marker_Pose_Classical_vs_Learned.pdf and paper/render/p-NN.jpg, then prints for every page the share of
the text area left empty at the bottom (large values = a figure or table was pushed to the next page).
"""
import glob
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOCX = os.path.join(HERE, "Marker_Pose_Classical_vs_Learned.docx")
NEW = DOCX[:-5] + "_new.docx"
if os.path.exists(NEW) and os.path.getmtime(NEW) > os.path.getmtime(DOCX):   # built while the paper was open in Word
    DOCX = NEW
PDF = os.path.join(HERE, "Marker_Pose_Classical_vs_Learned.pdf")
OUT = os.path.join(HERE, "render")
DPI = 90


def to_pdf():
    ps = ("$w = New-Object -ComObject Word.Application; $w.Visible = $false; "
          f"try {{ $d = $w.Documents.Open('{DOCX}', $false, $true); $d.SaveAs2('{PDF}', 17); "
          "$n = $d.ComputeStatistics(2); Write-Output \"pages $n\"; $d.Close($false) } finally { $w.Quit(); "
          "[void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($w); [GC]::Collect(); [GC]::WaitForPendingFinalizers(); Start-Sleep -Seconds 2 }")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=300)
    print(r.stdout.strip() or r.stderr.strip())


def pages():
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "p-*.jpg")):
        os.remove(f)
    subprocess.run(["pdftoppm", "-jpeg", "-r", str(DPI), PDF, os.path.join(OUT, "p")], check=True)
    return sorted(glob.glob(os.path.join(OUT, "p-*.jpg")))


def fill(files):
    top, bottom = int(1.0 * DPI), int((11.69 - 1.0) * DPI)            # text area between the 1 in margins
    rows = []
    for i, f in enumerate(files, 1):
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)[top:bottom]
        ink = np.where((g < 200).sum(axis=1) > 0)[0]
        empty = 1.0 if len(ink) == 0 else 1.0 - (ink.max() + 1) / g.shape[0]
        rows.append((i, empty))
    return rows


if __name__ == "__main__":
    if "--no-word" not in sys.argv:
        to_pdf()
    r = fill(pages())
    print("empty share at the bottom of each page [%]:")
    print("  " + "  ".join(f"{i}:{100 * e:.0f}" for i, e in r))
    bad = [i for i, e in r[:-1] if e > 0.14]
    print("pages with more than 14 % empty:", bad if bad else "none")
