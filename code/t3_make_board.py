"""t3_make_board.py -- Task 3, step 1: the chessboard target for camera calibration.

  python t3_make_board.py

Asks (Enter = [default]): number of inner corners (columns x rows), square size [mm], output folder.
Writes  <folder>/chessboard_<cols>x<rows>_<size>mm.png   (A4 landscape, print at 100 % scale, glue on rigid board)
        data/calibration/board.json                       what t3_calibrate.py needs (corners, square size)
After printing, measure one row of squares with a calliper and write measured_square_mm in board.json.
"""
import json
import os

import cv2
import numpy as np
from PIL import Image

from config import data_path, OUTPUT_DIR
from mvtools import ask, ask_folder, ask_number

A4_MM = (297.0, 210.0)          # landscape
DPI = 300


def make_board_image(cols, rows, square_mm, label=True):
    """A4-landscape image with a (cols+1) x (rows+1) chessboard (so cols x rows inner corners); returns (image, dpi)."""
    sq = max(1, round(square_mm / 25.4 * DPI))                     # whole pixels per square: sharp corners
    dpi = sq * 25.4 / square_mm                                    # dpi at which a square is exactly square_mm
    W, H = int(round(A4_MM[0] / 25.4 * dpi)), int(round(A4_MM[1] / 25.4 * dpi))
    bw, bh = (cols + 1) * sq, (rows + 1) * sq
    if bw > W - sq or bh > H - sq:
        raise ValueError(f"a {cols + 1}x{rows + 1} board of {square_mm} mm squares does not fit on A4 landscape "
                         f"({(cols + 1) * square_mm:.0f} x {(rows + 1) * square_mm:.0f} mm)")
    img = np.full((H, W), 255, np.uint8)
    x0, y0 = (W - bw) // 2, (H - bh) // 2
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                img[y0 + r * sq:y0 + (r + 1) * sq, x0 + c * sq:x0 + (c + 1) * sq] = 0
    if label and y0 > 70:
        cv2.putText(img, f"chessboard {cols}x{rows} inner corners, square {square_mm:g} mm - print at 100 %, do not scale",
                    (x0, H - y0 // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 90, 2, cv2.LINE_AA)
    return img, dpi


def main():
    print("Task 3 - chessboard for the camera calibration.  Press Enter to accept the [default].\n")
    cols = ask_number("1) Inner corners, columns", 9, int, 3)
    rows = ask_number("2) Inner corners, rows", 6, int, 3)
    square_mm = ask_number("3) Square size [mm] (A4 landscape: 25 fits 9x6)", 25.0, float, 5)
    out_dir = ask_folder("4) Folder for the board image", os.path.join(OUTPUT_DIR, "calibration"))
    try:
        img, dpi = make_board_image(cols, rows, square_mm)
    except ValueError as e:
        raise SystemExit(str(e))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"chessboard_{cols}x{rows}_{square_mm:g}mm.png")
    Image.fromarray(img).save(path, dpi=(dpi, dpi))

    found, _ = cv2.findChessboardCornersSB(img, (cols, rows))        # self-test: OpenCV must find the corners
    print(f"\nboard image -> {path}   self-test: {'OK, all corners found' if found else 'FAILED'}")

    cfg_path = data_path("calibration", "board.json")
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    old = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    same = (old.get("cols"), old.get("rows"), old.get("square_mm")) == (cols, rows, square_mm)
    json.dump(dict(cols=cols, rows=rows, square_mm=square_mm,
                   measured_square_mm=old.get("measured_square_mm") if same else None,
                   note="cols x rows = inner corners; replace measured_square_mm (null) by the calliper value of the PRINTED square"),
              open(cfg_path, "w"), indent=1)
    print(f"config -> {cfg_path}")
    print("Print at 100 % scale, glue on a rigid flat board, measure one square (or 5 squares / 5) and write it in board.json.")


if __name__ == "__main__":
    main()
