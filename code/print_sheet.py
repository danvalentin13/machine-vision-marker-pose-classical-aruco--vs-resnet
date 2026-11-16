"""print_sheet.py -- everything to print, in ONE PDF at exact physical size (A4, print at 100 % / "actual size").

  python print_sheet.py

Needs the outputs of t1_generate_markers.py (markers) and t3_make_board.py (chessboard).
Writes  output/print/print_sheet.pdf     page 1 marker F, page 2 marker M, page 3 chessboard
        output/print/page_N_*.png        the same pages as images
After printing: measure the black square of each marker and one chessboard square with a calliper and write the values in
data/markers.json (measured_side_mm) and data/calibration/board.json (measured_square_mm).
"""
import glob
import json
import os

import cv2
import numpy as np
from PIL import Image

from config import OUTPUT_DIR, data_path
from mvtools import ask_folder

A4 = (210.0, 297.0)                       # portrait [mm]


def page_with(img_path, caption, landscape=False):
    """Place the image, untouched, centred on a white A4 page whose pixel size matches the image's dpi."""
    im = Image.open(img_path).convert("L")
    dpi = im.info["dpi"][0]
    w_mm, h_mm = (A4[1], A4[0]) if landscape else A4
    W, H = int(round(w_mm / 25.4 * dpi)), int(round(h_mm / 25.4 * dpi))
    arr = np.full((H, W), 255, np.uint8)
    a = np.array(im)
    if a.shape[0] > H or a.shape[1] > W:
        raise SystemExit(f"{img_path} does not fit on A4")
    y0, x0 = (H - a.shape[0]) // 2, (W - a.shape[1]) // 2
    arr[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a
    cv2.putText(arr, caption, (int(W * 0.06), H - int(H * 0.03)), cv2.FONT_HERSHEY_SIMPLEX, W / 2300, 110, 2, cv2.LINE_AA)
    return Image.fromarray(arr), dpi


def main():
    cfg = json.load(open(data_path("markers.json")))
    markers_dir = os.path.join(OUTPUT_DIR, "markers")
    f_png = os.path.join(markers_dir, f"marker_F_id{cfg['id_fixed']}.png")
    m_png = os.path.join(markers_dir, f"marker_M_id{cfg['id_moving']}.png")
    boards = sorted(glob.glob(os.path.join(OUTPUT_DIR, "calibration", "chessboard_*.png")))
    if not (os.path.exists(f_png) and os.path.exists(m_png)) or not boards:
        raise SystemExit("run t1_generate_markers.py and t3_make_board.py first")
    out_dir = ask_folder("Folder for the print sheet", os.path.join(OUTPUT_DIR, "print"))
    os.makedirs(out_dir, exist_ok=True)
    side = cfg["nominal_side_mm"]
    pages = [
        (page_with(f_png, f"Marker F (fixed), id {cfg['id_fixed']}, {cfg['dictionary']} - black square {side:g} mm - print at 100 %"), "page_1_marker_F"),
        (page_with(m_png, f"Marker M (moving), id {cfg['id_moving']}, {cfg['dictionary']} - black square {side:g} mm - print at 100 %"), "page_2_marker_M"),
        (page_with(boards[-1], "", landscape=True), "page_3_chessboard"),
    ]
    for (im, dpi), name in pages:
        im.save(os.path.join(out_dir, name + ".png"), dpi=(dpi, dpi))
    first, rest = pages[0][0][0], [p[0][0] for p in pages[1:]]
    pdf = os.path.join(out_dir, "print_sheet.pdf")
    first.save(pdf, save_all=True, append_images=rest, resolution=pages[0][0][1])
    print(f"\n{pdf}")
    print("3 pages: marker F, marker M (A4 portrait), chessboard (A4 landscape).")
    print("Print at 100 % / 'actual size' (NOT 'fit to page'), then measure the black squares and one chessboard square with a calliper.")


if __name__ == "__main__":
    main()
