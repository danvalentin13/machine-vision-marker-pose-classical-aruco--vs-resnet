"""t1_generate_markers.py -- Task 1: generate the two fiducial markers (F fixed, M moving).

  python t1_t1_generate_markers.py

The program asks (press Enter to accept the [default]):
  1) folder for the marker images        2) folder for the dictionary table and sheet
  3) which dictionary                    4) id of F and id of M          5) printed side length

Writes
  <markers folder>/marker_F_id<ID>.png, marker_M_id<ID>.png   printable images (print at 100 %, no "fit to page")
  <dictionary folder>/dictionary.csv, .json, dictionary_sheet.pdf, sheet/page_N.png   every marker of the dictionary
  data/markers.json                                           dictionary, ids, side length (read by the later tasks)

The patterns come from OpenCV's predefined dictionaries (cv2.aruco.generateImageMarker); nothing is drawn by hand.
After printing, measure the black square with a calliper and write it in data/markers.json as
"measured_side_mm" (the pose code must use the measured value, not the nominal one).

Non-interactive use: every question can be given as an option, e.g.
  python t1_t1_generate_markers.py --markers-out X --dict-out Y --dictionary DICT_4X4_50 --id-fixed 0 --id-moving 1 --side-mm 80
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np
from PIL import Image

from config import data_path, OUTPUT_DIR
from mvtools import ask, ask_folder

DEFAULT_DICT = "DICT_6X6_250"
DEFAULT_IDS = (0, 23)
DEFAULT_SIDE_MM = 80.0          # assignment: 50-100 mm
QUIET_CELLS = 2                 # white quiet zone around the black square, in marker cells (>= 1 required)
TARGET_DPI = 300                # resolution used to choose the image size


# ----------------------------------------------------------------------------------------------
# asking the user
# ----------------------------------------------------------------------------------------------
def aruco_dictionary_names():
    return sorted(n for n in dir(cv2.aruco) if n.startswith("DICT_") and isinstance(getattr(cv2.aruco, n), int))


def ask_dictionary(default):
    names = aruco_dictionary_names()
    while True:
        name = ask("3) Dictionary (type ? to list the options)", default).upper()
        if name == "?":
            print("   " + "\n   ".join(names))
        elif name in names:
            return name
        else:
            print(f"   '{name}' is not an OpenCV dictionary. Type ? to list them.")


def ask_id(question, default, n_markers, taken=None):
    while True:
        text = ask(question, default)
        try:
            value = int(text)
        except ValueError:
            print("   enter a whole number")
            continue
        if not 0 <= value < n_markers:
            print(f"   this dictionary has ids 0..{n_markers - 1}")
        elif value == taken:
            print("   F and M must have different ids")
        else:
            return value


def ask_float(question, default):
    while True:
        try:
            value = float(ask(question, default).replace(",", "."))
            if value > 0:
                return value
        except ValueError:
            pass
        print("   enter a positive number")


# ----------------------------------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------------------------------
def make_marker_image(dictionary, marker_id: int, side_mm: float):
    """Marker with quiet zone as a uint8 image; returns (image, dpi) so that the black square is side_mm wide."""
    cells = dictionary.markerSize + 2                              # inner bits + the black border
    cell_px = max(1, round(side_mm / 25.4 * TARGET_DPI / cells))
    side_px = cell_px * cells                                      # whole number of pixels per cell: sharp edges
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    pad = QUIET_CELLS * cell_px
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    return img, side_px * 25.4 / side_mm                           # dpi at which the black square is exactly side_mm


def write_dictionary_files(dictionary, dict_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n = dictionary.markerSize
    rows = []
    for i in range(len(dictionary.bytesList)):
        bits = cv2.aruco.Dictionary.getBitsFromByteList(dictionary.bytesList[i:i + 1], n, 0).ravel()
        rows.append((i, "".join(str(int(b)) for b in bits)))
    with open(os.path.join(out_dir, "dictionary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", f"bits_{n}x{n}_row_major (1 = white, 0 = black)"])
        w.writerows(rows)
    with open(os.path.join(out_dir, "dictionary.json"), "w") as f:
        json.dump(dict(dictionary=dict_name, grid=f"{n}x{n}", n_markers=len(rows),
                       note="an id has no meaning besides its bit pattern; opencv chooses patterns that differ in many bits"),
                  f, indent=1)
    return rows


def write_dictionary_sheet(dictionary, dict_name, out_dir, highlight):
    """Contact sheet of the whole dictionary: every marker small, with its id under it (A4 pages, PDF + PNG).
    `highlight` = {id: role text}; those markers get an orange frame and the role in the label."""
    cols, rows, W, H, margin, header, msz = 6, 9, 1240, 1754, 60, 90, 110       # A4 at 150 dpi
    cw, ch = (W - 2 * margin) // cols, (H - 2 * margin - header) // rows
    font, gray, orange = cv2.FONT_HERSHEY_SIMPLEX, (170, 170, 170), (26, 115, 217)   # BGR
    n, per = len(dictionary.bytesList), cols * rows
    n_pages = -(-n // per)

    def centred(img, text, cx, y, scale, color, thick=1):
        w = cv2.getTextSize(text, font, scale, thick)[0][0]
        cv2.putText(img, text, (cx - w // 2, y), font, scale, color, thick, cv2.LINE_AA)

    pages = []
    for p in range(n_pages):
        page = np.full((H, W, 3), 255, np.uint8)
        cv2.putText(page, f"{dict_name}  -  {n} markers  ({dictionary.markerSize}x{dictionary.markerSize} bits)",
                    (margin, margin + 25), font, 0.9, (40, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(page, f"ids {p * per}-{min(n, (p + 1) * per) - 1}   page {p + 1}/{n_pages}   (orange = used in this project)",
                    (margin, margin + 60), font, 0.55, (110, 110, 110), 1, cv2.LINE_AA)
        for k in range(per):
            i = p * per + k
            if i >= n:
                break
            r, c = divmod(k, cols)
            x0, y0 = margin + c * cw + (cw - msz) // 2, margin + header + r * ch
            m = cv2.aruco.generateImageMarker(dictionary, i, msz)
            page[y0:y0 + msz, x0:x0 + msz] = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
            used = i in highlight
            cv2.rectangle(page, (x0 - 5, y0 - 5), (x0 + msz + 5, y0 + msz + 5), orange if used else gray, 3 if used else 1)
            label = f"ID {i}" + (f"  ({highlight[i]})" if used else "")
            centred(page, label, x0 + msz // 2, y0 + msz + 28, 0.6, orange if used else (40, 40, 40), 2 if used else 1)
        pages.append(page)

    sheet_dir = os.path.join(out_dir, "sheet")
    os.makedirs(sheet_dir, exist_ok=True)
    for old in os.listdir(sheet_dir):                              # a smaller dictionary must not leave old pages
        if old.startswith("page_") and old.endswith(".png"):
            os.remove(os.path.join(sheet_dir, old))
    for p, page in enumerate(pages, 1):
        cv2.imwrite(os.path.join(sheet_dir, f"page_{p}.png"), page)
    rgb = [Image.fromarray(cv2.cvtColor(pg, cv2.COLOR_BGR2RGB)) for pg in pages]
    rgb[0].save(os.path.join(out_dir, "dictionary_sheet.pdf"), save_all=True, append_images=rgb[1:], resolution=150)
    return n_pages


def generate(markers_dir, dict_dir, dict_name, id_fixed, id_moving, side_mm):
    os.makedirs(markers_dir, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))

    files = {}
    for role, mid in (("F", id_fixed), ("M", id_moving)):
        img, dpi = make_marker_image(dictionary, mid, side_mm)
        path = os.path.join(markers_dir, f"marker_{role}_id{mid}.png")
        Image.fromarray(img).save(path, dpi=(dpi, dpi))
        files[role] = (path, mid)
        print(f"{role}: id {mid}  ->  {path}  ({img.shape[1]}x{img.shape[0]} px)")

    rows = write_dictionary_files(dictionary, dict_name, dict_dir)
    print(f"dictionary {dict_name}: {len(rows)} markers  ->  {os.path.join(dict_dir, 'dictionary.csv')}")
    n_pages = write_dictionary_sheet(dictionary, dict_name, dict_dir, {id_fixed: "F fixed", id_moving: "M moving"})
    print(f"dictionary sheet: {n_pages} pages  ->  {os.path.join(dict_dir, 'dictionary_sheet.pdf')}")

    # config for the later tasks; the hand-measured side is kept only if nothing else changed
    cfg_path = data_path("markers.json")
    new = dict(dictionary=dict_name, id_fixed=id_fixed, id_moving=id_moving, nominal_side_mm=side_mm)
    measured = None
    if os.path.exists(cfg_path):
        old = json.load(open(cfg_path))
        if all(old.get(k) == v for k, v in new.items()):
            measured = old.get("measured_side_mm")
    with open(cfg_path, "w") as f:
        json.dump(dict(**new, measured_side_mm=measured, quiet_zone_cells=QUIET_CELLS,
                       note="replace measured_side_mm (null) with the calliper measurement of the PRINTED black square"),
                  f, indent=1)
    print(f"config -> {cfg_path}")
    # describe the marker folder itself, so later sessions (and t2_detect_live.py) can recognise these images
    with open(os.path.join(markers_dir, "markers_info.json"), "w") as f:
        json.dump(dict(dictionary=dict_name, id_fixed=id_fixed, id_moving=id_moving, nominal_side_mm=side_mm,
                       quiet_zone_cells=QUIET_CELLS), f, indent=1)

    # self-test: OpenCV must read back both ids from the saved images
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    for role, (path, mid) in files.items():
        _, ids, _ = detector.detectMarkers(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        found = [] if ids is None else ids.ravel().tolist()
        print(f"self-test {role}: detected ids {found}  {'OK' if found == [mid] else 'FAILED'}")


def main():
    ap = argparse.ArgumentParser(description="Task 1: generate the markers and the dictionary table. "
                                             "Options you leave out are asked in the terminal.")
    ap.add_argument("--markers-out", help="folder for the marker PNGs")
    ap.add_argument("--dict-out", help="folder for dictionary.csv / .json / sheet")
    ap.add_argument("--dictionary", help="OpenCV dictionary name, e.g. DICT_6X6_250")
    ap.add_argument("--id-fixed", type=int, help="id of the fixed marker F")
    ap.add_argument("--id-moving", type=int, help="id of the moving marker M")
    ap.add_argument("--side-mm", type=float, help="side of the black square when printed [mm]")
    a = ap.parse_args()

    print("Task 1 - generate markers.  Press Enter to accept the [default].\n")
    markers_dir = os.path.abspath(a.markers_out) if a.markers_out else \
        ask_folder("1) Folder for the marker images", os.path.join(OUTPUT_DIR, "markers"))
    dict_dir = os.path.abspath(a.dict_out) if a.dict_out else \
        ask_folder("2) Folder for the dictionary table and sheet", os.path.join(OUTPUT_DIR, "dictionary"))
    dict_name = a.dictionary.upper() if a.dictionary else ask_dictionary(DEFAULT_DICT)
    if dict_name not in aruco_dictionary_names():
        raise SystemExit(f"unknown dictionary {dict_name}")
    n_markers = len(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name)).bytesList)
    id_fixed = a.id_fixed if a.id_fixed is not None else ask_id("4a) Id of the FIXED marker F", DEFAULT_IDS[0], n_markers)
    id_moving = a.id_moving if a.id_moving is not None else \
        ask_id("4b) Id of the MOVING marker M", DEFAULT_IDS[1] if DEFAULT_IDS[1] != id_fixed and DEFAULT_IDS[1] < n_markers
               else (id_fixed + 1) % n_markers, n_markers, taken=id_fixed)
    side_mm = a.side_mm if a.side_mm else ask_float("5) Side of the black square when printed [mm] (50-100)", DEFAULT_SIDE_MM)
    if id_fixed == id_moving or not (0 <= id_fixed < n_markers and 0 <= id_moving < n_markers):
        raise SystemExit(f"ids must be different and in 0..{n_markers - 1}")
    print()
    generate(markers_dir, dict_dir, dict_name, id_fixed, id_moving, side_mm)


if __name__ == "__main__":
    main()
