"""Synthetic factory-monitoring dataset used in the STAR paper.

Each image is one of three diffusion-generated equipment-room templates
(Scene_template/clear.png, human.png, fire.png) with a procedurally rendered
machine status panel and room nameplate. The room label and panel readings are
drawn at random, so every image comes with exact ground truth.

    template   event               fire   technician
    clear      Normal operation    NO     ABSENT
    human      Technician event    NO     PRESENT
    fire       Fire event          YES    ABSENT

Usage (from the repository root):
    python factory_dataset/generate_factory_dataset.py --n 500 --seed 2027

Writes factory_dataset/images/factory_XXXX.png, annotations.json and
annotations.csv. The notebooks import load_dataset() / build_panel_records()
from this file.
"""
import argparse
import csv
import datetime
import glob
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_DIR = os.environ.get("SCENE_DIR", os.path.join(REPO_ROOT, "Scene_template"))
DATASET_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_WEIGHTS = {"clear": 34, "human": 33, "fire": 33}

TEMPLATE_HAZARDS = {
    "clear": {"fire": "NO",  "person": "NO"},
    "human": {"fire": "NO",  "person": "YES"},
    "fire":  {"fire": "YES", "person": "NO"},
}

EVENT_OF_TEMPLATE = {
    "clear": "Normal operation",
    "human": "Technician event",
    "fire":  "Fire event",
}

# Overlay regions as fractions of the template size (measured on clear.png).
SCREEN = (0.392, 0.214, 0.602, 0.483)
PLATE  = (0.434, 0.536, 0.556, 0.614)

ROOMS = ["PUMP ROOM", "BOILER ROOM", "COMPRESSOR ROOM",
         "CHILLER ROOM", "REACTOR ROOM", "UTILITY ROOM"]
MACHINE_PREFIX = ["MX", "CV", "PR", "RB", "WD"]

STATUS_COLORS = {"NORMAL": (40, 210, 110), "WARNING": (240, 190, 40), "FAULT": (240, 70, 70)}
POWER_COLORS  = {"GRID": (90, 190, 250), "BACKUP": (240, 170, 50)}

# The 7 tasks of Table I in the paper.
TECHNICIAN_Q = "In the room, is a maintenance technician PRESENT or ABSENT?"
QUESTIONS = [
    "Is the equipment status NORMAL, WARNING, or FAULT?",
    "What is the pressure reading?",
    "What is the temperature reading?",
    "Is the power on GRID or BACKUP?",
    "Which room is this?",
    "Is there a fire?",
    TECHNICIAN_Q,
]


# ─── Fonts: system DejaVu on Linux, otherwise the copy bundled with matplotlib ──
def _font_dir():
    linux = "/usr/share/fonts/truetype/dejavu"
    if os.path.exists(os.path.join(linux, "DejaVuSans-Bold.ttf")):
        return linux
    import matplotlib
    return os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")

_FONT_DIR = _font_dir()

def F(sz, bold=True):
    return ImageFont.truetype(os.path.join(_FONT_DIR, f"DejaVuSans{'-Bold' if bold else ''}.ttf"), sz)

def M(sz):
    return ImageFont.truetype(os.path.join(_FONT_DIR, "DejaVuSansMono-Bold.ttf"), sz)


def load_templates(scene_dir=SCENE_DIR):
    found = {}
    for p in glob.glob(os.path.join(scene_dir, "*.png")) + glob.glob(os.path.join(scene_dir, "*.jpg")):
        stem = os.path.splitext(os.path.basename(p))[0].lower()
        if stem in TEMPLATE_HAZARDS:
            found[stem] = Image.open(p).convert("RGB")
    missing = [t for t in TEMPLATE_HAZARDS if t not in found]
    if missing:
        raise FileNotFoundError(f"Missing template(s) {missing} in {scene_dir} "
                                f"(expected clear.png, human.png, fire.png)")
    return found


# ─── Overlays ────────────────────────────────────────────────────────────────
def fill_dashboard(scr, rec):
    w, h = scr.size
    d = ImageDraw.Draw(scr)
    d.rectangle((0, 0, w, h), fill=(10, 16, 24))
    d.rectangle((0, 0, w, h), outline=(40, 70, 90), width=max(2, w // 120))
    pad = int(w * 0.05)

    # header
    d.rectangle((0, 0, w, int(h * 0.16)), fill=(18, 30, 44))
    d.text((pad, int(h * 0.03)), f"UNIT {rec['machine_id']}", font=F(int(h * 0.10)), fill=(120, 210, 235))
    d.text((w - pad - int(w * 0.24), int(h * 0.04)), "HMI", font=F(int(h * 0.08)), fill=(90, 120, 140))

    # equipment status
    sc = STATUS_COLORS[rec["equipment_status"]]
    d.text((pad, int(h * 0.21)), "STATUS", font=F(int(h * 0.085)), fill=(150, 170, 185))
    d.ellipse((pad, int(h * 0.31), pad + int(h * 0.13), int(h * 0.44)), fill=sc)
    d.text((pad + int(h * 0.18), int(h * 0.30)), rec["equipment_status"], font=F(int(h * 0.16)), fill=sc)

    # pressure / temperature
    colw = (w - 2 * pad) / 2.0
    lf, vf = F(int(h * 0.055)), M(int(h * 0.085))
    ry = int(h * 0.50)
    for i, (lab, val) in enumerate([("PRESSURE", rec["pressure"]), ("TEMP", rec["temp"])]):
        x = int(pad + i * colw)
        d.text((x, ry), lab, font=lf, fill=(120, 150, 170))
        d.text((x, ry + int(h * 0.075)), val, font=vf, fill=(220, 235, 245))

    # power source
    py = int(h * 0.70)
    d.text((pad, py), "POWER", font=lf, fill=(120, 150, 170))
    d.text((int(pad + colw), py - int(h * 0.012)), rec["power"],
           font=F(int(h * 0.095)), fill=POWER_COLORS[rec["power"]])

    # footer date/time
    d.rectangle((0, int(h * 0.86), w, h), fill=(18, 30, 44))
    d.text((pad, int(h * 0.87)), rec["dt"], font=M(int(h * 0.075)), fill=(150, 200, 220))
    return scr


def fill_plate(pl, rec):
    w, h = pl.size
    d = ImageDraw.Draw(pl)
    text = rec["room"]
    sz = int(h * 0.42)
    while sz > 8 and d.textlength(text, font=F(sz)) > w * 0.88:
        sz -= 1
    font = F(sz)
    bb = d.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((w - tw) // 2 - bb[0], (h - th) // 2 - bb[1]), text, font=font, fill=(20, 22, 26))
    return pl


def render_panel(rec, templates):
    im = templates[rec["template"]].copy()
    W, H = im.size
    for box, fn in [(SCREEN, fill_dashboard), (PLATE, fill_plate)]:
        x1, y1, x2, y2 = int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H)
        im.paste(fn(im.crop((x1, y1, x2, y2)), rec), (x1, y1))
    return im


# ─── Ground truth ────────────────────────────────────────────────────────────
def make_panel_record(rng, template=None):
    tname = template or rng.choices(list(TEMPLATE_WEIGHTS), weights=list(TEMPLATE_WEIGHTS.values()), k=1)[0]
    fire, person = TEMPLATE_HAZARDS[tname]["fire"], TEMPLATE_HAZARDS[tname]["person"]

    status = "FAULT" if fire == "YES" else rng.choices(["NORMAL", "WARNING", "FAULT"], weights=[70, 22, 8], k=1)[0]
    if fire == "YES" or status == "FAULT":
        alert = "CRITICAL"
    elif person == "YES" or status == "WARNING":
        alert = "CAUTION"
    else:
        alert = "CLEAR"

    dt = datetime.datetime(2026, rng.randint(1, 12), rng.randint(1, 28),
                           rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59))
    return {
        "template": tname,
        "fire": fire,
        "person": person,
        "machine_id": f"{rng.choice(MACHINE_PREFIX)}-{rng.randint(100, 999)}",
        "room": rng.choice(ROOMS),
        "equipment_status": status,
        "alert_level": alert,
        "pressure": f"{rng.uniform(2.0, 6.0):.1f} bar",
        "temp": f"{rng.randint(40, 95)} C",
        "power": rng.choices(["GRID", "BACKUP"], weights=[80, 20], k=1)[0],
        "dt": dt.strftime("%Y-%m-%d  %H:%M:%S"),
    }


def technician_presence_label(meta):
    return "PRESENT" if meta["person"] == "YES" else "ABSENT"


def answer_for_question(meta, q):
    return {
        "What is the machine ID?": meta["machine_id"],
        "Is the equipment status NORMAL, WARNING, or FAULT?": meta["equipment_status"],
        "What is the pressure reading?": meta["pressure"],
        "What is the temperature reading?": meta["temp"],
        "Is the power on GRID or BACKUP?": meta["power"],
        "Which room is this?": meta["room"],
        "Is there a fire?": meta["fire"],
        TECHNICIAN_Q: technician_presence_label(meta),
    }[q]


# ─── Build / save / load ─────────────────────────────────────────────────────
def build_panel_records(n_panels=300, seed=None, templates=None):
    """Render n_panels images in memory. seed=None gives a new dataset every call."""
    templates = templates or load_templates()
    rng = random.Random(seed)
    out = []
    for i in range(n_panels):
        rec = make_panel_record(rng)
        out.append({"panel_id": i, "img": render_panel(rec, templates), "meta": rec})
    return out


def save_dataset(out_dir=DATASET_DIR, n_panels=500, seed=2027):
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    templates = load_templates()
    rng = random.Random(seed)
    entries = []
    for i in range(n_panels):
        meta = make_panel_record(rng)
        fname = f"factory_{i:04d}.png"
        render_panel(meta, templates).save(os.path.join(img_dir, fname), optimize=True)
        entries.append({
            "panel_id": i,
            "file": f"images/{fname}",
            "event": EVENT_OF_TEMPLATE[meta["template"]],
            "meta": meta,
            "qa": [{"question": q, "answer": answer_for_question(meta, q)} for q in QUESTIONS],
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_panels}")

    with open(os.path.join(out_dir, "annotations.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "n_images": n_panels, "questions": QUESTIONS, "images": entries}, f, indent=1)

    cols = ["file", "event", "template", "room", "equipment_status", "pressure", "temperature",
            "power", "fire", "technician", "machine_id", "alert_level", "datetime"]
    with open(os.path.join(out_dir, "annotations.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for e in entries:
            m = e["meta"]
            w.writerow([e["file"], e["event"], m["template"], m["room"], m["equipment_status"],
                        m["pressure"], m["temp"], m["power"], m["fire"], technician_presence_label(m),
                        m["machine_id"], m["alert_level"], m["dt"]])
    print(f"saved {n_panels} images + annotations to {out_dir}")


def load_dataset(data_dir=DATASET_DIR, n_panels=None):
    """Load the released images as panel_records: [{panel_id, img, meta}, ...]."""
    with open(os.path.join(data_dir, "annotations.json"), encoding="utf-8") as f:
        ann = json.load(f)
    out = []
    for e in ann["images"][:n_panels]:
        img = Image.open(os.path.join(data_dir, e["file"])).convert("RGB")
        out.append({"panel_id": e["panel_id"], "img": img, "meta": e["meta"]})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500, help="number of images")
    ap.add_argument("--seed", type=int, default=2027, help="random seed")
    ap.add_argument("--out", default=DATASET_DIR, help="output directory")
    args = ap.parse_args()
    save_dataset(args.out, args.n, args.seed)
