# Fills the film panels of Fig. 1 with real crops of one development-half target, its donor and the four edit arms, cut at one 256-pixel window.
# Run: python3 paper_values/isbi_fig1_real_crops.py render | check

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from paper_values import edit_example as p73

HER = ROOT / "figures_source" / "framework_drawing.png"
HER_SHA256 = "148dfd4dc2c86c09b33f36a715af7485bd929c4722a24211afe35245abb89654"
OUT = ROOT / "paper_isbi" / "figures" / "combined.png"
DONOR_ID = "00008590_000.png"
DONOR_FILE = ROOT / "review" / "fig1_real_crops" / "donor" / DONOR_ID
DONOR_SHA256 = "4bcb07fce5f1b714c89515ac48a0238743dfb1910ff7114b696255c22d4529bd"

ORANGE, GOLD, BLUE, GREEN = (217, 96, 15), (227, 160, 8), (42, 127, 192), (30, 158, 122)
OUTLINE_PX = 3.0
HATCH_PX, HATCH_PERIOD = 1.6, 9.0
SS = 4
WINDOW = 256
TOP_FILL, RIGHT_FILL, EDGE = 58, 27, 17

def crop_origin(tube_mask: np.ndarray, disk_mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(tube_mask | disk_mask)
    cy, cx = (int(ys.min()) + int(ys.max())) // 2, (int(xs.min()) + int(xs.max())) // 2
    y0, x0 = cy - WINDOW // 2, cx - WINDOW // 2
    h, w = tube_mask.shape
    assert 0 <= y0 and y0 + WINDOW <= h and 0 <= x0 and x0 + WINDOW <= w, "the window leaves the film"
    assert y0 <= ys.min() and ys.max() < y0 + WINDOW and x0 <= xs.min() and xs.max() < x0 + WINDOW, "the window does not hold both shapes"
    return y0, x0

def _components(a: np.ndarray, value: int, min_px: int = 300) -> list[tuple[int, int, int, int]]:
    m = (a[:, :, 0] == value) & (a[:, :, 1] == value) & (a[:, :, 2] == value)
    lab, _ = ndi.label(m)
    out = []
    for i, o in enumerate(ndi.find_objects(lab)):
        if int((lab[o] == i + 1).sum()) > min_px:
            out.append((o[0].start, o[0].stop, o[1].start, o[1].stop))
    return sorted(out)

def face_outline(g: np.ndarray, top: tuple, right: tuple) -> tuple[int, int, int, int]:
    ty0, ty1, tx0, tx1 = top
    ry0, ry1, rx0, rx1 = right
    ymid = (ty1 + ry1) // 2
    left = next(x for x in range(tx0 - 8, tx0 + 4) if g[ymid, x] == EDGE and g[ymid, x - 1] >= 150)
    xs = left + 14
    top_e = next(y for y in range(ty1 - 3, ty1 + 6) if g[y, xs] == EDGE)
    right_e = next(x for x in range(rx0 - 6, rx0 + 4) if g[top_e + 25, x] == EDGE and g[top_e + 25, x + 1] == 25)
    bottom = next(y for y in range(ry1 - 3, ry1 + 12) if g[y, xs] == EDGE and g[y + 1, xs] >= 150)
    return left, top_e, right_e, bottom

def silhouette(shape: tuple, rect: tuple, shear: int) -> np.ndarray:
    from PIL import ImageDraw
    l, t, r, b = rect
    im = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(im).polygon([(l, b), (r, b), (r + shear, b - shear), (r + shear, t - shear), (l + shear, t - shear), (l, t)], fill=255)
    return np.array(im) > 0

def outline_layer(mask_crop: np.ndarray, size: int, hatch: bool) -> np.ndarray:
    big = size * SS
    m = np.array(Image.fromarray(mask_crop.astype(np.uint8) * 255).resize((big, big), Image.BILINEAR)) > 127
    sd = ndi.distance_transform_edt(~m) - ndi.distance_transform_edt(m)
    w = OUTLINE_PX * SS / 2.0
    cover = np.abs(sd) <= w
    if hatch:
        yy, xx = np.mgrid[0:big, 0:big]
        lines = (((xx + yy) % (HATCH_PERIOD * SS)) < HATCH_PX * SS) & (sd < 0)
        cover = cover | lines
    return cover.reshape(size, SS, size, SS).mean(axis=(1, 3))

def paste(canvas: np.ndarray, rect: tuple, crop_gray: np.ndarray, visible: np.ndarray | None, layers: list[tuple[np.ndarray, tuple]]) -> None:
    l, t, r, b = rect
    size = r - l - 1
    assert b - t - 1 == size, "the front faces of the cubes are square"
    film = np.array(Image.fromarray(crop_gray).resize((size, size), Image.LANCZOS)).astype(float)
    rgb = np.repeat(film[:, :, None], 3, axis=2)
    for cov, color in layers:
        rgb = rgb * (1 - cov[:, :, None]) + np.array(color, dtype=float)[None, None, :] * cov[:, :, None]
    region = canvas[t + 1:b, l + 1:r]
    new = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    if visible is None:
        canvas[t + 1:b, l + 1:r] = new
    else:
        vis = visible[t + 1:b, l + 1:r]
        region[vis] = new[vis]

def cube_rects(her: np.ndarray) -> dict[str, tuple[int, int, int, int]]:
    a = her.astype(int)
    g = a[:, :, 0]
    tops, rights = _components(a, TOP_FILL), _components(a, RIGHT_FILL)
    assert len(tops) == len(rights) == 13, (len(tops), len(rights))
    index = {"target": 0, "tube": 3, "tube_tissue": 4, "disk": 5, "disk_tissue": 6, "donor_front": 10, "m1_front": 12}
    expect_top_y = {0: 196, 3: 214, 4: 214, 5: 502, 6: 502, 10: 579, 12: 1053}
    for k, i in index.items():
        assert tops[i][0] == expect_top_y[i], f"cube {k} is not where the drawing has it"
    rect = {k: face_outline(g, tops[i], rights[i]) for k, i in index.items()}
    off = (16, -16)
    l, t, r, b = rect["donor_front"]
    rect["donor_mid"] = (l + off[0], t + off[1], r + off[0], b + off[1])
    rect["donor_back"] = (l + 2 * off[0], t + 2 * off[1], r + 2 * off[0], b + 2 * off[1])
    l, t, r, b = rect["m1_front"]
    rect["m1_back"] = (l + off[0], t + off[1], r + off[0], b + off[1])
    return rect

def film_areas(rect: dict, shape: tuple) -> np.ndarray:
    allowed = np.zeros(shape[:2], bool)
    for l, t, r, b in rect.values():
        allowed[t + 1:b, l + 1:r] = True
    return allowed

def render() -> tuple[np.ndarray, dict]:
    from cxr_shortcuts.data.reserved_guard import assert_not_reserved

    assert hashlib.sha256(HER.read_bytes()).hexdigest() == HER_SHA256, "the framework drawing is not the file this figure starts from (sha256 differs)"
    iid = p73.chosen_id()
    assert_not_reserved([f"{iid}.png", DONOR_ID], "isbi_fig1_real_crops.render")
    comp = p73.INSPECT / f"{iid}.npz"
    assert hashlib.sha256(comp.read_bytes()).hexdigest() == p73.EXPECTED_SHA256, "the inspection composite is not the one the rule selected"
    assert hashlib.sha256(DONOR_FILE.read_bytes()).hexdigest() == DONOR_SHA256, "the donor film is not the one this figure was made from (sha256 differs)"
    with np.load(comp) as z:
        y0, x0 = crop_origin(z["tube_mask"], z["disk_mask"])
        w = WINDOW
        cut = lambda arr: np.ascontiguousarray(arr[y0:y0 + w, x0:x0 + w])
        crops = {"before": cut(z["before"]), "tube": cut(z["tube"]), "tube_tissue": cut(z["tube_tissue"][0]), "disk": cut(z["disk"]), "disk_tissue": cut(z["disk_tissue"][0])}
        masks = {"tube": cut(z["tube_mask"]), "disk": cut(z["disk_mask"])}
        shape = z["before"].shape
    donor = np.array(Image.open(DONOR_FILE).convert("L"))
    assert donor.shape == shape, "the donor film has another size than the target"
    crops["donor"] = cut(donor)

    pic = Image.open(HER)
    if pic.mode == "RGBA":
        assert int(np.array(pic)[:, :, 3].min()) == 255, "the drawing has a transparent pixel"
    her = np.array(pic.convert("RGB"))
    out = her.copy()
    rect = cube_rects(her)
    h, wd = out.shape[:2]
    shear = {"big": 13, "small": 8}

    def sil(k: str, s: int) -> np.ndarray:
        return silhouette((h, wd), rect[k], s)

    def visible_behind(k: str, in_front: list[str]) -> np.ndarray:
        cover = np.zeros((h, wd), bool)
        for f in in_front:
            cover |= ndi.binary_dilation(sil(f, shear["small"]), iterations=1)
        film = np.zeros((h, wd), bool)
        l, t, r, b = rect[k]
        film[t + 1:b, l + 1:r] = True
        return film & ~cover

    paste(out, rect["target"], crops["before"], None, [])
    paste(out, rect["tube"], crops["tube"], None, [(outline_layer(masks["tube"], rect["tube"][2] - rect["tube"][0] - 1, False), ORANGE)])
    paste(out, rect["tube_tissue"], crops["tube_tissue"], None, [(outline_layer(masks["tube"], rect["tube_tissue"][2] - rect["tube_tissue"][0] - 1, True), BLUE)])
    paste(out, rect["disk"], crops["disk"], None, [(outline_layer(masks["disk"], rect["disk"][2] - rect["disk"][0] - 1, False), GOLD)])
    paste(out, rect["disk_tissue"], crops["disk_tissue"], None, [(outline_layer(masks["disk"], rect["disk_tissue"][2] - rect["disk_tissue"][0] - 1, True), GREEN)])
    paste(out, rect["donor_front"], crops["donor"], None, [])
    paste(out, rect["donor_mid"], crops["donor"], visible_behind("donor_mid", ["donor_front"]), [])
    paste(out, rect["donor_back"], crops["donor"], visible_behind("donor_back", ["donor_front", "donor_mid"]), [])
    paste(out, rect["m1_front"], crops["tube"], None, [])
    paste(out, rect["m1_back"], crops["before"], visible_behind("m1_back", ["m1_front"]), [])

    changed = np.any(out != her, axis=2)
    stray = int((changed & ~film_areas(rect, out.shape)).sum())
    assert stray == 0, f"{stray} pixels outside the film areas changed"
    return out, {"id": iid, "donor": DONOR_ID, "window": (y0, x0), "changed_px": int(changed.sum()), "rects": rect}

def write(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, "RGB").save(path, dpi=(300, 300), optimize=True)

def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("render", "check"):
        print("Usage: python3 paper_values/isbi_fig1_real_crops.py render | check")
        return 2
    arr, info = render()
    if cmd == "render":
        write(arr, OUT)
        print(f"wrote {OUT} ({arr.shape[1]} by {arr.shape[0]}); target {info['id']}, donor {info['donor']}, window {info['window']}, {info['changed_px']} pixels changed, all inside the cubes' film areas")
        return 0
    tmp = OUT.with_suffix(".check.png")
    write(arr, tmp)
    same = OUT.exists() and tmp.read_bytes() == OUT.read_bytes()
    tmp.unlink()
    print("combined.png is current" if same else "combined.png differs from a fresh render")
    return 0 if same else 1

if __name__ == "__main__":
    sys.exit(main())
