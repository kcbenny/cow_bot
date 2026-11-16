import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cow_core import Health, Inventory, InventoryPanelUnavailable, SafetyError

ROOT = Path(__file__).resolve().parent
PROFILE = ROOT / "cow_profile"
HOVER = (0, 29, 500, 11)
WORLD = (520, 110, 1120, 750)
PANEL = (1676, 722, 200, 6)
INVENTORY_TAB = (1558, 998, 20, 30)
INVENTORY_TAB_POINT = (1568, 1013)
CENTERS = [(1713 + col * 42, 752 + row * 36) for row in range(7) for col in range(4)]


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def crop(image: np.ndarray, rect: tuple[int, int, int, int] | list[int]) -> np.ndarray:
    x, y, width, height = rect
    if min(x, y) < 0 or min(width, height) <= 0:
        raise SafetyError(f"Invalid ROI: {rect}")
    if x + width > image.shape[1] or y + height > image.shape[0]:
        raise SafetyError(f"ROI outside screenshot: {rect}")
    return image[y:y + height, x:x + width]


def text_mask(image: np.ndarray) -> np.ndarray:
    return image.max(axis=2) >= 180


def runs(mask: np.ndarray) -> list[np.ndarray]:
    columns = np.flatnonzero(mask.any(axis=0))
    if not len(columns):
        return []
    return list(np.split(columns, np.where(np.diff(columns) > 1)[0] + 1))


def learn_line(alphabet: dict[str, list], image: np.ndarray, label: str) -> None:
    mask = text_mask(crop(image, HOVER))
    columns = runs(mask)
    letters = label.replace(" ", "")
    if len(columns) != len(letters):
        raise SafetyError(f"Cannot learn native font from {label!r}")
    for character, cols in zip(letters, columns):
        glyph = mask[:, cols[0]:cols[-1] + 1].astype(int).tolist()
        variants = alphabet.setdefault(character, [])
        if glyph not in variants:
            variants.append(glyph)


def similar(actual: np.ndarray, expected: np.ndarray, tolerance: float = 3) -> bool:
    if actual.shape != expected.shape:
        return False
    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    return bool(delta.mean() <= tolerance and np.mean(delta.max(axis=2) <= 12) >= 0.98)


def bootstrap(profile: Path = PROFILE) -> None:
    profile.mkdir(parents=True, exist_ok=True)
    font_path = profile / "font.json"
    alphabet: dict[str, list] = (
        json.loads(font_path.read_text(encoding="utf-8")) if font_path.exists() else {}
    )
    labels = {
        "attack_cow": "Attack Cow (level-2) / 3 more options",
        "attack_cowcalf": "Attack Cow calf (level-2) / 3 more options",
        "take_cowhide": "Take Cowhide / 7 more options",
        "bank_booth": "Bank Bank booth / 4 more options",
        "gate_closed": "Open Gate / 2 more options",
        "gate_opened": "Close Gate / 3 more options",
        "before_take_cowhide": "Chop down Willow tree / 2 more options",
        "after_take_cowhide": "Walk here / 2 more options",
        "attacking_cowcalf": "Walk here",
    }
    for name, label in labels.items():
        learn_line(alphabet, load_image(ROOT / "images" / f"{name}.png"), label)
    empty = load_image(ROOT / "images" / "bank_booth.png")
    for i, (x, y) in enumerate(CENTERS):
        save_image(profile / f"empty_{i}.png", crop(empty, (x - 16, y - 16, 32, 32)))
    save_image(profile / "panel.png", crop(empty, PANEL))
    hide = load_image(ROOT / "images" / "before_take_cowhide.png")
    save_image(profile / "Cowhide.png", crop(hide, (1697, 736, 32, 32)))
    save_image(profile / "inventory_tab_open.png", crop(hide, INVENTORY_TAB))
    closed_tab = load_image(ROOT / "images" / "inventory_tab_closed.png")
    if closed_tab.shape[:2] != (INVENTORY_TAB[3], INVENTORY_TAB[2]):
        raise SafetyError("Closed inventory-tab reference must be 20x30 pixels")
    save_image(profile / "inventory_tab_closed.png", closed_tab)
    hover_tab = load_image(ROOT / "images" / "inventory_tab_hover.png")
    if hover_tab.shape[:2] != (INVENTORY_TAB[3], INVENTORY_TAB[2]):
        raise SafetyError("Hovered inventory-tab reference must be 20x30 pixels")
    save_image(profile / "inventory_tab_hover.png", hover_tab)
    for target, filename in (("Cow", "attacking_cow"), ("Cow calf", "attacking_cowcalf")):
        image = load_image(ROOT / "images" / f"{filename}.png")
        save_image(profile / f"hud_{target.replace(' ', '_')}.png", crop(image, (4, 50, 145, 16)))
    dead = load_image(ROOT / "images" / "take_cowhide.png")
    white = crop(dead, (60, 70, 34, 12)).min(axis=2) >= 180
    zero_cols = runs(white)[0]
    zero = white[:, zero_cols[0]:zero_cols[-1] + 1]
    bank_video = cv2.VideoCapture(str(ROOT / "deposit_inventory_button.mp4"))
    try:
        bank_video.set(cv2.CAP_PROP_POS_MSEC, 5000)
        ok, frame = bank_video.read()
        if not ok:
            raise SafetyError("Cannot read supplied deposit video")
        bank = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        bank_video.release()
    learn_line(alphabet, bank, "Deposit inventory")
    save_image(profile / "bank_header.png", crop(bank, (760, 51, 158, 22)))
    save_image(profile / "deposit_frame.png", bank)
    save_image(profile / "deposit_inventory.png", crop(bank, (1000, 806, 24, 18)))
    menu = load_image(ROOT / "images" / "rightclick_menu.png")
    for item, y in (("Cowhide", 554), ("Bones", 570)):
        mask = text_mask(crop(menu, (804, y, 116, 10)))
        columns = np.flatnonzero(mask.any(axis=0))
        save_image(
            profile / f"menu_{item}.png",
            crop(menu, (804, y, int(columns[-1]) + 1, 10)),
        )
    font_path.write_text(json.dumps(alphabet), encoding="utf-8")
    (profile / "zero.json").write_text(json.dumps(zero.astype(int).tolist()), encoding="utf-8")
    if not (profile / "settings.json").exists():
        settings = {"screen": [1920, 1080], "search": list(WORLD), "loot_radius": 130}
        (profile / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")


class Vision:
    def __init__(self, profile: Path = PROFILE) -> None:
        self.profile = profile
        self.settings = json.loads((profile / "settings.json").read_text(encoding="utf-8"))
        data = json.loads((profile / "font.json").read_text(encoding="utf-8"))
        self.font = {
            char: [np.asarray(variant, dtype=bool) for variant in variants]
            for char, variants in data.items()
        }
        self.zero = np.asarray(json.loads((profile / "zero.json").read_text()), dtype=bool)
        self.empty_slots = [load_image(profile / f"empty_{i}.png") for i in range(28)]
        self.panel = load_image(profile / "panel.png")
        self.inventory_tab_templates = [
            load_image(profile / "inventory_tab_open.png"),
            load_image(profile / "inventory_tab_closed.png"),
            load_image(profile / "inventory_tab_hover.png"),
        ]
        self.items = {
            item: load_image(profile / f"{item}.png")
            for item in ("Cowhide", "Bones") if (profile / f"{item}.png").is_file()
        }
        self.huds = {
            target: text_mask(load_image(profile / f"hud_{target.replace(' ', '_')}.png"))
            for target in ("Cow", "Cow calf")
        }
        self.bank_header = load_image(profile / "bank_header.png")

    def check_geometry(self, image: np.ndarray) -> None:
        width, height = self.settings["screen"]
        if abs(image.shape[1] - width) > 1 or abs(image.shape[0] - height) > 1:
            raise SafetyError("Resolution changed; recalibration required")

    def hover_mask(self, image: np.ndarray) -> np.ndarray:
        return text_mask(crop(image, HOVER))

    def hover(self, image: np.ndarray) -> str:
        mask = self.hover_mask(image)
        result = ""
        previous = None
        for columns in runs(mask):
            glyph = mask[:, columns[0]:columns[-1] + 1]
            matches = {
                char for char, variants in self.font.items()
                if any(glyph.shape == v.shape and np.array_equal(glyph, v) for v in variants)
            }
            if len(matches) != 1:
                return ""
            if matches == {"/"}:
                break
            if previous is not None and columns[0] - previous > 4:
                result += " "
            result += matches.pop()
            previous = columns[-1]
        return result.split(" / ")[0].strip()

    def health(self, image: np.ndarray, *, compressed: bool = False) -> Health | None:
        threshold = 150 if compressed else 180
        title = crop(image, (4, 50, 145, 16)).max(axis=2) >= threshold
        matches = [name for name, mask in self.huds.items() if np.array_equal(title, mask)]
        if not matches:
            return None
        digits = crop(image, (60, 70, 34, 12)).min(axis=2) >= threshold
        columns = runs(digits)
        if len(columns) != 3:
            return None
        glyph = digits[:, columns[0][0]:columns[0][-1] + 1]
        is_zero = glyph.shape == self.zero.shape and np.array_equal(glyph, self.zero)
        bar = crop(image, (7, 69, 138, 15))
        green = (bar[:, :, 1] > 150) & (bar[:, :, 0] < 80) & (bar[:, :, 2] < 80)
        if is_zero and not green.any():
            return Health(matches[0], False)
        if not is_zero and np.count_nonzero(green) >= 10:
            return Health(matches[0], True)
        return None

    def hud_clear(self, image: np.ndarray) -> bool:
        area = crop(image, (4, 50, 145, 39))
        return bool(np.mean(area.max(axis=2) < 30) >= 0.98)

    def inventory(self, image: np.ndarray) -> Inventory:
        if not self.inventory_panel_visible(image):
            raise InventoryPanelUnavailable("Inventory panel is hidden, obstructed, or changed")
        slots = []
        for i, (x, y) in enumerate(CENTERS):
            cell = crop(image, (x - 16, y - 16, 32, 32))
            if similar(cell, self.empty_slots[i]):
                slots.append("EMPTY")
                continue
            area = crop(image, (x - 18, y - 18, 36, 36))
            matches = []
            for name, template in self.items.items():
                scores = cv2.matchTemplate(area, template, cv2.TM_SQDIFF_NORMED)
                if float(scores.min()) <= 0.035:
                    matches.append(name)
            slots.append(matches[0] if len(matches) == 1 else "UNKNOWN")
        return Inventory(tuple(slots))

    def inventory_panel_visible(self, image: np.ndarray) -> bool:
        return similar(crop(image, PANEL), self.panel)

    def inventory_tab_visible(self, image: np.ndarray) -> bool:
        actual = crop(image, INVENTORY_TAB)
        actual_gray = cv2.cvtColor(actual, cv2.COLOR_RGB2GRAY)
        best_score = max(
            float(cv2.matchTemplate(
                actual_gray,
                cv2.cvtColor(template, cv2.COLOR_RGB2GRAY),
                cv2.TM_CCOEFF_NORMED,
            ).max())
            for template in self.inventory_tab_templates
        )
        return best_score >= 0.82

    def bank_open(self, image: np.ndarray) -> bool:
        return similar(crop(image, (760, 51, 158, 22)), self.bank_header, 6)

    def grid(self, center: tuple[int, int], radius: int | None = None):
        x, y, width, height = self.settings["search"]
        step = 12 if radius is not None else 24
        points = [
            (px, py)
            for py in range(y + 12, y + height - 12, step)
            for px in range(x + 12, x + width - 12, step)
            if radius is None or (px - center[0]) ** 2 + (py - center[1]) ** 2 <= radius ** 2
        ]
        yield from sorted(points, key=lambda p: (p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2)

    def loot_grid(
        self,
        center: tuple[int, int],
        radius: int,
        fine_radius: int = 48,
    ):
        x, y, width, height = self.settings["search"]
        phases = (
            ("fine", 12, 0, min(radius, fine_radius)),
            ("coarse", 24, min(radius, fine_radius), radius),
        )
        for name, step, inner_radius, outer_radius in phases:
            inner_squared = inner_radius ** 2
            outer_squared = outer_radius ** 2
            points = [
                (px, py)
                for py in range(y + 12, y + height - 12, step)
                for px in range(x + 12, x + width - 12, step)
                if (
                    (distance_squared := (px - center[0]) ** 2 + (py - center[1]) ** 2)
                    <= outer_squared
                    and (name == "fine" or distance_squared > inner_squared)
                )
            ]
            yield from (
                (point, name)
                for point in sorted(
                    points,
                    key=lambda p: (p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2,
                )
            )
