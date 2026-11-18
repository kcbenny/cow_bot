import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cow_core import Health, Inventory, InventoryPanelUnavailable, SafetyError

ROOT = Path(__file__).resolve().parent
PROFILE = ROOT / "cow_profile"
HOVER = (0, 29, 500, 11)
WORLD = (520, 110, 1120, 750)
ROUTE_CLICK_AREA = (0, 24, 1920, 1016)
ROUTE_ANCHOR_SEARCH = (0, 40, 1676, 830)
PANEL = (1676, 722, 200, 6)
INVENTORY_TAB = (1558, 998, 20, 30)
INVENTORY_TAB_POINT = (1568, 1013)
INVENTORY_VERTICAL_OFFSETS = range(-12, 13)
CHAT = (0, 870, 520, 130)
CENTERS = [(1713 + col * 42, 752 + row * 36) for row in range(7) for col in range(4)]
COWHIDE_HSV_LOWER = np.asarray([0, 0, 160], dtype=np.uint8)
COWHIDE_HSV_UPPER = np.asarray([20, 50, 255], dtype=np.uint8)
COWHIDE_MIN_CONTOUR_AREA = 80
COWHIDE_MAX_CONTOUR_AREA = 300


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


def learn_line(
    alphabet: dict[str, list], image: np.ndarray, label: str, *, prefix_only: bool = False,
) -> None:
    mask = text_mask(crop(image, HOVER))
    columns = runs(mask)
    letters = label.replace(" ", "")
    if len(columns) < len(letters) or (not prefix_only and len(columns) != len(letters)):
        raise SafetyError(f"Cannot learn native font from {label!r}")
    for character, cols in zip(letters, columns[:len(letters)]):
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
        busy_message = load_image(ROOT / "images" / "busy_fighting_chat.png")
        self.busy_message_template = (busy_message.max(axis=2) < 100).astype(np.uint8) * 255

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
        title = crop(image, (4, 50, 145, 16)).max(axis=2) >= 180
        if any(np.array_equal(title, mask) for mask in self.huds.values()):
            return False
        bar = crop(image, (7, 69, 138, 15))
        green = (bar[:, :, 1] > 150) & (bar[:, :, 0] < 80) & (bar[:, :, 2] < 80)
        red = (bar[:, :, 0] > 150) & (bar[:, :, 1] < 80) & (bar[:, :, 2] < 80)
        return int(np.count_nonzero(green | red)) < 100

    def player_motion(self, previous: np.ndarray, current: np.ndarray) -> int:
        if previous.shape != current.shape:
            raise SafetyError("Player-motion frames have different dimensions")
        width, height = current.shape[1], current.shape[0]
        anchor_x, anchor_y = self.settings.get("player_anchor", [942, 520])
        left, top = int(anchor_x) - 32, int(anchor_y) - 40
        if left < 0 or top < 0 or left + 64 > width or top + 80 > height:
            raise SafetyError("Player anchor is outside the screenshot; recalibration required")
        before = cv2.cvtColor(crop(previous, (left, top, 64, 80)), cv2.COLOR_RGB2GRAY)
        after = cv2.cvtColor(crop(current, (left, top, 64, 80)), cv2.COLOR_RGB2GRAY)
        return int(np.count_nonzero(cv2.absdiff(before, after) > 40))

    def busy_message_rows(self, image: np.ndarray) -> list[int]:
        chat = crop(image, CHAT)
        mask = (chat.max(axis=2) < 100).astype(np.uint8) * 255
        scores = cv2.matchTemplate(
            mask, self.busy_message_template, cv2.TM_CCOEFF_NORMED,
        )
        matched_rows = [
            row for row in range(scores.shape[0])
            if float(scores[row].max()) >= 0.97
        ]
        distinct_rows = []
        for row in matched_rows:
            if not distinct_rows or row - distinct_rows[-1] > 4:
                distinct_rows.append(row)
        return distinct_rows

    def busy_message_count(self, image: np.ndarray) -> int:
        return len(self.busy_message_rows(image))

    def busy_message_added(
        self, before: np.ndarray, after: np.ndarray, before_count: int | None = None,
    ) -> bool:
        after_rows = self.busy_message_rows(after)
        previous_count = (
            self.busy_message_count(before) if before_count is None else before_count
        )
        if len(after_rows) > previous_count:
            return True
        if not after_rows:
            return False
        before_chat = crop(before, CHAT)
        after_chat = crop(after, CHAT)
        if np.array_equal(before_chat, after_chat):
            return False
        last_row = after_rows[-1]
        row_height = self.busy_message_template.shape[0]
        # When chat is full, a new busy line scrolls older lines up while keeping the same visible count.
        has_lower_text = bool(
            (after_chat[last_row + row_height + 2:110].max(axis=2) < 100).any()
        )
        if has_lower_text:
            return False
        return not np.array_equal(
            before_chat[last_row:last_row + row_height],
            after_chat[last_row:last_row + row_height],
        )

    def target_in_combat(self, image: np.ndarray, point: tuple[int, int]) -> bool:
        x, y = point
        height, width = image.shape[:2]
        left, top = max(0, x - 65), max(0, y - 90)
        right, bottom = min(width, x + 65), min(height, y + 20)
        if right <= left or bottom <= top:
            return False
        roi = image[top:bottom, left:right]
        green = (roi[:, :, 0] < 30) & (roi[:, :, 1] > 220) & (roi[:, :, 2] < 30)
        red = (roi[:, :, 0] > 220) & (roi[:, :, 1] < 30) & (roi[:, :, 2] < 30)
        mask = (green | red).astype(np.uint8) * 255
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for index in range(1, count):
            _, _, bw, bh, area = stats[index]
            if 26 <= bw <= 34 and 4 <= bh <= 6 and area >= 120:
                return True
        return False

    def inventory(self, image: np.ndarray) -> Inventory:
        offset = self.inventory_panel_offset(image)
        if offset is None:
            raise InventoryPanelUnavailable("Inventory panel is hidden, obstructed, or changed")
        slots = []
        for i, (x, y) in enumerate(CENTERS):
            y += offset
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

    def inventory_panel_offset(self, image: np.ndarray) -> int | None:
        x, y, width, height = PANEL
        matches = [
            offset
            for offset in INVENTORY_VERTICAL_OFFSETS
            if similar(
                crop(image, (x, y + offset, width, height)),
                self.panel,
            )
        ]
        return matches[0] if len(matches) == 1 else None

    def inventory_panel_visible(self, image: np.ndarray) -> bool:
        return self.inventory_panel_offset(image) is not None

    def inventory_tab_offset(self, image: np.ndarray) -> int | None:
        x, y, width, height = INVENTORY_TAB
        templates = [
            cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
            for template in self.inventory_tab_templates
        ]
        scores = []
        for offset in INVENTORY_VERTICAL_OFFSETS:
            actual = cv2.cvtColor(
                crop(image, (x, y + offset, width, height)),
                cv2.COLOR_RGB2GRAY,
            )
            scores.append(max(
                float(cv2.matchTemplate(actual, template, cv2.TM_CCOEFF_NORMED).max())
                for template in templates
            ))
        ranked = sorted(scores, reverse=True)
        if ranked[0] < 0.82 or ranked[0] - ranked[1] < 0.1:
            return None
        return list(INVENTORY_VERTICAL_OFFSETS)[scores.index(ranked[0])]

    def inventory_tab_visible(self, image: np.ndarray) -> bool:
        return self.inventory_tab_offset(image) is not None

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
        distance = 0.0
        angle = 0.0
        visited = set()
        while distance <= radius:
            point = (
                round(center[0] + distance * np.cos(angle)),
                round(center[1] + distance * np.sin(angle)),
            )
            if (
                x + 12 <= point[0] < x + width - 12
                and y + 12 <= point[1] < y + height - 12
                and point not in visited
            ):
                visited.add(point)
                yield point, "fine" if distance <= fine_radius else "coarse"
            spacing = 12 if distance <= fine_radius else 24
            angle_step = min(1.0, spacing / max(distance, spacing))
            distance += spacing * angle_step / (2 * np.pi)
            angle += angle_step

    def loot_candidates(
        self,
        image: np.ndarray,
        center: tuple[int, int],
        radius: int,
        fine_radius: int = 48,
    ) -> list[tuple[tuple[int, int], str]]:
        self.check_geometry(image)
        search_x, search_y, search_width, search_height = self.settings["search"]
        left = max(search_x + 12, center[0] - radius)
        top = max(search_y + 12, center[1] - radius)
        right = min(search_x + search_width - 12, center[0] + radius + 1)
        bottom = min(search_y + search_height - 12, center[1] + radius + 1)
        if right <= left or bottom <= top:
            return []

        region = crop(image, (left, top, right - left, bottom - top))
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, COWHIDE_HSV_LOWER, COWHIDE_HSV_UPPER)
        circle = np.zeros(mask.shape, dtype=np.uint8)
        cv2.circle(
            circle,
            (center[0] - left, center[1] - top),
            radius,
            255,
            thickness=-1,
        )
        mask = cv2.bitwise_and(mask, circle)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        red_accent = cv2.bitwise_or(
            cv2.inRange(
                hsv,
                np.asarray([0, 70, 45], dtype=np.uint8),
                np.asarray([20, 255, 240], dtype=np.uint8),
            ),
            cv2.inRange(
                hsv,
                np.asarray([170, 70, 45], dtype=np.uint8),
                np.asarray([179, 255, 240], dtype=np.uint8),
            ),
        )

        candidates = []
        fallback_candidates = []
        seen = set()
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, width, height = cv2.boundingRect(contour)
            aspect_ratio = width / height
            red_pixels = cv2.countNonZero(red_accent[y:y + height, x:x + width])
            is_full_cowhide = (
                COWHIDE_MIN_CONTOUR_AREA < area < COWHIDE_MAX_CONTOUR_AREA
                and 12 <= width <= 26
                and 10 <= height <= 22
                and 0.7 <= aspect_ratio <= 1.8
                and red_pixels >= 5
            )
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            centroid_x = moments["m10"] / moments["m00"]
            centroid_y = moments["m01"] / moments["m00"]
            distance_squared = (
                (left + centroid_x - center[0]) ** 2
                + (top + centroid_y - center[1]) ** 2
            )
            # Partially occluded cowhide near the kill point/player can have a smaller contour.
            is_partial_near_center = (
                not is_full_cowhide
                and distance_squared <= fine_radius ** 2
                and 35 <= area <= COWHIDE_MIN_CONTOUR_AREA
                and 7 <= width <= 26
                and 6 <= height <= 22
                and red_pixels >= 2
            )
            if not is_full_cowhide and not is_partial_near_center:
                continue
            interior_points = [
                (candidate_x, candidate_y)
                for candidate_y in range(
                    max(y, math.ceil(centroid_y - 5)),
                    min(y + height - 1, math.floor(centroid_y + 5)) + 1,
                )
                for candidate_x in range(
                    max(x, math.ceil(centroid_x - 5)),
                    min(x + width - 1, math.floor(centroid_x + 5)) + 1,
                )
                if cv2.pointPolygonTest(
                    contour, (float(candidate_x), float(candidate_y)), False,
                ) >= 0
            ]
            if interior_points:
                candidate_x, candidate_y = random.choice(interior_points)
            else:
                contour_points = contour.reshape(-1, 2)
                nearest = contour_points[
                    np.argmin(
                        (contour_points[:, 0] - centroid_x) ** 2
                        + (contour_points[:, 1] - centroid_y) ** 2
                    )
                ]
                candidate_x, candidate_y = map(int, nearest)
            point = (left + candidate_x, top + candidate_y)
            point_distance_squared = (
                (point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2
            )
            if point_distance_squared > radius ** 2 or point in seen:
                continue
            seen.add(point)
            distance = np.sqrt(point_distance_squared)
            entry = (point, "fine" if distance <= fine_radius else "coarse")
            if is_full_cowhide:
                candidates.append(entry)
            else:
                fallback_candidates.append(entry)
        sort_key = lambda candidate: (
            (candidate[0][0] - center[0]) ** 2
            + (candidate[0][1] - center[1]) ** 2,
            candidate[0][1],
            candidate[0][0],
        )
        return sorted(candidates, key=sort_key) + sorted(fallback_candidates, key=sort_key)
