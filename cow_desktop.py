import ctypes
from ctypes import wintypes
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageGrab

from cow_core import HealthLock, InventoryPanelUnavailable, Mode, PickupCheck, SafetyError
from cow_vision import (
    INVENTORY_TAB_POINT, Vision, crop, load_image, save_image, similar, text_mask,
)


def foreground_window_problem(title: str, minimized: bool) -> str | None:
    problems = []
    if title != "Old School RuneScape":
        problems.append(f"unexpected foreground title={title[:120]!r}")
    if minimized:
        problems.append("foreground window is minimized")
    if not problems:
        return None
    return "Foreground-window check failed: " + "; ".join(problems)


@dataclass(frozen=True)
class Frame:
    number: int
    at: float
    image: np.ndarray


class Desktop:
    def __init__(self, vision: Vision, live: bool = False) -> None:
        if not hasattr(ctypes, "windll"):
            raise SafetyError("Desktop operation requires Windows")
        ctypes.windll.user32.SetProcessDPIAware()
        user = ctypes.windll.user32
        user.GetForegroundWindow.restype = wintypes.HWND
        user.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user.IsIconic.argtypes = [wintypes.HWND]
        user.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        import pyautogui
        self.mouse = pyautogui
        self.mouse.FAILSAFE = True
        self.mouse.PAUSE = 0
        self.vision = vision
        self.live = live
        self.frames: deque[Frame] = deque(maxlen=20)
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.error: Exception | None = None
        self.worker: threading.Thread | None = None
        self.sequence = 0
        self.cursor: tuple[int, int] | None = None
        self.last_pick_menu_failure: str | None = None

    def check(self) -> None:
        user = ctypes.windll.user32
        if self.stop.is_set() or user.GetAsyncKeyState(0x77) & 0x8000:
            raise SafetyError("Stopped by F8 or shutdown")
        hwnd = user.GetForegroundWindow()
        title = ctypes.create_unicode_buffer(512)
        user.GetWindowTextW(hwnd, title, len(title))
        problem = foreground_window_problem(title.value, bool(user.IsIconic(hwnd)))
        if problem is not None:
            raise SafetyError(problem)
        rect = wintypes.RECT()
        user.GetWindowRect(hwnd, ctypes.byref(rect))
        width, height = self.vision.settings["screen"]
        if abs(rect.left) > 10 or abs(rect.top) > 10 or rect.right < width - 10 or rect.bottom < height - 60:
            raise SafetyError("OSRS must use the calibrated maximized window layout")

    def snapshot(self) -> np.ndarray:
        self.check()
        image = np.asarray(ImageGrab.grab()).copy()
        self.vision.check_geometry(image)
        return image

    def _capture(self) -> None:
        try:
            while not self.stop.is_set():
                image = self.snapshot()
                self.sequence += 1
                frame = Frame(self.sequence, time.monotonic(), image)
                with self.condition:
                    self.frames.append(frame)
                    self.condition.notify_all()
                self.stop.wait(0.10)
        except (OSError, SafetyError) as error:
            self.error = error
            with self.condition:
                self.condition.notify_all()

    def __enter__(self):
        self.worker = threading.Thread(target=self._capture, name="cow-capture", daemon=True)
        self.worker.start()
        try:
            self.fresh()
        except (SafetyError, OSError):
            self.__exit__()
            raise
        return self

    def __exit__(self, *args) -> None:
        self.stop.set()
        if self.worker:
            self.worker.join(timeout=3)

    def fresh(self, after: float | None = None) -> Frame:
        after = time.monotonic() if after is None else after
        deadline = time.monotonic() + 2
        with self.condition:
            while True:
                if self.error is not None:
                    raise SafetyError(f"Capture failed: {self.error}") from self.error
                for frame in self.frames:
                    if frame.at > after:
                        return frame
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SafetyError("No fresh capture within two seconds")
                self.condition.wait(remaining)

    def _input_check(self) -> None:
        if not self.live:
            raise SafetyError("Input is disabled; use --live explicitly")
        self.check()
        if self.cursor is not None and tuple(self.mouse.position()) != self.cursor:
            raise SafetyError("External mouse movement detected")

    def move(self, point: tuple[int, int]) -> None:
        self._input_check()
        x, y = point
        width, height = self.vision.settings["screen"]
        if not 1 <= x < width - 1 or not 24 <= y < height - 48:
            raise SafetyError("Mouse target is outside the game")
        self.mouse.moveTo(x, y, duration=0.07)
        self.cursor = tuple(self.mouse.position())

    def guarded_click(
        self, point: tuple[int, int], expected: str, *, idle: bool = False,
        landmark: tuple[tuple[int, int, int, int], np.ndarray] | None = None,
    ) -> bool:
        self.move(point)
        previous = None
        for _ in range(2):
            frame = self.fresh()
            if self.vision.hover(frame.image) != expected:
                return False
            if idle and not self.vision.hud_clear(frame.image):
                return False
            mask = self.vision.hover_mask(frame.image)
            if previous is not None and not np.array_equal(mask, previous):
                return False
            previous = mask
        self._input_check()
        final = self.snapshot()
        if not np.array_equal(self.vision.hover_mask(final), previous):
            return False
        if idle and not self.vision.hud_clear(final):
            return False
        if landmark is not None and not similar(crop(final, landmark[0]), landmark[1], 6):
            return False
        self._input_check()
        self.mouse.click()
        return True

    def key(self, key: str) -> None:
        self._input_check()
        self.mouse.press(key)

    def reopen_inventory_tab(self) -> Frame | None:
        frame = self.fresh()
        if self.vision.inventory_panel_visible(frame.image):
            return None
        if not self.vision.inventory_tab_visible(frame.image):
            raise SafetyError("Inventory hidden and backpack tab could not be verified")

        self.move(INVENTORY_TAB_POINT)
        for _ in range(2):
            frame = self.fresh()
            if self.vision.inventory_panel_visible(frame.image):
                return frame
            if not self.vision.inventory_tab_visible(frame.image):
                raise SafetyError("Backpack tab changed before reopen click")

        final = self.snapshot()
        if self.vision.inventory_panel_visible(final):
            return self.fresh()
        if not self.vision.inventory_tab_visible(final):
            raise SafetyError("Backpack tab failed its final visual guard")

        self._input_check()
        self.mouse.click()

        deadline = time.monotonic() + 3
        previous = None
        while time.monotonic() < deadline:
            frame = self.fresh()
            if frame.at > deadline:
                raise SafetyError("Inventory reopen verification timed out")
            if not self.vision.inventory_panel_visible(frame.image):
                previous = None
                continue
            current = self.vision.inventory(frame.image)
            if current == previous:
                return frame
            previous = current
        raise SafetyError("Inventory panel did not reopen with two matching readings")

    def pick_menu(self, point: tuple[int, int], expected: str, items: tuple[str, ...]) -> str | None:
        self.last_pick_menu_failure = None
        self.move(point)
        mask = None
        for _ in range(2):
            frame = self.fresh()
            if self.vision.hover(frame.image) != expected:
                self.last_pick_menu_failure = "hover_text_changed"
                return None
            current = self.vision.hover_mask(frame.image)
            if mask is not None and not np.array_equal(mask, current):
                self.last_pick_menu_failure = "hover_text_unstable"
                return None
            mask = current
        self._input_check()
        if not np.array_equal(self.vision.hover_mask(self.snapshot()), mask):
            self.last_pick_menu_failure = "final_hover_guard_failed"
            return None
        self._input_check()
        self.mouse.click(button="right")
        frame = self.fresh()
        x, y = point
        left, top = max(520, x - 220), max(100, y - 220)
        right, bottom = min(1640, x + 220), min(860, y + 240)
        rect = (left, top, right - left, bottom - top)
        region = text_mask(crop(frame.image, rect)).astype(np.uint8) * 255
        for item in items:
            template = text_mask(load_image(self.vision.profile / f"menu_{item}.png"))
            scores = cv2.matchTemplate(
                region, template.astype(np.uint8) * 255, cv2.TM_SQDIFF_NORMED,
            )
            points = np.argwhere(scores <= 0.00001)
            if len(points) != 1:
                continue
            row, col = points[0]
            row_rect = (int(left + col), int(top + row), template.shape[1], template.shape[0])
            destination = (row_rect[0] + template.shape[1] // 2, row_rect[1] + 5)
            self.move(destination)
            for _ in range(2):
                if not np.array_equal(text_mask(crop(self.fresh().image, row_rect)), template):
                    self.last_pick_menu_failure = "menu_row_changed"
                    return None
            self._input_check()
            final = self.snapshot()
            if not np.array_equal(text_mask(crop(final, row_rect)), template):
                self.last_pick_menu_failure = "menu_final_guard_failed"
                return None
            self._input_check()
            self.mouse.click()
            return item
        self.last_pick_menu_failure = "menu_item_not_found_or_ambiguous"
        return None


class Audit:
    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.index = 0

    def event(self, state: str, frame: Frame | None = None, **data) -> None:
        self.index += 1
        record = {"event": self.index, "at": time.time(), "state": state, **data}
        if frame is not None:
            name = f"{self.index:05d}_{state}.png"
            save_image(self.directory / name, frame.image)
            record["screenshot"] = name
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=True) + "\n")
        print(json.dumps(record, ensure_ascii=True), flush=True)


def stable_inventory(desktop: Desktop):
    a = desktop.vision.inventory(desktop.fresh().image)
    b = desktop.vision.inventory(desktop.fresh().image)
    if a != b:
        raise SafetyError("Inventory did not produce two identical fresh readings")
    return b


def locate(image: np.ndarray, template: np.ndarray, rect: list[int]) -> tuple[int, int]:
    region = crop(image, rect)
    if template.shape[0] > region.shape[0] or template.shape[1] > region.shape[1]:
        raise SafetyError("Route anchor is larger than its search area")
    scores = cv2.matchTemplate(region, template, cv2.TM_SQDIFF_NORMED)
    _, _, best, _ = cv2.minMaxLoc(scores)
    x, y = best
    if not similar(region[y:y + template.shape[0], x:x + template.shape[1]], template, 6):
        raise SafetyError("Route origin anchor not found")
    alternatives = scores.copy()
    alternatives[max(0, y - 6):y + 7, max(0, x - 6):x + 7] = 1
    if float(alternatives.min()) < 0.015:
        raise SafetyError("Route anchor is ambiguous")
    return x + rect[0], y + rect[1]


class Route:
    def __init__(self, path: Path, vision: Vision) -> None:
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8"))
        if self.data.get("screen") != vision.settings["screen"]:
            raise SafetyError("Route resolution does not match the profile")
        for direction in ("to_bank", "to_cows"):
            steps = self.data.get(direction)
            if not isinstance(steps, list) or not steps:
                raise SafetyError(f"Route requires nonempty {direction} steps")
            for step in steps:
                if not 1 <= step["timeout"] <= 45 or not step["hover"]:
                    raise SafetyError("Invalid route timeout or action")
                for key in ("origin", "arrival"):
                    load_image(path.parent / step[key])
        self.vision = vision

    def run(self, direction: str, desktop: Desktop, audit: Audit) -> None:
        for index, step in enumerate(self.data[direction]):
            origin = load_image(self.path.parent / step["origin"])
            arrival = load_image(self.path.parent / step["arrival"])
            frame = desktop.fresh()
            if similar(crop(frame.image, step["arrival_roi"]), arrival, 6):
                second = desktop.fresh()
                if similar(crop(second.image, step["arrival_roi"]), arrival, 6):
                    audit.event("ROUTE_ALREADY_ARRIVED", second, direction=direction, step=index)
                    continue
            anchor = locate(frame.image, origin, step["search"])
            point = (anchor[0] + step["offset"][0], anchor[1] + step["offset"][1])
            # Reacquire after hover settles; never trust the earlier anchor alone.
            desktop.move(point)
            check = desktop.fresh()
            if locate(check.image, origin, step["search"]) != anchor:
                raise SafetyError("Route anchor moved during verification")
            origin_rect = (anchor[0], anchor[1], origin.shape[1], origin.shape[0])
            if not desktop.guarded_click(point, step["hover"], landmark=(origin_rect, origin)):
                raise SafetyError(f"Route action rejected: {step['hover']}")
            audit.event("ROUTE_CLICK", check, direction=direction, step=index, action=step["hover"])
            deadline = time.monotonic() + step["timeout"]
            confirmations = 0
            while time.monotonic() < deadline:
                frame = desktop.fresh()
                if frame.at > deadline:
                    raise SafetyError(f"Route step {direction}:{index} timed out")
                matches = similar(crop(frame.image, step["arrival_roi"]), arrival, 6)
                confirmations = confirmations + 1 if matches else 0
                if confirmations >= 3:
                    audit.event("ROUTE_ARRIVED", frame, direction=direction, step=index)
                    break
            else:
                raise SafetyError(f"Route step {direction}:{index} timed out")


class Bot:
    def __init__(self, desktop: Desktop, mode: Mode, route: Route | None, audit: Audit) -> None:
        self.desktop = desktop
        self.vision = desktop.vision
        self.mode = mode
        self.route = route
        self.audit = audit
        self.failures = 0
        self.kills = 0
        self.pickups = 0
        self.center = tuple(self.vision.settings.get("player_anchor", [942, 520]))
        self.last_kill_point: tuple[int, int] | None = None

    def read_inventory(self):
        try:
            return stable_inventory(self.desktop)
        except InventoryPanelUnavailable:
            frame = self.desktop.reopen_inventory_tab()
            if frame is None:
                raise SafetyError("Inventory panel state changed while attempting recovery")
            inventory = stable_inventory(self.desktop)
            self.audit.event(
                "INVENTORY_REOPENED",
                frame,
                cowhide=inventory.count("Cowhide"),
                bones=inventory.count("Bones"),
                empty=inventory.count("EMPTY"),
            )
            return inventory

    def bank(self) -> None:
        if self.route is None:
            raise SafetyError("Inventory full; no calibrated banking route is loaded")
        self.audit.event("WALK_TO_BANK", self.desktop.fresh())
        self.route.run("to_bank", self.desktop, self.audit)
        frame = self.desktop.fresh()
        if not self.vision.bank_open(frame.image):
            raise SafetyError("Route ended without a verified bank interface")
        inventory = stable_inventory(self.desktop)
        inventory.require_allowed(self.mode)
        # Deposit is only permitted for approved contents, never equipment or supplies.
        if inventory.empty:
            raise SafetyError("Unexpected empty inventory before deposit")
        button = load_image(self.vision.profile / "deposit_inventory.png")
        anchor = locate(frame.image, button, [588, 46, 484, 800])
        point = (anchor[0] + button.shape[1] // 2, anchor[1] + button.shape[0] // 2)
        if not self.desktop.guarded_click(
            point, "Deposit inventory",
            landmark=((760, 51, 158, 22), self.vision.bank_header),
        ):
            raise SafetyError("Deposit inventory final guard rejected")
        self.audit.event("DEPOSIT_CLICK", frame)
        deadline = time.monotonic() + 4
        confirmations = 0
        while time.monotonic() < deadline:
            frame = self.desktop.fresh()
            if frame.at > deadline:
                raise SafetyError("Deposit verification timed out")
            if not self.vision.bank_open(frame.image):
                raise SafetyError("Bank closed before deposit verification")
            empty = self.vision.inventory(frame.image).empty
            confirmations = confirmations + 1 if empty else 0
            if confirmations >= 2:
                break
        else:
            raise SafetyError("Deposit did not produce 28 empty slots")
        self.audit.event("DEPOSIT_VERIFIED", frame)
        self.desktop.key("esc")
        deadline = time.monotonic() + 3
        while self.vision.bank_open(self.desktop.fresh().image):
            if time.monotonic() >= deadline:
                raise SafetyError("Bank interface failed to close")
        self.route.run("to_cows", self.desktop, self.audit)
        if not stable_inventory(self.desktop).empty:
            raise SafetyError("Inventory changed during return journey")
        self.audit.event("RETURNED_TO_COWS", self.desktop.fresh())

    def attack(self) -> bool:
        clear_deadline = time.monotonic() + 15
        clear = 0
        while time.monotonic() < clear_deadline:
            frame = self.desktop.fresh()
            clear = clear + 1 if self.vision.hud_clear(frame.image) else 0
            if clear >= 2:
                break
        else:
            raise SafetyError("Combat HUD did not clear before next attack")
        search_deadline = time.monotonic() + 30
        for point in self.vision.grid(self.center):
            if time.monotonic() >= search_deadline:
                return False
            self.desktop.move(point)
            if time.monotonic() >= search_deadline:
                return False
            frame = self.desktop.fresh()
            if frame.at > search_deadline:
                return False
            action = self.vision.hover(frame.image)
            targets = {"Attack Cow (level-2)": "Cow", "Attack Cow calf (level-2)": "Cow calf"}
            if action not in targets:
                continue
            if not self.desktop.guarded_click(point, action, idle=True):
                continue
            lock = HealthLock(time.monotonic(), targets[action])
            self.audit.event("ATTACK_CLICK", frame, action=action, point=point)
            while True:
                frame = self.desktop.fresh()
                outcome = lock.observe(frame.number, frame.at, self.vision.health(frame.image))
                if outcome == "KILLED":
                    self.kills += 1
                    self.last_kill_point = point
                    self.audit.event("KILL_VERIFIED", frame, kills=self.kills)
                    return True
                if outcome == "MISSED":
                    self.audit.event("ATTACK_MISSED", frame)
                    return False
        return False

    def loot(self) -> None:
        if not self.mode.items:
            return
        baseline = self.read_inventory()
        baseline.require_allowed(self.mode)
        center = self.last_kill_point or self.center
        radius = self.vision.settings["loot_radius"]
        fine_radius = self.vision.settings.get("loot_fine_radius", 48)
        started = time.monotonic()
        deadline = started + 25
        scanned = {"fine": 0, "coarse": 0}
        hover_matches = 0
        ignored_item_matches = 0
        farthest_distance = 0.0
        last_frame = None
        target_actions = {f"Take {item}" for item in self.mode.items}

        def finish(reason: str) -> None:
            self.audit.event(
                "LOOT_SEARCH_FINISHED",
                last_frame,
                center=list(center),
                scanned_points=sum(scanned.values()),
                fine_points=scanned["fine"],
                coarse_points=scanned["coarse"],
                hover_matches=hover_matches,
                ignored_item_matches=ignored_item_matches,
                farthest_distance=round(farthest_distance, 1),
                elapsed_seconds=round(time.monotonic() - started, 2),
                stop_reason=reason,
            )

        for point, phase in self.vision.loot_grid(center, radius, fine_radius):
            if time.monotonic() >= deadline:
                finish("time_budget")
                return
            if baseline.full:
                finish("inventory_full")
                return
            self.desktop.move(point)
            if time.monotonic() >= deadline:
                finish("time_budget")
                return
            frame = self.desktop.fresh()
            if frame.at > deadline:
                finish("time_budget")
                return
            last_frame = frame
            scanned[phase] += 1
            distance = math.hypot(point[0] - center[0], point[1] - center[1])
            farthest_distance = max(farthest_distance, distance)
            action = self.vision.hover(frame.image)
            if action in {"Take Cowhide", "Take Bones", "Take Raw beef"} and action not in target_actions:
                ignored_item_matches += 1
                continue
            if action not in target_actions:
                continue
            hover_matches += 1
            baseline = self.read_inventory()
            baseline.require_allowed(self.mode)
            item = self.desktop.pick_menu(point, action, self.mode.items)
            if item is None:
                reason = getattr(self.desktop, "last_pick_menu_failure", None)
                if not isinstance(reason, str):
                    reason = "unspecified"
                self.audit.event(
                    "PICKUP_MENU_REJECTED",
                    self.desktop.fresh(),
                    action=action,
                    reason=reason,
                )
                continue
            check = PickupCheck(baseline, item, time.monotonic())
            self.audit.event("PICKUP_CLICK", self.desktop.fresh(), item=item)
            while True:
                frame = self.desktop.fresh()
                if check.observe(frame.number, frame.at, self.vision.inventory(frame.image)):
                    self.pickups += 1
                    baseline = self.vision.inventory(frame.image)
                    self.audit.event("PICKUP_VERIFIED", frame, item=item, pickups=self.pickups)
                    if self.mode == Mode.COWHIDE_ONLY and item == "Cowhide":
                        finish("target_collected")
                        return
                    break
        finish("region_exhausted")

    def run(self, max_kills: int) -> None:
        for item in self.mode.items:
            if item not in self.vision.items:
                raise SafetyError(f"Calibrate the {item} inventory sprite first")
        initial = self.read_inventory()
        if self.mode != Mode.COMBAT_ONLY:
            initial.require_allowed(self.mode)
        self.audit.event("READY", self.desktop.fresh(), mode=self.mode.value)
        while self.kills < max_kills:
            if self.mode.items:
                inventory = self.read_inventory()
                inventory.require_allowed(self.mode)
                if inventory.full:
                    self.bank()
                    continue
            if not self.attack():
                self.failures += 1
                self.audit.event("SEARCH_OR_ATTACK_FAILED", self.desktop.fresh(), failures=self.failures)
                if self.failures >= 5:
                    raise SafetyError("Five consecutive search/attack failures")
                continue
            self.failures = 0
            self.loot()
        self.audit.event("KILL_LIMIT_REACHED", self.desktop.fresh(), kills=self.kills)
