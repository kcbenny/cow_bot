import ctypes
from ctypes import wintypes
import json
import math
import random
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

PLAYER_STOP_TIMEOUT = 30
PLAYER_MOTION_THRESHOLD = 80
PLAYER_STABLE_FRAMES = 7
BUSY_TARGET_BLACKLIST_SECONDS = 20
BUSY_TARGET_BLACKLIST_RADIUS = 36
CAPTURE_INTERVAL_SECONDS = 0.05
MENU_RENDER_TIMEOUT_SECONDS = 1.0


def menu_template_location(
    image: np.ndarray, template: np.ndarray, rect: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    template_mask = text_mask(template).astype(np.uint8) * 255
    region = text_mask(crop(image, rect)).astype(np.uint8) * 255
    if template_mask.size == 0 or template_mask.shape[0] > region.shape[0] or template_mask.shape[1] > region.shape[1]:
        return None
    scores = cv2.matchTemplate(region, template_mask, cv2.TM_SQDIFF_NORMED)
    points = np.argwhere(scores <= 0.00001)
    if len(points) != 1:
        return None
    row, col = points[0]
    return (
        rect[0] + int(col), rect[1] + int(row),
        template_mask.shape[1], template_mask.shape[0],
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
    started_at: float | None = None


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
                captured_at = time.monotonic()
                image = self.snapshot()
                self.sequence += 1
                frame = Frame(self.sequence, time.monotonic(), image, captured_at)
                with self.condition:
                    self.frames.append(frame)
                    self.condition.notify_all()
                self.stop.wait(CAPTURE_INTERVAL_SECONDS)
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
                    captured_at = frame.started_at if frame.started_at is not None else frame.at
                    if captured_at > after:
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

    def _validate_mouse_point(self, point: tuple[int, int]) -> None:
        x, y = point
        width, height = self.vision.settings["screen"]
        if not 1 <= x < width - 1 or not 24 <= y < height - 48:
            raise SafetyError("Mouse target is outside the game")

    def move(self, point: tuple[int, int], *, duration: float = 0.07) -> None:
        self._input_check()
        self._validate_mouse_point(point)
        x, y = point
        self.mouse.moveTo(x, y, duration=duration)
        self.cursor = tuple(self.mouse.position())

    def move_loot(self, point: tuple[int, int]) -> None:
        self._validate_mouse_point(point)
        start = self.cursor or tuple(self.mouse.position())
        width, height = self.vision.settings["screen"]
        clamp = lambda x, y: (
            min(max(round(x), 1), width - 2),
            min(max(round(y), 24), height - 49),
        )
        dx, dy = point[0] - start[0], point[1] - start[1]
        distance = math.hypot(dx, dy)
        if distance == 0:
            self.move(point, duration=random.uniform(0.10, 0.16))
            return

        overshoot = random.uniform(1.0, 3.0)
        overshoot_point = clamp(
            point[0] + dx / distance * overshoot,
            point[1] + dy / distance * overshoot,
        )
        lateral = random.uniform(-2.0, 2.0)
        control_point = clamp(
            start[0] + (overshoot_point[0] - start[0]) * 0.58 - dy / distance * lateral,
            start[1] + (overshoot_point[1] - start[1]) * 0.58 + dx / distance * lateral,
        )
        for destination, duration in (
            (control_point, random.uniform(0.10, 0.16)),
            (overshoot_point, random.uniform(0.10, 0.18)),
            (point, random.uniform(0.08, 0.14)),
        ):
            self.move(destination, duration=duration)

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

    def route_menu_click(
        self, point: tuple[int, int], template: np.ndarray,
        landmark: tuple[tuple[int, int, int, int], np.ndarray],
    ) -> bool:
        self.move(point)
        frame = self.fresh()
        if not similar(crop(frame.image, landmark[0]), landmark[1], 6):
            return False
        self._input_check()
        self.mouse.click(button="right")

        width, height = self.vision.settings["screen"]
        x, y = point
        left, top = max(0, x - 280), max(24, y - 280)
        right, bottom = min(width, x + 280), min(height - 48, y + 280)
        rect = (left, top, right - left, bottom - top)
        deadline = time.monotonic() + MENU_RENDER_TIMEOUT_SECONDS
        row_rect = None
        while time.monotonic() < deadline:
            frame = self.fresh()
            row_rect = menu_template_location(frame.image, template, rect)
            if row_rect is not None:
                break
        if row_rect is None:
            return False

        destination = (row_rect[0] + row_rect[2] // 2, row_rect[1] + row_rect[3] + 3)
        self.move(destination)
        expected = text_mask(template)
        for _ in range(2):
            if not np.array_equal(text_mask(crop(self.fresh().image, row_rect)), expected):
                return False
        self._input_check()
        if not np.array_equal(text_mask(crop(self.snapshot(), row_rect)), expected):
            return False
        self._input_check()
        self.mouse.click()
        return True

    def reopen_inventory_tab(self) -> Frame | None:
        frame = self.fresh()
        if self.vision.inventory_panel_visible(frame.image):
            return None
        offset = self.vision.inventory_tab_offset(frame.image)
        if offset is None:
            raise SafetyError("Inventory hidden and backpack tab could not be verified")

        self.move((INVENTORY_TAB_POINT[0], INVENTORY_TAB_POINT[1] + offset))
        for _ in range(2):
            frame = self.fresh()
            if self.vision.inventory_panel_visible(frame.image):
                return frame
            if self.vision.inventory_tab_offset(frame.image) != offset:
                raise SafetyError("Backpack tab changed before reopen click")

        final = self.snapshot()
        if self.vision.inventory_panel_visible(final):
            return self.fresh()
        if self.vision.inventory_tab_offset(final) != offset:
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
        x, y = point
        left, top = max(520, x - 220), max(100, y - 220)
        right, bottom = min(1640, x + 220), min(860, y + 240)
        rect = (left, top, right - left, bottom - top)
        menu_deadline = time.monotonic() + MENU_RENDER_TIMEOUT_SECONDS
        while time.monotonic() < menu_deadline:
            frame = self.fresh()
            if frame.at > menu_deadline:
                break
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
                if expected == "Take Cowhide":
                    delay = min(0.45, max(0.20, random.gauss(0.325, 0.05)))
                    time.sleep(delay)
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
                if step["hover"] in ("Climb-up Staircase", "Climb-down Staircase"):
                    if not step.get("menu_action"):
                        raise SafetyError("Stair route step is missing its verified menu-action template")
                    load_image(path.parent / step["menu_action"])
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
            if step.get("menu_action"):
                template = load_image(self.path.parent / step["menu_action"])
                clicked = desktop.route_menu_click(point, template, (origin_rect, origin))
            else:
                clicked = desktop.guarded_click(
                    point, step["hover"], landmark=(origin_rect, origin),
                )
            if not clicked:
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
        self.blacklisted_targets: list[tuple[tuple[int, int], float]] = []

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
            now = time.monotonic()
            self.blacklisted_targets = [
                (target, expires_at)
                for target, expires_at in self.blacklisted_targets
                if expires_at > now
            ]
            if any(
                (point[0] - target[0]) ** 2 + (point[1] - target[1]) ** 2
                <= BUSY_TARGET_BLACKLIST_RADIUS ** 2
                for target, _ in self.blacklisted_targets
            ):
                continue
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
            if self.vision.target_in_combat(frame.image, point) is True:
                self.blacklisted_targets.append((
                    point, time.monotonic() + BUSY_TARGET_BLACKLIST_SECONDS,
                ))
                self.audit.event(
                    "TARGET_BLACKLISTED",
                    frame,
                    point=list(point),
                    duration_seconds=BUSY_TARGET_BLACKLIST_SECONDS,
                    reason="target_in_combat",
                )
                continue
            attack_frame_image = frame.image
            busy_messages_before = self.vision.busy_message_count(attack_frame_image)
            if not self.desktop.guarded_click(point, action, idle=True):
                continue
            self.audit.event("ATTACK_CLICK", frame, action=action, point=point)
            target = targets[action]
            stop_deadline = frame.at + PLAYER_STOP_TIMEOUT
            previous_image = frame.image
            stable_frames = 0
            saw_positive = False
            lock = None
            while True:
                frame = self.desktop.fresh()
                if (
                    self.vision.busy_message_count(frame.image) > busy_messages_before
                    or self.vision.busy_message_added(
                        attack_frame_image, frame.image, busy_messages_before,
                    ) is True
                ):
                    self.blacklisted_targets.append((
                        point, time.monotonic() + BUSY_TARGET_BLACKLIST_SECONDS,
                    ))
                    self.audit.event(
                        "TARGET_BLACKLISTED",
                        frame,
                        point=list(point),
                        duration_seconds=BUSY_TARGET_BLACKLIST_SECONDS,
                        message="Someone else is fighting that",
                    )
                    return False
                hp = self.vision.health(frame.image)
                if hp is not None and hp.target != target:
                    raise SafetyError("Combat HUD target changed")
                if lock is None:
                    if hp is not None and hp.positive:
                        saw_positive = True
                    if frame.at > stop_deadline:
                        self.audit.event("ATTACK_MISSED", frame, reason="player_stop_timeout")
                        return False
                    motion = self.vision.player_motion(previous_image, frame.image)
                    stable_frames = (
                        stable_frames + 1
                        if motion <= PLAYER_MOTION_THRESHOLD else 0
                    )
                    previous_image = frame.image
                    if stable_frames < PLAYER_STABLE_FRAMES:
                        continue
                    self.audit.event(
                        "PLAYER_STOPPED", frame, stable_frames=stable_frames,
                    )
                    lock = HealthLock(frame.at, target)
                    lock.seen_positive = saw_positive
                outcome = lock.observe(frame.number, frame.at, hp)
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
        color_candidate_points: list[tuple[int, int]] = []
        pending_color_candidates: deque[tuple[tuple[int, int], str]] = deque()
        target_actions = {f"Take {item}" for item in self.mode.items}
        spiral_actions = target_actions - {"Take Cowhide"}

        def finish(reason: str) -> None:
            self.audit.event(
                "LOOT_SEARCH_FINISHED",
                last_frame,
                center=list(center),
                scanned_points=sum(scanned.values()),
                fine_points=scanned["fine"],
                coarse_points=scanned["coarse"],
                color_candidates=len(color_candidate_points),
                color_candidate_points=[list(point) for point in color_candidate_points],
                hover_matches=hover_matches,
                ignored_item_matches=ignored_item_matches,
                farthest_distance=round(farthest_distance, 1),
                elapsed_seconds=round(time.monotonic() - started, 2),
                stop_reason=reason,
            )

        def enqueue_color_candidates(image: np.ndarray) -> None:
            for point, phase in self.vision.loot_candidates(
                image, center, radius, fine_radius,
            ):
                if any(
                    (point[0] - previous[0]) ** 2 + (point[1] - previous[1]) ** 2 <= 225
                    for previous in color_candidate_points
                ):
                    continue
                color_candidate_points.append(point)
                pending_color_candidates.append((point, phase))

        last_frame = self.desktop.fresh()
        if last_frame.at > deadline or time.monotonic() >= deadline:
            finish("time_budget")
            return
        enqueue_color_candidates(last_frame.image)
        occluded_probes: deque[tuple[tuple[int, int], str]] = deque()
        if "Cowhide" in self.mode.items:
            anchor_x, anchor_y = self.vision.settings.get("player_anchor", [942, 520])
            screen_w, screen_h = self.vision.settings.get("screen", [1920, 1080])
            raw_probes = [
                (int(anchor_x), int(anchor_y) - 16),
                (int(anchor_x), int(anchor_y) - 28),
                (int(anchor_x) - 14, int(anchor_y) - 20),
                (int(anchor_x) + 14, int(anchor_y) - 20),
                (int(anchor_x), int(anchor_y) - 40),
                (int(anchor_x), int(anchor_y)),
                center,
                (center[0], center[1] - 14),
            ]
            seen_probes: set[tuple[int, int]] = set()
            for probe in raw_probes:
                if (
                    probe in seen_probes
                    or not 1 <= probe[0] < screen_w - 1
                    or not 24 <= probe[1] < screen_h - 48
                ):
                    continue
                seen_probes.add(probe)
                occluded_probes.append((probe, "fine"))
        spiral_candidates = None
        if spiral_actions:
            spiral_candidates = iter(
                (point, phase)
                for point, phase in self.vision.loot_grid(center, radius, fine_radius)
            )
        while True:
            if time.monotonic() >= deadline:
                finish("time_budget")
                return
            if baseline.full:
                finish("inventory_full")
                return
            is_color_candidate = bool(pending_color_candidates)
            is_occluded_probe = False
            if is_color_candidate:
                point, phase = pending_color_candidates.popleft()
            else:
                frame = self.desktop.fresh()
                if frame.at > deadline or time.monotonic() >= deadline:
                    last_frame = frame
                    finish("time_budget")
                    return
                last_frame = frame
                enqueue_color_candidates(frame.image)
                elapsed = max(time.monotonic(), frame.at) - started
                if pending_color_candidates:
                    point, phase = pending_color_candidates.popleft()
                    is_color_candidate = True
                elif occluded_probes and elapsed >= 2.0:
                    point, phase = occluded_probes.popleft()
                    while any(
                        (point[0] - previous[0]) ** 2 + (point[1] - previous[1]) ** 2 <= 64
                        for previous in color_candidate_points
                    ):
                        if not occluded_probes:
                            point = None
                            break
                        point, phase = occluded_probes.popleft()
                    if point is None:
                        continue
                    is_occluded_probe = True
                elif spiral_candidates is not None:
                    try:
                        point, phase = next(spiral_candidates)
                    except StopIteration:
                        spiral_candidates = None
                        if "Cowhide" not in self.mode.items:
                            finish("region_exhausted")
                            return
                        continue
                elif "Cowhide" not in self.mode.items:
                    finish("region_exhausted")
                    return
                elif not occluded_probes and elapsed >= 10.0:
                    finish("region_exhausted")
                    return
                else:
                    continue
            if is_color_candidate or is_occluded_probe:
                self.desktop.move_loot(point)
            else:
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
            if not is_color_candidate and not is_occluded_probe and action not in spiral_actions:
                ignored_item_matches += 1
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
