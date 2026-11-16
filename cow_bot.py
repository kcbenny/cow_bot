import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from cow_core import HealthLock, Mode, SafetyError
from cow_vision import (
    CENTERS, PROFILE, ROOT, Vision, bootstrap, crop, learn_line,
    load_image, save_image, similar,
)


def select_rect(image: np.ndarray, title: str) -> list[int]:
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1280, 720)
    try:
        rect = list(map(int, cv2.selectROI(title, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))))
    finally:
        cv2.destroyAllWindows()
    if rect[2] < 24 or rect[3] < 16:
        raise SafetyError("Selection cancelled or too small (minimum 24 x 16)")
    return rect


def select_point(image: np.ndarray) -> tuple[int, int]:
    name = "Click the intended game destination; ESC cancels"
    chosen = []
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1280, 720)

    def clicked(event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            chosen.append((x, y))

    cv2.setMouseCallback(name, clicked)
    cv2.imshow(name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    try:
        while not chosen:
            if cv2.waitKey(20) & 0xFF == 27 or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                raise SafetyError("Point selection cancelled")
        return chosen[0]
    finally:
        cv2.destroyAllWindows()


def countdown(message: str, seconds: int = 5) -> None:
    if seconds <= 0:
        raise ValueError("Countdown seconds must be positive")
    print(f"ALERT: {message}", flush=True)
    for remaining in range(seconds, 0, -1):
        print(f"Starting in {remaining}...", flush=True)
        time.sleep(1)


def countdown_snapshot(desktop):
    countdown("Switch to maximized OSRS and hold the intended hover text for capture.")
    return desktop.snapshot()


def record_route(path: Path) -> None:
    from cow_desktop import Desktop, Route

    if path.exists():
        raise SafetyError("Route already exists; choose a new output filename")
    vision = Vision()
    desktop = Desktop(vision)
    data = {"screen": vision.settings["screen"], "to_bank": [], "to_cows": []}
    path.parent.mkdir(parents=True, exist_ok=True)
    asset_dir = path.parent / f"{path.stem}_assets"
    asset_dir.mkdir(exist_ok=False)
    alphabet = json.loads((PROFILE / "font.json").read_text(encoding="utf-8"))
    allowed = {
        "Walk here", "Open Gate", "Climb-up Staircase", "Climb-down Staircase", "Bank Bank booth",
    }
    for direction in ("to_bank", "to_cows"):
        print(f"\nRecord {direction}. Manually perform each step; the recorder sends NO inputs.")
        if direction == "to_cows":
            print("Manually deposit everything and CLOSE the bank before recording the return.")
        count = int(input("Number of steps (include gates and each staircase separately): "))
        if not 1 <= count <= 40:
            raise SafetyError("Step count must be between 1 and 40")
        for index in range(count):
            input(f"Step {index + 1}: prepare the starting location. Press Enter to capture.")
            before = countdown_snapshot(desktop)
            print("Select a distinctive STATIC landmark near the destination (not an NPC or cursor).")
            origin_roi = select_rect(before, "Select origin landmark; ENTER confirms")
            origin = crop(before, origin_roi)
            if float(origin.std()) < 12:
                raise SafetyError("Origin landmark has insufficient visual detail")
            point = select_point(before)
            world = vision.settings["search"]
            if not world[0] <= point[0] < world[0] + world[2] or not world[1] <= point[1] < world[1] + world[3]:
                raise SafetyError("Route click must lie inside the configured game-world search ROI")
            print("Type the COMPLETE top-left hover line from the captured screenshot.")
            print("For example: Open Gate / 2 more options; Walk here may have no suffix.")
            label = input("Hover line: ").strip()
            action = label.split(" / ")[0]
            if action not in allowed:
                raise SafetyError(f"Unsupported navigation action: {action}")
            learn_line(alphabet, before, label)
            input("Manually perform this ONE game action and wait until fully arrived. Press Enter.")
            after = countdown_snapshot(desktop)
            print("Select a location-specific ARRIVAL signature, e.g. minimap scenery near its centre.")
            print("It MUST differ from this same rectangle before moving. Avoid moving dots/NPCs.")
            arrival_roi = select_rect(after, "Select unique arrival signature; ENTER confirms")
            arrival = crop(after, arrival_roi)
            if float(arrival.std()) < 12 or similar(crop(before, arrival_roi), arrival, 12):
                raise SafetyError("Arrival signature is blank or already present before moving")
            timeout = float(input("Arrival timeout in seconds (stairs: 5; walking: up to 45): ") or "15")
            if not 1 <= timeout <= 45:
                raise SafetyError("Timeout must be between 1 and 45 seconds")
            stem = f"{direction}_{index:02d}"
            origin_path, arrival_path = asset_dir / f"{stem}_origin.png", asset_dir / f"{stem}_arrival.png"
            save_image(origin_path, origin)
            save_image(arrival_path, arrival)
            data[direction].append({
                "hover": action,
                "origin": str(origin_path.relative_to(path.parent)),
                "arrival": str(arrival_path.relative_to(path.parent)),
                "arrival_roi": arrival_roi,
                "offset": [point[0] - origin_roi[0], point[1] - origin_roi[1]],
                "search": world,
                "timeout": timeout,
            })
    if data["to_bank"][-1]["hover"] != "Bank Bank booth":
        raise SafetyError("Last to_bank step must open the bank via Bank Bank booth")
    (PROFILE / "font.json").write_text(json.dumps(alphabet), encoding="utf-8")
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    Route(path, Vision())
    print(f"Saved calibrated route: {path}")


def calibrate_item(path: Path, slot: int) -> None:
    if not 1 <= slot <= 28:
        raise SafetyError("Inventory slot must be in 1..28, ordered left-to-right, top-to-bottom")
    vision = Vision()
    image = load_image(path)
    vision.check_geometry(image)
    inventory = vision.inventory(image)
    if inventory.slots[slot - 1] != "UNKNOWN":
        raise SafetyError("Select an occupied, unrecognised slot containing ONLY Bones")
    x, y = CENTERS[slot - 1]
    save_image(PROFILE / "Bones.png", crop(image, (x - 16, y - 16, 32, 32)))
    print("Saved Bones sprite. Verify its classification with inspect before using bones modes.")


def inspect_image(
    vision: Vision, image: np.ndarray, *, passive: bool = False, compressed: bool = False,
) -> dict:
    vision.check_geometry(image)
    inventory = None
    inventory_error = None
    try:
        inventory = vision.inventory(image)
    except SafetyError as error:
        if not passive:
            raise
        inventory_error = str(error)
    hp = vision.health(image, compressed=compressed)
    return {
        "hover": vision.hover(image),
        "health": None if hp is None else {"target": hp.target, "positive": hp.positive},
        "hud_clear": vision.hud_clear(image),
        "bank_open": vision.bank_open(image),
        "inventory": None if inventory is None else {
            item: inventory.count(item) for item in sorted(set(inventory.slots))
        },
        "inventory_error": inventory_error,
    }


def replay(vision: Vision, path: Path, *, client_video: bool = False) -> None:
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise SafetyError(f"Cannot open video: {path}")
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise SafetyError("Video has no valid frame rate")
    stride = max(1, round(fps / 10))
    number, kills = 0, 0
    lock = None
    previous = None
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            number += 1
            if number % stride:
                continue
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if client_video:
                if image.shape[:2] != (1012, 1920):
                    raise SafetyError("--client-video is only calibrated for the supplied 1920x1012 clips")
                # The supplied client recordings omit the 23-pixel Windows title bar.
                image = np.pad(image, ((23, 45), (0, 0), (0, 0)))
            status = inspect_image(vision, image, passive=True, compressed=True)
            hp = vision.health(image, compressed=True)
            at = number / fps
            if lock is None and hp is not None and hp.positive:
                lock = HealthLock(at - 0.01, hp.target)
            if lock is not None:
                event = lock.observe(number, at, hp)
                if event == "KILLED":
                    kills += 1
                    lock = None
                    print(json.dumps({"seconds": at, "event": "OBSERVED_KILL", "kills": kills}))
                elif event == "MISSED":
                    lock = None
            if status != previous:
                print(json.dumps({"seconds": round(at, 2), **status}))
                previous = status
    finally:
        video.release()
    if not number:
        raise SafetyError("Video contains no readable frames")
    print(json.dumps({"frames_read": number, "observed_kills": kills, "inputs_sent": 0}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Official-client, visual-only OSRS cow bot")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Build native templates from the supplied images and video")
    inspect = sub.add_parser("inspect", help="Inspect a screenshot without controlling the game")
    inspect.add_argument("image", type=Path)
    video = sub.add_parser("replay", help="Passively replay a video; no game or input required")
    video.add_argument("video", type=Path)
    video.add_argument("--client-video", action="store_true", help="Supplied 1920x1012 client-area clips only")
    bones = sub.add_parser("calibrate-bones")
    bones.add_argument("image", type=Path)
    bones.add_argument("--slot", type=int, required=True)
    route = sub.add_parser("record-route", help="Manually calibrate both directions; no game inputs")
    route.add_argument("--output", type=Path, default=PROFILE / "route.json")
    observe = sub.add_parser("observe", help="Read game frames, never move or click")
    observe.add_argument("--seconds", type=int, default=30)
    run = sub.add_parser("run", help="Requires --live; looting needs a route unless --no-bank")
    run.add_argument("--live", action="store_true")
    run.add_argument("--mode", choices=[m.value for m in Mode])
    run.add_argument("--route", type=Path, default=PROFILE / "route.json")
    run.add_argument("--no-bank", action="store_true", help="Do not load a bank route; stop if inventory fills")
    run.add_argument("--max-kills", type=int, default=10)
    args = parser.parse_args()
    if args.command == "init":
        bootstrap()
        print(f"Native templates created: {PROFILE}")
        return
    vision = Vision()
    if args.command == "inspect":
        print(json.dumps(inspect_image(vision, load_image(args.image)), indent=2))
    elif args.command == "replay":
        replay(vision, args.video, client_video=args.client_video)
    elif args.command == "calibrate-bones":
        calibrate_item(args.image, args.slot)
    elif args.command == "record-route":
        record_route(args.output.resolve())
    elif args.command in ("run", "observe"):
        from cow_desktop import Audit, Bot, Desktop, Route
        if args.command == "run":
            if not args.live:
                raise SafetyError("No inputs sent. Add --live only after offline and passive checks.")
            if args.max_kills <= 0:
                raise SafetyError("--max-kills must be positive")
            if args.mode is None:
                modes = list(Mode)
                for index, mode in enumerate(modes, 1):
                    print(f"{index}. {mode.value}")
                choice = int(input("Loot mode [1]: ") or "1")
                if not 1 <= choice <= len(modes):
                    raise SafetyError("Mode must be 1..4")
                mode = modes[choice - 1]
            else:
                mode = Mode(args.mode)
            route = Route(args.route.resolve(), vision) if mode.items and not args.no_bank else None
            if args.no_bank:
                print("Banking disabled. No route will be loaded; a full inventory causes SAFE_STOP.")
        elif args.seconds <= 0:
            raise SafetyError("--seconds must be positive")
        directory = ROOT / "cow_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        audit = Audit(directory)
        desktop = Desktop(vision, live=args.command == "run")
        if args.command == "run":
            countdown(
                "LIVE BOT will control the game. Switch to maximized OSRS now; "
                "do not touch the mouse after countdown. F8 / Ctrl+C stops."
            )
        else:
            countdown(
                "PASSIVE OBSERVATION only; no inputs will be sent. "
                "Switch to the calibrated maximized OSRS window."
            )
        print("F8 / Ctrl+C stops. Live mode also stops on external mouse movement or focus loss.")
        try:
            with desktop:
                if args.command == "run":
                    Bot(desktop, mode, route, audit).run(args.max_kills)
                else:
                    deadline = time.monotonic() + args.seconds
                    previous = None
                    while time.monotonic() < deadline:
                        frame = desktop.fresh()
                        status = inspect_image(vision, frame.image, passive=True)
                        if status != previous:
                            audit.event("OBSERVATION", None, **status)
                            previous = status
        except (SafetyError, KeyboardInterrupt, OSError, desktop.mouse.FailSafeException) as error:
            frame = desktop.frames[-1] if desktop.frames else None
            audit.event("SAFE_STOP", frame, reason=str(error) or "Ctrl+C")
            raise


if __name__ == "__main__":
    try:
        main()
    except (SafetyError, OSError, ValueError, KeyError) as error:
        raise SystemExit(f"ERROR: {error}") from error
