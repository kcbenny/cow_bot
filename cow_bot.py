import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from cow_core import HealthLock, Mode, SafetyError
from cow_vision import (
    CENTERS, PROFILE, ROOT, ROUTE_ANCHOR_SEARCH, ROUTE_CLICK_AREA, Vision, bootstrap,
    crop, learn_line, load_image, save_image, similar, text_mask,
)


def select_rect(
    image: np.ndarray, title: str, minimum: tuple[int, int] = (24, 16),
) -> list[int]:
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1280, 720)
    try:
        rect = list(map(int, cv2.selectROI(title, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))))
    finally:
        cv2.destroyAllWindows()
    if rect[2] == 0 or rect[3] == 0:
        raise SafetyError("Selection cancelled")
    if rect[2] < minimum[0] or rect[3] < minimum[1]:
        raise SafetyError(f"Selection too small (minimum {minimum[0]} x {minimum[1]})")
    return rect


def preview_point_to_image(
    point: tuple[int, int], image_shape: tuple[int, ...], preview_shape: tuple[int, ...],
) -> tuple[int, int]:
    image_height, image_width = image_shape[:2]
    preview_height, preview_width = preview_shape[:2]
    x = round((point[0] + 0.5) * image_width / preview_width - 0.5)
    y = round((point[1] + 0.5) * image_height / preview_height - 0.5)
    return min(max(x, 0), image_width - 1), min(max(y, 0), image_height - 1)


def point_inside_rect(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    x, y = point
    left, top, width, height = rect
    return left <= x < left + width and top <= y < top + height


def select_point(
    image: np.ndarray, allowed: tuple[int, int, int, int],
) -> tuple[int, int]:
    name = "Click destination INSIDE green box; ESC cancels"
    preview_size = (1280, 720)
    preview = cv2.resize(image, preview_size, interpolation=cv2.INTER_AREA)
    x, y, width, height = allowed
    scale_x, scale_y = preview_size[0] / image.shape[1], preview_size[1] / image.shape[0]
    top_left = (round(x * scale_x), round(y * scale_y))
    bottom_right = (round((x + width) * scale_x), round((y + height) * scale_y))
    preview_bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    cv2.rectangle(preview_bgr, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(
        preview_bgr, "Click destination inside green box",
        (top_left[0] + 8, top_left[1] + 24), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 0), 2,
    )
    chosen = []
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, preview_size[0], preview_size[1])

    def clicked(event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            chosen.append(preview_point_to_image((x, y), image.shape, preview.shape))

    cv2.setMouseCallback(name, clicked)
    cv2.imshow(name, preview_bgr)
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


def countdown_snapshot(desktop, message: str):
    countdown(message)
    return desktop.snapshot()


def route_step_count(direction: str) -> int:
    while True:
        raw = input(
            f"輸入 {direction} 路線總步數（1–40；行路、開閘、每段樓梯及最後開銀行各算一步）："
        ).strip()
        try:
            count = int(raw)
        except ValueError:
            print("請只輸入 1 到 40 的整數，例如 6。")
            continue
        if 1 <= count <= 40:
            return count
        print("步數必須是 1 到 40 的整數，請重新輸入。")


def route_progress_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.progress.json")


def save_route_progress(progress_path: Path, progress: dict) -> None:
    temporary = progress_path.with_name(f"{progress_path.name}.tmp")
    temporary.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    temporary.replace(progress_path)


def load_route_progress(
    path: Path, asset_dir: Path, screen: list[int],
) -> tuple[dict, bool, bool]:
    if path.exists():
        raise SafetyError("Route already exists; choose a different output filename")
    progress_path = route_progress_path(path)
    resumed = progress_path.exists()
    reused_assets = asset_dir.exists()
    if resumed:
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SafetyError(f"Cannot read route progress file: {progress_path}") from error
        data, counts = progress.get("data"), progress.get("counts")
        if (
            progress.get("version") != 1
            or not isinstance(data, dict)
            or data.get("screen") != screen
            or not isinstance(data.get("to_bank"), list)
            or not isinstance(data.get("to_cows"), list)
            or not isinstance(counts, dict)
        ):
            raise SafetyError(f"Route progress file is invalid or uses another screen: {progress_path}")
        for direction, count in counts.items():
            if direction not in ("to_bank", "to_cows") or not isinstance(count, int) or not 1 <= count <= 40:
                raise SafetyError(f"Route progress has an invalid step count: {progress_path}")
            if len(data[direction]) > count:
                raise SafetyError(f"Route progress has too many recorded steps: {progress_path}")
        if not asset_dir.is_dir():
            raise SafetyError(f"Route progress assets are missing: {asset_dir}")
        return progress, True, reused_assets

    asset_dir.mkdir(parents=True, exist_ok=True)
    progress = {
        "version": 1,
        "data": {"screen": screen, "to_bank": [], "to_cows": []},
        "counts": {},
    }
    save_route_progress(progress_path, progress)
    return progress, False, reused_assets


def show_route_resume_context(asset_dir: Path, direction: str, step: int) -> None:
    stem = f"{direction}_{step:02d}"
    previous = f"{direction}_{step - 1:02d}"
    candidates = [
        asset_dir / f"{stem}_before_screenshot.png",
        asset_dir / f"{stem}_arrival_screenshot.png",
        asset_dir / f"{previous}_arrival_screenshot.png",
    ]
    screenshots = [candidate for candidate in candidates if candidate.is_file()]
    if not screenshots:
        print("這一步尚未留下截圖；將從該步重新開始。")
        return
    print("以下顯示目前未完成步驟及上一個已完成步驟的截圖；按任意鍵關閉每張圖後繼續。")
    for screenshot in screenshots:
        image = load_image(screenshot)
        preview = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
        title = f"恢復校準：{screenshot.name}"
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def report_route_resume(path: Path, error: BaseException) -> None:
    progress_path = route_progress_path(path)
    if not progress_path.is_file():
        return
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(f"無法讀取續錄進度檔：{progress_path}")
        return
    data, counts = progress.get("data"), progress.get("counts")
    if not isinstance(data, dict) or not isinstance(counts, dict):
        print(f"續錄進度檔格式無效：{progress_path}")
        return

    pending = None
    for direction in ("to_bank", "to_cows"):
        recorded = data.get(direction)
        if not isinstance(recorded, list):
            continue
        count = counts.get(direction)
        if count is None:
            pending = (direction, len(recorded) + 1, None, len(recorded))
            break
        if len(recorded) < count:
            pending = (direction, len(recorded) + 1, count, len(recorded))
            break

    print(f"\n路線錄製中止：{str(error) or '使用者中止錄製'}")
    if pending is None:
        print("兩個方向的步驟已記錄完；下次啟動會重新驗證並完成路線檔。")
    else:
        direction, step, count, completed = pending
        heading = "牛欄去銀行" if direction == "to_bank" else "銀行返回牛欄"
        total = f"/{count}" if count is not None else "（尚未輸入總步數）"
        print(f"續錄位置：{heading}第 {step}{total} 步；此前 {completed} 步已保存。")
        asset_dir = path.parent / f"{path.stem}_assets"
        for suffix in ("before_screenshot.png", "arrival_screenshot.png"):
            screenshot = asset_dir / f"{direction}_{step - 1:02d}_{suffix}"
            if screenshot.is_file():
                print(f"本步已有截圖可查看：{screenshot}")
    print(
        "下次請原樣重跑同一命令（不要改路線名稱）即可續錄：\n"
        f'py -3 cow_bot.py record-route --output "{path}"'
    )


def route_step_prompt(direction: str, step: int, total: int) -> bool:
    while True:
        raw = input(
            f"{direction}：第 {step}/{total} 步。按 Enter 開始這一步；"
            "如果整趟往返已完成，輸入 finish 結束錄製："
        ).strip()
        if not raw:
            return True
        if raw.casefold() == "finish":
            return False
        print("請按 Enter 開始這一步，或只輸入 finish 結束整趟往返錄製。")


def validate_recorded_route(data: dict) -> None:
    if not data["to_bank"] or not data["to_cows"]:
        raise SafetyError(
            "路線未保存：輸入 finish 前必須完成牛欄到銀行及銀行返回牛欄兩段路線。"
        )
    if data["to_bank"][-1]["hover"] != "Bank Bank booth":
        raise SafetyError("去銀行路線最後一步必須是 Bank Bank booth（開銀行）；路線未保存。")


def save_route_landmarks(
    origin_path: Path, origin: np.ndarray, arrival_path: Path, arrival: np.ndarray,
) -> None:
    save_image(origin_path, origin)
    save_image(arrival_path, arrival)


def split_finished_round_trip(data: dict) -> bool:
    if data["to_cows"]:
        return False
    bank_steps = [
        index for index, step in enumerate(data["to_bank"])
        if step["hover"] == "Bank Bank booth"
    ]
    if len(bank_steps) != 1 or bank_steps[0] == len(data["to_bank"]) - 1:
        return False
    bank_index = bank_steps[0]
    data["to_cows"] = data["to_bank"][bank_index + 1:]
    data["to_bank"] = data["to_bank"][:bank_index + 1]
    return True


def arrival_timeout(action: str) -> float:
    is_stair = action.startswith("Climb-")
    default = 5 if is_stair else 15
    guidance = (
        "樓梯建議 5 秒；直接按 Enter 使用 5 秒"
        if is_stair else
        "行路建議 15 秒，最長 45 秒；直接按 Enter 使用 15 秒"
    )
    while True:
        raw = input(f"到站確認期限（{guidance}；只輸入數字）：").strip()
        if not raw:
            return float(default)
        try:
            timeout = float(raw)
        except ValueError:
            print("格式錯誤：請只輸入秒數，例如 15；不要輸入說明文字。")
            continue
        if 1 <= timeout <= 45:
            return timeout
        print("秒數必須介乎 1 至 45，請重新輸入。")


def learn_route_hover_action(
    alphabet: dict, image: np.ndarray, screenshot_path: Path, label: str,
) -> str:
    allowed = {
        "Walk here", "Open Gate", "Climb-up Staircase", "Climb-down Staircase", "Bank Bank booth",
    }
    action = label.split("/", 1)[0].strip()
    if action not in allowed or action in ("Climb-up Staircase", "Climb-down Staircase"):
        raise ValueError("Unsupported hover action")
    try:
        learn_line(alphabet, image, action, prefix_only=True)
    except SafetyError as error:
        raise ValueError(str(error)) from error
    return action


def route_hover_action(alphabet: dict, image: np.ndarray, screenshot_path: Path) -> str:
    while True:
        label = input(
            "請輸入截圖左上角斜線前的完整動作文字；`/ N more options` 尾碼可略過。"
            f"截圖：{screenshot_path}\n動作文字："
        ).strip()
        try:
            return learn_route_hover_action(alphabet, image, screenshot_path, label)
        except ValueError as error:
            print(
                f"文字與截圖字形不符或不支援（{error}）。請確認斜線前的完整動作文字；"
                "若是樓梯右鍵選單且左上角空白，請輸入 Climb-up Staircase "
                "或 Climb-down Staircase。"
            )


def route_stair_action(
    image: np.ndarray, screenshot_path: Path, asset_path: Path, action: str,
) -> str:
    target = "Top-floor Staircase" if action == "Climb-up Staircase" else "Bottom-floor Staircase"
    while True:
        print(
            f"這是第二個框選，與前面的起點地標不同。樓梯動作不用抄左上角；"
            f"請只框選單中單獨一列「{target}」。"
            f"「{action}」是另一列，不要把兩列一起框。"
        )
        try:
            rect = select_rect(
                image,
                f"截圖預覽：框選 {target} 選單列；ENTER/SPACE 確認",
                minimum=(24, 6),
            )
        except SafetyError as error:
            if str(error).startswith("Selection too small"):
                print("選框太小；請只重選這一整列選單文字，其他已完成的選擇不必重做。")
                continue
            raise
        selected = crop(image, rect)
        mask = text_mask(selected)
        rows, columns = np.where(mask)
        if not len(rows):
            print("選取範圍內沒有可辨識的選單文字，請重選該列。")
            continue
        top, bottom = int(rows.min()), int(rows.max()) + 1
        left, right = int(columns.min()), int(columns.max()) + 1
        if bottom - top > 18 or right - left < 30:
            print("請只框住一整列樓梯選項，並包含完整文字；不要框到其他選項。")
            continue
        template = np.repeat(
            (mask[top:bottom, left:right].astype(np.uint8) * 255)[:, :, None],
            3,
            axis=2,
        )
        save_image(asset_path, template)
        return str(asset_path.relative_to(screenshot_path.parents[1]))


def route_action(
    alphabet: dict, image: np.ndarray, screenshot_path: Path, asset_path: Path,
) -> tuple[str, str | None]:
    stair_actions = {
        "Climb-up Staircase": "Climb-up Staircase",
        "Top-floor Staircase": "Climb-up Staircase",
        "Climb-down Staircase": "Climb-down Staircase",
        "Bottom-floor Staircase": "Climb-down Staircase",
    }
    while True:
        label = input(
            "輸入截圖左上角完整動作文字；若是樓梯右鍵選單，"
            "可輸入 Climb-up Staircase / Top-floor Staircase "
            "或 Climb-down Staircase / Bottom-floor Staircase："
        ).strip()
        action = label.split(" / ")[0]
        if label in stair_actions:
            stair_action = stair_actions[label]
            return stair_action, route_stair_action(
                image, screenshot_path, asset_path, stair_action,
            )
        try:
            return learn_route_hover_action(alphabet, image, screenshot_path, label), None
        except ValueError as error:
            print(
                f"文字與截圖字形不符或不支援（{error}）。請重新輸入；"
                "一般動作只需輸入斜線前的文字，"
                "樓梯右鍵選單可輸入 Climb-up Staircase / Top-floor Staircase "
                "或 Climb-down Staircase / Bottom-floor Staircase。"
            )


def record_route(path: Path) -> None:
    from cow_desktop import Desktop, Route

    vision = Vision()
    path.parent.mkdir(parents=True, exist_ok=True)
    asset_dir = path.parent / f"{path.stem}_assets"
    progress, resumed, reused_assets = load_route_progress(
        path, asset_dir, vision.settings["screen"],
    )
    desktop = Desktop(vision)
    progress_file = route_progress_path(path)
    data = progress["data"]
    counts = progress["counts"]
    if resumed:
        print(f"找到未完成進度，會從同一路線繼續：{path}")
    elif reused_assets:
        print("找到舊截圖資料夾但沒有進度檔；保留資料夾，從第一步開始記錄。")
    alphabet = json.loads((PROFILE / "font.json").read_text(encoding="utf-8"))
    finish_requested = False
    resume_context_shown = False
    for direction in ("to_bank", "to_cows"):
        heading = "牛欄去銀行" if direction == "to_bank" else "銀行返回牛欄"
        print(f"\n===== 記錄：{heading} =====")
        print("你要逐步手動帶路。程式只截圖和記錄，絕不移動滑鼠或操作遊戲。")
        if direction == "to_cows" and not data["to_cows"]:
            print("開始前：請你手動存倉並關閉銀行，角色留在銀行位置。")
        print("步數算法：每段行路、每道閘、每段樓梯各 1 步；去銀行最後的開銀行動作也算 1 步。")
        if direction not in counts:
            count = route_step_count(heading)
            counts[direction] = count
            save_route_progress(progress_file, progress)
        else:
            count = counts[direction]
            print(f"已保存此方向的預估步數：{count}；已完成：{len(data[direction])} 步。")
        completed = len(data[direction])
        if completed > count:
            raise SafetyError(f"Saved route progress exceeds the planned step count for {direction}")
        if completed == count:
            print(f"{heading} 已完成全部 {count} 步。")
            continue
        if resumed and not resume_context_shown:
            print(f"恢復位置：{heading}，接下來第 {completed + 1}/{count} 步。")
            show_route_resume_context(asset_dir, direction, completed)
            resume_context_shown = True
        for index in range(completed, count):
            stem = f"{direction}_{index:02d}"
            print(f"\n--- {heading}：第 {index + 1}/{count} 步 ---")
            if not route_step_prompt(heading, index + 1, count):
                finish_requested = True
                print("收到 finish：結束整趟往返錄製。")
                break
            if direction == "to_bank" and index == count - 1:
                print("注意：這必須是最後一步，動作要用 Bank Bank booth 開銀行。")
            print(
                "截圖前準備：一般動作請懸停目標；樓梯請先右鍵樓梯並保持選單打開。"
                "樓梯右鍵選單時左上角沒有文字是正常的，程式之後會讓你選要記錄的樓梯列。"
            )
            before = countdown_snapshot(
                desktop,
                "5 秒後截取起點畫面。切到最大化 OSRS，保持畫面不動；"
                "樓梯步驟請保持右鍵選單開啟。程式不會操作遊戲。",
            )
            before_path = asset_dir / f"{stem}_before_screenshot.png"
            save_image(before_path, before)
            print(f"起點截圖已保存：{before_path}")
            print(
                "在彈出的截圖預覽視窗中，用滑鼠拖框圈住附近清楚、固定的景物；"
                "放開滑鼠後按 Enter/Space。不要在遊戲畫面上畫框，也不要選人物、牛或游標。"
            )
            origin_roi = select_rect(before, "截圖預覽：框選起點固定地標；ENTER/SPACE 確認")
            origin = crop(before, origin_roi)
            if float(origin.std()) < 12:
                raise SafetyError("Origin landmark has insufficient visual detail")
            print(
                "目的地選擇視窗會再次顯示同一張截圖。請在預覽圖的目的地位置按一下；"
                "這只記錄座標，不會在遊戲裡點擊。"
            )
            world = vision.settings["search"]
            print(
                "目的地選擇視窗會再次顯示截圖；綠框涵蓋整個遊戲視窗客戶區，"
                "不是行走距離或步數限制。請在綠框內、遊戲視窗範圍按一下；"
                "不要選 Windows 標題列或工作列。座標會自動換算回原始截圖大小。"
            )
            world = ROUTE_CLICK_AREA
            while True:
                point = select_point(before, world)
                if point_inside_rect(point, world):
                    break
                print("你點在綠框外；請在較大的綠框內重新選目的地。")
            anchor_search = ROUTE_ANCHOR_SEARCH
            action, menu_action = route_action(
                alphabet,
                before,
                before_path,
                asset_dir / f"{stem}_menu_action.png",
            )
            input(
                f"現在切回遊戲，手動完成一次「{action}」並等角色到達這一步的目的地；"
                "程式不會代你點擊。到達後回這裡按 Enter 拍抵達畫面。"
            )
            after = countdown_snapshot(
                desktop,
                "5 秒後截取抵達畫面。保持最大化 OSRS 在前景及畫面穩定；程式不會操作遊戲。",
            )
            after_path = asset_dir / f"{stem}_arrival_screenshot.png"
            save_image(after_path, after)
            print(f"抵達截圖已保存：{after_path}")
            print(
                "在彈出的抵達截圖預覽視窗中，框選一小塊能證明已到達此步目的地的固定景物，"
                "例如獨特地形或圍欄。移動前同一位置必須不同；不要選人物、牛或會移動的點。"
            )
            arrival_roi = select_rect(after, "截圖預覽：框選抵達地標；ENTER/SPACE 確認")
            arrival = crop(after, arrival_roi)
            if float(arrival.std()) < 12 or similar(crop(before, arrival_roi), arrival, 12):
                raise SafetyError("Arrival signature is blank or already present before moving")
            print("這是程式日後執行路線時的到站等待上限，不是叫你現在等待。")
            timeout = arrival_timeout(action)
            origin_path, arrival_path = asset_dir / f"{stem}_origin.png", asset_dir / f"{stem}_arrival.png"
            save_route_landmarks(origin_path, origin, arrival_path, arrival)
            data[direction].append({
                "hover": action,
                "origin": str(origin_path.relative_to(path.parent)),
                "arrival": str(arrival_path.relative_to(path.parent)),
                "arrival_roi": arrival_roi,
                "offset": [point[0] - origin_roi[0], point[1] - origin_roi[1]],
                "search": anchor_search,
                "timeout": timeout,
            })
            if menu_action is not None:
                data[direction][-1]["menu_action"] = menu_action
            (PROFILE / "font.json").write_text(json.dumps(alphabet), encoding="utf-8")
            save_route_progress(progress_file, progress)
        if finish_requested:
            break
    if finish_requested and split_finished_round_trip(data):
        print("已按開銀行步驟自動分開去程與回程。")
    validate_recorded_route(data)
    (PROFILE / "font.json").write_text(json.dumps(alphabet), encoding="utf-8")
    temporary_route = path.with_name(f"{path.stem}.tmp.json")
    temporary_route.write_text(json.dumps(data, indent=2), encoding="utf-8")
    Route(temporary_route, Vision())
    temporary_route.replace(path)
    progress_file.unlink()
    print(f"\n校準完成，路線已保存：{path}")


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
    offset = vision.inventory_panel_offset(image)
    if offset is None:
        raise SafetyError("Inventory panel position could not be verified")
    save_image(PROFILE / "Bones.png", crop(image, (x - 16, y + offset - 16, 32, 32)))
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
        output = args.output.resolve()
        try:
            record_route(output)
        except (SafetyError, OSError, ValueError, KeyError, cv2.error) as error:
            report_route_resume(output, error)
            raise
        except KeyboardInterrupt as error:
            report_route_resume(output, error)
            raise
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
    except (SafetyError, OSError, ValueError, KeyError, cv2.error) as error:
        raise SystemExit(f"ERROR: {error}") from error
