import contextlib
import io
import json
import tempfile
import threading
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import Mock, call, patch

import cv2
import numpy as np

from cow_bot import (
    arrival_timeout, countdown, main, point_inside_rect, preview_point_to_image, replay,
    load_route_progress, report_route_resume, route_action, route_hover_action, route_progress_path,
    route_recalibration_target_area, route_step_count, route_step_prompt, route_stair_action,
    save_route_progress, select_rect,
    save_route_landmarks, split_finished_round_trip, recalibrate_route_step,
    learn_route_hover_action,
    verify_route_step_start, verify_route_step_arrival,
    show_route_resume_context, validate_recorded_route,
)
from cow_core import (
    Health, HealthLock, Inventory, InventoryPanelUnavailable, Mode, PickupCheck, SafetyError,
)
from cow_desktop import (
    Bot, Desktop, Frame, Route, foreground_window_problem, locate, menu_template_location,
    project_route_point, registered_route_arrival, route_scene_homography,
    route_reference_search, route_target_search, stable_route_target,
)
from cow_vision import (
    PROFILE, ROOT, Vision, bootstrap, crop, decode_hover_line, learn_line, load_image,
    save_image, text_mask,
)


def inventory(hides=0, bones=0, other=0):
    return Inventory(tuple(
        ["Cowhide"] * hides + ["Bones"] * bones + ["UNKNOWN"] * other
        + ["EMPTY"] * (28 - hides - bones - other)
    ))


def hover_image(text, shape=(1080, 1920, 3)):
    image = np.zeros(shape, dtype=np.uint8)
    x = 8
    for character in text:
        if character == " ":
            x += 5
            continue
        glyph = np.zeros((6, 5), dtype=np.uint8)
        glyph[0, :] = 255
        glyph[5, :] = 255
        code = ord(character)
        for row in range(4):
            for column in range(3):
                if code & (1 << (row * 3 + column)):
                    glyph[row + 1, column + 1] = 255
        image[31:37, x:x + 5] = glyph[:, :, None]
        x += 7
    return image


class CountdownTests(unittest.TestCase):
    def test_countdown_displays_each_second_and_waits_one_second_each(self):
        output = io.StringIO()
        with patch("cow_bot.time.sleep") as sleep:
            with contextlib.redirect_stdout(output):
                countdown("LIVE BOT will control the game.")
        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "ALERT: LIVE BOT will control the game.",
                "Starting in 5...",
                "Starting in 4...",
                "Starting in 3...",
                "Starting in 2...",
                "Starting in 1...",
            ],
        )
        self.assertEqual(sleep.call_count, 5)
        sleep.assert_has_calls([call(1)] * 5)

    def test_route_step_count_reprompts_until_valid_integer(self):
        output = io.StringIO()
        with patch("builtins.input", side_effect=["many", "0", "3"]), contextlib.redirect_stdout(output):
            self.assertEqual(route_step_count("牛欄去銀行"), 3)
        self.assertIn("請只輸入 1 到 40 的整數", output.getvalue())
        self.assertIn("步數必須是 1 到 40", output.getvalue())

    def test_step_four_recalibration_uses_the_annotated_safe_area(self):
        self.assertEqual(
            route_recalibration_target_area("to_bank", 4, (1080, 1920, 3)),
            (820, 740, 370, 180),
        )
        self.assertEqual(
            route_recalibration_target_area("to_bank", 3, (1080, 1920, 3)),
            (0, 24, 1920, 896),
        )
        with self.assertRaisesRegex(SafetyError, "outside the safe screen"):
            route_recalibration_target_area("to_bank", 4, (900, 1920, 3))

    def test_select_rect_accepts_short_menu_row_and_reports_cancel_separately(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with patch("cow_bot.cv2.namedWindow"), patch("cow_bot.cv2.resizeWindow"):
            with patch("cow_bot.cv2.selectROI", return_value=(10, 10, 80, 8)):
                with patch("cow_bot.cv2.destroyAllWindows"):
                    self.assertEqual(select_rect(image, "menu", minimum=(24, 6)), [10, 10, 80, 8])
            with patch("cow_bot.cv2.selectROI", return_value=(10, 10, 80, 8)):
                with patch("cow_bot.cv2.destroyAllWindows"):
                    with self.assertRaisesRegex(SafetyError, "minimum 24 x 16"):
                        select_rect(image, "landmark")

    def test_route_step_prompt_accepts_enter_or_finish_only(self):
        output = io.StringIO()
        with patch("builtins.input", side_effect=["skip", ""]), contextlib.redirect_stdout(output):
            self.assertTrue(route_step_prompt("牛欄去銀行", 2, 30))
        self.assertIn("請按 Enter 開始這一步", output.getvalue())
        with patch("builtins.input", return_value="finish"):
            self.assertFalse(route_step_prompt("銀行返回牛欄", 26, 30))

    def test_finish_only_accepts_a_complete_round_trip_route(self):
        incomplete = {"to_bank": [{"hover": "Bank Bank booth"}], "to_cows": []}
        with self.assertRaisesRegex(SafetyError, "必須完成牛欄到銀行及銀行返回牛欄"):
            validate_recorded_route(incomplete)
        complete = {
            "to_bank": [{"hover": "Bank Bank booth"}],
            "to_cows": [{"hover": "Walk here"}],
        }
        validate_recorded_route(complete)

    def test_finish_splits_combined_round_trip_after_bank_step(self):
        outbound = [{"hover": "Walk here"}, {"hover": "Bank Bank booth"}]
        return_trip = [{"hover": "Walk here"}, {"hover": "Climb-down Staircase"}]
        data = {"to_bank": outbound + return_trip, "to_cows": []}

        self.assertTrue(split_finished_round_trip(data))
        self.assertEqual(data, {"to_bank": outbound, "to_cows": return_trip})
        validate_recorded_route(data)

    def test_finish_does_not_split_when_bank_is_missing_ambiguous_or_last(self):
        for steps in (
            [{"hover": "Walk here"}, {"hover": "Walk here"}],
            [{"hover": "Bank Bank booth"}, {"hover": "Walk here"},
             {"hover": "Bank Bank booth"}],
            [{"hover": "Walk here"}, {"hover": "Bank Bank booth"}],
        ):
            data = {"to_bank": steps, "to_cows": []}
            self.assertFalse(split_finished_round_trip(data))
            self.assertEqual(data["to_cows"], [])

    def test_save_route_landmarks_persists_both_step_signatures(self):
        origin = np.full((12, 14, 3), 50, dtype=np.uint8)
        arrival = np.full((10, 13, 3), 200, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            origin_path = Path(directory) / "origin.png"
            arrival_path = Path(directory) / "arrival.png"
            save_route_landmarks(origin_path, origin, arrival_path, arrival)
            np.testing.assert_array_equal(load_image(origin_path), origin)
            np.testing.assert_array_equal(load_image(arrival_path), arrival)

    def test_route_progress_resumes_same_output_and_preserves_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            route = root / "route.json"
            assets = root / "route_assets"
            assets.mkdir()
            progress, resumed, reused_assets = load_route_progress(route, assets, [1920, 1080])
            self.assertFalse(resumed)
            self.assertTrue(reused_assets)
            progress["counts"]["to_bank"] = 30
            progress["data"]["to_bank"].append({"hover": "Walk here"})
            save_route_progress(route_progress_path(route), progress)

            restored, resumed, _ = load_route_progress(route, assets, [1920, 1080])

            self.assertTrue(resumed)
            self.assertEqual(restored["counts"]["to_bank"], 30)
            self.assertEqual(restored["data"]["to_bank"], [{"hover": "Walk here"}])

    def test_route_resume_displays_incomplete_and_previous_step_screenshots(self):
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory)
            save_image(assets / "to_bank_01_before_screenshot.png", np.zeros((1080, 1920, 3), dtype=np.uint8))
            save_image(assets / "to_bank_00_arrival_screenshot.png", np.zeros((1080, 1920, 3), dtype=np.uint8))
            with patch("cow_bot.cv2.namedWindow") as window:
                with patch("cow_bot.cv2.imshow"), patch("cow_bot.cv2.waitKey", return_value=0):
                    with patch("cow_bot.cv2.destroyWindow"):
                        show_route_resume_context(assets, "to_bank", 1)
            self.assertEqual(window.call_count, 2)

    def test_route_error_reports_saved_position_and_same_resume_command(self):
        with tempfile.TemporaryDirectory() as directory:
            route = Path(directory) / "route.json"
            assets = Path(directory) / "route_assets"
            assets.mkdir()
            (assets / "to_bank_02_before_screenshot.png").write_bytes(b"saved image")
            (assets / "to_bank_01_arrival_screenshot.png").write_bytes(b"previous image")
            progress = {
                "version": 1,
                "data": {
                    "screen": [1920, 1080],
                    "to_bank": [{"hover": "Walk here"}, {"hover": "Walk here"}],
                    "to_cows": [],
                },
                "counts": {"to_bank": 30},
            }
            save_route_progress(route_progress_path(route), progress)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                report_route_resume(route, ValueError("selection failed"))
            self.assertIn("牛欄去銀行第 3/30 步；此前 2 步已保存", output.getvalue())
            self.assertIn("to_bank_02_before_screenshot.png", output.getvalue())
            self.assertIn(f'--output "{route}"', output.getvalue())

    def test_arrival_timeout_defaults_and_reprompts(self):
        with patch("builtins.input", return_value=""):
            self.assertEqual(arrival_timeout("Walk here"), 15)
            self.assertEqual(arrival_timeout("Climb-up Staircase"), 5)

        output = io.StringIO()
        with patch("builtins.input", side_effect=["(stairs: 5, walking: 50)", "50", "15"]), contextlib.redirect_stdout(output):
            self.assertEqual(arrival_timeout("Walk here"), 15)
        self.assertEqual(output.getvalue().count("請只輸入秒數"), 1)
        self.assertEqual(output.getvalue().count("秒數必須介乎 1 至 45"), 1)

    def test_preview_click_is_scaled_back_to_original_screenshot_coordinates(self):
        self.assertEqual(
            preview_point_to_image((640, 360), (1080, 1920, 3), (720, 1280, 3)),
            (960, 540),
        )
        self.assertEqual(
            preview_point_to_image((1279, 719), (1080, 1920, 3), (720, 1280, 3)),
            (1919, 1079),
        )

    def test_route_destination_area_covers_client_but_excludes_windows_chrome(self):
        self.assertTrue(point_inside_rect((0, 24), (0, 24, 1920, 1016)))
        self.assertTrue(point_inside_rect((1919, 1039), (0, 24, 1920, 1016)))
        self.assertFalse(point_inside_rect((500, 20), (0, 24, 1920, 1016)))
        self.assertFalse(point_inside_rect((500, 1040), (0, 24, 1920, 1016)))

    def test_recalibrate_route_cli_dispatches_one_based_step(self):
        argv = [
            "cow_bot.py", "recalibrate-route", "--route", "cow_profile\\route_v17.json",
            "--direction", "to_bank", "--step", "4",
        ]
        vision = Mock()
        with patch("sys.argv", argv), patch("cow_bot.Vision", return_value=vision):
            with patch("cow_bot.recalibrate_route_step") as recalibrate:
                main()
        recalibrate.assert_called_once_with(
            Path("cow_profile\\route_v17.json"), "to_bank", 4,
        )

    def test_bad_hover_text_reprompts_instead_of_aborting_route(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        output = io.StringIO()
        with patch("builtins.input", side_effect=["Walk here", "Walk here / 1 more options"]):
            with patch("cow_bot.learn_line", side_effect=[SafetyError("wrong glyph count"), None]):
                with contextlib.redirect_stdout(output):
                    action = route_hover_action({}, image, Path("route_assets/step.png"))
        self.assertEqual(action, "Walk here")
        self.assertIn("文字與截圖字形不符或不支援", output.getvalue())

    def test_route_hover_learning_rejects_action_that_disagrees_with_screenshot(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory)
            bootstrap(profile)
            alphabet = json.loads((profile / "font.json").read_text(encoding="utf-8"))
            original = json.loads(json.dumps(alphabet))
            image = load_image(ROOT / "images" / "after_take_cowhide.png")

            with self.assertRaisesRegex(ValueError, "Screenshot hover is 'Walk here'"):
                learn_route_hover_action(
                    alphabet, image, Path("route_assets/step.png"), "Open Gate",
                )

            self.assertEqual(alphabet, original)

    def test_gate_recalibration_screenshot_keeps_its_actual_hover_label(self):
        alphabet = json.loads((PROFILE / "font.json").read_text(encoding="utf-8"))
        image = load_image(
            PROFILE / "route_v17_assets" / "to_bank_01_recal_20261121_010157_329447"
            / "to_bank_01_recal_20261121_010157_329447_before_screenshot.png",
        )
        self.assertEqual(decode_hover_line(alphabet, image), "Walk here")

    def test_learn_line_rejects_ambiguous_glyphs_without_partial_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory)
            bootstrap(profile)
            alphabet = json.loads((profile / "font.json").read_text(encoding="utf-8"))
            original = json.loads(json.dumps(alphabet))
            image = load_image(ROOT / "images" / "after_take_cowhide.png")

            with self.assertRaisesRegex(SafetyError, "Glyph for 'O' is ambiguous"):
                learn_line(alphabet, image, "Open Gate", prefix_only=True)

            self.assertEqual(alphabet, original)

    def test_hover_action_ignores_optional_more_options_suffix(self):
        visible_line = "Bank Bank booth / 7 more options"
        image = hover_image(visible_line)

        for label in ("Bank Bank booth", "Bank Bank booth / 4 more options",
                      "Bank Bank booth / 27 more options"):
            alphabet = {}
            with patch("builtins.input", return_value=label):
                action = route_hover_action(alphabet, image, Path("route_assets/step.png"))
            self.assertEqual(action, "Bank Bank booth")
            self.assertEqual(set(alphabet), set("Bankbooth"))

    def test_learn_line_prefix_only_ignores_trailing_option_glyphs(self):
        image = hover_image("Bank Bank booth / 4 more options", (40, 500, 3))
        alphabet = {}
        learn_line(alphabet, image, "Bank Bank booth", prefix_only=True)
        self.assertEqual(set(alphabet), set("Bankbooth"))

    def test_stair_action_uses_menu_row_when_hover_text_is_unavailable(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with patch("builtins.input", return_value="Bottom-floor Staircase"):
            with patch("cow_bot.route_stair_action", return_value="route_assets/step_menu.png") as record:
                action, menu = route_action(
                    {}, image, Path("C:/profile/route_assets/step.png"),
                    Path("C:/profile/route_assets/step_menu.png"),
                )
        self.assertEqual((action, menu), ("Climb-down Staircase", "route_assets/step_menu.png"))
        record.assert_called_once()
        self.assertEqual(record.call_args.args[3], "Climb-down Staircase")

    def test_stair_menu_template_rejects_both_rows_then_accepts_target_row(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.putText(image, "Climb-down Staircase", (80, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_8)
        cv2.putText(image, "Bottom-floor Staircase", (80, 218),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_8)
        with tempfile.TemporaryDirectory() as directory:
            screenshot = Path(directory) / "route_assets" / "before.png"
            asset = Path(directory) / "route_assets" / "menu.png"
            output = io.StringIO()
            with patch(
                "cow_bot.select_rect",
                side_effect=[[70, 185, 250, 40], [70, 202, 250, 16]],
            ) as select:
                with contextlib.redirect_stdout(output):
                    path = route_stair_action(image, screenshot, asset, "Climb-down Staircase")
            template = load_image(asset)
            self.assertEqual(path, str(Path("route_assets") / "menu.png"))
            self.assertEqual(select.call_count, 2)
            self.assertIn("不要把兩列一起框", output.getvalue())
            self.assertGreater(np.count_nonzero(text_mask(template)), 0)

    def test_menu_template_match_requires_one_exact_option(self):
        template = np.zeros((8, 16, 3), dtype=np.uint8)
        template[1:7, 2:14] = 255
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[30:38, 40:56] = template
        self.assertEqual(menu_template_location(image, template, (0, 0, 100, 100)), (40, 30, 16, 8))
        image[60:68, 40:56] = template
        self.assertIsNone(menu_template_location(image, template, (0, 0, 100, 100)))


class ForegroundWindowTests(unittest.TestCase):
    def test_foreground_failure_reports_title_and_minimized_state_separately(self):
        self.assertIsNone(foreground_window_problem("Old School RuneScape", False))
        self.assertEqual(
            foreground_window_problem("PowerShell", False),
            "Foreground-window check failed: unexpected foreground title='PowerShell'",
        )
        self.assertEqual(
            foreground_window_problem("Old School RuneScape", True),
            "Foreground-window check failed: foreground window is minimized",
        )
        self.assertEqual(
            foreground_window_problem("Other window", True),
            "Foreground-window check failed: unexpected foreground title='Other window'; "
            "foreground window is minimized",
        )


class HealthTests(unittest.TestCase):
    def test_requires_positive_then_two_independent_zeros(self):
        lock = HealthLock(0, "Cow")
        self.assertEqual(lock.observe(1, .1, Health("Cow", False)), "WAIT")
        self.assertEqual(lock.observe(2, .2, Health("Cow", True)), "WAIT")
        self.assertEqual(lock.observe(3, .3, Health("Cow", False)), "WAIT")
        self.assertEqual(lock.observe(3, .4, Health("Cow", False)), "WAIT")
        self.assertEqual(lock.observe(4, .5, Health("Cow", False)), "KILLED")

    def test_missing_frame_interrupts_zero_streak(self):
        lock = HealthLock(0, "Cow")
        for i, hp in enumerate([True, False, None, False], 1):
            result = lock.observe(i, i / 10, None if hp is None else Health("Cow", hp))
            self.assertEqual(result, "WAIT")
        self.assertEqual(lock.observe(5, .5, Health("Cow", False)), "KILLED")

    def test_stale_capture_and_stale_hud_do_not_confirm_kill(self):
        lock = HealthLock(10, "Cow")
        self.assertEqual(lock.observe(1, 9, Health("Cow", True)), "WAIT")
        self.assertEqual(lock.observe(2, 10.1, Health("Cow", False)), "WAIT")
        self.assertEqual(lock.observe(3, 13.1, Health("Cow", False)), "WAIT")
        self.assertEqual(lock.observe(4, 20.1, Health("Cow", False)), "MISSED")

    def test_target_change_and_combat_timeout_stop(self):
        with self.assertRaisesRegex(SafetyError, "target changed"):
            HealthLock(0, "Cow").observe(1, .1, Health("Cow calf", True))
        lock = HealthLock(0, "Cow")
        lock.observe(1, 1, Health("Cow", True))
        with self.assertRaisesRegex(SafetyError, "timed out"):
            lock.observe(2, 31, None)

    def test_first_positive_after_start_deadline_is_not_accepted(self):
        self.assertEqual(HealthLock(0, "Cow").observe(1, 10.1, Health("Cow", True)), "MISSED")

    def test_positive_health_within_ten_seconds_starts_kill_confirmation(self):
        self.assertEqual(
            HealthLock(0, "Cow").observe(1, 9.9, Health("Cow", True)),
            "WAIT",
        )

    def test_lock_can_begin_on_the_first_frame_after_player_stops(self):
        lock = HealthLock(12, "Cow")
        self.assertEqual(lock.observe(1, 12, Health("Cow", True)), "WAIT")
        self.assertTrue(lock.seen_positive)


class PickupTests(unittest.TestCase):
    def test_requires_exact_increment_and_two_fresh_confirmations(self):
        check = PickupCheck(inventory(2), "Cowhide", 0)
        self.assertFalse(check.observe(1, .1, inventory(2)))
        self.assertFalse(check.observe(2, .2, inventory(3)))
        self.assertFalse(check.observe(2, .3, inventory(3)))
        self.assertTrue(check.observe(3, .4, inventory(3)))

    def test_wrong_item_extra_items_and_timeout_stop(self):
        for current in (inventory(2, 1), inventory(4), inventory(3, other=1)):
            with self.subTest(current=current):
                with self.assertRaisesRegex(SafetyError, "unexpected"):
                    PickupCheck(inventory(2), "Cowhide", 0).observe(1, .1, current)
        with self.assertRaisesRegex(SafetyError, "fifteen seconds"):
            PickupCheck(inventory(), "Cowhide", 0).observe(1, 15.1, inventory())

    def test_full_and_unapproved_inventory_rejected(self):
        with self.assertRaises(SafetyError):
            PickupCheck(inventory(28), "Cowhide", 0)
        with self.assertRaises(SafetyError):
            inventory(1, other=1).require_allowed(Mode.COWHIDE_ONLY)
        with self.assertRaises(SafetyError):
            inventory(1, 1).require_allowed(Mode.COWHIDE_ONLY)
        inventory(1, 1).require_allowed(Mode.COWHIDE_AND_BONES)

    def test_late_second_confirmation_cannot_pass_timeout(self):
        check = PickupCheck(inventory(1), "Cowhide", 0)
        self.assertFalse(check.observe(1, 14.9, inventory(2)))
        with self.assertRaisesRegex(SafetyError, "fifteen seconds"):
            check.observe(2, 15.1, inventory(2))


class VisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_profile = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp_profile.cleanup)
        cls.profile = Path(cls.temp_profile.name)
        bootstrap(cls.profile)
        cls.vision = Vision(cls.profile)

    def image(self, name):
        return load_image(ROOT / "images" / f"{name}.png")

    def test_reference_hover_commands_are_exact(self):
        expected = {
            "attack_cow": "Attack Cow (level-2)",
            "attack_cowcalf": "Attack Cow calf (level-2)",
            "take_cowhide": "Take Cowhide",
            "bank_booth": "Bank Bank booth",
            "gate_closed": "Open Gate",
            "gate_opened": "Close Gate",
            "before_take_cowhide": "Chop down Willow tree",
        }
        for name, action in expected.items():
            with self.subTest(name=name):
                self.assertEqual(self.vision.hover(self.image(name)), action)

    def test_hover_ignores_bright_pixels_outside_the_action_text(self):
        image = self.image("attack_cow").copy()
        image[35, 366] = [255, 255, 255]
        self.assertEqual(self.vision.hover(image), "Attack Cow (level-2)")

    def test_real_hud_and_idle_discrimination(self):
        self.assertEqual(self.vision.health(self.image("attacking_cow")), Health("Cow", True))
        self.assertEqual(self.vision.health(self.image("attacking_cowcalf")), Health("Cow calf", True))
        self.assertEqual(self.vision.health(self.image("take_cowhide")), Health("Cow", False))
        self.assertIsNone(self.vision.health(self.image("attack_cow")))
        self.assertTrue(self.vision.hud_clear(self.image("attack_cow")))
        self.assertFalse(self.vision.hud_clear(self.image("take_cowhide")))
        corrupt = self.image("attacking_cow")
        corrupt[70:82, 60:94] = 0
        self.assertIsNone(self.vision.health(corrupt))
        self.assertFalse(self.vision.hud_clear(corrupt))
        scenery = load_image(
            ROOT / "cow_runs" / "20261117_013622_347050" / "00001_READY.png",
        )
        self.assertTrue(self.vision.hud_clear(scenery))

    def test_actual_before_after_hide_increment_not_stack_ocr(self):
        before = self.vision.inventory(self.image("before_take_cowhide"))
        after = self.vision.inventory(self.image("after_take_cowhide"))
        self.assertEqual(before.count("Cowhide"), 1)
        self.assertEqual(after.count("Cowhide"), 2)
        self.assertEqual(after.count("EMPTY"), before.count("EMPTY") - 1)
        check = PickupCheck(before, "Cowhide", 0)
        self.assertFalse(check.observe(1, .1, after))
        self.assertTrue(check.observe(2, .2, after))
        with self.assertRaisesRegex(SafetyError, "Unapproved"):
            before.require_allowed(Mode.COWHIDE_ONLY)

    def test_hidden_inventory_is_not_treated_as_empty(self):
        hidden = self.image("bank_booth")
        hidden[722:728, 1676:1876] = 0
        with self.assertRaisesRegex(InventoryPanelUnavailable, "panel"):
            self.vision.inventory(hidden)

    def test_inventory_tab_icon_is_recognized_in_open_and_closed_states(self):
        open_image = self.image("before_take_cowhide")
        closed_icon = load_image(ROOT / "images" / "inventory_tab_closed.png")
        closed_image = open_image.copy()
        closed_image[998:1028, 1558:1578] = closed_icon
        closed_image[722:728, 1676:1876] = 0
        self.assertTrue(self.vision.inventory_tab_visible(open_image))
        self.assertTrue(self.vision.inventory_tab_visible(closed_image))
        self.assertTrue(self.vision.inventory_panel_visible(open_image))
        self.assertFalse(self.vision.inventory_panel_visible(closed_image))
        hovered_image = open_image.copy()
        hovered_image[998:1028, 1558:1578] = load_image(
            ROOT / "images" / "inventory_tab_hover.png",
        )
        self.assertTrue(self.vision.inventory_tab_visible(hovered_image))
        unknown = open_image.copy()
        unknown[998:1028, 1558:1578] = 0
        self.assertFalse(self.vision.inventory_tab_visible(unknown))

    def test_inventory_detection_tracks_bounded_vertical_offset(self):
        image = self.image("before_take_cowhide")
        shifted = image.copy()
        shifted[8:] = image[:-8]

        self.assertEqual(self.vision.inventory_panel_offset(shifted), 8)
        self.assertEqual(self.vision.inventory_tab_offset(shifted), 8)
        self.assertTrue(self.vision.inventory_panel_visible(shifted))
        self.assertTrue(self.vision.inventory_tab_visible(shifted))
        self.assertEqual(self.vision.inventory(shifted), self.vision.inventory(image))

    def test_reopen_inventory_requires_verified_icon_and_two_readings(self):
        desktop = object.__new__(Desktop)
        vision = Mock()
        vision.inventory_panel_visible.side_effect = [False, False, False, False, True, True]
        vision.inventory_tab_offset.return_value = 8
        contents = inventory(2)
        vision.inventory.return_value = contents
        desktop.vision = vision
        frames = [Frame(index, index / 10, np.zeros((1080, 1920, 3), dtype=np.uint8))
                  for index in (1, 2, 3, 5, 6)]
        desktop.fresh = Mock(side_effect=frames)
        desktop.snapshot = Mock(return_value=np.zeros((1080, 1920, 3), dtype=np.uint8))
        desktop.move = Mock()
        desktop._input_check = Mock()
        desktop.mouse = Mock()

        with patch("cow_desktop.time.monotonic", return_value=.2):
            reopened = desktop.reopen_inventory_tab()

        self.assertEqual(reopened.number, 6)
        desktop.move.assert_called_once_with((1568, 1021))
        desktop.mouse.click.assert_called_once_with()
        self.assertEqual(vision.inventory_tab_offset.call_count, 4)
        self.assertEqual(vision.inventory.call_count, 2)

    def test_reopen_inventory_does_not_click_if_panel_appears_while_aiming(self):
        desktop = object.__new__(Desktop)
        desktop.vision = Mock()
        desktop.vision.inventory_panel_visible.side_effect = [False, True]
        desktop.vision.inventory_tab_offset.return_value = 8
        desktop.fresh = Mock(side_effect=[
            Frame(1, .1, np.zeros((1080, 1920, 3), dtype=np.uint8)),
            Frame(2, .2, np.zeros((1080, 1920, 3), dtype=np.uint8)),
        ])
        desktop.move = Mock()
        desktop.mouse = Mock()

        reopened = desktop.reopen_inventory_tab()

        self.assertEqual(reopened.number, 2)
        desktop.mouse.click.assert_not_called()

    def test_reopen_inventory_refuses_unrecognized_tab_without_clicking(self):
        desktop = object.__new__(Desktop)
        desktop.vision = Mock()
        desktop.vision.inventory_panel_visible.return_value = False
        desktop.vision.inventory_tab_offset.return_value = None
        desktop.fresh = Mock(return_value=Frame(
            1, .1, np.zeros((1080, 1920, 3), dtype=np.uint8),
        ))
        desktop.move = Mock()
        desktop.mouse = Mock()
        with self.assertRaisesRegex(SafetyError, "tab could not be verified"):
            desktop.reopen_inventory_tab()
        desktop.move.assert_not_called()
        desktop.mouse.click.assert_not_called()

    def test_bot_reopens_hidden_inventory_before_resuming_loot_search(self):
        vision = Mock()
        vision.settings = {"loot_radius": 130, "loot_fine_radius": 48}
        vision.loot_candidates.side_effect = [
            [((1036, 554), "fine")],
            [],
        ]
        vision.loot_grid.return_value = iter([((1036, 554), "fine")])
        vision.hover.return_value = "Walk here"
        desktop, audit = Mock(), Mock()
        desktop.vision = vision
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        desktop.fresh.side_effect = [
            Frame(1, .1, image),
            Frame(2, .2, image),
            Frame(3, 25.1, image),
        ]
        desktop.reopen_inventory_tab.return_value = Frame(
            2, .2, np.zeros((1080, 1920, 3), dtype=np.uint8),
        )
        bot = Bot(desktop, Mode.COWHIDE_ONLY, None, audit)
        bot.last_kill_point = (1036, 554)
        with patch(
            "cow_desktop.stable_inventory",
            side_effect=[InventoryPanelUnavailable("hidden"), inventory(2)],
        ):
            with patch("cow_desktop.time.monotonic", return_value=0):
                bot.loot()

        desktop.reopen_inventory_tab.assert_called_once()
        vision.loot_grid.assert_not_called()
        states = [call.args[0] for call in audit.event.call_args_list]
        self.assertEqual(states[0], "INVENTORY_REOPENED")
        self.assertEqual(states[1], "LOOT_SEARCH_FINISHED")

    def test_bot_startup_reopens_hidden_inventory_before_ready(self):
        vision = Mock()
        vision.settings = {}
        vision.items = {"Cowhide": np.zeros((32, 32, 3), dtype=np.uint8)}
        desktop, audit = Mock(), Mock()
        desktop.vision = vision
        desktop.reopen_inventory_tab.return_value = Frame(
            1, .1, np.zeros((1080, 1920, 3), dtype=np.uint8),
        )
        desktop.fresh.return_value = Frame(
            2, .2, np.zeros((1080, 1920, 3), dtype=np.uint8),
        )
        bot = Bot(desktop, Mode.COWHIDE_ONLY, None, audit)
        with patch(
            "cow_desktop.stable_inventory",
            side_effect=[InventoryPanelUnavailable("hidden"), inventory(2)],
        ):
            bot.run(0)

        desktop.reopen_inventory_tab.assert_called_once()
        states = [call.args[0] for call in audit.event.call_args_list]
        self.assertEqual(states, ["INVENTORY_REOPENED", "READY", "KILL_LIMIT_REACHED"])

    def test_deposit_video_has_bank_empty_inventory_and_unique_button(self):
        image = load_image(self.profile / "deposit_frame.png")
        self.assertTrue(self.vision.bank_open(image))
        self.assertTrue(self.vision.inventory(image).empty)
        self.assertEqual(self.vision.hover(image), "Deposit inventory")
        button = load_image(self.profile / "deposit_inventory.png")
        self.assertEqual(locate(image, button, [588, 46, 484, 800]), (1000, 806))
        self.assertFalse(self.vision.bank_open(self.image("bank_booth")))

    def test_menu_templates_match_only_the_approved_row(self):
        image = self.image("rightclick_menu")
        region = text_mask(crop(image, (790, 530, 150, 160))).astype(np.uint8) * 255
        for item, expected_y in (("Cowhide", 554), ("Bones", 570)):
            template = text_mask(load_image(self.profile / f"menu_{item}.png")).astype(np.uint8) * 255
            scores = cv2.matchTemplate(region, template, cv2.TM_SQDIFF_NORMED)
            positions = np.argwhere(scores <= .00001)
            self.assertEqual(len(positions), 1)
            self.assertEqual(int(positions[0][0]) + 530, expected_y)

    def test_duplicate_or_missing_waypoint_never_picks_arbitrarily(self):
        rng = np.random.default_rng(42)
        template = rng.integers(0, 256, (25, 30, 3), dtype=np.uint8)
        image = np.zeros((100, 150, 3), dtype=np.uint8)
        with self.assertRaises(SafetyError):
            locate(image, template, [0, 0, 150, 100])
        image[10:35, 10:40] = template
        self.assertEqual(locate(image, template, [0, 0, 150, 100]), (10, 10))
        image[60:85, 100:130] = template
        with self.assertRaisesRegex(SafetyError, "ambiguous"):
            locate(image, template, [0, 0, 150, 100])

    def test_geometry_rejected_instead_of_scaling_blindly(self):
        with self.assertRaisesRegex(SafetyError, "Resolution"):
            self.vision.check_geometry(np.zeros((720, 1280, 3), dtype=np.uint8))

    def test_player_motion_measures_only_the_configured_anchor_region(self):
        previous = np.zeros((1080, 1920, 3), dtype=np.uint8)
        current = previous.copy()
        current[480:490, 910:920] = 255
        self.assertEqual(self.vision.player_motion(previous, current), 100)
        self.assertEqual(self.vision.player_motion(previous, previous), 0)

    def test_busy_message_template_counts_repeated_chat_lines(self):
        image = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        glyphs = self.vision.busy_message_template == 255
        for top in (880, 894):
            line = image[top:top + glyphs.shape[0], 56:56 + glyphs.shape[1]]
            line[glyphs] = 0
        self.assertEqual(self.vision.busy_message_count(image), 2)
        self.assertEqual(
            self.vision.busy_message_count(np.full_like(image, 255)),
            0,
        )

    def test_loot_grid_spirals_outward_from_kill_center(self):
        center = (1036, 554)
        points = list(self.vision.loot_grid(center, radius=130, fine_radius=48))
        phases = [phase for _, phase in points]
        fine = [point for point, phase in points if phase == "fine"]
        coarse = [point for point, phase in points if phase == "coarse"]
        self.assertEqual(points[0], (center, "fine"))
        self.assertTrue(fine)
        self.assertTrue(coarse)
        self.assertEqual(phases, ["fine"] * len(fine) + ["coarse"] * len(coarse))
        self.assertEqual(len(points), len(set(point for point, _ in points)))
        self.assertTrue(all(
            (x - center[0]) ** 2 + (y - center[1]) ** 2 <= 48 ** 2 for x, y in fine
        ))
        self.assertTrue(all(
            48 ** 2 < (x - center[0]) ** 2 + (y - center[1]) ** 2 <= 130 ** 2
            for x, y in coarse
        ))
        distances = [
            np.hypot(point[0] - center[0], point[1] - center[1])
            for point, _ in points
        ]
        self.assertTrue(all(
            current + 1 >= previous for previous, current in zip(distances, distances[1:])
        ))
        self.assertGreater(distances[-1], 120)
        self.assertTrue({
            -1, 1,
        }.issubset({int(np.sign(point[0] - center[0])) for point, _ in points}))
        self.assertTrue({
            -1, 1,
        }.issubset({int(np.sign(point[1] - center[1])) for point, _ in points}))

    def test_loot_candidates_use_bright_low_saturation_blobs_inside_radius(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        center = (940, 578)
        image[570:587, 962:979] = 220
        image[574:577, 968:971] = (220, 20, 20)
        image[570:587, 1020:1037] = (0, 0, 255)
        image[570:587, 1085:1102] = 220
        candidates = self.vision.loot_candidates(image, center, radius=130)
        self.assertEqual(len(candidates), 1)
        point, phase = candidates[0]
        self.assertLessEqual(abs(point[0] - 970), 5)
        self.assertLessEqual(abs(point[1] - 578), 5)
        self.assertEqual(phase, "fine")

    def test_loot_candidates_match_the_reference_cowhide_sprite(self):
        image = self.image("take_cowhide")
        candidates = self.vision.loot_candidates(image, (982, 519), radius=130)
        self.assertTrue(any(
            abs(point[0] - 980) <= 5 and abs(point[1] - 517) <= 5
            for point, _ in candidates
        ))

    def test_loot_candidates_allows_partially_occluded_cowhide_near_center(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        center = (940, 520)
        image[500:508, 936:945] = 220
        image[502:505, 939:942] = (220, 20, 20)
        candidates = self.vision.loot_candidates(image, center, radius=130, fine_radius=48)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0][1], "fine")

    def test_loot_search_uses_kill_point_and_logs_scan_metrics(self):
        vision = Mock()
        vision.settings = {"loot_radius": 130, "loot_fine_radius": 48}
        vision.loot_candidates.return_value = []
        vision.loot_grid.return_value = iter([
            ((1036, 554), "fine"),
            ((1060, 554), "coarse"),
        ])
        vision.hover.return_value = "Walk here"
        desktop, audit = Mock(), Mock()
        desktop.vision = vision
        desktop.fresh.return_value = Frame(1, 1, np.zeros((1080, 1920, 3), dtype=np.uint8))
        bot = Bot(desktop, Mode.BONES_ONLY, None, audit)
        bot.last_kill_point = (1036, 554)
        with patch("cow_desktop.stable_inventory", return_value=inventory()):
            with patch("cow_desktop.time.monotonic", return_value=0):
                bot.loot()
        vision.loot_grid.assert_called_once_with((1036, 554), 130, 48)
        event, frame = audit.event.call_args.args
        metrics = audit.event.call_args.kwargs
        self.assertEqual(event, "LOOT_SEARCH_FINISHED")
        self.assertEqual(frame.number, 1)
        self.assertEqual(metrics["center"], [1036, 554])
        self.assertEqual(metrics["scanned_points"], 2)
        self.assertEqual(metrics["fine_points"], 1)
        self.assertEqual(metrics["coarse_points"], 1)
        self.assertEqual(metrics["color_candidates"], 0)
        self.assertEqual(metrics["hover_matches"], 0)
        self.assertEqual(metrics["farthest_distance"], 24)
        self.assertEqual(metrics["stop_reason"], "region_exhausted")

    def test_cowhide_only_stops_after_verified_pickup(self):
        vision = Mock()
        vision.settings = {"loot_radius": 130, "loot_fine_radius": 48}
        vision.loot_candidates.return_value = [((1036, 554), "fine")]
        vision.loot_grid.return_value = iter([
            ((1036, 554), "fine"),
            ((1048, 554), "fine"),
        ])
        vision.hover.return_value = "Take Cowhide"
        vision.inventory.side_effect = [inventory(1), inventory(1), inventory(1)]
        desktop, audit = Mock(), Mock()
        desktop.vision = vision
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        desktop.fresh.side_effect = [
            Frame(1, .1, image),
            Frame(2, .2, image),
            Frame(3, .3, image),
            Frame(4, .4, image),
            Frame(5, .5, image),
            Frame(6, .6, image),
        ]
        desktop.pick_menu.return_value = "Cowhide"
        bot = Bot(desktop, Mode.COWHIDE_ONLY, None, audit)
        bot.last_kill_point = (1036, 554)
        with patch("cow_desktop.stable_inventory", side_effect=[inventory(), inventory()]):
            with patch("cow_desktop.time.monotonic", return_value=0):
                bot.loot()

        self.assertEqual(vision.hover.call_count, 1)
        vision.loot_grid.assert_not_called()
        desktop.move_loot.assert_called_once_with((1036, 554))
        desktop.pick_menu.assert_called_once_with(
            (1036, 554), "Take Cowhide", ("Cowhide",),
        )
        self.assertEqual(bot.pickups, 1)
        finish_calls = [
            call for call in audit.event.call_args_list
            if call.args and call.args[0] == "LOOT_SEARCH_FINISHED"
        ]
        self.assertEqual(len(finish_calls), 1)
        self.assertEqual(finish_calls[0].kwargs["scanned_points"], 1)
        self.assertEqual(finish_calls[0].kwargs["stop_reason"], "target_collected")

    def test_cowhide_search_rechecks_frames_for_a_late_color_candidate(self):
        vision = Mock()
        vision.settings = {"loot_radius": 130, "loot_fine_radius": 48}
        vision.loot_candidates.side_effect = [
            [],
            [((1036, 554), "fine")],
        ]
        vision.hover.return_value = "Take Cowhide"
        vision.inventory.side_effect = [inventory(1), inventory(1), inventory(1)]
        desktop, audit = Mock(), Mock()
        desktop.vision = vision
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        desktop.fresh.side_effect = [
            Frame(1, .1, image),
            Frame(2, .2, image),
            Frame(3, .3, image),
            Frame(4, .4, image),
            Frame(5, .5, image),
            Frame(6, .6, image),
            Frame(7, 25.1, image),
        ]
        desktop.pick_menu.return_value = "Cowhide"
        bot = Bot(desktop, Mode.COWHIDE_ONLY, None, audit)
        bot.last_kill_point = (1036, 554)
        with patch("cow_desktop.stable_inventory", side_effect=[inventory(), inventory()]):
            with patch("cow_desktop.time.monotonic", return_value=0):
                bot.loot()

        self.assertEqual(bot.pickups, 1)
        desktop.pick_menu.assert_called_once_with(
            (1036, 554), "Take Cowhide", ("Cowhide",),
        )
        vision.loot_grid.assert_not_called()
        finish_call = next(
            call for call in audit.event.call_args_list
            if call.args and call.args[0] == "LOOT_SEARCH_FINISHED"
        )
        self.assertEqual(finish_call.kwargs["color_candidates"], 1)
        self.assertEqual(finish_call.kwargs["color_candidate_points"], [[1036, 554]])

    def test_cowhide_search_probes_above_player_when_color_mask_misses_occluded_hide(self):
        vision = Mock()
        vision.settings = {
            "loot_radius": 130,
            "loot_fine_radius": 48,
            "player_anchor": [942, 520],
            "screen": [1920, 1080],
        }
        vision.loot_candidates.return_value = []
        vision.hover.return_value = "Take Cowhide"
        vision.inventory.side_effect = [inventory(1), inventory(1), inventory(1)]
        desktop, audit = Mock(), Mock()
        desktop.vision = vision
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        desktop.fresh.side_effect = [
            Frame(1, .1, image),
            Frame(2, 2.2, image),
            Frame(3, 2.3, image),
            Frame(4, 2.4, image),
            Frame(5, 2.5, image),
            Frame(6, 2.6, image),
        ]
        desktop.pick_menu.return_value = "Cowhide"
        bot = Bot(desktop, Mode.COWHIDE_ONLY, None, audit)
        bot.last_kill_point = (964, 458)
        with patch("cow_desktop.stable_inventory", side_effect=[inventory(), inventory()]):
            with patch("cow_desktop.time.monotonic", return_value=0):
                bot.loot()

        self.assertEqual(bot.pickups, 1)
        desktop.move_loot.assert_called_once_with((942, 504))
        desktop.pick_menu.assert_called_once_with(
            (942, 504), "Take Cowhide", ("Cowhide",),
        )
        vision.loot_grid.assert_not_called()
        finish_call = next(
            call for call in audit.event.call_args_list
            if call.args and call.args[0] == "LOOT_SEARCH_FINISHED"
        )
        self.assertEqual(finish_call.kwargs["stop_reason"], "target_collected")

    def test_cowhide_mode_ignores_bones_hover_candidates(self):
        vision = Mock()
        vision.settings = {"loot_radius": 130}
        vision.loot_candidates.return_value = [((1036, 554), "fine")]
        vision.loot_grid.return_value = iter([((1036, 554), "fine")])
        vision.hover.return_value = "Take Bones"
        desktop, audit = Mock(), Mock()
        desktop.vision = vision
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        desktop.fresh.side_effect = [
            Frame(1, .1, image),
            Frame(2, .2, image),
            Frame(3, 25.1, image),
        ]
        bot = Bot(desktop, Mode.COWHIDE_ONLY, None, audit)
        bot.last_kill_point = (1036, 554)
        with patch("cow_desktop.stable_inventory", return_value=inventory()):
            with patch("cow_desktop.time.monotonic", return_value=0):
                bot.loot()

        desktop.pick_menu.assert_not_called()
        vision.loot_grid.assert_not_called()
        metrics = audit.event.call_args.kwargs
        self.assertEqual(metrics["hover_matches"], 0)
        self.assertEqual(metrics["ignored_item_matches"], 1)

    def test_search_continues_after_five_consecutive_failed_attacks(self):
        desktop, audit = Mock(), Mock()
        desktop.vision = self.vision
        desktop.fresh.return_value = Frame(
            1, 1, np.zeros((1080, 1920, 3), dtype=np.uint8),
        )
        bot = Bot(desktop, Mode.COMBAT_ONLY, None, audit)
        attempts = {"count": 0}

        def attack():
            attempts["count"] += 1
            if attempts["count"] <= 5:
                return False
            bot.kills += 1
            return True

        bot.attack = Mock(side_effect=attack)
        bot.loot = Mock()
        with patch("cow_desktop.stable_inventory", return_value=inventory()):
            bot.run(1)

        self.assertEqual(attempts["count"], 6)
        self.assertEqual(
            [call.kwargs["failures"] for call in audit.event.call_args_list
             if call.args and call.args[0] == "SEARCH_OR_ATTACK_FAILED"],
            [1, 2, 3, 4, 5],
        )
        bot.loot.assert_called_once()

    def test_attack_search_gets_its_full_budget_after_hud_clear(self):
        vision = Mock()
        vision.settings = {"player_anchor": [942, 520]}
        vision.grid.return_value = iter([(1036, 554)])
        vision.hud_clear.side_effect = [False, True, True]
        vision.hover.return_value = "Attack Cow (level-2)"
        vision.busy_message_count.return_value = 0
        vision.player_motion.side_effect = [100, 100, 100] + [0] * 7
        vision.health.side_effect = [Health("Cow", True)] * 10 + [
            Health("Cow", False), Health("Cow", False),
        ]
        desktop, audit = Mock(), Mock()
        desktop.vision = vision
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frames = [
            Frame(1, .1, image),
            Frame(2, 7.5, image),
            Frame(3, 15.1, image),
            Frame(4, 15.2, image),
            Frame(5, 20.3, image),
            Frame(6, 25.1, image),
            Frame(7, 25.2, image),
            Frame(8, 25.3, image),
            Frame(9, 25.4, image),
            Frame(10, 25.5, image),
            Frame(11, 25.6, image),
            Frame(12, 25.7, image),
            Frame(13, 25.8, image),
            Frame(14, 25.9, image),
            Frame(15, 26.0, image),
            Frame(16, 26.1, image),
        ]
        clock = {"now": 0}

        def fresh():
            frame = frames.pop(0)
            clock["now"] = frame.at
            return frame

        desktop.fresh.side_effect = fresh
        desktop.guarded_click.return_value = True
        bot = Bot(desktop, Mode.COMBAT_ONLY, None, audit)
        with patch("cow_desktop.time.monotonic", side_effect=lambda: clock["now"]):
            self.assertTrue(bot.attack())

        desktop.guarded_click.assert_called_once_with(
            (1036, 554), "Attack Cow (level-2)", idle=True,
        )
        self.assertEqual(bot.kills, 1)
        self.assertIn(
            "PLAYER_STOPPED",
            [call.args[0] for call in audit.event.call_args_list],
        )

    def test_new_busy_message_blacklists_nearby_target_for_later_search(self):
        vision = Mock()
        vision.settings = {"player_anchor": [942, 520]}
        vision.grid.side_effect = lambda center: iter([
            (1036, 554), (1060, 554),
        ])
        vision.hud_clear.return_value = True
        vision.hover.return_value = "Attack Cow (level-2)"
        vision.busy_message_count.side_effect = [0, 1]
        desktop, audit = Mock(), Mock()
        desktop.vision = vision
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        desktop.fresh.side_effect = [
            Frame(1, .1, image),
            Frame(2, .2, image),
            Frame(3, .3, image),
            Frame(4, .4, image),
            Frame(5, .5, image),
            Frame(6, .6, image),
        ]
        desktop.guarded_click.return_value = True
        bot = Bot(desktop, Mode.COMBAT_ONLY, None, audit)
        with patch("cow_desktop.time.monotonic", return_value=0):
            self.assertFalse(bot.attack())
            self.assertFalse(bot.attack())

        desktop.guarded_click.assert_called_once()
        self.assertEqual(bot.blacklisted_targets, [((1036, 554), 20)])
        self.assertIn(
            "TARGET_BLACKLISTED",
            [call.args[0] for call in audit.event.call_args_list],
        )

    def test_attack_miss_without_a_new_busy_message_does_not_blacklist(self):
        vision = Mock()
        vision.settings = {"player_anchor": [942, 520]}
        vision.grid.return_value = iter([(1036, 554)])
        vision.hud_clear.return_value = True
        vision.hover.return_value = "Attack Cow (level-2)"
        vision.target_in_combat.return_value = False
        vision.busy_message_count.return_value = 0
        vision.busy_message_added.return_value = False
        vision.player_motion.return_value = 0
        vision.health.return_value = None
        desktop, audit = Mock(), Mock()
        desktop.vision = vision
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        desktop.fresh.side_effect = [
            Frame(1, .1, image),
            Frame(2, .2, image),
            Frame(3, .3, image),
            *[Frame(number, number / 10, image) for number in range(4, 11)],
            Frame(11, 11.1, image),
        ]
        desktop.guarded_click.return_value = True
        bot = Bot(desktop, Mode.COMBAT_ONLY, None, audit)
        with patch("cow_desktop.time.monotonic", return_value=0):
            self.assertFalse(bot.attack())

        self.assertEqual(bot.blacklisted_targets, [])
        self.assertIn(
            "ATTACK_MISSED",
            [call.args[0] for call in audit.event.call_args_list],
        )

    def test_target_in_combat_detects_overhead_health_bar(self):
        busy = load_image(
            ROOT / "cow_runs" / "20261117_011210_170702" / "00002_ATTACK_CLICK.png",
        )
        self.assertTrue(self.vision.target_in_combat(busy, (940, 482)))
        self.assertFalse(self.vision.target_in_combat(self.image("attack_cow"), (940, 482)))

    def test_attack_skips_and_blacklists_target_already_in_combat_without_clicking(self):
        vision = Mock()
        vision.settings = {"player_anchor": [942, 520]}
        vision.grid.side_effect = lambda center: iter([(1036, 554), (1120, 554)])
        vision.hud_clear.return_value = True
        vision.hover.return_value = "Attack Cow (level-2)"
        vision.target_in_combat.side_effect = [True, False]
        vision.busy_message_count.return_value = 0
        vision.busy_message_added.return_value = False
        vision.player_motion.return_value = 0
        vision.health.side_effect = [Health("Cow", True)] * 7 + [
            Health("Cow", False), Health("Cow", False),
        ]
        desktop, audit = Mock(), Mock()
        desktop.vision = vision
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        desktop.fresh.side_effect = [
            Frame(number, number / 10, image) for number in range(1, 20)
        ]
        desktop.guarded_click.return_value = True
        bot = Bot(desktop, Mode.COMBAT_ONLY, None, audit)
        with patch("cow_desktop.time.monotonic", return_value=0):
            self.assertTrue(bot.attack())

        desktop.guarded_click.assert_called_once_with(
            (1120, 554), "Attack Cow (level-2)", idle=True,
        )
        self.assertEqual(bot.blacklisted_targets, [((1036, 554), 20)])
        self.assertIn(
            "TARGET_BLACKLISTED",
            [call.args[0] for call in audit.event.call_args_list],
        )

    def test_busy_message_added_detects_scrolled_full_chat(self):
        base = load_image(
            ROOT / "cow_runs" / "20261117_011210_170702" / "00002_ATTACK_CLICK.png",
        )
        scrolled = base.copy()
        x, y, w, h = (0, 870, 520, 130)
        chat = scrolled[y:y + h, x:x + w]
        rows = self.vision.busy_message_rows(base)
        self.assertEqual(len(rows), 4)
        row_h = self.vision.busy_message_template.shape[0]
        last_row = rows[-1]
        # Scroll visible busy lines up by one line height and place a fresh copy at the bottom.
        step = rows[1] - rows[0]
        chat[rows[0]:last_row + row_h - step] = chat[rows[0] + step:last_row + row_h]
        chat[last_row - 1:last_row - 1 + row_h] = chat[last_row:last_row + row_h]
        # Shift timestamp pixel slightly to simulate a new minute/scroll difference on the last row.
        chat[last_row:last_row + 2, 12:18] = 255
        self.assertFalse(self.vision.busy_message_added(base, base))
        self.assertTrue(self.vision.busy_message_added(base, scrolled))

    def test_found_cowhide_is_processed_even_if_search_budget_expires(self):
        vision = Mock()
        vision.settings = {"loot_radius": 130}
        vision.loot_candidates.return_value = [((1036, 554), "fine")]
        vision.loot_grid.return_value = iter([((1036, 554), "fine")])
        clock = {"now": 0}

        def find_cowhide(image):
            clock["now"] = 26
            return "Take Cowhide"

        vision.hover.side_effect = find_cowhide
        vision.inventory.side_effect = [inventory(1), inventory(1), inventory(1)]
        desktop, audit = Mock(), Mock()
        desktop.vision = vision
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        desktop.fresh.side_effect = [
            Frame(1, .1, image),
            Frame(2, .2, image),
            Frame(3, 26.1, image),
            Frame(4, 26.2, image),
            Frame(5, 26.3, image),
        ]
        desktop.pick_menu.return_value = "Cowhide"
        bot = Bot(desktop, Mode.COWHIDE_ONLY, None, audit)
        bot.last_kill_point = (1036, 554)
        with patch("cow_desktop.stable_inventory", return_value=inventory()):
            with patch("cow_desktop.time.monotonic", side_effect=lambda: clock["now"]):
                bot.loot()

        desktop.pick_menu.assert_called_once_with(
            (1036, 554), "Take Cowhide", ("Cowhide",),
        )
        self.assertEqual(bot.pickups, 1)
        states = [call.args[0] for call in audit.event.call_args_list]
        self.assertIn("PICKUP_VERIFIED", states)
        self.assertEqual(states[-1], "LOOT_SEARCH_FINISHED")

    def test_loot_search_does_not_inspect_a_point_after_time_budget(self):
        vision = Mock()
        vision.settings = {"loot_radius": 130}
        vision.loot_candidates.return_value = []
        vision.loot_grid.return_value = iter([((1036, 554), "fine")])
        vision.hover.return_value = "Walk here"
        desktop, audit = Mock(), Mock()
        desktop.vision = vision
        desktop.fresh.return_value = Frame(
            1, .1, np.zeros((1080, 1920, 3), dtype=np.uint8),
        )
        clock = {"now": 0}
        desktop.move.side_effect = lambda point: clock.update(now=26)
        bot = Bot(desktop, Mode.BONES_ONLY, None, audit)
        bot.last_kill_point = (1036, 554)
        with patch("cow_desktop.stable_inventory", return_value=inventory()):
            with patch("cow_desktop.time.monotonic", side_effect=lambda: clock["now"]):
                bot.loot()
        self.assertEqual(desktop.fresh.call_count, 2)
        vision.hover.assert_not_called()
        metrics = audit.event.call_args.kwargs
        self.assertEqual(metrics["scanned_points"], 0)
        self.assertEqual(metrics["stop_reason"], "time_budget")
        self.assertEqual(metrics["elapsed_seconds"], 26)

    def fake_desktop(self, final):
        desktop = object.__new__(Desktop)
        desktop.vision = self.vision
        desktop.move = Mock()
        desktop._input_check = Mock()
        desktop.snapshot = Mock(return_value=final)
        desktop.mouse = Mock()
        image = self.image("attack_cow")
        desktop.fresh = Mock(side_effect=[Frame(1, 1, image), Frame(2, 2, image)])
        return desktop

    def test_loot_move_overshoots_then_corrects_to_target(self):
        desktop = object.__new__(Desktop)
        desktop.vision = Mock(settings={"screen": (1920, 1080)})
        desktop.cursor = (500, 500)
        desktop.mouse = Mock()
        desktop.move = Mock()
        with patch(
            "cow_desktop.random.uniform",
            side_effect=[2.0, 0.0, 0.12, 0.14, 0.10],
        ):
            desktop.move_loot((600, 550))

        calls = desktop.move.call_args_list
        self.assertEqual(len(calls), 3)
        overshoot_point = calls[1].args[0]
        self.assertGreater(overshoot_point[0], 600)
        self.assertGreater(overshoot_point[1], 550)
        self.assertEqual(calls[-1].args[0], (600, 550))

    def test_cowhide_pick_menu_uses_bounded_normal_reaction_delay(self):
        desktop = self.fake_desktop(self.image("attack_cow"))
        hover, menu = self.image("take_cowhide"), self.image("rightclick_menu")
        images = [hover, hover, menu, menu, menu]
        desktop.fresh.side_effect = [
            Frame(i, i, image) for i, image in enumerate(images, 1)
        ]
        desktop.snapshot.side_effect = [hover, menu]
        with patch("cow_desktop.random.gauss", return_value=0.1):
            with patch("cow_desktop.time.sleep") as sleep:
                result = desktop.pick_menu((860, 540), "Take Cowhide", ("Cowhide",))

        self.assertEqual(result, "Cowhide")
        sleep.assert_called_once_with(0.2)

    def test_capture_timestamps_frame_before_snapshot(self):
        desktop = object.__new__(Desktop)
        desktop.stop = threading.Event()
        desktop.frames = deque(maxlen=20)
        desktop.condition = threading.Condition()
        desktop.sequence = 0
        desktop.error = None
        clock = {"now": 1.0}

        def snapshot():
            clock["now"] = 2.0
            desktop.stop.set()
            return np.zeros((1080, 1920, 3), dtype=np.uint8)

        desktop.snapshot = snapshot
        with patch("cow_desktop.time.monotonic", side_effect=lambda: clock["now"]):
            desktop._capture()
        self.assertEqual(desktop.frames[0].started_at, 1.0)
        self.assertEqual(desktop.frames[0].at, 2.0)

    def test_final_guard_blocks_changed_target_and_stale_positive_hud(self):
        for final in (self.image("attack_cowcalf"), self.image("attacking_cow")):
            with self.subTest():
                desktop = self.fake_desktop(final)
                self.assertFalse(desktop.guarded_click((900, 500), "Attack Cow (level-2)", idle=True))
                desktop.mouse.click.assert_not_called()

    def test_final_guard_sends_one_click_only_for_unchanged_verified_target(self):
        desktop = self.fake_desktop(self.image("attack_cow"))
        self.assertTrue(desktop.guarded_click((900, 500), "Attack Cow (level-2)", idle=True))
        desktop.mouse.click.assert_called_once_with()

    def test_final_guard_waits_for_transient_hover_text_to_settle(self):
        desktop = object.__new__(Desktop)
        desktop.vision = Mock()
        desktop.move = Mock()
        desktop._input_check = Mock()
        desktop.fresh = Mock(side_effect=[
            Frame(index, index / 20, np.zeros((1080, 1920, 3), dtype=np.uint8))
            for index in range(1, 4)
        ])
        final = np.zeros((1080, 1920, 3), dtype=np.uint8)
        desktop.snapshot = Mock(return_value=final)
        desktop.mouse = Mock()
        desktop.vision.hover.side_effect = [
            "Inventory", "Walk here", "Walk here", "Walk here",
        ]
        mask = np.ones((11, 300), dtype=np.uint8)
        desktop.vision.hover_mask.side_effect = [mask, mask, mask]

        with patch("cow_desktop.time.monotonic", side_effect=[0, .01, .1, .2]):
            with patch("cow_desktop.time.sleep") as sleep:
                self.assertTrue(desktop.guarded_click((713, 671), "Walk here"))

        self.assertEqual(desktop.fresh.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        desktop.mouse.click.assert_called_once_with()

    def test_final_guard_rejects_hover_that_never_matches(self):
        desktop = object.__new__(Desktop)
        desktop.vision = Mock()
        desktop.vision.hover.return_value = "Inventory"
        desktop.move = Mock()
        desktop._input_check = Mock()
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        desktop.fresh = Mock(side_effect=[
            Frame(index, index / 20, image) for index in range(1, 4)
        ])
        desktop.snapshot = Mock(return_value=image)
        desktop.mouse = Mock()

        with patch("cow_desktop.time.monotonic", side_effect=[0, .01, .2, .4, .76]):
            with patch("cow_desktop.time.sleep"):
                self.assertFalse(desktop.guarded_click((713, 671), "Walk here"))

        desktop.snapshot.assert_not_called()
        desktop.mouse.click.assert_not_called()

    def test_route_guard_reacquires_expected_hover_nearby_before_click(self):
        desktop = object.__new__(Desktop)
        desktop.vision = Mock(settings={"screen": (1920, 1080)})
        desktop.vision.hover.return_value = "Walk here"
        desktop.move = Mock()
        desktop.fresh = Mock(return_value=Frame(
            1, 1, np.zeros((1080, 1920, 3), dtype=np.uint8),
        ))
        desktop.guarded_click = Mock(side_effect=[False, True])

        self.assertEqual(
            desktop.guarded_route_click((713, 671), "Walk here"),
            (713, 665),
        )
        self.assertEqual(
            desktop.guarded_click.call_args_list,
            [
                call((713, 671), "Walk here", landmark=None),
                call((713, 665), "Walk here", landmark=None),
            ],
        )

    def test_route_guard_never_clicks_nearby_when_expected_hover_is_absent(self):
        desktop = object.__new__(Desktop)
        desktop.vision = Mock(settings={"screen": (1920, 1080)})
        desktop.vision.hover.return_value = "Take Spinach roll"
        desktop.move = Mock()
        desktop.fresh = Mock(return_value=Frame(
            1, 1, np.zeros((1080, 1920, 3), dtype=np.uint8),
        ))
        desktop.guarded_click = Mock(return_value=False)

        self.assertIsNone(desktop.guarded_route_click((713, 671), "Walk here"))
        desktop.guarded_click.assert_called_once_with(
            (713, 671), "Walk here", landmark=None,
        )

    def test_bones_only_chooses_bones_from_a_cowhide_stack(self):
        desktop = self.fake_desktop(self.image("attack_cow"))
        hover, menu = self.image("take_cowhide"), self.image("rightclick_menu")
        images = [hover, hover, menu, menu, menu]
        desktop.fresh.side_effect = [Frame(i, i, image) for i, image in enumerate(images, 1)]
        desktop.snapshot.side_effect = [hover, menu]
        result = desktop.pick_menu((860, 540), "Take Cowhide", ("Bones",))
        self.assertEqual(result, "Bones")
        self.assertEqual(desktop.mouse.click.call_count, 2)
        self.assertEqual(desktop.mouse.click.call_args_list[0].kwargs, {"button": "right"})
        self.assertEqual(desktop.mouse.click.call_args_list[1].kwargs, {})

    def test_pick_menu_retries_when_first_post_click_frame_has_no_menu(self):
        desktop = self.fake_desktop(self.image("attack_cow"))
        hover, menu = self.image("take_cowhide"), self.image("rightclick_menu")
        images = [hover, hover, self.image("attack_cow"), menu, menu, menu]
        desktop.fresh.side_effect = [
            Frame(i, i, image) for i, image in enumerate(images, 1)
        ]
        desktop.snapshot.side_effect = [hover, menu]
        result = desktop.pick_menu((860, 540), "Take Cowhide", ("Bones",))
        self.assertEqual(result, "Bones")
        self.assertEqual(desktop.mouse.click.call_count, 2)
        self.assertEqual(desktop.mouse.click.call_args_list[0].kwargs, {"button": "right"})
        self.assertEqual(desktop.mouse.click.call_args_list[1].kwargs, {})

    def test_menu_final_guard_does_not_left_click_a_disappeared_item(self):
        desktop = self.fake_desktop(self.image("attack_cow"))
        desktop.key = Mock()
        hover, menu = self.image("take_cowhide"), self.image("rightclick_menu")
        images = [hover, hover, menu, menu, menu]
        desktop.fresh.side_effect = [Frame(i, i, image) for i, image in enumerate(images, 1)]
        desktop.snapshot.side_effect = [hover, hover]
        self.assertIsNone(desktop.pick_menu((860, 540), "Take Cowhide", ("Bones",)))
        desktop.mouse.click.assert_called_once_with(button="right")
        desktop.key.assert_not_called()

    def test_route_timeout_does_not_send_another_click(self):
        origin = crop(self.image("gate_closed"), (1000, 180, 40, 40))
        before = self.image("gate_closed")
        arrival = np.full((24, 24, 3), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            from cow_vision import save_image
            save_image(path / "origin.png", origin)
            save_image(path / "arrival.png", arrival)
            step = {
                "origin": "origin.png", "arrival": "arrival.png", "arrival_roi": [700, 500, 24, 24],
                "offset": [0, 0], "search": [520, 110, 1120, 750], "hover": "Open Gate",
                "timeout": 1,
            }
            (path / "route.json").write_text(json.dumps({
                "screen": [1920, 1080], "to_bank": [step], "to_cows": [step],
            }))
            route = Route(path / "route.json", self.vision)
            desktop = Mock()
            desktop.fresh.return_value = Frame(1, 1, before)
            desktop.hover_is_stable.return_value = False
            desktop.guarded_route_click.return_value = (1000, 180)
            with patch("cow_desktop.locate", return_value=(1000, 180)):
                with patch("cow_desktop.time.monotonic", side_effect=[0, .1, 2]):
                    with self.assertRaisesRegex(SafetyError, "timed out"):
                        route.run("to_bank", desktop, Mock())
            desktop.guarded_route_click.assert_called_once()

    def test_open_gate_step_is_skipped_when_close_gate_hover_is_stable(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        gate_origin = np.full((24, 24, 3), 100, dtype=np.uint8)
        walk_origin = np.full((24, 24, 3), 150, dtype=np.uint8)
        arrival = np.full((24, 24, 3), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name, saved in (
                ("gate_origin.png", gate_origin), ("walk_origin.png", walk_origin),
                ("arrival.png", arrival),
            ):
                save_image(path / name, saved)
            shared = {
                "arrival": "arrival.png", "arrival_roi": [700, 500, 24, 24],
                "offset": [0, 0], "search": [0, 40, 1676, 830], "timeout": 1,
            }
            gate_step = {**shared, "origin": "gate_origin.png", "hover": "Open Gate"}
            walk_step = {**shared, "origin": "walk_origin.png", "hover": "Walk here"}
            route_path = path / "route.json"
            route_path.write_text(json.dumps({
                "screen": [1920, 1080],
                "to_bank": [gate_step, walk_step], "to_cows": [walk_step],
            }))
            route = Route(route_path, self.vision)
            desktop = Mock()
            desktop.fresh.return_value = Frame(1, 1, image)
            desktop.hover_is_stable.return_value = True
            desktop.guarded_route_click.return_value = (200, 200)
            audit = Mock()

            with patch(
                "cow_desktop.locate",
                side_effect=[(100, 100), (100, 100), (200, 200), (200, 200)],
            ):
                with patch("cow_desktop.time.monotonic", side_effect=[0, 2]):
                    with self.assertRaisesRegex(SafetyError, "timed out"):
                        route.run("to_bank", desktop, audit)

            desktop.hover_is_stable.assert_called_once()
            hover_args = desktop.hover_is_stable.call_args.args
            self.assertEqual(hover_args[:2], ((100, 100), "Close Gate"))
            self.assertEqual(hover_args[2][0], (100, 100, 24, 24))
            np.testing.assert_array_equal(hover_args[2][1], gate_origin)
            desktop.guarded_route_click.assert_called_once()
            click_args = desktop.guarded_route_click.call_args
            self.assertEqual(click_args.args, ((200, 200), "Walk here"))
            self.assertEqual(
                click_args.kwargs["landmark"][0], (200, 200, 24, 24),
            )
            np.testing.assert_array_equal(
                click_args.kwargs["landmark"][1], walk_origin,
            )
            audit.event.assert_any_call(
                "ROUTE_GATE_ALREADY_OPEN", desktop.fresh.return_value,
                direction="to_bank", step=0,
            )

    def test_stable_hover_requires_three_identical_close_gate_readings(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        desktop = object.__new__(Desktop)
        desktop.vision = Mock()
        desktop.move = Mock()
        desktop.fresh = Mock(return_value=Frame(1, 1, image))
        desktop.snapshot = Mock(return_value=image)
        desktop._input_check = Mock()
        mask = np.ones((11, 80), dtype=bool)
        desktop.vision.hover_mask.return_value = mask
        desktop.vision.hover.side_effect = ["Close Gate", "Close Gate", "Close Gate"]

        self.assertTrue(desktop.hover_is_stable((500, 400), "Close Gate"))

        desktop.vision.hover.side_effect = ["Close Gate", "Open Gate"]
        self.assertFalse(desktop.hover_is_stable((500, 400), "Close Gate"))

    def test_registered_arrival_confirmations_survive_skipped_frames(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        origin = np.full((24, 24, 3), 100, dtype=np.uint8)
        arrival = np.full((24, 24, 3), 200, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name, saved in (
                ("before.png", image), ("arrival_screenshot.png", image),
                ("origin.png", origin), ("arrival.png", arrival),
            ):
                save_image(path / name, saved)
            step = {
                "origin": "origin.png", "arrival": "arrival.png",
                "reference": "before.png", "arrival_screenshot": "arrival_screenshot.png",
                "arrival_roi": [700, 500, 24, 24], "offset": [0, 0],
                "search": [0, 40, 1676, 830], "hover": "Walk here", "timeout": 1,
            }
            route_path = path / "route.json"
            route_path.write_text(json.dumps({
                "screen": [1920, 1080], "to_bank": [step], "to_cows": [step],
            }))
            route = Route(route_path, self.vision)
            timestamps = [0, 0.01, 0.02, 0.05, 0.10, 0.16, 0.21, 0.26, 0.37]
            frames = [
                Frame(index, at, image) for index, at in enumerate(timestamps, 1)
            ]
            desktop = Mock()
            desktop.fresh.side_effect = frames
            desktop.guarded_route_click.return_value = (500, 300)
            audit = Mock()

            with patch("cow_desktop.locate", return_value=(500, 300)):
                with patch("cow_desktop.registered_route_arrival", return_value=True) as aligned:
                    with patch("cow_desktop.time.monotonic", return_value=0):
                        route.run("to_bank", desktop, audit)

            self.assertEqual(aligned.call_count, 3)
            audit.event.assert_any_call(
                "ROUTE_ARRIVED", frames[8],
                direction="to_bank", step=0,
            )

    def test_route_scene_alignment_projects_points_and_arrival_roi(self):
        rng = np.random.default_rng(42)
        reference = rng.integers(0, 256, (420, 520, 3), dtype=np.uint8)
        transform = np.float32([
            [1.05, 0.02, 12], [-0.01, 1.10, 8], [0, 0, 1],
        ])
        current = cv2.warpPerspective(reference, transform, (520, 420))
        rect = [0, 0, 520, 420]

        matrix = route_scene_homography(reference, current, rect)
        self.assertIsNotNone(matrix)
        actual = project_route_point(matrix, (260, 210), rect)
        expected = cv2.perspectiveTransform(
            np.float32([[[260, 210]]]), transform,
        )[0, 0]
        self.assertLess(np.linalg.norm(np.asarray(actual) - expected), 3)
        self.assertTrue(registered_route_arrival(
            reference, current, [200, 150, 100, 80], rect,
        ))

    def test_recalibrate_route_step_changes_only_the_selected_step(self):
        rng = np.random.default_rng(42)
        before = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        after = before.copy()
        after[100:116, 100:124] = rng.integers(
            0, 256, (16, 24, 3), dtype=np.uint8,
        )
        origin = np.full((20, 20, 3), 80, dtype=np.uint8)
        arrival = np.full((20, 20, 3), 120, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile"
            profile.mkdir()
            (profile / "font.json").write_text("{}", encoding="utf-8")
            save_image(root / "origin.png", origin)
            save_image(root / "arrival.png", arrival)
            step = {
                "origin": "origin.png", "arrival": "arrival.png",
                "arrival_roi": [300, 300, 20, 20], "offset": [10, 10],
                "search": [0, 40, 1676, 830], "hover": "Walk here", "timeout": 15,
            }
            route_path = root / "route.json"
            original = {
                "screen": [1920, 1080],
                "to_bank": [dict(step, label=f"out-{index}") for index in range(4)],
                "to_cows": [dict(step, label="return")],
            }
            route_path.write_text(json.dumps(original), encoding="utf-8")

            events = []
            desktop = Mock()
            desktop.snapshot.return_value = before
            with patch("cow_bot.Vision", return_value=self.vision):
                with patch("cow_bot.PROFILE", profile):
                    with patch("cow_desktop.Desktop", return_value=desktop):
                        with patch(
                            "cow_bot.countdown",
                            side_effect=lambda *args: events.append("countdown"),
                        ):
                            with patch(
                                "cow_bot.verify_route_step_start",
                                side_effect=lambda *args: events.append("verify"),
                            ):
                                with patch(
                                    "cow_bot.verify_route_step_arrival",
                                    side_effect=lambda *args: events.append("verify_arrival"),
                                ):
                                    with patch("builtins.input", return_value=""):
                                        with patch(
                                            "cow_bot.countdown_snapshot",
                                            side_effect=lambda *args: (
                                                events.append("arrival_snapshot"), after,
                                            )[1],
                                        ):
                                            with patch(
                                                "cow_bot.select_rect",
                                                side_effect=[
                                                    [10, 10, 40, 30], [100, 100, 24, 16],
                                                ],
                                            ):
                                                with patch(
                                                    "cow_bot.select_point", return_value=(900, 800),
                                                ) as select_destination:
                                                    with patch(
                                                        "cow_bot.route_action",
                                                        return_value=("Walk here", None),
                                                    ):
                                                        with patch(
                                                            "cow_bot.arrival_timeout",
                                                            return_value=15,
                                                        ):
                                                            recalibrate_route_step(
                                                                route_path, "to_bank", 4,
                                                            )

            self.assertEqual(events[:2], ["countdown", "verify"])
            self.assertLess(events.index("arrival_snapshot"), events.index("verify_arrival"))
            updated = json.loads(route_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["to_bank"][:3], original["to_bank"][:3])
            self.assertEqual(updated["to_cows"], original["to_cows"])
            self.assertNotEqual(updated["to_bank"][3], original["to_bank"][3])
            self.assertEqual(updated["to_bank"][3]["offset"], [890, 790])
            self.assertTrue((root / updated["to_bank"][3]["origin"]).is_file())
            self.assertTrue((root / updated["to_bank"][3]["arrival"]).is_file())
            select_destination.assert_called_once_with(before, (820, 740, 370, 180))
            Route(route_path, self.vision)

    def test_step_recalibration_alerts_only_after_previous_arrival_is_verified(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        arrival = np.full((16, 24, 3), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_image(path / "previous_arrival.png", arrival)
            save_image(path / "previous_arrival_screenshot.png", image)
            step = {
                "origin": "unused.png", "arrival": "previous_arrival.png",
                "arrival_screenshot": "previous_arrival_screenshot.png",
                "arrival_roi": [100, 100, 24, 16], "search": [0, 40, 1676, 830],
            }
            data = {"to_bank": [step, step, step, step]}
            desktop = Mock()
            desktop.snapshot.return_value = image
            output = io.StringIO()

            with patch("cow_desktop.registered_route_arrival", return_value=True) as matches:
                with patch("cow_bot.time.sleep"):
                    with contextlib.redirect_stdout(output):
                        verify_route_step_start(path / "route.json", data, "to_bank", 4, desktop)

            self.assertEqual(matches.call_count, 3)
            self.assertIn("ALERT: 已確認在 to_bank 第 4 步的正確起點", output.getvalue())

    def test_step_recalibration_refuses_unverified_start_without_alert(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        arrival = np.full((16, 24, 3), 255, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_image(path / "previous_arrival.png", arrival)
            save_image(path / "previous_arrival_screenshot.png", image)
            step = {
                "origin": "unused.png", "arrival": "previous_arrival.png",
                "arrival_screenshot": "previous_arrival_screenshot.png",
                "arrival_roi": [100, 100, 24, 16], "search": [0, 40, 1676, 830],
            }
            data = {"to_bank": [step, step, step, step]}
            desktop = Mock()
            desktop.snapshot.return_value = image
            output = io.StringIO()

            with patch("cow_desktop.registered_route_arrival", return_value=False):
                with patch("cow_bot.ROUTE_START_VERIFY_TIMEOUT_SECONDS", 0.1):
                    with patch("cow_bot.time.monotonic", side_effect=[0, 0, 1]):
                        with patch("cow_bot.time.sleep"):
                            with contextlib.redirect_stdout(output):
                                with self.assertRaisesRegex(
                                    SafetyError, "could not be verified",
                                ):
                                    verify_route_step_start(
                                        path / "route.json", data, "to_bank", 4, desktop,
                                    )

            self.assertNotIn("ALERT:", output.getvalue())

    def test_step_recalibration_alerts_after_three_arrival_landmark_matches(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        roi = [100, 100, 24, 16]
        arrival = np.full((16, 24, 3), 255, dtype=np.uint8)
        image[100:116, 100:124] = arrival
        desktop = Mock()
        desktop.snapshot.return_value = image
        output = io.StringIO()

        with patch("cow_desktop.registered_route_arrival", return_value=False) as registered:
            with patch("cow_bot.time.sleep"):
                with contextlib.redirect_stdout(output):
                    verify_route_step_arrival(
                        "to_bank", 3, image, roi, arrival, [0, 40, 1676, 830], desktop,
                    )

        self.assertEqual(desktop.snapshot.call_count, 3)
        registered.assert_not_called()
        self.assertIn(
            "ALERT: 已確認 to_bank 第 3 步的抵達地標，連續 3 次畫面符合",
            output.getvalue(),
        )
        self.assertIn("不代表角色位於原畫面的綠色點選框", output.getvalue())

    def test_step_recalibration_refuses_unverified_arrival_without_alert(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        arrival = np.full((16, 24, 3), 255, dtype=np.uint8)
        desktop = Mock()
        desktop.snapshot.return_value = image
        output = io.StringIO()

        with patch("cow_desktop.registered_route_arrival", return_value=False):
            with patch("cow_bot.ROUTE_ARRIVAL_VERIFY_TIMEOUT_SECONDS", 0.1):
                with patch("cow_bot.time.monotonic", side_effect=[0, 0, 1]):
                    with patch("cow_bot.time.sleep"):
                        with contextlib.redirect_stdout(output):
                            with self.assertRaisesRegex(
                                SafetyError, "Arrival landmark .* could not be verified",
                            ):
                                verify_route_step_arrival(
                                    "to_bank", 3, image, [100, 100, 24, 16], arrival,
                                    [0, 40, 1676, 830], desktop,
                                )

        self.assertNotIn("ALERT: 已確認", output.getvalue())

    def test_registered_route_target_waits_until_projected_point_stabilizes(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frames = [Frame(index, index / 20, image) for index in range(1, 6)]
        desktop = Mock()
        desktop.fresh.side_effect = frames
        desktop.move = Mock()
        projected = [(100, 100), (118, 100), (119, 101), (120, 100), (121, 100)]

        with patch("cow_desktop.route_scene_homography", return_value=np.eye(3)):
            with patch("cow_desktop.project_route_point", side_effect=projected):
                with patch("cow_desktop.time.monotonic", return_value=0):
                    point, check = stable_route_target(
                        image, (500, 300), [0, 40, 1676, 830], desktop,
                    )

        self.assertEqual(point, (120, 100))
        self.assertIs(check, frames[4])
        desktop.move.assert_called_once_with((120, 100))

    def test_route_reference_search_includes_anchors_below_scene_search_boundary(self):
        rng = np.random.default_rng(42)
        reference = np.zeros((1080, 1920, 3), dtype=np.uint8)
        origin = rng.integers(0, 256, (45, 380, 3), dtype=np.uint8)
        reference[834:879, 230:610] = origin
        search = [0, 40, 1676, 830]

        with self.assertRaisesRegex(SafetyError, "Route origin anchor not found"):
            locate(reference, origin, search)

        expanded = route_reference_search(search, reference.shape)
        anchor = locate(reference, origin, expanded)
        self.assertEqual(anchor, (230, 834))
        target_search = route_target_search(
            search, (anchor[0] + 399, anchor[1] + 168), reference.shape,
        )
        self.assertEqual(target_search, [0, 40, 1676, 992])
        self.assertEqual(
            project_route_point(np.eye(3), (629, 1002), target_search),
            (629, 1002),
        )

    def test_route_scene_alignment_rejects_unrelated_scenes(self):
        rng = np.random.default_rng(21)
        reference = rng.integers(0, 256, (420, 520, 3), dtype=np.uint8)
        unrelated = np.random.default_rng(22).integers(
            0, 256, (420, 520, 3), dtype=np.uint8,
        )
        self.assertIsNone(route_scene_homography(
            reference, unrelated, [0, 0, 520, 420],
        ))

    def test_route_uses_scene_projection_when_entry_anchor_is_missing(self):
        before = np.zeros((1080, 1920, 3), dtype=np.uint8)
        origin = np.full((24, 24, 3), 100, dtype=np.uint8)
        arrival = np.full((24, 24, 3), 200, dtype=np.uint8)
        current = self.image("attack_cow")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_image(path / "to_bank_00_before_screenshot.png", before)
            save_image(path / "origin.png", origin)
            save_image(path / "arrival.png", arrival)
            step = {
                "origin": "origin.png", "arrival": "arrival.png",
                "reference": "to_bank_00_before_screenshot.png",
                "arrival_roi": [700, 500, 24, 24], "offset": [10, 10],
                "search": [0, 40, 1676, 830], "hover": "Walk here", "timeout": 1,
            }
            route_path = path / "route.json"
            route_path.write_text(json.dumps({
                "screen": [1920, 1080], "to_bank": [step], "to_cows": [step],
            }))
            route = Route(route_path, self.vision)
            desktop = Mock()
            desktop.fresh.return_value = Frame(1, 1, current)
            desktop.guarded_route_click.return_value = (516, 310)
            audit = Mock()

            def locate_side_effect(image, template, rect):
                if np.array_equal(image, before):
                    return 500, 300
                raise SafetyError("Route origin anchor not found")

            check_frame = Frame(2, 2, current)
            with patch("cow_desktop.locate", side_effect=locate_side_effect):
                with patch(
                    "cow_desktop.stable_route_target",
                    return_value=((510, 310), check_frame),
                ) as settle:
                    with patch("cow_desktop.time.monotonic", side_effect=[0, 2]):
                        with self.assertRaisesRegex(SafetyError, "timed out"):
                            route.run("to_bank", desktop, audit)

            settle.assert_called_once()
            desktop.guarded_route_click.assert_called_once_with(
                (510, 310), "Walk here", landmark=None,
            )
            audit.event.assert_any_call(
                "ROUTE_SCENE_REALIGNED", check_frame,
                direction="to_bank", step=0, point=[510, 310],
            )
            audit.event.assert_any_call(
                "ROUTE_HOVER_REACQUIRED", desktop.fresh.return_value,
                direction="to_bank", step=0, action="Walk here",
                original_point=[510, 310], point=[516, 310],
            )

    def test_bank_returns_only_after_all_slots_verified_empty(self):
        desktop = Mock()
        desktop.vision = Mock(wraps=self.vision)
        desktop.vision.profile = self.profile
        desktop.vision.bank_header = self.vision.bank_header
        desktop.vision.settings = self.vision.settings
        bank_frame = load_image(self.profile / "deposit_frame.png")
        desktop.fresh.return_value = Frame(1, 1, bank_frame)
        desktop.vision.bank_open.side_effect = [True, True, True, False]
        desktop.vision.inventory.side_effect = [inventory(), inventory()]
        route, audit = Mock(), Mock()
        bot = Bot(desktop, Mode.COWHIDE_ONLY, route, audit)
        with patch("cow_desktop.stable_inventory", side_effect=[inventory(28), inventory()]):
            bot.bank()
        self.assertEqual([call.args[0] for call in route.run.call_args_list], ["to_bank", "to_cows"])
        desktop.guarded_click.assert_called_once()
        desktop.key.assert_called_once_with("esc")
        events = [call.args[0] for call in audit.event.call_args_list]
        self.assertLess(events.index("DEPOSIT_VERIFIED"), events.index("RETURNED_TO_COWS"))

    def test_no_bank_cli_skips_route_loading_and_preserves_kill_limit(self):
        argv = [
            "cow_bot.py", "run", "--live", "--mode", "COWHIDE_ONLY",
            "--max-kills", "1", "--no-bank",
        ]
        with patch("sys.argv", argv), patch("cow_bot.Vision", return_value=self.vision):
            with patch("cow_desktop.Route") as route, patch("cow_desktop.Bot") as bot:
                with patch("cow_desktop.Desktop"), patch("cow_desktop.Audit"):
                    with patch("cow_bot.time.sleep"), contextlib.redirect_stdout(io.StringIO()):
                        main()
        route.assert_not_called()
        self.assertEqual(bot.call_args.args[1], Mode.COWHIDE_ONLY)
        self.assertIsNone(bot.call_args.args[2])
        bot.return_value.run.assert_called_once_with(1)

    def test_no_route_stops_on_full_inventory_before_any_input(self):
        desktop, audit = Mock(), Mock()
        desktop.vision = self.vision
        bot = Bot(desktop, Mode.COWHIDE_ONLY, None, audit)
        bot.attack = Mock()
        bot.loot = Mock()
        with patch("cow_desktop.stable_inventory", return_value=inventory(28)):
            with self.assertRaisesRegex(SafetyError, "Inventory full"):
                bot.run(1)
        bot.attack.assert_not_called()
        bot.loot.assert_not_called()
        desktop.guarded_click.assert_not_called()
        self.assertNotIn("WALK_TO_BANK", [call.args[0] for call in audit.event.call_args_list])

    def test_no_route_runs_one_kill_and_loot_then_stops(self):
        desktop, audit = Mock(), Mock()
        desktop.vision = self.vision
        bot = Bot(desktop, Mode.COWHIDE_ONLY, None, audit)

        def attack():
            bot.kills += 1
            return True

        bot.attack = Mock(side_effect=attack)
        bot.loot = Mock()
        bot.bank = Mock()
        with patch("cow_desktop.stable_inventory", return_value=inventory()):
            bot.run(1)
        bot.attack.assert_called_once()
        bot.loot.assert_called_once()
        bot.bank.assert_not_called()
        self.assertEqual(audit.event.call_args.args[0], "KILL_LIMIT_REACHED")

    def test_foreign_items_stop_before_deposit(self):
        desktop = Mock()
        desktop.vision = self.vision
        desktop.fresh.return_value = Frame(1, 1, load_image(self.profile / "deposit_frame.png"))
        bot = Bot(desktop, Mode.COWHIDE_ONLY, Mock(), Mock())
        with patch("cow_desktop.stable_inventory", return_value=inventory(27, other=1)):
            with self.assertRaisesRegex(SafetyError, "Unapproved"):
                bot.bank()
        desktop.guarded_click.assert_not_called()
        desktop.key.assert_not_called()

    def test_failed_deposit_does_not_close_bank_or_return_to_cows(self):
        desktop = Mock()
        desktop.vision = Mock(wraps=self.vision)
        desktop.vision.profile = self.profile
        desktop.vision.bank_header = self.vision.bank_header
        desktop.vision.settings = self.vision.settings
        desktop.vision.inventory.return_value = inventory(28)
        desktop.fresh.return_value = Frame(1, .2, load_image(self.profile / "deposit_frame.png"))
        route = Mock()
        bot = Bot(desktop, Mode.COWHIDE_ONLY, route, Mock())
        with patch("cow_desktop.stable_inventory", return_value=inventory(28)):
            with patch("cow_desktop.time.monotonic", side_effect=[0, .1, 5]):
                with self.assertRaisesRegex(SafetyError, "28 empty"):
                    bot.bank()
        desktop.key.assert_not_called()
        self.assertEqual([call.args[0] for call in route.run.call_args_list], ["to_bank"])

    def test_real_combat_video_replay_observes_kill_without_input(self):
        output = io.StringIO()
        with patch("cow_desktop.Desktop", side_effect=AssertionError("Replay must not access the desktop")) as control:
            with contextlib.redirect_stdout(output):
                replay(self.vision, ROOT / "full_attack_cow_and_take_cowhide.mp4", client_video=True)
        control.assert_not_called()
        summary = json.loads(output.getvalue().splitlines()[-1])
        self.assertGreaterEqual(summary["observed_kills"], 1)


if __name__ == "__main__":
    unittest.main()
