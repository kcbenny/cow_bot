import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import cv2
import numpy as np

from cow_bot import countdown, main, replay
from cow_core import (
    Health, HealthLock, Inventory, InventoryPanelUnavailable, Mode, PickupCheck, SafetyError,
)
from cow_desktop import Bot, Desktop, Frame, Route, foreground_window_problem, locate
from cow_vision import ROOT, Vision, bootstrap, crop, load_image, text_mask


def inventory(hides=0, bones=0, other=0):
    return Inventory(tuple(
        ["Cowhide"] * hides + ["Bones"] * bones + ["UNKNOWN"] * other
        + ["EMPTY"] * (28 - hides - bones - other)
    ))


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
        with self.assertRaisesRegex(SafetyError, "four seconds"):
            PickupCheck(inventory(), "Cowhide", 0).observe(1, 4.1, inventory())

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
        self.assertFalse(check.observe(1, 3.9, inventory(2)))
        with self.assertRaisesRegex(SafetyError, "four seconds"):
            check.observe(2, 4.1, inventory(2))


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

    def test_reopen_inventory_requires_verified_icon_and_two_readings(self):
        desktop = object.__new__(Desktop)
        vision = Mock()
        vision.inventory_panel_visible.side_effect = [False, False, False, False, True, True]
        vision.inventory_tab_visible.return_value = True
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
        desktop.move.assert_called_once_with((1568, 1013))
        desktop.mouse.click.assert_called_once_with()
        self.assertEqual(vision.inventory_tab_visible.call_count, 4)
        self.assertEqual(vision.inventory.call_count, 2)

    def test_reopen_inventory_does_not_click_if_panel_appears_while_aiming(self):
        desktop = object.__new__(Desktop)
        desktop.vision = Mock()
        desktop.vision.inventory_panel_visible.side_effect = [False, True]
        desktop.vision.inventory_tab_visible.return_value = True
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
        desktop.vision.inventory_tab_visible.return_value = False
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
        vision.loot_grid.return_value = iter([((1036, 554), "fine")])
        vision.hover.return_value = "Walk here"
        desktop, audit = Mock(), Mock()
        desktop.vision = vision
        desktop.fresh.return_value = Frame(
            1, .1, np.zeros((1080, 1920, 3), dtype=np.uint8),
        )
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
        self.assertEqual(vision.loot_grid.call_count, 1)
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

    def test_loot_grid_starts_at_kill_center_then_expands_coarse(self):
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
        x0, y0 = 532, 122
        self.assertTrue(all((x - x0) % 12 == 0 and (y - y0) % 12 == 0 for x, y in fine))
        self.assertTrue(all((x - x0) % 24 == 0 and (y - y0) % 24 == 0 for x, y in coarse))

    def test_loot_search_uses_kill_point_and_logs_scan_metrics(self):
        vision = Mock()
        vision.settings = {"loot_radius": 130, "loot_fine_radius": 48}
        vision.loot_grid.return_value = iter([
            ((1036, 554), "fine"),
            ((1060, 554), "coarse"),
        ])
        vision.hover.return_value = "Walk here"
        desktop, audit = Mock(), Mock()
        desktop.vision = vision
        desktop.fresh.return_value = Frame(1, 1, np.zeros((1080, 1920, 3), dtype=np.uint8))
        bot = Bot(desktop, Mode.COWHIDE_ONLY, None, audit)
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
        self.assertEqual(metrics["hover_matches"], 0)
        self.assertEqual(metrics["farthest_distance"], 24)
        self.assertEqual(metrics["stop_reason"], "region_exhausted")

    def test_cowhide_only_stops_after_verified_pickup(self):
        vision = Mock()
        vision.settings = {"loot_radius": 130, "loot_fine_radius": 48}
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
        ]
        desktop.pick_menu.return_value = "Cowhide"
        bot = Bot(desktop, Mode.COWHIDE_ONLY, None, audit)
        bot.last_kill_point = (1036, 554)
        with patch("cow_desktop.stable_inventory", side_effect=[inventory(), inventory()]):
            with patch("cow_desktop.time.monotonic", return_value=0):
                bot.loot()

        self.assertEqual(vision.hover.call_count, 1)
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

    def test_cowhide_mode_ignores_bones_hover_candidates(self):
        vision = Mock()
        vision.settings = {"loot_radius": 130}
        vision.loot_grid.return_value = iter([((1036, 554), "fine")])
        vision.hover.return_value = "Take Bones"
        desktop, audit = Mock(), Mock()
        desktop.vision = vision
        desktop.fresh.return_value = Frame(
            1, .1, np.zeros((1080, 1920, 3), dtype=np.uint8),
        )
        bot = Bot(desktop, Mode.COWHIDE_ONLY, None, audit)
        bot.last_kill_point = (1036, 554)
        with patch("cow_desktop.stable_inventory", return_value=inventory()):
            with patch("cow_desktop.time.monotonic", return_value=0):
                bot.loot()

        desktop.pick_menu.assert_not_called()
        metrics = audit.event.call_args.kwargs
        self.assertEqual(metrics["hover_matches"], 0)
        self.assertEqual(metrics["ignored_item_matches"], 1)
        self.assertEqual(metrics["stop_reason"], "region_exhausted")

    def test_attack_search_gets_its_full_budget_after_hud_clear(self):
        vision = Mock()
        vision.settings = {"player_anchor": [942, 520]}
        vision.grid.return_value = iter([(1036, 554)])
        vision.hud_clear.side_effect = [False, True, True]
        vision.hover.return_value = "Attack Cow (level-2)"
        vision.health.side_effect = [
            Health("Cow", True),
            Health("Cow", False),
            Health("Cow", False),
        ]
        desktop, audit = Mock(), Mock()
        desktop.vision = vision
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frames = [
            Frame(1, .1, image),
            Frame(2, 7.5, image),
            Frame(3, 15.1, image),
            Frame(4, 15.2, image),
            Frame(5, 15.3, image),
            Frame(6, 15.4, image),
            Frame(7, 15.5, image),
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

    def test_found_cowhide_is_processed_even_if_search_budget_expires(self):
        vision = Mock()
        vision.settings = {"loot_radius": 130}
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
            Frame(2, 26.1, image),
            Frame(3, 26.2, image),
            Frame(4, 26.3, image),
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
        vision.loot_grid.return_value = iter([((1036, 554), "fine")])
        vision.hover.return_value = "Walk here"
        desktop, audit = Mock(), Mock()
        desktop.vision = vision
        clock = {"now": 0}
        desktop.move.side_effect = lambda point: clock.update(now=26)
        bot = Bot(desktop, Mode.COWHIDE_ONLY, None, audit)
        bot.last_kill_point = (1036, 554)
        with patch("cow_desktop.stable_inventory", return_value=inventory()):
            with patch("cow_desktop.time.monotonic", side_effect=lambda: clock["now"]):
                bot.loot()
        desktop.fresh.assert_not_called()
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
            desktop.guarded_click.return_value = True
            with patch("cow_desktop.locate", return_value=(1000, 180)):
                with patch("cow_desktop.time.monotonic", side_effect=[0, .1, 2]):
                    with self.assertRaisesRegex(SafetyError, "timed out"):
                        route.run("to_bank", desktop, Mock())
            desktop.guarded_click.assert_called_once()

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
