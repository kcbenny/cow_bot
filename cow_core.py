from dataclasses import dataclass
from enum import Enum


class SafetyError(RuntimeError):
    pass


class InventoryPanelUnavailable(SafetyError):
    pass


class Mode(str, Enum):
    COWHIDE_ONLY = "COWHIDE_ONLY"
    BONES_ONLY = "BONES_ONLY"
    COWHIDE_AND_BONES = "COWHIDE_AND_BONES"
    COMBAT_ONLY = "COMBAT_ONLY"

    @property
    def items(self) -> tuple[str, ...]:
        return {
            Mode.COWHIDE_ONLY: ("Cowhide",),
            Mode.BONES_ONLY: ("Bones",),
            Mode.COWHIDE_AND_BONES: ("Cowhide", "Bones"),
            Mode.COMBAT_ONLY: (),
        }[self]


@dataclass(frozen=True)
class Health:
    target: str
    positive: bool


@dataclass(frozen=True)
class Inventory:
    slots: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.slots) != 28:
            raise ValueError("Inventory must contain exactly 28 slots")

    def count(self, item: str) -> int:
        return self.slots.count(item)

    @property
    def full(self) -> bool:
        return "EMPTY" not in self.slots

    @property
    def empty(self) -> bool:
        return all(item == "EMPTY" for item in self.slots)

    def require_allowed(self, mode: Mode) -> None:
        foreign = set(self.slots) - {"EMPTY", *mode.items}
        if foreign:
            raise SafetyError(f"Unapproved inventory contents: {sorted(foreign)}")


class HealthLock:
    """Only fresh positive HP followed by two fresh zero frames confirms a kill."""

    def __init__(self, started_at: float, target: str) -> None:
        self.started_at = started_at
        self.target = target
        self.seen_positive = False
        self.zeros = 0
        self.last_id = -1
        self.last_time = started_at - 1e-9

    def observe(self, frame_id: int, captured_at: float, hp: Health | None) -> str:
        if frame_id <= self.last_id or captured_at <= self.last_time:
            return "WAIT"
        self.last_id = frame_id
        self.last_time = captured_at
        elapsed = captured_at - self.started_at
        if elapsed > 30:
            raise SafetyError("Combat timed out; outcome is unknown")
        if not self.seen_positive and elapsed > 10:
            return "MISSED"
        if hp is not None and hp.target != self.target:
            raise SafetyError("Combat HUD target changed")
        if hp is not None and hp.positive:
            self.seen_positive = True
            self.zeros = 0
        elif hp is not None and self.seen_positive:
            self.zeros += 1
            if self.zeros >= 2:
                return "KILLED"
        else:
            self.zeros = 0
        return "WAIT"


class PickupCheck:
    def __init__(self, baseline: Inventory, item: str, clicked_at: float) -> None:
        if baseline.full:
            raise SafetyError("Cannot pick up into a full inventory")
        self.baseline = baseline
        self.item = item
        self.clicked_at = clicked_at
        self.last_id = -1
        self.last_time = clicked_at
        self.confirmations = 0

    def observe(self, frame_id: int, captured_at: float, current: Inventory) -> bool:
        if frame_id <= self.last_id or captured_at <= self.last_time:
            return False
        self.last_id = frame_id
        self.last_time = captured_at
        if captured_at - self.clicked_at > 15:
            raise SafetyError("Pickup not verified within fifteen seconds")
        expected = self.baseline.count(self.item) + 1
        unchanged = all(
            current.count(item) == self.baseline.count(item)
            for item in set(self.baseline.slots + current.slots) - {self.item, "EMPTY"}
        )
        correct = (
            current.count(self.item) == expected
            and current.count("EMPTY") == self.baseline.count("EMPTY") - 1
            and unchanged
        )
        if current != self.baseline and not correct:
            raise SafetyError("Pickup changed unexpected inventory contents")
        self.confirmations = self.confirmations + 1 if correct else 0
        if self.confirmations >= 2:
            return True
        return False
