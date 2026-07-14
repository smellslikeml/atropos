"""
Tests for the floor-guarded reward composition.

Exercises the deterministic reward-floor *override* from "Designing Reward
Signals for Portable Query Generation" (arXiv:2606.27291): a completion the
floor flags as degenerate has its reward clamped regardless of the base score.

These tests wire through the existing ``CombinedReward`` call site and the
existing ``RewardFloor`` detector, not just the new module, to prove the
integration rather than a stand-alone unit.
"""

import pytest

from atroposlib.envs.reward_fns.combined_reward import CombinedReward
from atroposlib.envs.reward_fns.floor_guarded_reward import (
    FloorGuardedReward,
    apply_reward_floor,
)
from atroposlib.envs.reward_fns.registry import registry
from atroposlib.envs.reward_fns.reward_floor import RewardFloor
from atroposlib.envs.reward_fns.reward_function import RewardFunction


@registry.register
class ConstantStubReward(RewardFunction):
    """Test stub: returns a fixed (high) reward for every completion.

    Stands in for a permissive LLM-judge that a reward-hacking completion has
    successfully exploited.
    """

    def __init__(self, value: float = 5.0, weight: float = 1.0, **kwargs):
        super().__init__(weight=weight, **kwargs)
        self.value = value

    def compute(self, completions, **kwargs):
        return [self.value] * len(completions)


# A completion that is a verbatim copy of the reference -> reward hacking.
HACK_TEXT = "Senior backend engineer with ten years of distributed systems."
# A genuinely different, substantive completion.
CLEAN_TEXT = "Experienced platform lead skilled in scaling reliable services."


class TestApplyRewardFloor:
    """Unit tests for the deterministic override helper."""

    def test_clamps_only_masked_entries(self):
        rewards = [5.0, 3.0, 0.9]
        mask = [True, False, True]
        out = apply_reward_floor(rewards, mask, floor_value=-1.0)
        assert out == [-1.0, 3.0, -1.0]

    def test_no_flags_passes_through(self):
        rewards = [0.2, 0.7]
        out = apply_reward_floor(rewards, [False, False], floor_value=-1.0)
        assert out == [0.2, 0.7]

    def test_shorter_mask_is_safe(self):
        rewards = [1.0, 2.0, 3.0]
        out = apply_reward_floor(rewards, [True], floor_value=-1.0)
        assert out == [-1.0, 2.0, 3.0]


class TestFloorGuardedReward:
    """The override composition itself."""

    def test_registry_integration(self):
        # The registry lazily loads reward classes by file name, so the created
        # instance is a RewardFunction (base identity is shared across loads).
        reward = registry.create(
            {
                "type": "floor_guarded",
                "base": {"type": "constantstub", "value": 5.0},
                "floor_value": -1.0,
            }
        )
        assert isinstance(reward, RewardFunction)
        assert type(reward).__name__ == "FloorGuardedReward"
        assert reward.compute([CLEAN_TEXT], reference=HACK_TEXT) == [5.0]

    def test_override_beats_high_base(self):
        """A high base reward cannot rescue a verbatim-copy completion."""
        reward = FloorGuardedReward(
            base={"type": "constantstub", "value": 5.0},
            floor={"type": "reward_floor", "rules": ["verbatim_copy"]},
            floor_value=-1.0,
        )
        scores = reward.compute([HACK_TEXT], reference=HACK_TEXT)
        assert scores == [-1.0]

    def test_clean_completion_keeps_base(self):
        reward = FloorGuardedReward(
            base={"type": "constantstub", "value": 5.0},
            floor={"type": "reward_floor", "rules": ["verbatim_copy"]},
            floor_value=-1.0,
        )
        scores = reward.compute([CLEAN_TEXT], reference=HACK_TEXT)
        assert scores == [5.0]

    def test_default_floor_is_reward_floor(self):
        reward = FloorGuardedReward(base={"type": "constantstub"})
        assert isinstance(reward.floor, RewardFloor)

    def test_empty_completions(self):
        reward = FloorGuardedReward(base={"type": "constantstub"})
        assert reward.compute([]) == []


class TestCombinedRewardFloorOverride:
    """The call-site wiring in CombinedReward."""

    def test_additive_floor_lets_hack_survive(self):
        """Baseline: routing the floor additively does NOT stop the hack.

        This is the failure mode the paper's override corrects -- a large base
        reward plus an additive penalty can still net positive.
        """
        additive = CombinedReward(
            rewards=[
                {"type": "constantstub", "value": 5.0},
                {
                    "type": "reward_floor",
                    "rules": ["verbatim_copy"],
                    "penalty_value": -1.0,
                },
            ],
            normalization="none",
        )
        scores = additive.compute([HACK_TEXT], reference=HACK_TEXT)
        # 5.0 + (-1.0) = 4.0 -> the reward-hacking completion is still rewarded.
        assert scores[0] > 0

    def test_override_floor_clamps_hack(self):
        """Override wiring clamps the same completion below zero."""
        guarded = CombinedReward(
            rewards=[{"type": "constantstub", "value": 5.0}],
            normalization="none",
            floor={"type": "reward_floor", "rules": ["verbatim_copy"]},
            floor_value=-1.0,
        )
        scores = guarded.compute([HACK_TEXT], reference=HACK_TEXT)
        assert scores == [-1.0]

    def test_override_floor_preserves_clean(self):
        guarded = CombinedReward(
            rewards=[{"type": "constantstub", "value": 5.0}],
            normalization="none",
            floor={"type": "reward_floor", "rules": ["verbatim_copy"]},
            floor_value=-1.0,
        )
        scores = guarded.compute([CLEAN_TEXT], reference=HACK_TEXT)
        assert scores == [5.0]

    def test_no_floor_is_unchanged(self):
        """Regression: without a floor, CombinedReward behaves as before."""
        combined = CombinedReward(
            rewards=[{"type": "constantstub", "value": 2.0}],
            normalization="none",
        )
        assert combined.floor is None
        assert combined.compute([HACK_TEXT], reference=HACK_TEXT) == [2.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
