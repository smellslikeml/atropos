"""
Tests for the reward floor module.

Tests the RewardFloor class and its degeneracy detectors, ensuring
integration with the reward function registry and proper behavior
for various degeneracy patterns.
"""

import pytest

from atroposlib.envs.reward_fns.registry import registry
from atroposlib.envs.reward_fns.reward_floor import (
    DegeneracyDetector,
    RewardFloor,
    reward_floor,
)
from atroposlib.envs.reward_fns.reward_function import RewardFunction


class TestDegeneracyDetector:
    """Test individual degeneracy detector functions."""

    def test_verbatim_copy_ratio_no_copy(self):
        """Test that non-copying text returns low ratio."""
        completion = "The capital of France is Paris and it's beautiful."
        reference = "Python is a programming language."
        ratio = DegeneracyDetector.verbatim_copy_ratio(completion, reference)
        assert ratio < 0.5

    def test_verbatim_copy_ratio_full_copy(self):
        """Test that verbatim copy returns high ratio."""
        completion = "The capital of France is Paris."
        reference = "The capital of France is Paris."
        ratio = DegeneracyDetector.verbatim_copy_ratio(completion, reference)
        assert ratio > 0.9

    def test_verbatim_copy_ratio_partial_copy(self):
        """Test that partial copy returns medium ratio."""
        completion = "The capital of France is Paris and it's beautiful."
        reference = "The capital of France is Paris."
        ratio = DegeneracyDetector.verbatim_copy_ratio(completion, reference)
        assert 0.5 < ratio < 0.9

    def test_verbatim_copy_ratio_with_prompt(self):
        """Test verbatim copy detection against prompt."""
        completion = "Here is the answer you asked for."
        prompt = "Here is the answer you asked for."
        ratio = DegeneracyDetector.verbatim_copy_ratio(completion, prompt=prompt)
        assert ratio > 0.9

    def test_is_empty_or_minimal_empty_string(self):
        """Test that empty string is detected."""
        assert DegeneracyDetector.is_empty_or_minimal("") is True

    def test_is_empty_or_minimal_whitespace(self):
        """Test that whitespace-only is detected."""
        assert DegeneracyDetector.is_empty_or_minimal("   ") is True

    def test_is_empty_or_minimal_too_short(self):
        """Test that very short text is detected."""
        assert DegeneracyDetector.is_empty_or_minimal("ab", min_length=10) is True

    def test_is_empty_or_minimal_valid_text(self):
        """Test that valid text passes."""
        assert (
            DegeneracyDetector.is_empty_or_minimal(
                "This is a valid response with enough content."
            )
            is False
        )

    def test_consecutive_repetition_ratio_no_repetition(self):
        """Test that normal text has low repetition ratio."""
        text = "The quick brown fox jumps over the lazy dog."
        ratio = DegeneracyDetector.consecutive_repetition_ratio(text)
        assert ratio == 0.0

    def test_consecutive_repetition_ratio_high_repetition(self):
        """Test that stuttering text has high repetition ratio."""
        # Use a pattern that creates consecutive window repeats
        text = "hello world hello world hello world hello world"
        ratio = DegeneracyDetector.consecutive_repetition_ratio(text, window=2)
        # With 2-word window "hello world" repeating 4 times, we get 2 repeats out of 3 checks
        assert ratio > 0.5

    def test_template_exploitation_no_template(self):
        """Test that normal content has low exploitation score."""
        content = "After careful analysis, the answer is 42 because of the mathematical properties."
        score = DegeneracyDetector.template_exploitation_score(content)
        assert score < 0.5

    def test_template_exploitation_high_template(self):
        """Test that template-heavy content has high exploitation score."""
        content = "\\boxed{42} #### 42"
        score = DegeneracyDetector.template_exploitation_score(content)
        assert score > 0.7


class TestRewardFloor:
    """Test the RewardFloor class."""

    def test_registry_integration(self):
        """Test that RewardFloor is registered in the registry."""
        # Create via registry (returns a RewardFunction instance)
        floor = registry.create("reward_floor", floor_value=0.5)
        # Should be a RewardFunction (might be wrapped by LegacyFunctionWrapper)
        assert isinstance(floor, RewardFunction)
        assert floor.name == "rewardfloor"

    def test_registry_integration_with_params(self):
        """Test registry creation with parameters."""
        floor = registry.create(
            "reward_floor",
            rules=["verbatim_copy"],
            floor_value=0.3,
            verbatim_threshold=0.9,
        )
        assert isinstance(floor, RewardFunction)
        assert floor.floor_value == 0.3
        assert floor.verbatim_threshold == 0.9

    def test_is_reward_function_subclass(self):
        """Test that RewardFloor inherits from RewardFunction."""
        floor = RewardFloor()
        assert isinstance(floor, RewardFunction)

    def test_compute_empty_completions(self):
        """Test that empty completions list returns empty rewards."""
        floor = RewardFloor()
        rewards = floor.compute([])
        assert rewards == []

    def test_compute_verbatim_copy_triggers_floor(self):
        """Test that verbatim copying triggers the reward floor."""
        floor = RewardFloor(
            rules=["verbatim_copy"],
            floor_value=0.2,
            verbatim_threshold=0.85,
        )

        completions = [
            {"role": "assistant", "content": "The answer is 42."},
            {"role": "assistant", "content": "The answer is 42."},
        ]

        rewards = floor.compute(completions, reference="The answer is 42.")
        # First completion is a verbatim copy
        assert rewards[0] == 0.2

    def test_compute_normal_content_passes_floor(self):
        """Test that normal content passes the floor (returns 1.0)."""
        floor = RewardFloor(
            rules=["verbatim_copy"],
            floor_value=0.2,
        )

        completions = [
            {"role": "assistant", "content": "After analysis, I conclude 42."},
        ]

        rewards = floor.compute(completions, reference="The capital is Paris.")
        # Normal content should pass
        assert rewards[0] == 1.0

    def test_compute_empty_minimal_triggers_floor(self):
        """Test that empty/minimal completions trigger floor."""
        floor = RewardFloor(
            rules=["empty_minimal"],
            floor_value=0.1,
            min_completion_length=10,
        )

        completions = [{"role": "assistant", "content": "short"}]
        rewards = floor.compute(completions)
        assert rewards[0] == 0.1

    def test_compute_repetition_triggers_floor(self):
        """Test that repetitive content triggers floor."""
        floor = RewardFloor(
            rules=["repetition"],
            floor_value=0.3,
            repetition_threshold=0.5,  # Threshold lower than expected ratio
        )

        # Use a pattern with consecutive 3-word repetition
        # Need enough words for window=3 detection (at least 6 words)
        # Pattern: "one two three" repeated creates consecutive 3-word matches
        completions = [
            {"role": "assistant", "content": "one two three one two three one two three one two three"}
        ]
        rewards = floor.compute(completions)
        # High repetition should trigger floor (ratio ~0.78 > 0.5)
        assert rewards[0] == 0.3

    def test_compute_multiple_rules(self):
        """Test floor with multiple rules enabled."""
        floor = RewardFloor(
            rules=["verbatim_copy", "empty_minimal", "repetition"],
            floor_value=0.0,
        )

        # Empty completion should trigger empty_minimal rule
        completions = [{"role": "assistant", "content": ""}]
        rewards = floor.compute(completions)
        assert rewards[0] == 0.0

    def test_compute_custom_detector(self):
        """Test custom detector function."""
        def always_degenerate(content, reference):
            return True

        floor = RewardFloor(
            rules=[],
            floor_value=0.5,
            custom_detectors={"always": always_degenerate},
        )

        completions = [{"role": "assistant", "content": "normal content"}]
        rewards = floor.compute(completions)
        # Custom detector should trigger
        assert rewards[0] == 0.5

    def test_compute_with_different_completion_formats(self):
        """Test that different completion formats are handled."""
        floor = RewardFloor(rules=["empty_minimal"], floor_value=0.0)

        # String format
        rewards1 = floor.compute(["valid content"])
        assert rewards1[0] == 1.0

        # Dict format with role/content
        rewards2 = floor.compute([{"role": "assistant", "content": "valid content"}])
        assert rewards2[0] == 1.0

        # Dict format with message wrapper
        rewards3 = floor.compute(
            [{"message": {"role": "assistant", "content": "valid content"}}]
        )
        assert rewards3[0] == 1.0

    def test_penalty_value(self):
        """Test that penalty_value is applied instead of floor_value."""
        floor = RewardFloor(
            rules=["verbatim_copy"],
            floor_value=0.5,
            penalty_value=-1.0,
        )

        completions = [{"role": "assistant", "content": "exact copy"}]
        rewards = floor.compute(completions, reference="exact copy")
        # Penalty value should be applied
        assert rewards[0] == -1.0

    def test_weight_application(self):
        """Test that weight is properly applied to rewards."""
        floor = RewardFloor(
            rules=["empty_minimal"],
            floor_value=0.5,
            weight=2.0,
        )

        completions = [{"role": "assistant", "content": ""}]
        rewards = floor(completions)  # Using __call__ which applies weight
        # 0.5 * 2.0 = 1.0
        assert rewards[0] == 1.0

    def test_error_handling(self):
        """Test that errors are handled gracefully."""
        floor = RewardFloor(rules=["verbatim_copy"])

        # Detector that raises an exception
        def broken_detector(content, reference):
            raise ValueError("Intentional error")

        floor.custom_detectors = {"broken": broken_detector}

        completions = [{"role": "assistant", "content": "test"}]
        # Should not raise, should return conservative value
        rewards = floor.compute(completions)
        assert len(rewards) == 1
        assert isinstance(rewards[0], float)

    def test_batch_reference_handling(self):
        """Test handling of batch references."""
        floor = RewardFloor(
            rules=["verbatim_copy"],
            floor_value=0.0,
            verbatim_threshold=0.9,
        )

        completions = [
            {"role": "assistant", "content": "copy one"},
            {"role": "assistant", "content": "copy two"},
        ]

        # References as list matching completions
        references = ["copy one", "different text"]
        rewards = floor.compute(completions, reference=references)

        # First should trigger floor (verbatim copy)
        assert rewards[0] == 0.0
        # Second should pass (not a verbatim copy)
        assert rewards[1] == 1.0


class TestLegacyFunction:
    """Test the legacy reward_floor function."""

    def test_legacy_function_basic(self):
        """Test that legacy function works."""
        completions = [{"role": "assistant", "content": "test"}]
        rewards = reward_floor(completions, floor_value=0.5)
        assert len(rewards) == len(completions)

    def test_legacy_function_with_rules(self):
        """Test legacy function with rules parameter."""
        completions = [{"role": "assistant", "content": ""}]
        rewards = reward_floor(
            completions, rules=["empty_minimal"], floor_value=0.0
        )
        assert rewards[0] == 0.0
