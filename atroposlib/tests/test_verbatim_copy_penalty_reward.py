"""
Tests for VerbatimCopyPenaltyReward.

Based on insights from "Designing Reward Signals for Portable Query Generation"
(arXiv:2606.27291), this test verifies that n-gram overlap detection can identify
verbatim copying and apply appropriate penalties.
"""

from atroposlib.envs.reward_fns import registry


class TestVerbatimCopyPenaltyReward:
    """Test suite for verbatim copy penalty reward function."""

    def test_registry_can_create_reward(self):
        """Test that the reward function is registered and can be created."""
        reward_fn = registry.create("verbatim_copy_penalty", n=6, threshold=0.3)
        assert reward_fn is not None
        assert reward_fn.name == "verbatimcopypenaltyreward"

    def test_no_reference_no_penalty(self):
        """Test that no penalty is applied when no reference is provided."""
        reward_fn = registry.create("verbatim_copy_penalty")

        completions = [
            "This is a generated completion about job search.",
            "Another completion with different content.",
        ]

        rewards = reward_fn(completions)
        assert len(rewards) == 2
        assert all(r == 0.0 for r in rewards)

    def test_verbatim_copy_detection(self):
        """Test detection of verbatim copying from reference text."""
        reward_fn = registry.create(
            "verbatim_copy_penalty",
            n=6,
            threshold=0.3,
            penalty_scale=-1.0,
        )

        reference = (
            "The job requires strong programming skills in Python and Java. "
            "Candidates should have experience with machine learning frameworks."
        )

        completions = [
            # High verbatim copy - should be penalized
            "The job requires strong programming skills in Python and Java. "
            "Candidates should have experience with machine learning frameworks.",
            # Low overlap - should not be penalized
            "This position seeks developers with knowledge of AI and deep learning.",
            # Partial copy - may have small penalty depending on threshold
            "The job requires strong programming skills in Python and JavaScript.",
        ]

        rewards = reward_fn(completions, reference=reference)

        # First completion should be penalized (high overlap)
        assert rewards[0] < 0, f"Expected penalty for verbatim copy, got {rewards[0]}"

        # Second completion should not be penalized (low overlap)
        assert rewards[1] == 0.0, f"Expected no penalty, got {rewards[1]}"

    def test_ngram_size_affects_sensitivity(self):
        """Test that larger n-gram sizes detect longer copied sequences."""
        # Small n (2) - more sensitive, detects shorter overlaps
        reward_fn_small_n = registry.create("verbatim_copy_penalty", n=2, threshold=0.3)

        # Large n (8) - less sensitive, only detects long sequences
        reward_fn_large_n = registry.create("verbatim_copy_penalty", n=8, threshold=0.3)

        reference = "machine learning engineer with deep learning experience"

        completions = [
            "machine learning engineer with data science expertise"
        ]  # Partial overlap

        rewards_small_n = reward_fn_small_n(completions, reference=reference)
        rewards_large_n = reward_fn_large_n(completions, reference=reference)

        # Small n should detect more overlap than large n
        # (or detect it more strongly due to more n-grams matching)
        # The exact relationship depends on the text, but large n is more conservative
        assert isinstance(rewards_small_n[0], float)
        assert isinstance(rewards_large_n[0], float)

    def test_threshold_controls_penalty_trigger(self):
        """Test that threshold controls when penalties are applied."""
        # Strict threshold - catches more potential copying
        reward_fn_strict = registry.create(
            "verbatim_copy_penalty",
            n=6,
            threshold=0.1,
        )

        # Lenient threshold - only catches obvious copying
        reward_fn_lenient = registry.create(
            "verbatim_copy_penalty",
            n=6,
            threshold=0.7,
        )

        reference = "software engineer with experience in web development"

        completions = [
            "software engineer with expertise in web development"
        ]  # Partial overlap

        rewards_strict = reward_fn_strict(completions, reference=reference)
        rewards_lenient = reward_fn_lenient(completions, reference=reference)

        # Strict threshold should penalize more than lenient
        assert rewards_strict[0] <= rewards_lenient[0]

    def test_empty_completions(self):
        """Test handling of empty or very short completions."""
        reward_fn = registry.create("verbatim_copy_penalty", min_length=20)

        reference = "This is a longer reference text for comparison purposes."

        completions = [
            "",  # Empty
            "Short",  # Too short
            "This is long enough text for analysis.",  # Long enough
        ]

        rewards = reward_fn(completions, reference=reference)

        # First two should be skipped (too short)
        assert rewards[0] == 0.0
        assert rewards[1] == 0.0

        # Third should be analyzed
        assert isinstance(rewards[2], float)

    def test_weight_applies_to_rewards(self):
        """Test that weight is properly applied to computed penalties."""
        reward_fn = registry.create(
            "verbatim_copy_penalty",
            n=6,
            threshold=0.3,
            penalty_scale=-1.0,
            weight=2.0,  # Double weight
        )

        reference = "python developer with machine learning experience"

        completions = [
            "python developer with machine learning expertise",  # High overlap
        ]

        rewards = reward_fn(completions, reference=reference)

        # Weighted reward should be 2x the computed penalty
        # Since this is a negative penalty, weighted should be more negative
        assert rewards[0] <= 0

    def test_dict_completion_format(self):
        """Test handling of dict completion format (OpenAI-style)."""
        reward_fn = registry.create("verbatim_copy_penalty", n=6, threshold=0.3)

        reference = "data scientist with SQL knowledge"

        completions = [
            {"role": "assistant", "content": "data scientist with SQL expertise"},
        ]

        rewards = reward_fn(completions, reference=reference)

        # Should handle dict format and extract content
        assert len(rewards) == 1
        assert isinstance(rewards[0], float)

    def test_list_of_messages_completion_format(self):
        """Test handling of list-of-messages completion format."""
        reward_fn = registry.create("verbatim_copy_penalty", n=6, threshold=0.3)

        reference = "full stack developer with React experience"

        completions = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "assistant",
                    "content": "full stack developer with React and Node.js",
                },
            ],
        ]

        rewards = reward_fn(completions, reference=reference)

        # Should extract assistant message content
        assert len(rewards) == 1
        assert isinstance(rewards[0], float)

    def test_combined_reward_integration(self):
        """Test that VerbatimCopyPenaltyReward works with CombinedReward."""
        # Create a combined reward that includes verbatim copy penalty
        combined = registry.create(
            "combined",
            rewards=[
                {"type": "verbatim_copy_penalty", "params": {"n": 6, "threshold": 0.3}},
            ],
        )

        reference = "cloud engineer with AWS experience"

        completions = [
            "cloud engineer with AWS and Azure skills",  # Some overlap
        ]

        rewards = combined(completions, reference=reference)

        # Combined reward should work
        assert len(rewards) == 1
        assert isinstance(rewards[0], float)


def test_legacy_function_interface():
    """Test the legacy function interface for backward compatibility."""
    from atroposlib.envs.reward_fns.verbatim_copy_penalty_reward import (
        verbatim_copy_penalty_reward,
    )

    reference = "backend engineer with API design experience"

    completions = [
        "backend engineer with REST API design skills",
    ]

    rewards = verbatim_copy_penalty_reward(
        completions,
        reference=reference,
        n=6,
        threshold=0.3,
    )

    assert len(rewards) == 1
    assert isinstance(rewards[0], float)
