"""
Tests for the CDE (Curiosity-Driven Exploration) reward function.
"""

import pytest

from atroposlib.envs.reward_fns.registry import registry


class TestCDEReward:
    """Test suite for CDE reward function."""

    def test_registry_registration(self):
        """Test that CDE reward is properly registered."""
        # Create a CDE reward, which triggers loading from the file
        cde_reward = registry.create("cde")
        assert cde_reward is not None
        # Now 'cde' should be in the registered names
        reward_names = registry.list_registered()
        assert "cde" in reward_names, "CDE reward should be registered as 'cde'"

    def test_create_from_registry(self):
        """Test creating CDE reward from registry."""
        cde_reward = registry.create("cde")
        assert cde_reward is not None
        # The name property uses the class name lowercased
        assert cde_reward.name == "cdereward"

    def test_create_with_custom_weights(self):
        """Test creating CDE reward with custom weights."""
        cde_reward = registry.create(
            "cde",
            novelty_weight=0.7,
            entropy_weight=0.2,
            surprise_weight=0.1,
        )
        assert cde_reward.novelty_weight == 0.7
        assert cde_reward.entropy_weight == 0.2
        assert cde_reward.surprise_weight == 0.1

    def test_compute_simple_completions(self):
        """Test CDE reward on simple string completions."""
        cde_reward = registry.create("cde")
        completions = ["Hello world", "Goodbye world", "Another test"]

        rewards = cde_reward.compute(completions)

        assert len(rewards) == len(completions)
        assert all(isinstance(r, float) for r in rewards)

    def test_compute_dict_format_completions(self):
        """Test CDE reward on dict-format completions (assistant messages)."""
        cde_reward = registry.create("cde")
        completions = [
            {"role": "assistant", "content": "First response"},
            {"role": "assistant", "content": "Second response"},
        ]

        rewards = cde_reward.compute(completions)

        assert len(rewards) == len(completions)
        assert all(isinstance(r, float) for r in rewards)

    def test_empty_completions(self):
        """Test CDE reward on empty completions."""
        cde_reward = registry.create("cde")
        completions = ["", "   ", "short"]

        rewards = cde_reward.compute(completions)

        assert len(rewards) == len(completions)
        # Empty/short completions should get low curiosity scores
        assert all(r <= 0.5 for r in rewards)

    def test_weight_application(self):
        """Test that weight is properly applied."""
        cde_reward = registry.create("cde", weight=2.0)
        completions = ["Test completion"]

        # The __call__ method applies weight
        weighted_rewards = cde_reward(completions)
        raw_rewards = cde_reward.compute(completions)

        assert len(weighted_rewards) == len(raw_rewards)
        # Weighted rewards should be 2x the raw rewards
        for w, r in zip(weighted_rewards, raw_rewards):
            assert abs(w - 2.0 * r) < 1e-6

    def test_novelty_detection(self):
        """Test that novel content gets higher curiosity scores."""
        cde_reward = registry.create("cde", novelty_weight=1.0, entropy_weight=0.0, surprise_weight=0.0)

        # First batch - all novel (use longer content)
        first_batch = [
            "This is a unique piece of content with many words here",
            "Another completely different sentence with lots of words",
            "Third unique paragraph containing many distinct words here"
        ]
        first_rewards = cde_reward.compute(first_batch)

        # Second batch with similar content
        second_batch = [
            "This is a unique piece of content with many words here",
            "Some new and different content for variety in testing",
            "Yet another distinct piece of writing for our novelty test"
        ]
        second_rewards = cde_reward.compute(second_batch)

        # The repeated content should have lower novelty
        assert second_rewards[0] < first_rewards[0], "Repeated content should have lower novelty"

    def test_entropy_bonus(self):
        """Test entropy bonus computation."""
        # High diversity content vs low diversity content
        cde_reward = registry.create(
            "cde",
            novelty_weight=0.0,
            entropy_weight=1.0,
            surprise_weight=0.0,
        )

        # Use longer content for proper analysis
        high_diversity = [
            "The quick brown fox jumps over lazy dogs and then runs away quickly",
            "Many unique words exist here in this particular sentence that we wrote"
        ]
        low_diversity = [
            "the the the the the the the the the the the the the the the the",
            "same same same same same same same same same same same same same"
        ]

        high_rewards = cde_reward.compute(high_diversity)
        low_rewards = cde_reward.compute(low_diversity)

        # High diversity should get higher entropy bonus
        assert high_rewards[0] > low_rewards[0], "High diversity should get higher entropy"

    def test_with_value_predictions(self):
        """Test CDE reward with value predictions."""
        cde_reward = registry.create(
            "cde",
            novelty_weight=0.0,
            entropy_weight=0.0,
            surprise_weight=1.0,
        )

        # Use content long enough to pass min_tokens threshold
        completions = ["This is a longer test completion with many words"]

        # Higher value = higher surprise (normalized by 10)
        rewards = cde_reward.compute(completions, values=[5.0])
        assert rewards[0] > 0, "Positive value should produce surprise"
        # Value 5.0 should give surprise of 0.5
        assert abs(rewards[0] - 0.5) < 0.01, f"Expected 0.5, got {rewards[0]}"

    def test_combined_reward_integration(self):
        """Test that CDE works with CombinedReward."""
        combined = registry.create(
            "combined",
            rewards=[
                {"type": "cde", "params": {"novelty_weight": 0.5}},
                {"type": "format"},
            ],
        )

        completions = ["<think Test</think", "<answer>42</answer>"]

        rewards = combined.compute(completions)
        assert len(rewards) == len(completions)

    def test_history_tracking(self):
        """Test that completion history is tracked for novelty."""
        cde_reward = registry.create("cde")

        # Add multiple batches to build history
        for i in range(5):
            cde_reward.compute([f"Batch {i} content"])

        # History should be maintained
        assert len(cde_reward._completion_history) > 0

        # But should be bounded
        assert len(cde_reward._completion_history) <= cde_reward._history_size

    def test_legacy_function(self):
        """Test the legacy function wrapper."""
        from atroposlib.envs.reward_fns.cde_reward import cde_reward

        completions = ["Test completion"]
        rewards = cde_reward(completions)

        assert len(rewards) == len(completions)
        assert all(isinstance(r, float) for r in rewards)

    def test_custom_ngram_sizes(self):
        """Test CDE with custom n-gram sizes."""
        cde_reward = registry.create("cde", ngram_sizes=[2, 5, 7])
        assert cde_reward.ngram_sizes == [2, 5, 7]

        completions = ["This is a longer test completion with many words"]
        rewards = cde_reward.compute(completions)

        assert len(rewards) == len(completions)

    def test_call_with_kwargs(self):
        """Test calling CDE with various kwargs."""
        cde_reward = registry.create("cde")
        completions = ["Test"]

        # Should handle extra kwargs gracefully
        rewards = cde_reward.compute(completions, extra_param="ignored")
        assert len(rewards) == len(completions)
