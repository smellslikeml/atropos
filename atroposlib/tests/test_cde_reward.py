"""
Tests for the CDE (Curiosity-Driven Exploration) reward function.

These tests pin the paper's two curiosity signals (arXiv:2509.09675v1):
  - actor-wise: perplexity over the actor's own generated response
  - critic-wise: variance of value estimates across a multi-head critic
"""

import pytest

from atroposlib.envs.reward_fns.registry import registry

LONG_TEXT = "this is a sufficiently long completion with plenty of distinct tokens"


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
        """Test creating CDE reward with custom component weights."""
        cde_reward = registry.create("cde", actor_weight=0.7, critic_weight=0.2)
        assert cde_reward.actor_weight == 0.7
        assert cde_reward.critic_weight == 0.2

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
        # Completions below min_tokens get no exploration bonus
        assert all(r == 0.0 for r in rewards)

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

    # --- Actor-wise bonus: perplexity over the actor's own response ---

    def test_actor_bonus_exact_from_logprobs(self):
        """Actor bonus equals normalized log-perplexity of the response."""
        cde_reward = registry.create("cde", actor_weight=1.0, critic_weight=0.0)

        # Mean logprob -1.0 -> log PPL = 1.0 -> bonus = 1.0 / 10.0
        rewards = cde_reward.compute([LONG_TEXT], logprobs=[[-1.0] * 8])
        assert rewards[0] == pytest.approx(0.1)

    def test_actor_bonus_penalizes_overconfidence(self):
        """Paper's key property: overconfident (low-perplexity) outputs get
        little exploration bonus; uncertain (high-perplexity) ones get more."""
        cde_reward = registry.create("cde", actor_weight=1.0, critic_weight=0.0)

        confident = cde_reward.compute([LONG_TEXT], logprobs=[[-0.01] * 8])[0]
        uncertain = cde_reward.compute([LONG_TEXT], logprobs=[[-4.0] * 8])[0]

        assert uncertain > confident
        assert confident == pytest.approx(0.001)

    def test_actor_bonus_proxy_novel_vs_repeated(self):
        """Proxy path (no logprobs): content repeated from history has low
        surprisal under the bigram model; novel content scores higher."""
        cde_reward = registry.create(
            "cde", actor_weight=1.0, critic_weight=0.0, min_tokens=3
        )
        seed = "alpha beta gamma delta epsilon zeta eta theta iota kappa"

        cde_reward.compute([seed])  # seed the proxy corpus
        repeated = cde_reward.compute([seed])[0]
        novel = cde_reward.compute(
            ["completely unrelated tokens absent from the corpus so far"]
        )[0]

        assert repeated < novel, "Repeated content should have lower curiosity"

    def test_actor_bonus_proxy_cold_start_diversity(self):
        """Empty history: surprisal reduces to log |vocab|, so lexically
        diverse text scores above repetitive text (diversity promotion)."""
        diverse_reward = registry.create(
            "cde", actor_weight=1.0, critic_weight=0.0, min_tokens=3
        ).compute(["one two three four five six seven eight nine ten"])[0]
        repetitive_reward = registry.create(
            "cde", actor_weight=1.0, critic_weight=0.0, min_tokens=3
        ).compute(["same same same same same same same same same same"])[0]

        assert diverse_reward > repetitive_reward

    # --- Critic-wise bonus: multi-head value variance ---

    def test_critic_bonus_multi_head_variance(self):
        """Critic bonus grows with the spread of multi-head value estimates."""
        cde_reward = registry.create("cde", actor_weight=0.0, critic_weight=1.0)

        uncertain = cde_reward.compute([LONG_TEXT], values=[[1.0, 3.0, 5.0]])[0]
        certain = cde_reward.compute([LONG_TEXT], values=[[2.9, 3.0, 3.1]])[0]

        assert uncertain > certain
        # std of [2.9, 3.0, 3.1] = sqrt(0.02 / 3)
        assert certain == pytest.approx((0.02 / 3) ** 0.5)

    def test_critic_bonus_scalar_value_has_no_signal(self):
        """Paper: curiosity is the *variance across heads*. A single scalar
        value estimate carries no epistemic signal, and neither does its
        magnitude — value level is not curiosity."""
        cde_reward = registry.create("cde", actor_weight=0.0, critic_weight=1.0)

        assert cde_reward.compute([LONG_TEXT], values=[5.0])[0] == 0.0
        assert cde_reward.compute([LONG_TEXT])[0] == 0.0

    def test_min_tokens_guard(self):
        """Completions below the token threshold get no bonus."""
        cde_reward = registry.create("cde", min_tokens=10)

        rewards = cde_reward.compute(["too short"], logprobs=[[-1.0, -1.0]])

        assert rewards == [0.0]

    def test_combined_reward_integration(self):
        """Test that CDE works with CombinedReward."""
        combined = registry.create(
            "combined",
            rewards=[
                {"type": "cde", "params": {"actor_weight": 0.5}},
                {"type": "format"},
            ],
        )

        completions = ["<think Test</think", "<answer>42</answer>"]

        rewards = combined.compute(completions)
        assert len(rewards) == len(completions)

    def test_history_tracking(self):
        """Test that the proxy corpus is tracked and bounded."""
        cde_reward = registry.create("cde")

        # Add multiple batches to build history
        for i in range(5):
            cde_reward.compute(
                [f"Batch {i} content with enough tokens to pass the guard here"]
            )

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

    def test_call_with_kwargs(self):
        """Test calling CDE with various kwargs."""
        cde_reward = registry.create("cde")
        completions = ["Test"]

        # Should handle extra kwargs gracefully
        rewards = cde_reward.compute(completions, extra_param="ignored")
        assert len(rewards) == len(completions)
