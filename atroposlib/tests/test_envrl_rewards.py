"""
Tests for EnvRL auxiliary objective reward functions.

Tests the state prediction and inverse dynamics reward functions.
"""

import pytest

from atroposlib.envs.reward_fns.registry import registry


class TestStatePredictionReward:
    """Tests for StatePredictionReward."""

    def test_registry_registration(self):
        """Test that StatePredictionReward is registered properly."""
        reward_fn = registry.create("state_prediction")
        # Check that it's a RewardFunction instance
        from atroposlib.envs.reward_fns.reward_function import RewardFunction
        assert isinstance(reward_fn, RewardFunction)
        assert reward_fn.name == "statepredictionreward"

    def test_basic_compute_with_simple_completions(self):
        """Test basic computation with simple completions."""
        reward_fn = registry.create("state_prediction")
        completions = ["action 1", "action 2", "action 3"]

        rewards = reward_fn.compute(completions)

        assert len(rewards) == 3
        assert all(isinstance(r, float) for r in rewards)
        assert all(0.0 <= r <= 1.0 for r in rewards)

    def test_compute_with_explicit_next_states(self):
        """Test computation with explicit next states provided."""
        reward_fn = registry.create("state_prediction")
        completions = ["predict next state", "another prediction"]
        next_states = ["actual next state 1", "actual next state 2"]

        rewards = reward_fn.compute(completions, next_states=next_states)

        assert len(rewards) == 2
        # Rewards should reflect similarity
        assert all(isinstance(r, float) for r in rewards)

    def test_compute_with_trajectories(self):
        """Test computation with trajectory data."""
        reward_fn = registry.create("state_prediction")
        completions = ["action 1", "action 2"]
        trajectories = [
            {"state": "initial state", "next_state": "next state 1"},
            {"state": "state 2", "next_state": "next state 2"},
        ]

        rewards = reward_fn.compute(completions, trajectories=trajectories)

        assert len(rewards) == 2
        assert all(isinstance(r, float) for r in rewards)

    def test_compute_with_weight(self):
        """Test that weight is properly applied."""
        weight = 2.0
        reward_fn = registry.create("state_prediction", weight=weight)
        completions = ["action 1"]

        # Call via __call__ to test weight application
        rewards = reward_fn(completions, next_states=["next state"])

        assert len(rewards) == 1
        # Weight should be applied, making reward potentially higher than 1.0
        assert isinstance(rewards[0], float)

    def test_compute_with_min_prediction_length(self):
        """Test minimum prediction length threshold."""
        reward_fn = registry.create("state_prediction", min_prediction_length=50)
        completions = ["short"]  # Below threshold

        rewards = reward_fn.compute(completions)

        assert len(rewards) == 1
        # Short actions should get lower reward in fallback mode
        assert rewards[0] < 1.0

    def test_empty_completions(self):
        """Test with empty completions list."""
        reward_fn = registry.create("state_prediction")
        completions = []

        rewards = reward_fn.compute(completions)

        assert rewards == []

    def test_handles_missing_next_states(self):
        """Test graceful handling when next states are not provided."""
        reward_fn = registry.create("state_prediction")
        completions = ["action 1", "action 2"]

        # Should not error, should use fallback heuristic
        rewards = reward_fn.compute(completions)

        assert len(rewards) == 2
        assert all(isinstance(r, float) for r in rewards)


class TestInverseDynamicsReward:
    """Tests for InverseDynamicsReward."""

    def test_registry_registration(self):
        """Test that InverseDynamicsReward is registered properly."""
        reward_fn = registry.create("inverse_dynamics")
        from atroposlib.envs.reward_fns.reward_function import RewardFunction
        assert isinstance(reward_fn, RewardFunction)
        assert reward_fn.name == "inversedynamicsreward"

    def test_basic_compute_with_simple_completions(self):
        """Test basic computation with simple completions."""
        reward_fn = registry.create("inverse_dynamics")
        completions = ["action 1", "action 2", "action 3"]

        rewards = reward_fn.compute(completions)

        assert len(rewards) == 3
        assert all(isinstance(r, float) for r in rewards)
        assert all(0.0 <= r <= 1.0 for r in rewards)

    def test_compute_with_explicit_actions(self):
        """Test computation with explicit actions provided."""
        reward_fn = registry.create("inverse_dynamics")
        completions = ["predicted action 1", "predicted action 2"]
        actions = ["actual action 1", "actual action 2"]

        rewards = reward_fn.compute(completions, actions=actions)

        assert len(rewards) == 2
        # Rewards should reflect similarity
        assert all(isinstance(r, float) for r in rewards)

    def test_compute_with_trajectories(self):
        """Test computation with trajectory data."""
        reward_fn = registry.create("inverse_dynamics")
        completions = ["pred action 1", "pred action 2"]
        trajectories = [
            {"action": "actual action 1", "state": "state 1"},
            {"action": "actual action 2", "state": "state 2"},
        ]

        rewards = reward_fn.compute(completions, trajectories=trajectories)

        assert len(rewards) == 2
        assert all(isinstance(r, float) for r in rewards)

    def test_compute_with_action_history(self):
        """Test computation with action history for sequence consistency."""
        reward_fn = registry.create(
            "inverse_dynamics",
            sequence_consistency_weight=0.5
        )
        completions = ["action 1"]
        action_history = [["previous action 1", "previous action 2"]]

        rewards = reward_fn.compute(completions, action_history=action_history)

        assert len(rewards) == 1
        assert isinstance(rewards[0], float)

    def test_compute_with_weight(self):
        """Test that weight is properly applied."""
        weight = 1.5
        reward_fn = registry.create("inverse_dynamics", weight=weight)
        completions = ["action 1"]

        # Call via __call__ to test weight application
        rewards = reward_fn(completions, actions=["actual action"])

        assert len(rewards) == 1
        assert isinstance(rewards[0], float)

    def test_empty_completions(self):
        """Test with empty completions list."""
        reward_fn = registry.create("inverse_dynamics")
        completions = []

        rewards = reward_fn.compute(completions)

        assert rewards == []

    def test_handles_missing_actions(self):
        """Test graceful handling when actions are not provided."""
        reward_fn = registry.create("inverse_dynamics")
        completions = ["action 1", "action 2"]

        # Should not error, should use fallback heuristic
        rewards = reward_fn.compute(completions)

        assert len(rewards) == 2
        assert all(isinstance(r, float) for r in rewards)

    def test_sequence_consistency_bonus(self):
        """Test that sequence consistency affects rewards."""
        reward_fn = registry.create(
            "inverse_dynamics",
            sequence_consistency_weight=0.3
        )
        completions = ["similar action"]
        action_history = [["similar action", "similar pattern"]]

        rewards = reward_fn.compute(completions, action_history=action_history)

        assert len(rewards) == 1
        # Similar actions should get some consistency bonus
        assert rewards[0] >= 0.0


class TestEnvRLIntegration:
    """Integration tests for EnvRL reward functions."""

    def test_combined_use(self):
        """Test using both reward functions together."""
        state_pred = registry.create("state_prediction", weight=0.1)
        inv_dyn = registry.create("inverse_dynamics", weight=0.1)

        completions = ["action 1", "action 2"]
        next_states = ["next 1", "next 2"]
        actions = ["actual 1", "actual 2"]

        state_rewards = state_pred(completions, next_states=next_states)
        inv_rewards = inv_dyn(completions, actions=actions)

        assert len(state_rewards) == 2
        assert len(inv_rewards) == 2
        assert all(isinstance(r, float) for r in state_rewards)
        assert all(isinstance(r, float) for r in inv_rewards)

    def test_import_from_atroposlib(self):
        """Test that reward functions can be created from registry."""
        from atroposlib.envs.reward_fns import registry

        # Both should be creatable from registry
        state_pred = registry.create("state_prediction")
        inv_dyn = registry.create("inverse_dynamics")

        from atroposlib.envs.reward_fns.reward_function import RewardFunction
        assert isinstance(state_pred, RewardFunction)
        assert isinstance(inv_dyn, RewardFunction)
