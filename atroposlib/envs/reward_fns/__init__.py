"""
Reward functions for evaluating model outputs in various environments.

This module provides a framework for creating, composing, and applying reward functions
to evaluate model outputs. Reward functions can be used for both dataset environments
and online/gymnasium environments.

Key components:
- RewardFunction: Abstract base class for all reward functions
- RewardRegistry: Registry for registering and loading reward functions
- CombinedReward: Meta reward function that combines multiple reward functions

Built-in registered reward types include:
- accuracy, format, r1, cascading_r1_math, reasoning_steps,
  repetition_penalty, cosine_scaled, crossword_format, chandas_meter
- cde: Curiosity-Driven Exploration bonus for RLVR
  (https://arxiv.org/abs/2509.09675v1). Adds the paper's actor-wise
  perplexity bonus and critic-wise multi-head value-variance bonus on top
  of a verifiable reward. Accepts optional ``logprobs`` (per-completion
  token logprobs under the actor policy, for the exact actor bonus) and
  ``values`` (per-completion list of multi-head critic value estimates,
  for the critic bonus) kwargs at compute time; without them the actor
  bonus falls back to a bigram perplexity proxy and the critic bonus is 0.

Usage:
    # Define a reward function
    @registry.register
    class MyReward(RewardFunction):
        def compute(self, completions, **kwargs):
            # Implementation
            return [score for completion in completions]

    # Create and use a reward function
    reward_fn = registry.create("my_reward", weight=1.5)
    scores = reward_fn(completions, **kwargs)

    # Compose the CDE exploration bonus with a verifiable reward
    combined = CombinedReward(
        rewards=[
            {"type": "accuracy"},
            {"type": "cde", "params": {"actor_weight": 1.0, "critic_weight": 0.5}},
        ]
    )
    scores = combined(completions, logprobs=logprobs, values=values)
"""

from .combined_reward import CombinedReward
from .registry import registry
from .reward_function import RewardFunction

__all__ = ["RewardFunction", "registry", "CombinedReward"]
