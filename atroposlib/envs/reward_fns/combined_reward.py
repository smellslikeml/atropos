"""Combined reward function that combines multiple reward functions."""

import logging
from typing import Any, Dict, List, Optional, Union

from .floor_guarded_reward import apply_reward_floor
from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)


@registry.register
class CombinedReward(RewardFunction):
    """Meta reward function that combines multiple reward functions"""

    def __init__(
        self,
        rewards: List[Union[str, Dict]],
        normalization: str = "none",
        floor: Optional[Union[str, Dict]] = None,
        floor_value: float = -1.0,
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize with a list of reward functions to combine.

        Args:
            rewards: List of reward functions (names or config dicts)
            normalization: How to normalize rewards, one of:
                          - "none": No normalization
                          - "sum": Divide by sum of weights
                          - "minmax": Scale to range [0,1] based on min/max values
            floor: Optional deterministic reward-floor config (e.g.
                   ``{"type": "reward_floor", ...}``). When set, the combined
                   reward is clamped to ``floor_value`` for any completion the
                   floor flags as degenerate, as an override rather than an
                   additive penalty. Implements the S_r gating from
                   "Designing Reward Signals for Portable Query Generation"
                   (arXiv:2606.27291), which keeps the signal robust to
                   GRPO-style reward hacking.
            floor_value: Value assigned to degenerate completions when ``floor``
                         is set (the paper clamps to -1.0).
            weight: Weight for this combined reward
            **kwargs: Additional parameters
        """
        super().__init__(weight=weight, **kwargs)
        self.normalization = normalization
        self.reward_functions = []

        # Initialize all sub-reward functions
        for reward_config in rewards:
            self.reward_functions.append(registry.create(reward_config))

        # Optional deterministic floor applied as an override after combination.
        self.floor = registry.create(floor) if floor is not None else None
        self.floor_value = floor_value

    @property
    def name(self) -> str:
        """Get a descriptive name for this combined reward"""
        return f"combined({','.join(r.name for r in self.reward_functions)})"

    def set_wandb_logger(self, logger):
        """Propagate the WandB logger to all sub-rewards"""
        super().set_wandb_logger(logger)
        for reward_fn in self.reward_functions:
            reward_fn.set_wandb_logger(logger)
        if self.floor is not None:
            self.floor.set_wandb_logger(logger)

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        """Compute combined rewards by calling all sub-rewards"""
        if not completions:
            return []

        # Initialize with zeros
        combined_rewards = [0.0] * len(completions)

        # Collect all sub-reward values
        all_rewards = []
        for reward_fn in self.reward_functions:
            try:
                rewards = reward_fn.compute(completions, **kwargs)
                all_rewards.append(rewards)

                # Add to combined total (pre-normalization)
                for i, r in enumerate(rewards):
                    combined_rewards[i] += r
            except Exception as e:
                logger.error(f"Error computing reward for {reward_fn.name}: {e}")
                logger.exception(e)

        # Apply normalization if needed
        if self.normalization == "sum":
            total_weight = sum(r.weight for r in self.reward_functions)
            if total_weight > 0:
                combined_rewards = [r / total_weight for r in combined_rewards]
        elif self.normalization == "minmax":
            # Avoid division by zero
            reward_min = min(combined_rewards) if combined_rewards else 0
            reward_max = max(combined_rewards) if combined_rewards else 0
            if reward_max > reward_min:
                combined_rewards = [
                    (r - reward_min) / (reward_max - reward_min)
                    for r in combined_rewards
                ]

        # Deterministic floor override: clamp degenerate completions after
        # combination so a high sub-reward cannot rescue a reward-hacking
        # completion (arXiv:2606.27291).
        if self.floor is not None:
            degenerate_mask = self.floor.detect(completions, **kwargs)
            combined_rewards = apply_reward_floor(
                combined_rewards, degenerate_mask, self.floor_value
            )

        return combined_rewards
