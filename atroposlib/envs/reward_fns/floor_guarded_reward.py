"""
Deterministic reward-floor override for composed reward signals.

Implements the central mechanism from:

    "Designing Reward Signals for Portable Query Generation: A Case Study in
    Industrial Semantic Job Search" (arXiv:2606.27291)

The paper's total reward is a judge/rubric reward *gated* by a deterministic,
rule-based floor (S_r): when a degenerate reward-hacking pattern is detected,
the reward is clamped to a fixed floor value regardless of what the judge
awarded. The paper shows this override -- not an additive penalty -- is what
keeps the reward signal portable across critic-free optimizers, and in
particular stops GRPO's group-relative advantage normalization from
amplifying spurious judge rewards (a rule-based floor delivered a +0.147
quality gain that outweighed the choice of RL algorithm).

``FloorGuardedReward`` composes any base reward with a ``RewardFloor`` detector
using this override semantics, so a high base score can never rescue a
completion the floor has flagged as degenerate. This differs from routing the
floor through ``CombinedReward`` additively, where a large judge reward can
still net positive despite the penalty.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from .registry import registry
from .reward_floor import RewardFloor
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)


def apply_reward_floor(
    rewards: List[float],
    degenerate_mask: List[bool],
    floor_value: float = -1.0,
) -> List[float]:
    """
    Clamp rewards to ``floor_value`` wherever the degeneracy mask is set.

    This is the deterministic override at the heart of the paper's S_r: a
    flagged completion's reward is replaced outright, not merely penalized, so
    the judge/base reward cannot outweigh the floor.

    Args:
        rewards: Base rewards, one per completion.
        degenerate_mask: Booleans, True where the completion is degenerate.
        floor_value: Value assigned to degenerate completions.

    Returns:
        A new list of rewards with degenerate entries clamped to ``floor_value``.
    """
    return [
        floor_value if (i < len(degenerate_mask) and degenerate_mask[i]) else r
        for i, r in enumerate(rewards)
    ]


@registry.register
class FloorGuardedReward(RewardFunction):
    """
    A base reward gated by a deterministic reward floor (override semantics).

    Reproduces the paper's total reward ``r_total = S_r-gated r_rubric``: the
    base reward is computed normally, but any completion the floor flags as
    degenerate has its reward clamped to ``floor_value`` regardless of the base
    score. This is the override behavior the paper identifies as the portable,
    optimizer-agnostic fix for reward hacking.

    Usage:
        # Guard an accuracy (or LLM-judge / rubric) reward with the floor.
        reward = FloorGuardedReward(
            base={"type": "accuracy", "weight": 1.0},
            floor={"type": "reward_floor", "rules": ["verbatim_copy"]},
            floor_value=-1.0,
        )
        scores = reward(completions, reference=refs)

        # `base` may be a list, in which case it is wrapped in a CombinedReward.
        reward = FloorGuardedReward(base=[{"type": "accuracy"}, {"type": "format"}])
    """

    def __init__(
        self,
        base: Union[str, Dict, List[Union[str, Dict]]],
        floor: Optional[Union[str, Dict]] = None,
        floor_value: float = -1.0,
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the floor-guarded reward.

        Args:
            base: The reward being protected (name, config dict, or a list of
                  configs, which are wrapped in a ``CombinedReward``). Stands in
                  for the paper's r_rubric judge reward.
            floor: Config for the deterministic floor detector. Defaults to a
                   ``RewardFloor`` with its default rules (the paper's S_r).
            floor_value: Value assigned to completions the floor flags as
                         degenerate. The paper clamps to -1.0.
            weight: Weight for this reward function.
            **kwargs: Additional configuration.
        """
        super().__init__(weight=weight, **kwargs)

        # Base reward being protected (e.g. an LLM-judge / rubric reward).
        if isinstance(base, list):
            self.base = registry.create({"type": "combined", "rewards": base})
        else:
            self.base = registry.create(base)

        # Deterministic floor detector (paper's S_r).
        self.floor = registry.create(floor) if floor is not None else RewardFloor()
        self.floor_value = floor_value

    @property
    def name(self) -> str:
        """Descriptive name reflecting the guarded base reward."""
        return self._name or f"floor_guarded({self.base.name})"

    def set_wandb_logger(self, logger):
        """Propagate the WandB logger to the base and floor rewards."""
        super().set_wandb_logger(logger)
        self.base.set_wandb_logger(logger)
        self.floor.set_wandb_logger(logger)

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        """
        Compute base rewards, then clamp degenerate completions to the floor.

        Args:
            completions: List of completions to evaluate.
            **kwargs: Context forwarded to both the base reward and the floor
                      detector (e.g. ``reference``, ``prompt``, ``solution``).

        Returns:
            List of rewards with degenerate completions clamped to
            ``floor_value``.
        """
        if not completions:
            return []

        base_rewards = self.base.compute(completions, **kwargs)
        degenerate_mask = self.floor.detect(completions, **kwargs)
        return apply_reward_floor(base_rewards, degenerate_mask, self.floor_value)
