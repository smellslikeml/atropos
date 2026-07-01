"""Rank-calibrated reward shaping for score-based tasks without ground-truth solutions.

Adapted from RiVER -- "Reinforcement Learning without Ground-Truth Solutions can
Improve LLMs" (arXiv:2606.27369). RiVER trains LLMs on score-based optimization
tasks (e.g. AtCoder Heuristic Contest execution scores) using deterministic
execution feedback as continuous-valued supervision, with no ground-truth solution
required to assign a reward.

When group-relative RL is applied to such continuous rewards, RiVER identifies two
failure modes that this reward function counters directly at the reward layer:

* scale dominance -- raw score magnitudes differ wildly across instances, so a few
  large-magnitude instances dominate policy updates. We calibrate *within each
  instance group* (the completions passed to a single ``compute`` call), so every
  instance contributes on a comparable scale regardless of absolute magnitude.

* frequency dominance -- frequently sampled mediocre solutions can accumulate
  advantage and crowd out rare but stronger ones. A rank-based emphasis gives the
  top-ranked solver the full reward while keeping bounded, non-negative feedback
  for the other valid solvers.

The shaping is instance-wise: only completions evaluated against the same instance
are compared, so call this once per prompt/group. The lower-level group-relative
advantage code (see ``atroposlib.utils.advantages``) then centers these calibrated
rewards. Per-instance calibration makes that global centering benign -- the part
that would otherwise reintroduce scale dominance is removed here.
"""

import logging
import re
from typing import Any, List, Optional, Sequence, Union

from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)

Number = Union[int, float]

# Matches the last numeric token in a completion's content, used only as a
# convenience fallback when no explicit ``scores`` are provided.
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _coerce_scores(
    completions: List[Any],
    scores: Optional[Union[Number, Sequence[Number]]],
) -> List[float]:
    """Resolve one float score per completion from explicit values or parsed content.

    Execution-feedback scores are normally passed via the ``scores`` kwarg: a scalar
    is broadcast to every completion, and a list must align with ``completions``.
    When omitted, a numeric score is parsed from each completion's content as a
    convenience for environments that print a score in the model output.
    """
    n = len(completions)

    if scores is None:
        parsed: List[float] = []
        for completion in completions:
            content = RewardFunction.get_content(completion)
            tokens = _NUMBER_RE.findall(content)
            parsed.append(float(tokens[-1]) if tokens else 0.0)
        return parsed

    if isinstance(scores, (int, float)):
        return [float(scores)] * n

    score_list = list(scores)
    if len(score_list) != n:
        raise ValueError(
            f"scores has length {len(score_list)} but {n} completions were given"
        )
    return [float(s) for s in score_list]


@registry.register
class RankCalibratedReward(RewardFunction):
    """Calibrated, group-relative reward for continuous execution scores.

    The reward for each completion within the group is::

        c_i = (s_i - s_min) / (s_max - s_min)          # in [0, 1], instance-wise
        r_i = min_reward + (c_i ** emphasis) * (max_reward - min_reward)

    where ``s_i`` is the raw execution score of completion *i*. Min-max normalizing
    within the group makes the reward scale-invariant per instance (counters scale
    dominance); the ``emphasis`` exponent sharpens the gap so the top-ranked solver
    receives the full reward while the other valid solvers keep bounded feedback
    (counters frequency dominance). Set ``emphasis=1.0`` for pure linear
    calibration with no extra top-emphasis.
    """

    def __init__(
        self,
        emphasis: float = 2.0,
        min_reward: float = 0.0,
        max_reward: float = 1.0,
        higher_is_better: bool = True,
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            emphasis: Exponent >= 1 applied to the calibrated score; larger values
                emphasize the top-ranked solver. 1.0 disables emphasis.
            min_reward: Reward assigned to the weakest completion in the group.
            max_reward: Reward assigned to the top-ranked solver.
            higher_is_better: Whether larger raw scores are better.
            weight: Importance factor (applied by ``RewardFunction.__call__``).
            **kwargs: Additional ``RewardFunction`` configuration.
        """
        super().__init__(weight=weight, **kwargs)
        if emphasis < 1.0:
            raise ValueError("emphasis must be >= 1.0")
        if max_reward < min_reward:
            raise ValueError("max_reward must be >= min_reward")
        self.emphasis = float(emphasis)
        self.min_reward = float(min_reward)
        self.max_reward = float(max_reward)
        self.higher_is_better = bool(higher_is_better)

    def compute(
        self,
        completions: List[Any],
        scores: Optional[Union[Number, Sequence[Number]]] = None,
        **kwargs,
    ) -> List[float]:
        """Return rank-calibrated rewards aligned with ``completions``.

        Args:
            completions: The group of completions for a single instance.
            scores: Raw execution score per completion. A scalar is broadcast to
                all completions; a list must align with ``completions``. If
                omitted, a score is parsed from each completion's content.
            **kwargs: Additional context (ignored).

        Returns:
            One calibrated reward per completion, in ``[min_reward, max_reward]``.
        """
        if not completions:
            return []

        raw = _coerce_scores(completions, scores)
        lo = min(raw)
        hi = max(raw)
        span = hi - lo

        # No signal within the group: the completions are indistinguishable, so
        # return bounded, equal feedback (the advantage centers to ~0 downstream).
        if span <= 0.0:
            return [self.min_reward] * len(completions)

        rewards: List[float] = []
        for s in raw:
            if self.higher_is_better:
                calibrated = (s - lo) / span
            else:
                calibrated = (hi - s) / span
            emphasized = calibrated**self.emphasis
            rewards.append(
                self.min_reward + emphasized * (self.max_reward - self.min_reward)
            )
        return rewards


# Legacy function wrapper for backward compatibility with function-style use.
def rank_calibrated_reward(
    completions: List[Any],
    scores: Optional[Union[Number, Sequence[Number]]] = None,
    **kwargs,
) -> List[float]:
    """Legacy function wrapper for ``RankCalibratedReward``."""
    reward_fn = RankCalibratedReward()
    return reward_fn.compute(completions, scores=scores, **kwargs)
