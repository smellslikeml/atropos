"""Reward for the *accuracy of a model's self-assessment* of its own performance.

Adapted from "Reinforcement Learning with Metacognitive Feedback Elicits
Faithful Uncertainty Expression in LLMs" (arXiv:2606.32032). The paper's RLMF
paradigm hinges on a *metacognitive-accuracy* term that scores how well a model
judges its own performance::

    Z_g = 1 - (F_pred - F_gold) ** 2

where ``F_gold`` is the completion's measured faithful-calibration level and
``F_pred`` is the model's *own self-prediction* of that level. This is the
signal the paper calls monitoring "one level above output confidence": beyond
expressing calibrated confidence in an answer, the model estimates how well
calibrated it is, and is rewarded when that meta-estimate is accurate.

The sibling module :mod:`.metacognitive_reward` already delivers the paper's
*first-order* faithfulness (expressed confidence vs. correctness). This module
adds the *second-order* metacognitive-accuracy term that was the paper's
headline contribution, and it reuses ``MetacognitiveReward`` to supply
``F_gold`` so the two stay consistent. Like every reward here it auto-registers
and is resolvable from ``DatasetEnv.score`` via ``registry.create(...)``
(environments/dataset_environment/dataset_env.py), typically composed with the
first-order reward through ``CombinedReward``.

Mode-2 adaptation (substituted auxiliaries, per the implementation spec):
  * The paper elicits ``F_pred`` from a separate LLM self-judge. That learned
    estimator is replaced by a parameter-free parser that reads a self-reported
    assessment straight out of the completion (a ``<self_assessment>0.7</...>``
    tag or a ``Self-assessment: 70%`` line), reusing this package's existing
    confidence parser.
  * The paper's gold FC level (``F_gold``) is computed from intrinsic
    confidence via multi-sample estimation. That auxiliary is substituted with
    the target-native first-order faithfulness already provided by
    ``MetacognitiveReward`` (or a caller-supplied precomputed ``faithfulness`` /
    ``correctness`` signal), keeping the module dependency-light.

Intentionally out of scope (belongs to the trainer, not a reward): the paper's
asymmetric ``A_RLMF`` advantage re-weighting that only *amplifies*
above-average-faithfulness completions. Atropos is a reward/environments
framework and does not own the GRPO advantage computation, so this module
surfaces the metacognitive-accuracy *score* ``Z_g`` for the RL path to consume
like any other reward rather than injecting scaling into the optimizer.
"""

import logging
from typing import Any, List, Optional, Union

from .metacognitive_reward import MetacognitiveReward, _clamp_unit, parse_confidence
from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)


@registry.register
class MetacognitiveAccuracyReward(RewardFunction):
    """Reward completions for accurately self-predicting their own faithfulness.

    For each completion the reward parses ``F_pred`` (the model's self-assessed
    performance level) and compares it against ``F_gold`` (the measured level)
    with the paper's proper-score form ``Z_g = 1 - (F_pred - F_gold) ** 2``. A
    model that knows when it is well or poorly calibrated scores ~1.0; a model
    whose self-assessment is far from reality scores near 0.

    ``F_gold`` is resolved in priority order:
      1. a caller-supplied precomputed ``faithfulness`` signal (per item), else
      2. a caller-supplied ``correctness`` signal (per item), else
      3. the first-order faithfulness from :class:`MetacognitiveReward`,
         computed on the same completions from ``solution`` / ``ground_truth``.
    """

    def __init__(
        self,
        self_assessment_tag: str = "self_assessment",
        confidence_tag: str = "confidence",
        answer_tag: str = "answer",
        default_reward: float = 0.5,
        weight: float = 1.0,
        **kwargs,
    ):
        """Initialise the metacognitive-accuracy reward.

        Args:
            self_assessment_tag: XML tag / keyword naming the model's
                self-predicted performance level ``F_pred`` (default
                ``"self_assessment"``).
            confidence_tag: Tag for the first-order confidence statement,
                forwarded to :class:`MetacognitiveReward` when it computes
                ``F_gold`` (default ``"confidence"``).
            answer_tag: XML tag wrapping the answer, forwarded to
                :class:`MetacognitiveReward` for its proxy correctness check.
            default_reward: Neutral reward returned when a completion does not
                express any self-assessment (no metacognitive signal available).
            weight: Weight applied when combining with other rewards.
            **kwargs: Additional configuration.
        """
        super().__init__(weight=weight, **kwargs)
        self.self_assessment_tag = self_assessment_tag
        self.confidence_tag = confidence_tag
        self.answer_tag = answer_tag
        self.default_reward = _clamp_unit(default_reward)
        # Reuse the first-order reward to source F_gold consistently. Its
        # neutral default is irrelevant here: we only read F_gold for a
        # completion once it has expressed a self-assessment.
        self._gold_reward = MetacognitiveReward(
            confidence_tag=confidence_tag,
            answer_tag=answer_tag,
            default_reward=default_reward,
        )

    def _resolve_gold(
        self,
        completions: List[Any],
        faithfulness: Optional[Union[List[Any], Any]],
        correctness: Optional[Union[List[Any], Any]],
        solution: Optional[Union[str, List[str]]],
        ground_truth: Optional[Union[str, List[str]]],
    ) -> List[float]:
        """Return the measured F_gold level in [0, 1] for each completion."""
        n = len(completions)

        def _broadcast(signal: Union[List[Any], Any]) -> List[float]:
            values = signal if isinstance(signal, list) else [signal] * n
            return [_clamp_unit(float(v)) for v in values]

        if faithfulness is not None:
            return _broadcast(faithfulness)
        if correctness is not None:
            # Correctness is a bool/float in {0, 1}; treat it as the gold level.
            values = correctness if isinstance(correctness, list) else [correctness] * n
            return [_clamp_unit(float(bool(v))) for v in values]
        # Fall back to the first-order faithfulness the sibling reward computes.
        return self._gold_reward.compute(
            completions, solution=solution, ground_truth=ground_truth
        )

    def compute(
        self,
        completions: List[Any],
        solution: Optional[Union[str, List[str]]] = None,
        ground_truth: Optional[Union[str, List[str]]] = None,
        correctness: Optional[Union[List[Any], Any]] = None,
        faithfulness: Optional[Union[List[Any], Any]] = None,
        **kwargs,
    ) -> List[float]:
        """Score each completion by the accuracy of its self-assessment.

        Args:
            completions: Model completions to evaluate.
            solution / ground_truth: Reference answer(s) used to compute the
                first-order ``F_gold`` when neither ``faithfulness`` nor
                ``correctness`` is supplied.
            correctness: Optional precomputed correctness signal used as
                ``F_gold`` when ``faithfulness`` is absent.
            faithfulness: Optional precomputed first-order faithfulness signal
                used directly as ``F_gold`` (highest priority).
            **kwargs: Additional context (``item``, ``config``, ...).

        Returns:
            One metacognitive-accuracy score ``Z_g`` in [0, 1] per completion.
            Completions with no self-assessment receive ``default_reward``.
        """
        if not completions:
            return []

        golds = self._resolve_gold(
            completions, faithfulness, correctness, solution, ground_truth
        )

        rewards: List[float] = []
        scored = 0
        for idx, completion in enumerate(completions):
            content = self.get_content(completion)
            f_pred = parse_confidence(content, tag=self.self_assessment_tag)
            if f_pred is None:
                rewards.append(self.default_reward)
                continue
            f_gold = golds[idx]
            z_g = 1.0 - (f_pred - f_gold) ** 2
            rewards.append(_clamp_unit(z_g))
            scored += 1

        if scored:
            logger.info(
                "Metacognitive accuracy (Z_g): mean=%.3f over %d/%d self-assessed completions",
                sum(rewards) / len(rewards),
                scored,
                len(completions),
            )
        return rewards
