"""Reward function that scores completions by metacognitive faithfulness.

Adapted from "Reinforcement Learning with Metacognitive Feedback Elicits
Faithful Uncertainty Expression in LLMs" (arXiv:2606.32032). The paper's
RLMF paradigm refines completion rankings using a *metacognitive feedback*
signal: how faithfully a model's self-reported confidence matches whether
it was actually correct. This module delivers that core signal as a reward
function that auto-registers alongside ``accuracy_reward`` / ``format_reward``
and is invoked from ``DatasetEnv.score`` via ``registry.create(...)``
(environments/dataset_environment/dataset_env.py).

Mode-2 adaptation (substituted auxiliaries, per the implementation spec):
  * The paper elicits self-judgments with a separate LLM judge. That learned
    estimator is replaced by a parameter-free parser that reads a
    self-reported confidence score straight out of the completion text
    (a ``<confidence>0.8</confidence>`` tag or a ``Confidence: 80%`` line).
  * The paper's bespoke preference-optimization loop and its standalone
    faithful-calibration benchmark are out of scope: evaluation belongs in a
    downstream PR, and the faithfulness signal plugs into Atropos' existing
    GRPO / PipelineRL reward path through ``CombinedReward`` rather than a
    custom optimizer.

Core mechanism (kept at full fidelity): a Brier proper score rewards
calibrated confidence::

    faithfulness = 1 - (p - outcome) ** 2

where ``p`` in [0, 1] is the expressed confidence and ``outcome`` in {0, 1}
is whether the completion is correct. A model that is confidently right or
appropriately doubtful scores ~1.0; a model that is confidently wrong (or
skeptically right) scores near 0. For general tasks, combine this with
``accuracy_reward`` via ``CombinedReward`` so task correctness still
dominates -- mirroring the paper's "preserving accuracy" design.
"""

import logging
import re
from typing import Any, List, Optional, Union

from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)

# A confidence is a 1-3 digit integer, optionally with a decimal part.
_CONFIDENCE_NUMBER = r"(\d{1,3}(?:\.\d+)?)"


def _clamp_unit(value: float) -> float:
    """Clamp a scalar into the [0, 1] range used for confidence / reward."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def parse_confidence(text: str, tag: str = "confidence") -> Optional[float]:
    """Extract a self-reported confidence score in [0, 1] from ``text``.

    Recognised forms, in priority order:
      * an XML-style ``<tag>0.8</tag>`` or ``<tag>80%</tag>`` block,
      * a line-prefixed ``Confidence: 80%`` / ``confidence: 0.8`` statement,
      * a bracketed ``[confidence: 0.8]`` annotation.

    Returns ``None`` when no confidence statement is found.
    """
    if not isinstance(text, str):
        return None

    tag_re = re.escape(tag)
    patterns = [
        # <confidence>0.8</confidence> or <confidence>80%</confidence>
        rf"<{tag_re}>\s*{_CONFIDENCE_NUMBER}\s*%?\s*</{tag_re}>",
        # Confidence: 80% / confidence: 0.8  (at line start)
        rf"(?im)^\s*{tag_re}\s*[:=]\s*{_CONFIDENCE_NUMBER}\s*%?",
        # [confidence: 0.8] inline
        rf"\[\s*{tag_re}\s*[:=]\s*{_CONFIDENCE_NUMBER}\s*%?\s*\]",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if "%" in match.group(0):
            value /= 100.0
        return _clamp_unit(value)
    return None


def _normalize(text: str) -> str:
    """Lightweight normalisation for the proxy correctness comparison."""
    if not isinstance(text, str):
        return ""
    lowered = re.sub(r"\s+", " ", text.lower().strip())
    return lowered.strip("`'\".,;:!?")


def _extract_answer(text: str, answer_tag: str = "answer") -> str:
    """Best-effort answer extraction for the proxy correctness check."""
    if not isinstance(text, str):
        return ""
    tag_re = re.escape(answer_tag)
    match = re.search(rf"<{tag_re}>\s*(.*?)\s*</{tag_re}>", text, re.DOTALL)
    if match:
        return match.group(1)
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1]
    return text.strip()


def _is_correct(content: str, gold: Any, answer_tag: str = "answer") -> float:
    """Parameter-free correctness proxy in {0.0, 1.0}.

    Deliberately avoids ``math_verify`` / ``torch`` so the reward stays
    dependency-light. Callers needing exact verification should pass a
    precomputed ``correctness`` list (e.g. computed once via
    ``accuracy_reward``) and let this reward focus on the faithfulness signal.
    """
    if gold is None:
        return 0.0
    candidate = _normalize(_extract_answer(content, answer_tag))
    target = _normalize(str(gold))
    if not candidate or not target:
        return 0.0
    if candidate == target:
        return 1.0
    try:
        if abs(float(candidate) - float(target)) < 1e-6:
            return 1.0
    except ValueError:
        return 0.0
    return 0.0


@registry.register
class MetacognitiveReward(RewardFunction):
    """Reward completions for faithful (metacognitively calibrated) confidence.

    The reward is a Brier proper score over the model's expressed confidence
    versus its actual correctness. It is designed to compose with
    ``accuracy_reward`` (via ``CombinedReward``) for general tasks, surfacing
    the RLMF metacognitive-feedback signal inside Atropos' existing RL path.
    """

    def __init__(
        self,
        confidence_tag: str = "confidence",
        answer_tag: str = "answer",
        default_reward: float = 0.5,
        weight: float = 1.0,
        **kwargs,
    ):
        """Initialise the metacognitive reward.

        Args:
            confidence_tag: XML tag / keyword naming the self-reported
                confidence statement (default ``"confidence"``).
            answer_tag: XML tag wrapping the answer for the proxy correctness
                check (default ``"answer"``).
            default_reward: Neutral reward returned when a completion does not
                express any confidence (no shaping signal available).
            weight: Weight applied when combining with other rewards.
            **kwargs: Additional configuration.
        """
        super().__init__(weight=weight, **kwargs)
        self.confidence_tag = confidence_tag
        self.answer_tag = answer_tag
        self.default_reward = _clamp_unit(default_reward)

    def compute(
        self,
        completions: List[Any],
        solution: Optional[Union[str, List[str]]] = None,
        ground_truth: Optional[Union[str, List[str]]] = None,
        correctness: Optional[Union[List[Any], Any]] = None,
        **kwargs,
    ) -> List[float]:
        """Score each completion by faithfulness of its expressed confidence.

        Args:
            completions: Model completions to evaluate.
            solution / ground_truth: Reference answer(s) used for the proxy
                correctness check when ``correctness`` is not supplied.
            correctness: Optional precomputed correctness signal (bool / float
                / list thereof). When given, the reward stays purely on the
                metacognitive faithfulness and skips ground-truth matching.
            **kwargs: Additional context (``item``, ``config``, ...).

        Returns:
            One faithfulness score in [0, 1] per completion. Completions that
            do not express confidence receive ``default_reward``.
        """
        if not completions:
            return []

        # Normalise the optional precomputed correctness into a per-item list.
        correctness_per_item: List[Optional[float]] = [None] * len(completions)
        if correctness is not None:
            if isinstance(correctness, list):
                correctness_per_item = [None if c is None else float(bool(c)) for c in correctness]
            else:
                correctness_per_item = [float(bool(correctness))] * len(completions)

        gold = ground_truth if ground_truth is not None else solution
        golds = gold if isinstance(gold, list) else [gold] * len(completions)

        rewards: List[float] = []
        for idx, completion in enumerate(completions):
            content = self.get_content(completion)
            confidence = parse_confidence(content, tag=self.confidence_tag)
            if confidence is None:
                rewards.append(self.default_reward)
                continue

            if correctness_per_item[idx] is not None:
                outcome = correctness_per_item[idx]
            else:
                outcome = _is_correct(content, golds[idx], self.answer_tag)

            faithfulness = 1.0 - (confidence - outcome) ** 2
            rewards.append(_clamp_unit(faithfulness))

        if rewards:
            logger.info(
                "Metacognitive faithfulness: mean=%.3f over %d completions",
                sum(rewards) / len(rewards),
                len(rewards),
            )
        return rewards


# Legacy function wrapper for backward compatibility.
def metacognitive_reward(
    completions: List[Any],
    solution: Optional[Union[str, List[str]]] = None,
    ground_truth: Optional[Union[str, List[str]]] = None,
    correctness: Optional[Union[List[Any], Any]] = None,
    confidence_tag: str = "confidence",
    answer_tag: str = "answer",
    default_reward: float = 0.5,
    **kwargs,
) -> List[float]:
    """Legacy function wrapper for :class:`MetacognitiveReward`."""
    reward_fn = MetacognitiveReward(
        confidence_tag=confidence_tag,
        answer_tag=answer_tag,
        default_reward=default_reward,
    )
    return reward_fn.compute(
        completions,
        solution=solution,
        ground_truth=ground_truth,
        correctness=correctness,
        **kwargs,
    )
