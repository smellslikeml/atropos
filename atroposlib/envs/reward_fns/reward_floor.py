"""
Reward floor for preventing reward-hacking in GRPO training.

This module implements a deterministic, rule-based reward floor to mitigate
GRPO's vulnerability to spurious reward signals. As described in:

    "Designing Reward Signals for Portable Query Generation: A Case Study in
    Industrial Semantic Job Search" (arXiv:2606.27291)

GRPO's group-relative advantage normalization makes it uniquely sensitive to
reward hacking compared to other critic-free optimizers. This reward floor
applies deterministic penalties for degenerate behaviors that exploit LLM-as-judge
reward signals, preventing the policy from learning undesirable patterns.

The reward floor works by setting a maximum cap on rewards for completions that
exhibit degenerate behaviors, effectively preventing the policy from being
reinforced for exploiting judge flaws.
"""

import difflib
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set

from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)


class DegeneracyDetector:
    """
    Collection of deterministic rule-based detectors for degenerate behaviors.

    Each detector identifies a specific pattern of reward-hacking behavior:
    - Verbatim copying: Completion too similar to input/reference
    - Empty/minimal: Completion lacks substantive content
    - Excessive repetition: Same content repeated (stuttering)
    - Template exploitation: Completion relies on format rather than content
    """

    @staticmethod
    def verbatim_copy_ratio(
        completion: str,
        reference: Optional[str] = None,
        prompt: Optional[str] = None,
        threshold: float = 0.9,
    ) -> float:
        """
        Calculate how much of the completion is copied from reference/prompt.

        Uses SequenceMatcher to find the longest common subsequences between
        completion and reference text. High ratio suggests verbatim copying
        exploitation.

        Args:
            completion: The model's completion text
            reference: Optional reference answer to check against
            prompt: Optional prompt text to check against
            threshold: Ratio above which copying is considered degenerate

        Returns:
            float: Ratio of completion that is copied (0.0 to 1.0)
        """
        if not completion:
            return 0.0

        completion_clean = re.sub(r"\s+", " ", completion.lower().strip())
        max_ratio = 0.0

        # Check against reference if provided
        if reference:
            reference_clean = re.sub(r"\s+", " ", reference.lower().strip())
            if reference_clean:
                matcher = difflib.SequenceMatcher(
                    None, completion_clean, reference_clean
                )
                ratio = matcher.ratio()
                max_ratio = max(max_ratio, ratio)

        # Check against prompt if provided
        if prompt:
            prompt_clean = re.sub(r"\s+", " ", prompt.lower().strip())
            if prompt_clean:
                matcher = difflib.SequenceMatcher(None, completion_clean, prompt_clean)
                ratio = matcher.ratio()
                max_ratio = max(max_ratio, ratio)

        return max_ratio

    @staticmethod
    def is_empty_or_minimal(
        completion: str, min_length: int = 10, min_words: int = 2
    ) -> bool:
        """
        Detect if completion is empty or lacks substantive content.

        Args:
            completion: The model's completion text
            min_length: Minimum character length for non-degenerate
            min_words: Minimum word count for non-degenerate

        Returns:
            bool: True if completion is empty or minimal
        """
        if not completion:
            return True

        completion_clean = completion.strip()
        if len(completion_clean) < min_length:
            return True

        words = re.findall(r"\b\w+\b", completion_clean)
        if len(words) < min_words:
            return True

        return False

    @staticmethod
    def consecutive_repetition_ratio(completion: str, window: int = 3) -> float:
        """
        Detect excessive consecutive repetition (stuttering).

        Args:
            completion: The model's completion text
            window: Size of sliding window to check for repeats

        Returns:
            float: Ratio of positions that are repetitive (0.0 to 1.0)
        """
        if not completion:
            return 0.0

        words = re.findall(r"\b\w+\b", completion.lower())
        if len(words) < window * 2:
            return 0.0

        repetitive_count = 0
        total_checks = 0

        for i in range(len(words) - window):
            current_window = " ".join(words[i : i + window])
            next_window = " ".join(words[i + window : i + 2 * window])

            if current_window == next_window:
                repetitive_count += 1
            total_checks += 1

        return repetitive_count / total_checks if total_checks > 0 else 0.0

    @staticmethod
    def template_exploitation_score(
        completion: str,
        template_markers: Optional[Set[str]] = None,
        content_threshold: float = 0.2,
    ) -> float:
        """
        Detect if completion relies on template markers without real content.

        This checks if the completion consists primarily of formatting templates
        (like "The answer is: \boxed{}") without substantive reasoning.

        Args:
            completion: The model's completion text
            template_markers: Set of template marker patterns
            content_threshold: Minimum ratio of non-template content

        Returns:
            float: Score from 0.0 (no exploitation) to 1.0 (full exploitation)
        """
        if not completion:
            return 1.0

        # Default template markers for common answer formats
        if template_markers is None:
            template_markers = {
                r"\\boxed\{[^}]*\}",
                r"####\s*\S+",
                r"the answer is:?\s*$",
                r"answer:?\s*$",
            }

        completion_lower = completion.lower()

        # Count template matches
        template_chars = 0
        for pattern in template_markers:
            matches = re.findall(pattern, completion_lower, re.IGNORECASE)
            for match in matches:
                template_chars += len(match)

        # If template content exceeds threshold, flag as exploitation
        total_chars = len(completion)
        if total_chars == 0:
            return 1.0

        template_ratio = template_chars / total_chars
        content_ratio = 1.0 - template_ratio

        return 1.0 - min(1.0, content_ratio / content_threshold)


@registry.register
class RewardFloor(RewardFunction):
    """
    Deterministic reward floor to prevent reward-hacking in GRPO training.

    Implements the core insight from "Designing Reward Signals for Portable
    Query Generation" (arXiv:2606.27291): GRPO's group-relative advantage
    normalization makes it uniquely vulnerable to reward hacking. A rule-based
    reward floor corrects for spurious rewards assigned to degenerate behaviors.

    Usage:
        # Apply floor to verbatim copying
        floor = RewardFloor(rules=["verbatim_copy"], floor_value=0.3)

        # Apply floor to multiple degeneracy types
        floor = RewardFloor(
            rules=["verbatim_copy", "empty_minimal", "repetition"],
            floor_value=0.2
        )

        # Compose with existing reward via CombinedReward
        combined = CombinedReward([
            {"type": "accuracy", "weight": 1.0},
            {"type": "reward_floor", "weight": 1.0, "params": {...}}
        ])
    """

    def __init__(
        self,
        rules: Optional[List[str]] = None,
        floor_value: float = 0.0,
        penalty_value: float = 0.0,
        verbatim_threshold: float = 0.85,
        repetition_threshold: float = 0.3,
        min_completion_length: int = 10,
        min_completion_words: int = 2,
        custom_detectors: Optional[Dict[str, Callable]] = None,
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the reward floor.

        Args:
            rules: List of degeneracy rules to enforce. Options:
                   - "verbatim_copy": Penalize copying from reference/prompt
                   - "empty_minimal": Penalize empty or minimal completions
                   - "repetition": Penalize excessive consecutive repetition
                   - "template": Penalize template exploitation
                   If None, all rules are enabled.
            floor_value: Maximum reward value for degenerate completions.
                         Completions flagged as degenerate receive this value
                         instead of their original reward.
            penalty_value: Direct penalty value to apply (negative values reduce reward).
                          Alternative to floor_value; if set, this value is returned
                          for degenerate completions instead of capping.
            verbatim_threshold: Ratio (0-1) above which copying is flagged as degenerate
            repetition_threshold: Ratio (0-1) above which repetition is flagged
            min_completion_length: Minimum character length for non-degenerate
            min_completion_words: Minimum word count for non-degenerate
            custom_detectors: Optional dict of custom detector functions
            weight: Weight for this reward function
            **kwargs: Additional configuration
        """
        super().__init__(weight=weight, **kwargs)

        # Default: enable all rules if none specified
        self.rules = rules if rules is not None else [
            "verbatim_copy",
            "empty_minimal",
            "repetition",
        ]

        # Validate rules
        valid_rules = {"verbatim_copy", "empty_minimal", "repetition", "template"}
        for rule in self.rules:
            if rule not in valid_rules:
                logger.warning(f"Unknown rule '{rule}', ignoring")

        self.floor_value = floor_value
        self.penalty_value = penalty_value
        self.verbatim_threshold = verbatim_threshold
        self.repetition_threshold = repetition_threshold
        self.min_completion_length = min_completion_length
        self.min_completion_words = min_completion_words
        self.custom_detectors = custom_detectors or {}

    def detect(self, completions: List[Any], **kwargs) -> List[bool]:
        """
        Detect degenerate reward-hacking behavior in each completion.

        This is the deterministic, rule-based core of the reward floor. It
        returns a boolean mask (one entry per completion) so downstream
        composition can *clamp* rewards on a hard override rather than folding
        the floor in additively. See ``FloorGuardedReward`` for that wiring.

        Args:
            completions: List of completions to evaluate
            **kwargs: Additional context including:
                      - reference / solution: reference text for verbatim check
                      - prompt: Prompt text for verbatim check

        Returns:
            List of booleans, True where the completion is degenerate.
        """
        # Extract context for degeneracy detection
        reference = kwargs.get("reference") or kwargs.get("solution")
        prompt = kwargs.get("prompt")

        # Handle batch vs single reference
        if (
            reference
            and isinstance(reference, list)
            and len(reference) == len(completions)
        ):
            references = reference
        elif reference:
            references = [reference] * len(completions)
        else:
            references = [None] * len(completions)

        flags = []
        for i, completion in enumerate(completions):
            try:
                content = self.get_content(completion)
                ref_text = references[i] if i < len(references) else None

                # Check if completion passes all enabled rules
                is_degenerate = False

                # Rule: Verbatim copying
                if "verbatim_copy" in self.rules:
                    copy_ratio = DegeneracyDetector.verbatim_copy_ratio(
                        content,
                        reference=ref_text,
                        prompt=prompt,
                        threshold=self.verbatim_threshold,
                    )
                    if copy_ratio >= self.verbatim_threshold:
                        is_degenerate = True
                        logger.info(
                            f"Reward floor triggered: verbatim copying "
                            f"(ratio={copy_ratio:.2f} >= {self.verbatim_threshold})"
                        )

                # Rule: Empty or minimal completion
                if not is_degenerate and "empty_minimal" in self.rules:
                    if DegeneracyDetector.is_empty_or_minimal(
                        content,
                        min_length=self.min_completion_length,
                        min_words=self.min_completion_words,
                    ):
                        is_degenerate = True
                        logger.info(
                            "Reward floor triggered: empty or minimal completion"
                        )

                # Rule: Excessive repetition
                if not is_degenerate and "repetition" in self.rules:
                    rep_ratio = DegeneracyDetector.consecutive_repetition_ratio(
                        content
                    )
                    if rep_ratio >= self.repetition_threshold:
                        is_degenerate = True
                        logger.info(
                            f"Reward floor triggered: excessive repetition "
                            f"(ratio={rep_ratio:.2f} >= {self.repetition_threshold})"
                        )

                # Rule: Template exploitation
                if not is_degenerate and "template" in self.rules:
                    template_score = DegeneracyDetector.template_exploitation_score(
                        content
                    )
                    if template_score > 0.7:  # High exploitation
                        is_degenerate = True
                        logger.info(
                            f"Reward floor triggered: template exploitation "
                            f"(score={template_score:.2f})"
                        )

                # Check custom detectors
                if not is_degenerate:
                    for detector_name, detector_fn in self.custom_detectors.items():
                        try:
                            if detector_fn(content, ref_text):
                                is_degenerate = True
                                logger.info(
                                    f"Reward floor triggered: custom detector "
                                    f"'{detector_name}'"
                                )
                                break
                        except Exception as e:
                            logger.warning(
                                f"Custom detector '{detector_name}' failed: {e}"
                            )

                flags.append(is_degenerate)

            except Exception as e:
                logger.error(f"Error in reward floor detection: {e}")
                logger.exception(e)
                # On error, conservatively treat the completion as degenerate
                flags.append(True)

        return flags

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        """
        Compute reward floor values for completions.

        Args:
            completions: List of completions to evaluate
            **kwargs: Additional context (reference / solution / prompt) used
                      by :meth:`detect`.

        Returns:
            List of reward values. Degenerate completions receive
            ``penalty_value`` (if set) or ``floor_value``; others receive 1.0
            (passing the floor, then scaled by weight in ``__call__``).
        """
        degenerate_value = (
            self.penalty_value if self.penalty_value != 0 else self.floor_value
        )
        return [
            degenerate_value if flag else 1.0
            for flag in self.detect(completions, **kwargs)
        ]


# Legacy function for backward compatibility
def reward_floor(
    completions: List[Any],
    rules: Optional[List[str]] = None,
    floor_value: float = 0.0,
    **kwargs,
) -> List[float]:
    """
    Legacy function wrapper for RewardFloor.

    Args:
        completions: List of completions to evaluate
        rules: List of degeneracy rules to enforce
        floor_value: Maximum reward for degenerate completions
        **kwargs: Additional parameters

    Returns:
        List of reward values
    """
    reward_fn = RewardFloor(rules=rules, floor_value=floor_value)
    return reward_fn.compute(completions, **kwargs)
