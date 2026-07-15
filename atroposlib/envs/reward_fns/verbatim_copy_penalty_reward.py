"""
Reward function for penalizing verbatim copying in completions.

Based on insights from "Designing Reward Signals for Portable Query Generation:
A Case Study in Industrial Semantic Job Search" (arXiv:2606.27291).

The paper demonstrates that deterministic, rule-based reward floors can mitigate
reward hacking in RLAIF systems, particularly for GRPO-based optimization. This
function implements n-gram overlap detection as a proxy for verbatim copying,
providing a reward floor that corrects for spurious rewards assigned to
degenerate copying behaviors.

Key mechanism: When a completion contains high n-gram overlap with reference
or source material, it likely represents verbatim copying rather than genuine
generation. This function detects such patterns and applies a penalty, serving
as a reward floor that prevents policies from exploiting LLM-as-judge rubrics
through superficial copying.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set

from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)


@registry.register
class VerbatimCopyPenaltyReward(RewardFunction):
    """
    Reward function that penalizes verbatim copying in completions.

    Uses n-gram overlap analysis to detect when a completion contains
    substantial copied content from reference or source material. This
    implements a deterministic reward floor as described in the paper,
    which helps prevent reward hacking in RLAIF systems.

    The penalty scales based on:
    1. The proportion of n-grams in the completion that appear in reference text
    2. The length of copied sequences (longer copies = higher penalty)
    3. The concentration of copying (one long copy vs many short fragments)

    Attributes:
        n: N-gram size for overlap detection (default: 6, matching the paper)
        threshold: Minimum overlap ratio to trigger penalty (default: 0.3)
        penalty_scale: Maximum penalty to apply (default: -1.0)
        min_length: Minimum text length for analysis (default: 20 chars)
        reference_key: Key to lookup reference text in kwargs (default: "reference")
        stopwords: Set of words to ignore in n-gram analysis
    """

    def __init__(
        self,
        n: int = 6,
        threshold: float = 0.3,
        penalty_scale: float = -1.0,
        min_length: int = 20,
        reference_key: str = "reference",
        stopwords: Optional[Set[str]] = None,
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize verbatim copy penalty reward function.

        Args:
            n: N-gram size for overlap detection. Higher values detect longer
               copied sequences but may miss shorter copies. Default 6 matches
               the paper's 6-gram profile overlap approach.
            threshold: Minimum proportion of n-grams that must overlap with
                       reference text to trigger a penalty (0.0 to 1.0).
            penalty_scale: Maximum penalty to apply for verbatim copying.
                          Negative value creates a reward floor.
            min_length: Minimum character length for n-gram analysis.
                       Shorter texts are skipped to avoid false positives.
            reference_key: Key in kwargs to lookup reference/source text.
                          Can also pass "reference" directly in kwargs.
            stopwords: Set of common words to exclude from n-gram analysis.
            weight: Weight for this reward function when combining.
            **kwargs: Additional configuration.
        """
        super().__init__(weight=weight, **kwargs)
        self.n = max(1, min(n, 10))  # Clamp n to reasonable range [1, 10]
        self.threshold = max(0.0, min(threshold, 1.0))  # Clamp to [0, 1]
        self.penalty_scale = penalty_scale
        self.min_length = min_length
        self.reference_key = reference_key

        # Default stopwords (common function words that don't indicate copying)
        self.stopwords = stopwords or {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
            "may",
            "might",
            "can",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "what",
            "which",
            "who",
            "when",
            "where",
            "why",
            "how",
        }

    def _extract_ngrams(self, text: str, n: int) -> Set[str]:
        """
        Extract n-grams from text, excluding stopwords.

        Args:
            text: Input text to extract n-grams from
            n: N-gram size

        Returns:
            Set of n-grams (as space-separated strings)
        """
        # Normalize: lowercase and extract words
        words = re.findall(r"\b\w+\b", text.lower())

        # Filter out stopwords and short words
        content_words = [w for w in words if w not in self.stopwords and len(w) > 2]

        # Extract n-grams
        if len(content_words) < n:
            return set()

        ngrams = {
            " ".join(content_words[i : i + n])
            for i in range(len(content_words) - n + 1)
        }

        return ngrams

    def _analyze_copy_pattern(
        self,
        completion_ngrams: Set[str],
        reference_ngrams: Set[str],
        completion_length: int,
    ) -> Dict[str, float]:
        """
        Analyze the pattern of copying between completion and reference.

        Args:
            completion_ngrams: Set of n-grams from completion
            reference_ngrams: Set of n-grams from reference
            completion_length: Character length of completion

        Returns:
            Dict with analysis metrics including overlap ratio, penalty
        """
        if not completion_ngrams or not reference_ngrams:
            return {"overlap_ratio": 0.0, "penalty": 0.0}

        # Find overlapping n-grams
        overlapping_ngrams = completion_ngrams & reference_ngrams

        if not overlapping_ngrams:
            return {"overlap_ratio": 0.0, "penalty": 0.0}

        # Calculate overlap ratio
        overlap_ratio = len(overlapping_ngrams) / len(completion_ngrams)

        # Calculate penalty based on overlap ratio
        if overlap_ratio < self.threshold:
            # Below threshold - no penalty
            penalty = 0.0
        else:
            # Scale penalty from 0 to penalty_scale based on severity
            # Severity is how much we exceed the threshold
            severity = (overlap_ratio - self.threshold) / (1.0 - self.threshold)
            penalty = severity * self.penalty_scale

        return {
            "overlap_ratio": overlap_ratio,
            "overlapping_count": len(overlapping_ngrams),
            "penalty": penalty,
        }

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        """
        Calculate penalties for verbatim copying.

        Args:
            completions: List of completions to evaluate
            **kwargs: Additional context including:
                     - reference: Reference/source text to compare against
                     - Or any key matching self.reference_key

        Returns:
            List of penalty scores (negative values = penalties applied)
        """
        # Extract reference text from kwargs
        reference_text = kwargs.get(self.reference_key, kwargs.get("reference", ""))

        if not reference_text:
            # No reference provided - no penalties
            logger.debug("No reference text provided for verbatim copy detection")
            return [0.0] * len(completions)

        # Pre-compute reference n-grams for efficiency
        reference_ngrams = self._extract_ngrams(reference_text, self.n)

        if not reference_ngrams:
            # Reference too short after filtering - no meaningful comparison
            logger.debug("Reference text too short for n-gram analysis")
            return [0.0] * len(completions)

        rewards = []
        for completion in completions:
            try:
                # Extract content from completion
                content = self.get_content(completion)

                # Skip very short completions
                if len(content) < self.min_length:
                    logger.debug(
                        f"Completion too short for copy analysis: {len(content)} chars"
                    )
                    rewards.append(0.0)
                    continue

                # Extract n-grams from completion
                completion_ngrams = self._extract_ngrams(content, self.n)

                if not completion_ngrams:
                    # Completion too short after filtering
                    rewards.append(0.0)
                    continue

                # Analyze copy pattern and calculate penalty
                analysis = self._analyze_copy_pattern(
                    completion_ngrams,
                    reference_ngrams,
                    len(content),
                )

                penalty = analysis["penalty"]
                overlap_ratio = analysis["overlap_ratio"]

                # Log significant overlaps for monitoring
                if overlap_ratio >= self.threshold:
                    logger.info(
                        f"Verbatim copy detected: {overlap_ratio:.2%} overlap, "
                        f"penalty={penalty:.2f}"
                    )

                rewards.append(penalty)

            except Exception as e:
                logger.error(f"Error in verbatim copy analysis: {e}")
                logger.exception(e)
                rewards.append(0.0)

        return rewards


# Legacy function for backward compatibility
def verbatim_copy_penalty_reward(
    completions: List[Any],
    reference: str = "",
    n: int = 6,
    threshold: float = 0.3,
    penalty_scale: float = -1.0,
    **kwargs,
) -> List[float]:
    """
    Legacy function wrapper for VerbatimCopyPenaltyReward.

    Based on "Designing Reward Signals for Portable Query Generation:
    A Case Study in Industrial Semantic Job Search" (arXiv:2606.27291).

    Args:
        completions: List of completions to evaluate
        reference: Reference/source text to compare against for copying
        n: N-gram size for overlap detection (default: 6)
        threshold: Minimum overlap ratio to trigger penalty (default: 0.3)
        penalty_scale: Maximum penalty for verbatim copying (default: -1.0)
        **kwargs: Additional parameters

    Returns:
        List of penalties (negative values indicate copying detected)
    """
    reward_fn = VerbatimCopyPenaltyReward(
        n=n,
        threshold=threshold,
        penalty_scale=penalty_scale,
    )
    return reward_fn.compute(completions, reference=reference, **kwargs)
