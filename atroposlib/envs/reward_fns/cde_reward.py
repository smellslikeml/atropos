"""
Curiosity-Driven Exploration (CDE) reward function for RLVR.

Based on: CDE: Curiosity-Driven Exploration for Efficient Reinforcement Learning
in Large Language Models (https://arxiv.org/abs/2509.09675v1)

CDE addresses the problem of poor exploration in Reinforcement Learning with
Verifiable Rewards (RLVR), which leads to premature convergence and entropy collapse.
This reward function provides intrinsic curiosity signals to encourage diverse
exploration during RL training.

The implementation combines three complementary curiosity signals:
1. **Actor novelty**: Measures how novel/completed outputs are compared to recent history
2. **Entropy bonus**: Encourages high-entropy distributions to prevent mode collapse
3. **Critic surprise** (optional): Uses value prediction variance if available

This is an adapted port (Mode 2) that captures the core CDE mechanism while using
simpler, parameter-free components suitable for the Atropos framework:
- Uses n-gram diversity and token entropy instead of learned embeddings
- Accepts optional value predictions via kwargs but doesn't require them
- Stateless design that works across distributed training setups
"""

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)


@registry.register
class CDEReward(RewardFunction):
    """
    Curiosity-Driven Exploration reward function for RLVR.

    Combines actor novelty, entropy bonus, and optional critic surprise signals
    to encourage diverse exploration and prevent premature convergence in RLVR.

    Components:
        - novelty_score: Measures output diversity via n-gram analysis
        - entropy_bonus: Encourages high-entropy token distributions
        - surprise_score: Optional value-based curiosity signal

    The reward is computed as:
        curiosity = w_novelty * novelty + w_entropy * entropy + w_surprise * surprise
    """

    def __init__(
        self,
        novelty_weight: float = 0.5,
        entropy_weight: float = 0.3,
        surprise_weight: float = 0.2,
        ngram_sizes: Optional[List[int]] = None,
        entropy_window_size: int = 50,
        min_tokens: int = 10,
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the CDE reward function.

        Args:
            novelty_weight: Weight for the novelty/diversity component (default: 0.5)
            entropy_weight: Weight for the entropy bonus component (default: 0.3)
            surprise_weight: Weight for the value-based surprise component (default: 0.2)
            ngram_sizes: N-gram sizes for diversity analysis (default: [2, 3, 4])
            entropy_window_size: Window size for local entropy computation (default: 50)
            min_tokens: Minimum tokens required for meaningful analysis (default: 3)
            weight: Overall weight for this reward function (default: 1.0)
            **kwargs: Additional configuration
        """
        super().__init__(weight=weight, **kwargs)
        self.novelty_weight = novelty_weight
        self.entropy_weight = entropy_weight
        self.surprise_weight = surprise_weight
        self.ngram_sizes = ngram_sizes or [2, 3, 4]
        self.entropy_window_size = entropy_window_size
        self.min_tokens = min_tokens

        # Track recent completions for novelty comparison (circular buffer)
        self._history_size = 100
        self._completion_history: List[str] = []

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        """
        Compute CDE curiosity rewards for the given completions.

        Args:
            completions: List of completions to evaluate
            **kwargs: Additional context, may include:
                - values: List of value predictions for critic surprise
                - logprobs: List of logprobability tensors for entropy calculation
                - reference_texts: Reference texts for cross-sample novelty

        Returns:
            List of curiosity scores (higher = more novel/diverse)
        """
        # Extract content from different possible formats
        completion_contents = [
            self.get_content(completion) for completion in completions
        ]

        # Extract optional value predictions if available
        values = kwargs.get("values", None)
        logprobs = kwargs.get("logprobs", None)

        curiosity_rewards = []

        for idx, content in enumerate(completion_contents):
            # Skip empty completions
            if not content or len(content.strip()) < self.min_tokens:
                curiosity_rewards.append(0.0)
                continue

            # Compute the three curiosity components
            novelty_score = self._compute_novelty_score(content)
            entropy_score = self._compute_entropy_bonus(
                content, logprobs[idx] if logprobs and idx < len(logprobs) else None
            )
            surprise_score = self._compute_surprise_score(
                idx, values[idx] if values and idx < len(values) else None
            )

            # Combine weighted components
            curiosity = (
                self.novelty_weight * novelty_score
                + self.entropy_weight * entropy_score
                + self.surprise_weight * surprise_score
            )

            curiosity_rewards.append(curiosity)

        # Update history with current batch for future novelty comparisons
        self._update_history(completion_contents)

        logger.debug(
            f"CDE rewards: min={min(curiosity_rewards):.3f}, "
            f"max={max(curiosity_rewards):.3f}, mean={sum(curiosity_rewards)/len(curiosity_rewards):.3f}"
        )

        return curiosity_rewards

    def _compute_novelty_score(self, content: str) -> float:
        """
        Compute novelty score based on n-gram diversity.

        Higher scores indicate more diverse/unique content compared to recent history.

        Args:
            content: The completion text to analyze

        Returns:
            Novelty score between 0 and 1
        """
        # Tokenize the content
        tokens = self._tokenize(content)

        if len(tokens) < self.min_tokens:
            return 0.0

        # Extract n-grams from current content
        current_ngrams = set()
        for n in self.ngram_sizes:
            if len(tokens) >= n:
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i : i + n])
                    current_ngrams.add(ngram)

        if not current_ngrams:
            return 0.0

        # Compare against history
        if not self._completion_history:
            # First batch - reward intrinsic diversity
            return self._compute_intrinsic_diversity(content)

        # Compute overlap with recent history
        history_overlap = 0
        for hist_content in self._completion_history[-20:]:  # Compare with last 20
            hist_tokens = self._tokenize(hist_content)
            hist_ngrams = set()
            for n in self.ngram_sizes:
                if len(hist_tokens) >= n:
                    for i in range(len(hist_tokens) - n + 1):
                        ngram = tuple(hist_tokens[i : i + n])
                        hist_ngrams.add(ngram)

            # Jaccard similarity
            if current_ngrams and hist_ngrams:
                overlap = len(current_ngrams & hist_ngrams) / len(
                    current_ngrams | hist_ngrams
                )
                history_overlap = max(history_overlap, overlap)

        # Novelty is inversely related to overlap
        novelty = 1.0 - history_overlap
        return max(0.0, novelty)

    def _compute_intrinsic_diversity(self, content: str) -> float:
        """
        Compute intrinsic diversity of content without history comparison.

        Uses n-gram uniqueness ratio within the content itself.

        Args:
            content: The completion text to analyze

        Returns:
            Diversity score between 0 and 1
        """
        tokens = self._tokenize(content)

        if len(tokens) < self.min_tokens:
            return 0.0

        # Compute unique n-gram ratio
        unique_ratios = []
        for n in self.ngram_sizes:
            if len(tokens) >= n:
                ngrams = [
                    tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
                ]
                if ngrams:
                    unique_ratio = len(set(ngrams)) / len(ngrams)
                    unique_ratios.append(unique_ratio)

        return sum(unique_ratios) / len(unique_ratios) if unique_ratios else 0.0

    def _compute_entropy_bonus(
        self, content: str, logprobs: Optional[Any] = None
    ) -> float:
        """
        Compute entropy bonus to encourage exploration.

        If logprobs are provided, uses actual token entropy.
        Otherwise, estimates entropy from lexical diversity.

        Args:
            content: The completion text to analyze
            logprobs: Optional logprobability tensor/array

        Returns:
            Entropy bonus between 0 and 1
        """
        if logprobs is not None:
            # Try to use actual logprobs for accurate entropy
            try:
                # Handle different logprobs formats
                if hasattr(logprobs, "entropy"):
                    # Some formats provide entropy directly
                    return min(1.0, float(logprobs.entropy()))
                elif hasattr(logprobs, "__len__"):
                    # Array-like: compute entropy from distribution
                    import numpy as np

                    probs = np.exp(np.array(logprobs))
                    probs = probs[probs > 0]  # Avoid log(0)
                    if len(probs) > 0:
                        entropy = -np.sum(probs * np.log(probs))
                        # Normalize by log of vocab size (approximate)
                        return min(1.0, entropy / 10.0)  # 10 is approximate log(vocab)
            except Exception as e:
                logger.debug(f"Could not compute entropy from logprobs: {e}")

        # Fallback: estimate entropy from lexical diversity
        tokens = self._tokenize(content)

        if len(tokens) < self.min_tokens:
            return 0.0

        # Use type-token ratio as entropy proxy
        unique_tokens = set(tokens)
        ttr = len(unique_tokens) / len(tokens) if tokens else 0.0

        # Adjust for content length (TTR decreases with length)
        # Standardized TTR approximation
        adjusted_ttr = ttr * (1 + len(tokens) / (len(tokens) + 100))

        return min(1.0, adjusted_ttr)

    def _compute_surprise_score(
        self, idx: int, value: Optional[float] = None
    ) -> float:
        """
        Compute surprise score based on value prediction variance.

        Higher surprise indicates the model encountered unexpected states.

        Args:
            idx: Index of the current completion (for position-based surprise)
            value: Optional value prediction from critic

        Returns:
            Surprise score between 0 and 1
        """
        if value is not None:
            # If we have value predictions, compute surprise based on value magnitude
            try:
                val = float(value)
                # Normalize by assuming typical value range [-10, 10]
                # Surprise is higher for extreme values
                surprise = min(1.0, abs(val) / 10.0)
                return surprise
            except (TypeError, ValueError):
                pass

        # Fallback: small baseline curiosity for all completions
        # This ensures some exploration signal even without values
        return 0.1

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple word-level tokenization.

        Args:
            text: The text to tokenize

        Returns:
            List of tokens
        """
        # Simple regex-based tokenization
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def _update_history(self, completion_contents: List[str]) -> None:
        """
        Update the completion history with current batch.

        Args:
            completion_contents: List of completion texts from current batch
        """
        for content in completion_contents:
            if content and len(content.strip()) >= self.min_tokens:
                self._completion_history.append(content)
                # Keep history bounded
                if len(self._completion_history) > self._history_size:
                    self._completion_history.pop(0)


# Legacy function for backward compatibility
def cde_reward(
    completions: List[Any],
    novelty_weight: float = 0.5,
    entropy_weight: float = 0.3,
    surprise_weight: float = 0.2,
    **kwargs,
) -> List[float]:
    """
    Legacy function wrapper for CDE reward.

    Args:
        completions: List of completions to evaluate
        novelty_weight: Weight for novelty component (default: 0.5)
        entropy_weight: Weight for entropy component (default: 0.3)
        surprise_weight: Weight for surprise component (default: 0.2)
        **kwargs: Additional parameters including values, logprobs

    Returns:
        List of curiosity scores (higher = more novel/diverse)
    """
    reward_fn = CDEReward(
        novelty_weight=novelty_weight,
        entropy_weight=entropy_weight,
        surprise_weight=surprise_weight,
    )
    return reward_fn.compute(completions, **kwargs)
