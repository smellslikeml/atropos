"""
Dense Signal Evaluator - QVal methodology for Atropos.

This module implements QVal-inspired evaluation of dense supervision signals
by measuring their correlation with reference Q-values (or outcome-based proxies).

The core insight from QVal (arXiv:2606.32034v1) is that dense supervision methods
should be evaluated by how well their signals align with optimal Q-values from a
reference policy, rather than by downstream training performance.

This adaptation focuses on teacher distillation contexts where we want to evaluate
whether teacher-provided signals (logprobs, embeddings, etc.) actually correlate
with good outcomes.

Key substitutions from the original QVal paper:
- Reference Q-values: Monte Carlo returns from episode rewards (proxy for Q*)
- Learned Q-function: Simple ranking-based correlation metrics
- Multi-environment testbed: Focused evaluation on distillation signals
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import kendalltau, spearmanr

logger = logging.getLogger(__name__)


@dataclass
class SignalEvaluationResult:
    """Result of evaluating a dense supervision signal.

    Attributes:
        signal_name: Name of the evaluated signal
        spearman_rho: Spearman correlation coefficient between signal and reference
        kendall_tau: Kendall tau correlation coefficient
        num_samples: Number of state-action pairs evaluated
        p_value_spearman: P-value for Spearman correlation
        p_value_kendall: P-value for Kendall correlation
        metadata: Additional information about the evaluation
    """

    signal_name: str
    spearman_rho: float
    kendall_tau: float
    num_samples: int
    p_value_spearman: float
    p_value_kendall: float
    metadata: Dict[str, Any]

    def __str__(self) -> str:
        """Human-readable summary of evaluation results."""
        return (
            f"Signal '{self.signal_name}':\n"
            f"  Spearman ρ: {self.spearman_rho:.4f} (p={self.p_value_spearman:.4e})\n"
            f"  Kendall τ: {self.kendall_tau:.4f} (p={self.p_value_kendall:.4e})\n"
            f"  Samples: {self.num_samples}"
        )

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if correlation is statistically significant."""
        return self.p_value_spearman < alpha or self.p_value_kendall < alpha

    def is_positive_alignment(self) -> bool:
        """Check if signal shows positive Q-alignment (both correlations positive)."""
        return self.spearman_rho > 0 and self.kendall_tau > 0


class DenseSignalEvaluator:
    """
    Training-free evaluator for dense supervision signals.

    This evaluator measures Q-alignment: how well a supervision signal orders
    actions according to reference Q-values (or outcome-based proxies).

    Usage:
        evaluator = DenseSignalEvaluator()
        result = evaluator.evaluate(
            signal_scores=[0.8, 0.3, 0.9, ...],  # Teacher logprobs or other signal
            reference_values=[1.0, 0.0, 1.0, ...],  # Outcome rewards or MC returns
            signal_name="teacher_logprobs"
        )
    """

    def __init__(self, min_samples: int = 10, seed: Optional[int] = None):
        """
        Initialize the evaluator.

        Args:
            min_samples: Minimum number of samples required for evaluation
            seed: Random seed for reproducibility
        """
        self.min_samples = min_samples
        self.rng = np.random.default_rng(seed)

    def evaluate(
        self,
        signal_scores: Union[List[float], np.ndarray],
        reference_values: Union[List[float], np.ndarray],
        signal_name: str = "dense_signal",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SignalEvaluationResult:
        """
        Evaluate a dense supervision signal by measuring Q-alignment.

        Q-alignment is measured as the correlation between the signal's ordering
        of actions and the reference ordering (by Q-values or outcome returns).

        Args:
            signal_scores: Dense supervision scores for each state-action pair
            reference_values: Reference Q-values or outcome returns
            signal_name: Name identifier for the signal being evaluated
            metadata: Optional additional information to attach to results

        Returns:
            SignalEvaluationResult with correlation metrics and statistics

        Raises:
            ValueError: If inputs are invalid (wrong length, too few samples, etc.)
        """
        # Convert to numpy arrays
        signals = np.asarray(signal_scores, dtype=np.float64)
        references = np.asarray(reference_values, dtype=np.float64)

        # Validate inputs
        if signals.shape != references.shape:
            raise ValueError(
                f"Signal scores and reference values must have same shape. "
                f"Got signals={signals.shape}, references={references.shape}"
            )

        if len(signals) < self.min_samples:
            raise ValueError(
                f"Need at least {self.min_samples} samples for evaluation, "
                f"got {len(signals)}"
            )

        # Check for constant inputs (no variance)
        if np.std(signals) < 1e-10 or np.std(references) < 1e-10:
            logger.warning(
                f"Signal '{signal_name}' has near-zero variance in signals "
                f"or references. Correlation metrics may be unreliable."
            )

        # Compute Spearman correlation (primary metric)
        spearman_rho, p_value_spearman = spearmanr(signals, references)

        # Compute Kendall tau (secondary metric, more robust to outliers)
        kendall_tau, p_value_kendall = kendalltau(signals, references)

        # Handle NaN values (can occur with constant inputs)
        if np.isnan(spearman_rho):
            spearman_rho = 0.0
            p_value_spearman = 1.0
        if np.isnan(kendall_tau):
            kendall_tau = 0.0
            p_value_kendall = 1.0

        result = SignalEvaluationResult(
            signal_name=signal_name,
            spearman_rho=float(spearman_rho),
            kendall_tau=float(kendall_tau),
            num_samples=len(signals),
            p_value_spearman=float(p_value_spearman),
            p_value_kendall=float(p_value_kendall),
            metadata=metadata or {},
        )

        logger.info(f"Evaluated signal '{signal_name}': {result}")

        return result

    def evaluate_teacher_logprobs(
        self,
        teacher_logprobs: List[List[List[float]]],
        outcome_rewards: List[float],
        top_k: Optional[int] = None,
        signal_name: str = "teacher_logprobs",
    ) -> SignalEvaluationResult:
        """
        Evaluate teacher distillation signal from logprobs.

        Extracts a scalar signal from teacher logprobs and correlates with outcomes.

        Args:
            teacher_logprobs: Teacher logprobs from distillation, shape
                [sequence][position][top_k]. Each inner list contains logprobs for
                top-k tokens at that position.
            outcome_rewards: Outcome reward for each sequence
            top_k: Number of top tokens to aggregate over. If None, uses all available.
            signal_name: Name identifier for this evaluation

        Returns:
            SignalEvaluationResult with correlation metrics

        Note:
            The signal extracted is the average teacher logprob assigned to tokens
            that were actually selected (assuming first token in each top-k list
            corresponds to the selected token). This is a proxy for how well the
            teacher's confidence predicts good outcomes.
        """
        if len(teacher_logprobs) != len(outcome_rewards):
            raise ValueError(
                f"Number of sequences must match. "
                f"Got {len(teacher_logprobs)} teacher logprobs, "
                f"{len(outcome_rewards)} outcomes"
            )

        # Extract scalar signal: mean teacher logprob per sequence
        sequence_signals: List[float] = []
        for seq_idx, seq_logprobs in enumerate(teacher_logprobs):
            if seq_logprobs is None or len(seq_logprobs) == 0:
                logger.warning(f"Sequence {seq_idx} has no teacher logprobs, skipping")
                continue

            # Aggregate logprobs across positions
            position_means: List[float] = []
            for pos_logprobs in seq_logprobs:
                if pos_logprobs is None or len(pos_logprobs) == 0:
                    continue

                # Use first token's logprob as the signal (selected token)
                # If top_k is specified, average over top-k tokens
                if top_k is not None and top_k > 1:
                    k_logprobs = pos_logprobs[:top_k]
                    signal = float(np.mean(k_logprobs))
                else:
                    signal = float(pos_logprobs[0])

                position_means.append(signal)

            if position_means:
                sequence_signals.append(float(np.mean(position_means)))

        if len(sequence_signals) < self.min_samples:
            raise ValueError(
                f"Insufficient valid sequences after extracting signal: "
                f"{len(sequence_signals)} < {self.min_samples}"
            )

        # Align outcomes with valid sequences
        valid_indices = [
            i
            for i, seq in enumerate(teacher_logprobs)
            if seq is not None and len(seq) > 0
        ]
        aligned_outcomes = [outcome_rewards[i] for i in valid_indices]

        return self.evaluate(
            signal_scores=sequence_signals,
            reference_values=aligned_outcomes,
            signal_name=signal_name,
            metadata={
                "num_sequences": len(teacher_logprobs),
                "valid_sequences": len(sequence_signals),
            },
        )

    def evaluate_multiple_signals(
        self,
        signals_dict: Dict[str, Union[List[float], np.ndarray]],
        reference_values: Union[List[float], np.ndarray],
    ) -> Dict[str, SignalEvaluationResult]:
        """
        Evaluate multiple dense supervision signals against the same reference.

        Args:
            signals_dict: Mapping from signal name to signal scores
            reference_values: Reference Q-values or outcome returns

        Returns:
            Dictionary mapping signal names to evaluation results
        """
        results: Dict[str, SignalEvaluationResult] = {}

        for signal_name, signal_scores in signals_dict.items():
            try:
                result = self.evaluate(
                    signal_scores=signal_scores,
                    reference_values=reference_values,
                    signal_name=signal_name,
                )
                results[signal_name] = result
            except Exception as e:
                logger.error(f"Failed to evaluate signal '{signal_name}': {e}")
                results[signal_name] = None  # type: ignore

        return results

    def compare_signals(
        self,
        results: Dict[str, SignalEvaluationResult],
        metric: str = "spearman",
    ) -> List[Tuple[str, float]]:
        """
        Rank signals by their Q-alignment strength.

        Args:
            results: Dictionary of signal evaluation results
            metric: Which correlation metric to use for ranking
                ('spearman' or 'kendall')

        Returns:
            List of (signal_name, correlation_value) sorted by correlation strength
        """
        if metric not in ("spearman", "kendall"):
            raise ValueError(f"Invalid metric '{metric}'. Use 'spearman' or 'kendall'")

        attr_name = "spearman_rho" if metric == "spearman" else "kendall_tau"

        ranked = [
            (name, getattr(result, attr_name))
            for name, result in results.items()
            if result is not None
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked

    def summarize_results(self, results: Dict[str, SignalEvaluationResult]) -> str:
        """
        Generate a human-readable summary of multiple signal evaluations.

        Args:
            results: Dictionary of signal evaluation results

        Returns:
            Formatted summary string
        """
        lines = ["=" * 60, "DENSE SIGNAL EVALUATION SUMMARY", "=" * 60]

        # Rank by Spearman correlation
        ranked = self.compare_signals(results, metric="spearman")

        lines.append("\nSignals ranked by Q-alignment (Spearman ρ):")
        for rank, (name, rho) in enumerate(ranked, 1):
            result = results[name]
            significant = "✓" if result.is_significant() else "✗"
            positive = "↑" if result.is_positive_alignment() else "↓"
            lines.append(
                f"  {rank}. {name}: {rho:.4f} {significant} {positive} "
                f"(n={result.num_samples})"
            )

        # Summary statistics
        valid_results = [r for r in results.values() if r is not None]
        if valid_results:
            best = max(valid_results, key=lambda r: r.spearman_rho)
            worst = min(valid_results, key=lambda r: r.spearman_rho)

            lines.append("\n" + "-" * 60)
            lines.append(f"Best signal: {best.signal_name} (ρ={best.spearman_rho:.4f})")
            lines.append(
                f"Worst signal: {worst.signal_name} (ρ={worst.spearman_rho:.4f})"
            )

            # Count positive alignments
            n_positive = sum(1 for r in valid_results if r.is_positive_alignment())
            lines.append(
                f"Signals with positive Q-alignment: {n_positive}/{len(valid_results)}"
            )

        lines.append("=" * 60)

        return "\n".join(lines)


def compute_reference_values_from_rewards(
    rewards: List[List[float]],
    gamma: float = 1.0,
) -> List[float]:
    """
    Compute reference Q-value proxies from episode rewards.

    Uses Monte Carlo returns (discounted sum of future rewards) as a proxy
    for Q-values when a learned Q-function is not available.

    Args:
        rewards: List of episodes, each containing a list of stepwise rewards
        gamma: Discount factor

    Returns:
        List of return values, one per episode
    """
    returns: List[float] = []

    for episode_rewards in rewards:
        if not episode_rewards:
            continue

        # Compute discounted return
        episode_return = 0.0
        for i, reward in enumerate(reversed(episode_rewards)):
            episode_return = reward + gamma * episode_return

        returns.append(episode_return)

    return returns


def aggregate_token_level_signals(
    token_signals: List[List[float]],
    aggregation: str = "mean",
) -> List[float]:
    """
    Aggregate token-level signals to sequence-level scores.

    Args:
        token_signals: Token-level signals for each sequence
        aggregation: How to aggregate ('mean', 'max', 'min', 'last')

    Returns:
        One scalar score per sequence
    """
    if aggregation == "mean":
        return [float(np.mean(s)) if s else 0.0 for s in token_signals]
    elif aggregation == "max":
        return [float(np.max(s)) if s else 0.0 for s in token_signals]
    elif aggregation == "min":
        return [float(np.min(s)) if s else 0.0 for s in token_signals]
    elif aggregation == "last":
        return [float(s[-1]) if s else 0.0 for s in token_signals]
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
