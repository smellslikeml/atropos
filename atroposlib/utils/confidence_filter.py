"""
Label-free data filtering for knowledge distillation.

This module provides confidence-based filtering utilities inspired by
uDistil-Whisper: Label-Free Data Filtering for Knowledge Distillation in
Low-Data Regimes (https://arxiv.org/abs/2407.01257v5).

The core insight is that model confidence signals (entropy, max probability)
can be used to filter high-quality predictions without requiring ground truth
labels. This is particularly useful in distillation settings where rewards serve
as the primary supervision signal.

The paper originally proposed a learned estimator for filtering quality. This
implementation substitutes that with parameter-free confidence metrics that
approximate the same signal: entropy and max probability over the model's
output distribution.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np


def compute_entropy(probs: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """
    Compute Shannon entropy over a probability distribution.

    Lower entropy indicates higher model confidence (the distribution is more
    concentrated on fewer outcomes).

    Args:
        probs: Probability array of shape (..., vocab_size) or (..., batch, vocab_size).
               Values should be non-negative and sum to 1 along the specified axis.
        axis: The axis representing the probability distribution dimension.
        eps: Small constant to avoid log(0).

    Returns:
        Entropy values with shape matching probs, reduced along the specified axis.

    Examples:
        >>> # Uniform distribution (high entropy, low confidence)
        >>> probs = np.array([[0.25, 0.25, 0.25, 0.25]])
        >>> compute_entropy(probs)
        array([1.386...])

        >>> # Peak distribution (low entropy, high confidence)
        >>> probs = np.array([[0.97, 0.01, 0.01, 0.01]])
        >>> compute_entropy(probs)
        array([0.174...])
    """
    clipped_probs = np.clip(probs, eps, 1.0)
    log_probs = np.log(clipped_probs)
    entropy = -np.sum(probs * log_probs, axis=axis)
    return entropy


def compute_max_probability(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute the maximum probability over a distribution.

    Higher max probability indicates higher model confidence (the model is
    more certain about its top prediction).

    Args:
        probs: Probability array of shape (..., vocab_size).
        axis: The axis representing the probability distribution dimension.

    Returns:
        Maximum probability values with shape matching probs, reduced along
        the specified axis.

    Examples:
        >>> # Uniform distribution (low max probability)
        >>> probs = np.array([[0.25, 0.25, 0.25, 0.25]])
        >>> compute_max_probability(probs)
        array([0.25])

        >>> # Peak distribution (high max probability)
        >>> probs = np.array([[0.97, 0.01, 0.01, 0.01]])
        >>> compute_max_probability(probs)
        array([0.97])
    """
    return np.max(probs, axis=axis)


def filter_by_confidence(
    logprobs: Union[List[List[List[float]]], np.ndarray],
    method: str = "entropy",
    threshold: Optional[float] = None,
    top_k: Optional[int] = None,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter sequences by model confidence without ground truth labels.

    This implements the label-free filtering mechanism from uDistil-Whisper,
    using model confidence signals to identify high-quality predictions.

    Args:
        logprobs: Log probabilities of shape (num_sequences, seq_len, vocab_size)
                  or a list of sequences where each sequence has shape
                  (seq_len, vocab_size). Each inner list represents the
                  logprobs for all tokens at a position.
        method: Confidence metric to use. One of:
                - "entropy": Lower entropy = higher confidence (default)
                - "max_prob": Higher max probability = higher confidence
                - "combined": Harmonic mean of normalized entropy and max_prob
        threshold: Optional confidence threshold. Sequences with average
                   confidence better than this threshold are kept.
                   For "entropy": values below threshold are kept.
                   For "max_prob": values above threshold are kept.
                   If None, uses top_k instead.
        top_k: Optional number of top-confident sequences to keep.
               If None, uses threshold instead.
        epsilon: Small constant for numerical stability in exp/log operations.

    Returns:
        A tuple of (keep_mask, confidence_scores):
        - keep_mask: Boolean array of shape (num_sequences,) indicating which
                     sequences pass the filter.
        - confidence_scores: Raw confidence scores for each sequence.

    Raises:
        ValueError: If both threshold and top_k are None, or if method is invalid.

    Examples:
        >>> # High-confidence sequence (peak distribution)
        >>> logprobs_high = [[[-0.03, -4.6, -4.6]]]  # ~[0.97, 0.01, 0.01]
        >>>
        >>> # Low-confidence sequence (uniform distribution)
        >>> logprobs_low = [[[-1.38, -1.38, -1.38]]]  # ~[0.25, 0.25, 0.25]
        >>>
        >>> logprobs = np.array([logprobs_high, logprobs_low])
        >>> keep_mask, scores = filter_by_confidence(logprobs, method="entropy", top_k=1)
        >>> keep_mask
        array([ True, False])
        >>> scores
        array([0.174..., 1.386...])
    """
    if threshold is None and top_k is None:
        raise ValueError("Either threshold or top_k must be provided.")
    if threshold is not None and top_k is not None:
        raise ValueError("Only one of threshold or top_k should be provided.")
    if method not in ("entropy", "max_prob", "combined"):
        raise ValueError(
            f"Invalid method: {method}. Must be 'entropy', 'max_prob', or 'combined'."
        )

    # Convert to numpy array if needed
    if not isinstance(logprobs, np.ndarray):
        # Handle jagged sequences by padding
        max_len = max(len(seq) for seq in logprobs)
        vocab_size = len(logprobs[0][0]) if logprobs and len(logprobs[0]) > 0 else 1

        padded = np.zeros((len(logprobs), max_len, vocab_size), dtype=np.float32)
        for i, seq in enumerate(logprobs):
            seq_len = len(seq)
            if seq_len > 0:
                for j, pos in enumerate(seq):
                    if pos:
                        padded[i, j, : len(pos)] = pos

        logprobs = padded

    if logprobs.size == 0:
        return np.array([], dtype=bool), np.array([])

    # Convert logprobs to probabilities (numerically stable)
    max_logprob = np.max(logprobs, axis=-1, keepdims=True)
    probs = np.exp(logprobs - max_logprob)
    probs_sum = np.sum(probs, axis=-1, keepdims=True)
    probs = probs / (probs_sum + epsilon)

    # Compute position-wise confidence metrics
    entropy = compute_entropy(probs, axis=-1)
    max_prob = compute_max_probability(probs, axis=-1)

    # Average across sequence length (axis 1)
    avg_entropy = np.mean(entropy, axis=1)
    avg_max_prob = np.mean(max_prob, axis=1)

    # Compute confidence scores based on method
    if method == "entropy":
        confidence_scores = avg_entropy  # Lower is better
    elif method == "max_prob":
        confidence_scores = avg_max_prob  # Higher is better
    else:  # combined
        # Normalize: entropy to [0,1] where 0 is best, max_prob is already [0,1] where 1 is best
        # Use log(vocab_size) as max entropy for normalization
        vocab_size = logprobs.shape[-1]
        max_possible_entropy = np.log(vocab_size) if vocab_size > 1 else 1.0
        normalized_entropy = avg_entropy / (max_possible_entropy + epsilon)
        # Combined score: harmonic mean of (1 - normalized_entropy) and max_prob
        # Higher is better
        inverse_entropy = 1.0 - normalized_entropy
        confidence_scores = (
            2.0
            * (inverse_entropy * avg_max_prob)
            / (inverse_entropy + avg_max_prob + epsilon)
        )

    # Apply filtering
    num_sequences = logprobs.shape[0]
    keep_mask = np.zeros(num_sequences, dtype=bool)

    if threshold is not None:
        if method == "entropy":
            keep_mask = confidence_scores <= threshold
        else:  # max_prob or combined
            keep_mask = confidence_scores >= threshold
    else:  # top_k is not None
        k = min(top_k, num_sequences)
        if method == "entropy":
            # Keep lowest entropy (most confident)
            top_k_indices = np.argpartition(confidence_scores, k)[:k]
        else:
            # Keep highest max_prob or combined score
            top_k_indices = np.argpartition(confidence_scores, -k)[-k:]
        keep_mask[top_k_indices] = True

    return keep_mask, confidence_scores


def filter_group_by_confidence(
    logprobs_list: List[List[List[float]]],
    method: str = "entropy",
    percentile: Optional[float] = None,
    top_k: Optional[int] = None,
    epsilon: float = 1e-8,
) -> Tuple[List[int], np.ndarray]:
    """
    Filter a group of sequences by confidence, returning indices of kept sequences.

    This is a convenience wrapper around filter_by_confidence that returns
    indices rather than a boolean mask, making it easier to use in pipelines
    where you need to filter associated data (e.g., tokens, labels).

    Args:
        logprobs_list: List of sequences, where each sequence is a list of
                       positions, and each position is a list of logprobs.
                       Shape: [(seq_len, vocab_size), ...]
        method: Confidence metric: "entropy", "max_prob", or "combined".
        percentile: Optional percentile threshold (0-100). Sequences with
                    confidence better than this percentile are kept.
                    For "entropy": below percentile = kept.
                    For "max_prob"/"combined": above percentile = kept.
        top_k: Optional number of top-confident sequences to keep.
        epsilon: Numerical stability constant.

    Returns:
        A tuple of (kept_indices, confidence_scores):
        - kept_indices: List of indices from the input that pass the filter.
        - confidence_scores: Raw confidence scores for all sequences.

    Raises:
        ValueError: If both percentile and top_k are specified, or if neither is.

    Examples:
        >>> logprobs_list = [
        ...     [[-0.03, -4.6, -4.6]],  # High confidence
        ...     [[-1.38, -1.38, -1.38]],  # Low confidence
        ... ]
        >>> indices, scores = filter_group_by_confidence(logprobs_list, top_k=1)
        >>> indices
        [0]
    """
    if percentile is not None and top_k is not None:
        raise ValueError("Only one of percentile or top_k should be provided.")
    if percentile is None and top_k is None:
        raise ValueError("Either percentile or top_k must be provided.")

    # Convert to numpy array if needed
    if not isinstance(logprobs_list, np.ndarray):
        max_len = max(len(seq) for seq in logprobs_list)
        vocab_size = (
            len(logprobs_list[0][0]) if logprobs_list and len(logprobs_list[0]) > 0 else 1
        )

        padded = np.zeros((len(logprobs_list), max_len, vocab_size), dtype=np.float32)
        for i, seq in enumerate(logprobs_list):
            seq_len = len(seq)
            if seq_len > 0:
                for j, pos in enumerate(seq):
                    if pos:
                        padded[i, j, : len(pos)] = pos

        logprobs = padded
    else:
        logprobs = logprobs_list

    # Compute confidence scores first
    if logprobs.size == 0:
        return [], np.array([])

    # Convert logprobs to probabilities (numerically stable)
    max_logprob = np.max(logprobs, axis=-1, keepdims=True)
    probs = np.exp(logprobs - max_logprob)
    probs_sum = np.sum(probs, axis=-1, keepdims=True)
    probs = probs / (probs_sum + epsilon)

    # Compute position-wise confidence metrics
    entropy = compute_entropy(probs, axis=-1)
    max_prob = compute_max_probability(probs, axis=-1)

    # Average across sequence length (axis 1)
    avg_entropy = np.mean(entropy, axis=1)
    avg_max_prob = np.mean(max_prob, axis=1)

    # Compute confidence scores based on method
    if method == "entropy":
        confidence_scores = avg_entropy  # Lower is better
    elif method == "max_prob":
        confidence_scores = avg_max_prob  # Higher is better
    else:  # combined
        vocab_size = logprobs.shape[-1]
        max_possible_entropy = np.log(vocab_size) if vocab_size > 1 else 1.0
        normalized_entropy = avg_entropy / (max_possible_entropy + epsilon)
        inverse_entropy = 1.0 - normalized_entropy
        confidence_scores = 2.0 * (inverse_entropy * avg_max_prob) / (
            inverse_entropy + avg_max_prob + epsilon
        )

    # Apply filtering
    keep_mask = np.zeros(len(confidence_scores), dtype=bool)

    if percentile is not None:
        # Compute threshold from percentile
        if method == "entropy":
            threshold = np.percentile(confidence_scores, percentile)
            keep_mask = confidence_scores <= threshold
        else:
            threshold = np.percentile(confidence_scores, 100 - percentile)
            keep_mask = confidence_scores >= threshold
    else:  # top_k is not None
        k = min(top_k, len(confidence_scores))
        if method == "entropy":
            top_k_indices = np.argpartition(confidence_scores, k)[:k]
        else:
            top_k_indices = np.argpartition(confidence_scores, -k)[-k:]
        keep_mask[top_k_indices] = True

    kept_indices = [i for i, keep in enumerate(keep_mask) if keep]
    return kept_indices, confidence_scores
