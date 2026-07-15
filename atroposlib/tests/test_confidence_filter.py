"""
Tests for confidence-based filtering utilities.

Tests the label-free data filtering mechanisms from uDistil-Whisper:
Label-Free Data Filtering for Knowledge Distillation in Low-Data Regimes.
"""

import math

import numpy as np
import pytest

from atroposlib.utils.confidence_filter import (
    compute_entropy,
    compute_max_probability,
    filter_by_confidence,
    filter_group_by_confidence,
)


class TestComputeEntropy:
    """Tests for compute_entropy function."""

    def test_uniform_distribution_high_entropy(self):
        """Uniform distribution should have high entropy."""
        # 4-way uniform distribution
        probs = np.array([[0.25, 0.25, 0.25, 0.25]])
        entropy = compute_entropy(probs)
        # Expected: -4 * (0.25 * log(0.25)) = log(4) ≈ 1.386
        expected = np.log(4)
        assert math.isclose(entropy[0], expected, rel_tol=1e-5)

    def test_peak_distribution_low_entropy(self):
        """Peak distribution should have low entropy."""
        # One dominant outcome
        probs = np.array([[0.97, 0.01, 0.01, 0.01]])
        entropy = compute_entropy(probs)
        # Should be much lower than uniform
        assert entropy[0] < 0.5

    def test_binary_distribution(self):
        """Binary distribution entropy."""
        # 50/50 coin flip
        probs = np.array([[0.5, 0.5]])
        entropy = compute_entropy(probs)
        expected = np.log(2)  # ≈ 0.693
        assert math.isclose(entropy[0], expected, rel_tol=1e-5)

    def test_deterministic_zero_entropy(self):
        """Deterministic outcome should have zero entropy."""
        probs = np.array([[1.0, 0.0, 0.0, 0.0]])
        entropy = compute_entropy(probs)
        # Using eps in computation prevents exact zero, but should be very small
        assert entropy[0] < 0.01

    def test_multiple_sequences(self):
        """Entropy computation across multiple sequences."""
        probs = np.array(
            [
                [0.25, 0.25, 0.25, 0.25],  # High entropy
                [0.97, 0.01, 0.01, 0.01],  # Low entropy
            ]
        )
        entropy = compute_entropy(probs)
        assert entropy[0] > entropy[1]  # Uniform > Peak

    def test_custom_axis(self):
        """Entropy computation with custom axis."""
        probs = np.array(
            [
                [[0.25, 0.25, 0.25, 0.25]],
                [[0.97, 0.01, 0.01, 0.01]],
            ]
        )
        entropy = compute_entropy(probs, axis=-1)
        assert entropy.shape == (2, 1)
        assert entropy[0, 0] > entropy[1, 0]


class TestComputeMaxProbability:
    """Tests for compute_max_probability function."""

    def test_uniform_distribution_low_max(self):
        """Uniform distribution should have low max probability."""
        probs = np.array([[0.25, 0.25, 0.25, 0.25]])
        max_prob = compute_max_probability(probs)
        assert max_prob[0] == 0.25

    def test_peak_distribution_high_max(self):
        """Peak distribution should have high max probability."""
        probs = np.array([[0.97, 0.01, 0.01, 0.01]])
        max_prob = compute_max_probability(probs)
        assert max_prob[0] == 0.97

    def test_multiple_sequences(self):
        """Max probability across multiple sequences."""
        probs = np.array(
            [
                [0.25, 0.25, 0.25, 0.25],  # Low max
                [0.97, 0.01, 0.01, 0.01],  # High max
            ]
        )
        max_prob = compute_max_probability(probs)
        assert max_prob[0] < max_prob[1]

    def test_custom_axis(self):
        """Max probability with custom axis."""
        probs = np.array(
            [
                [[0.25, 0.25, 0.25, 0.25]],
                [[0.97, 0.01, 0.01, 0.01]],
            ]
        )
        max_prob = compute_max_probability(probs, axis=-1)
        assert max_prob.shape == (2, 1)
        assert max_prob[0, 0] < max_prob[1, 0]


class TestFilterByConfidence:
    """Tests for filter_by_confidence function."""

    def test_entropy_method_top_k(self):
        """Filter by entropy keeping top-k (lowest entropy)."""
        # Sequence 0: Low entropy (high confidence)
        logprobs_high = np.array([[[-0.03, -4.6, -4.6]]])  # ~[0.97, 0.01, 0.01]
        # Sequence 1: High entropy (low confidence)
        logprobs_low = np.array([[[-1.38, -1.38, -1.38]]])  # ~[0.25, 0.25, 0.25]

        logprobs = np.vstack([logprobs_high, logprobs_low])
        keep_mask, scores = filter_by_confidence(logprobs, method="entropy", top_k=1)

        assert keep_mask[0]  # High confidence kept
        assert not keep_mask[1]  # Low confidence dropped
        assert scores[0] < scores[1]  # Lower entropy for high confidence

    def test_max_prob_method_top_k(self):
        """Filter by max probability keeping top-k (highest max prob)."""
        logprobs_high = np.array([[[-0.03, -4.6, -4.6]]])  # High max prob
        logprobs_low = np.array([[[-1.38, -1.38, -1.38]]])  # Low max prob

        logprobs = np.vstack([logprobs_high, logprobs_low])
        keep_mask, scores = filter_by_confidence(logprobs, method="max_prob", top_k=1)

        assert keep_mask[0]
        assert not keep_mask[1]
        assert scores[0] > scores[1]  # Higher max prob for high confidence

    def test_entropy_threshold(self):
        """Filter by entropy with threshold."""
        logprobs_high = np.array([[[-0.03, -4.6, -4.6]]])  # Low entropy
        logprobs_low = np.array([[[-1.38, -1.38, -1.38]]])  # High entropy

        logprobs = np.vstack([logprobs_high, logprobs_low])
        # Threshold between the two entropies
        keep_mask, scores = filter_by_confidence(
            logprobs, method="entropy", threshold=0.5
        )

        assert keep_mask[0]  # Below threshold (kept)
        assert not keep_mask[1]  # Above threshold (dropped)

    def test_max_prob_threshold(self):
        """Filter by max probability with threshold."""
        logprobs_high = np.array([[[-0.03, -4.6, -4.6]]])  # High max prob
        logprobs_low = np.array([[[-1.38, -1.38, -1.38]]])  # Low max prob

        logprobs = np.vstack([logprobs_high, logprobs_low])
        keep_mask, scores = filter_by_confidence(
            logprobs, method="max_prob", threshold=0.5
        )

        assert keep_mask[0]  # Above threshold (kept)
        assert not keep_mask[1]  # Below threshold (dropped)

    def test_combined_method(self):
        """Filter using combined method (harmonic mean)."""
        logprobs_high = np.array([[[-0.03, -4.6, -4.6]]])  # High confidence
        logprobs_low = np.array([[[-1.38, -1.38, -1.38]]])  # Low confidence

        logprobs = np.vstack([logprobs_high, logprobs_low])
        keep_mask, scores = filter_by_confidence(logprobs, method="combined", top_k=1)

        assert keep_mask[0]
        assert not keep_mask[1]

    def test_multi_token_sequence(self):
        """Filtering across multi-token sequences."""
        # Sequence with consistent high confidence
        logprobs_high = np.array([[[-0.03, -4.6, -4.6], [-0.1, -3.0, -3.0]]])
        # Sequence with mixed confidence
        logprobs_mixed = np.array([[[-0.03, -4.6, -4.6], [-1.38, -1.38, -1.38]]])

        logprobs = np.vstack([logprobs_high, logprobs_mixed])
        keep_mask, _ = filter_by_confidence(logprobs, method="entropy", top_k=1)

        # Both sequences have same first position, but high has consistently low entropy
        assert keep_mask[0]  # Consistently high confidence

    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        logprobs = np.array([[[-0.03, -4.6, -4.6]]])
        with pytest.raises(ValueError, match="Invalid method"):
            filter_by_confidence(logprobs, method="invalid", top_k=1)

    def test_no_threshold_or_top_k_raises(self):
        """Missing threshold and top_k should raise ValueError."""
        logprobs = np.array([[[-0.03, -4.6, -4.6]]])
        with pytest.raises(ValueError, match="Either threshold or top_k"):
            filter_by_confidence(logprobs, method="entropy")

    def test_both_threshold_and_top_k_raises(self):
        """Both threshold and top_k should raise ValueError."""
        logprobs = np.array([[[-0.03, -4.6, -4.6]]])
        with pytest.raises(ValueError, match="Only one of"):
            filter_by_confidence(logprobs, method="entropy", threshold=0.5, top_k=1)

    def test_empty_logprobs(self):
        """Empty logprobs should return empty results."""
        keep_mask, scores = filter_by_confidence(
            np.array([]), method="entropy", top_k=1
        )
        assert keep_mask.size == 0
        assert scores.size == 0


class TestFilterGroupByConfidence:
    """Tests for filter_group_by_confidence function."""

    def test_returns_indices(self):
        """Should return indices of kept sequences."""
        logprobs_list = [
            [[-0.03, -4.6, -4.6]],  # High confidence
            [[-1.38, -1.38, -1.38]],  # Low confidence
        ]

        kept_indices, scores = filter_group_by_confidence(logprobs_list, top_k=1)

        assert kept_indices == [0]
        assert len(scores) == 2

    def test_percentile_filtering(self):
        """Percentile-based filtering."""
        logprobs_list = [
            [[-0.03, -4.6, -4.6]],  # High confidence
            [[-0.5, -2.0, -2.0]],  # Medium confidence
            [[-1.38, -1.38, -1.38]],  # Low confidence
        ]

        # Keep top 50% by entropy (lowest entropy)
        kept_indices, _ = filter_group_by_confidence(
            logprobs_list, method="entropy", percentile=50
        )

        assert len(kept_indices) == 2  # 50% of 3, rounded
        assert 0 in kept_indices  # Highest confidence kept

    def test_max_prob_percentile(self):
        """Percentile filtering with max_prob method."""
        logprobs_list = [
            [[-0.03, -4.6, -4.6]],  # High max prob
            [[-1.38, -1.38, -1.38]],  # Low max prob
        ]

        kept_indices, _ = filter_group_by_confidence(
            logprobs_list, method="max_prob", percentile=50
        )

        assert len(kept_indices) == 1
        assert kept_indices == [0]

    def test_invalid_percentile_and_top_k(self):
        """Both percentile and top_k should raise ValueError."""
        logprobs_list = [[[-0.03, -4.6, -4.6]]]
        with pytest.raises(ValueError, match="Only one of"):
            filter_group_by_confidence(logprobs_list, percentile=50, top_k=1)

    def test_no_percentile_or_top_k(self):
        """Missing both should raise ValueError."""
        logprobs_list = [[[-0.03, -4.6, -4.6]]]
        with pytest.raises(ValueError, match="Either percentile or top_k"):
            filter_group_by_confidence(logprobs_list)

    def test_combined_percentile(self):
        """Combined method with percentile."""
        logprobs_list = [
            [[-0.03, -4.6, -4.6]],  # High confidence
            [[-1.38, -1.38, -1.38]],  # Low confidence
        ]

        kept_indices, _ = filter_group_by_confidence(
            logprobs_list, method="combined", percentile=50
        )

        assert len(kept_indices) == 1
        assert 0 in kept_indices


class TestConfidenceIntegration:
    """Integration tests for confidence filtering in realistic scenarios."""

    def test_teacher_distillation_scenario(self):
        """Test filtering scenario similar to teacher distillation."""
        # Simulate teacher logprobs for multiple sequences
        # Each sequence has multiple positions with top-k logprobs
        logprobs_list = [
            # Sequence 0: High confidence throughout
            [[-0.03, -4.6, -4.6], [-0.05, -4.0, -4.0], [-0.02, -5.0, -5.0]],
            # Sequence 1: Low confidence throughout
            [[-1.38, -1.38, -1.38], [-1.39, -1.39, -1.39], [-1.37, -1.37, -1.37]],
            # Sequence 2: Medium confidence
            [[-0.5, -2.0, -2.0], [-0.6, -1.8, -1.8], [-0.4, -2.2, -2.2]],
        ]

        # Keep top 50% by entropy
        kept_indices, scores = filter_group_by_confidence(
            logprobs_list, method="entropy", percentile=50
        )

        # Should keep highest confidence sequences
        assert len(kept_indices) == 2  # 50% of 3 = 1.5, rounded to 2
        assert 0 in kept_indices  # Highest confidence kept
        assert 1 not in kept_indices  # Lowest confidence dropped

    def test_vocabulary_size_normalization(self):
        """Test that entropy normalization works with different vocab sizes."""
        # Small vocab (2 tokens)
        small_vocab = np.array([[[0.5, 0.5]]])

        # Large vocab (100 tokens, uniform distribution)
        large_vocab = np.array([[0.01] * 100])

        entropy_small = compute_entropy(small_vocab)
        entropy_large = compute_entropy(large_vocab)

        # Large vocab uniform should have higher entropy
        assert entropy_large[0] > entropy_small[0]

        # Max entropy for n vocab is log(n)
        expected_small = np.log(2)
        expected_large = np.log(100)

        assert np.isclose(entropy_small[0], expected_small, rtol=1e-5)
        assert np.isclose(entropy_large[0], expected_large, rtol=1e-5)
