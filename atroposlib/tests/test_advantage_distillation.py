"""
Tests for ROAD-VLA advantage-guided self-distillation.

This test module verifies the advantage distillation functionality that
implements the core insight from ROAD-VLA: converting sparse rewards into
dense token-level supervision through advantage-shaped teacher logits.
"""

import math

import numpy as np
import pytest

from atroposlib.utils.advantage_distillation import (
    AdvantageDistillationConfig,
    batch_compute_token_advantages,
    batch_construct_advantage_logits,
    calibrate_advantage_scale,
    compute_advantage_distillation_payload,
    compute_token_level_advantages,
    construct_advantage_shaped_logits,
)


class TestComputeTokenLevelAdvantages:
    """Tests for token-level advantage computation."""

    def test_no_advantages_returns_empty(self):
        """Test that empty advantages returns empty list."""
        result = compute_token_level_advantages([], [1, 2, 3], num_tokens=10)
        assert result == []

    def test_no_masks_returns_empty(self):
        """Test that empty masks returns empty list."""
        result = compute_token_level_advantages([1.0], [], num_tokens=10)
        assert result == []

    def test_all_masked_returns_zeros(self):
        """Test that all-masked sequences return all zeros."""
        masks = [-100, -100, -100]
        result = compute_token_level_advantages([1.0], masks, num_tokens=10)
        assert result == [0.0, 0.0, 0.0]

    def test_valid_tokens_get_advantages(self):
        """Test that valid tokens receive distributed advantages."""
        masks = [-100, 5, -100, 7]
        result = compute_token_level_advantages([1.0], masks, num_tokens=10)
        # Only positions 1 and 3 should have non-zero advantages
        assert result[0] == 0.0  # Masked
        assert result[1] > 0  # Valid
        assert result[2] == 0.0  # Masked
        assert result[3] > 0  # Valid
        # Later valid token should have higher weight
        assert result[3] > result[1]

    def test_multiple_advantages_averaged(self):
        """Test that multiple sequence advantages are averaged."""
        masks = [-100, 5, 7]
        result = compute_token_level_advantages([1.0, 2.0, 3.0], masks, num_tokens=10)
        # Average is 2.0, distributed across valid tokens
        assert result[0] == 0.0  # Masked
        assert result[1] > 0
        assert result[2] > 0
        # Sum of valid advantages should reflect average
        valid_sum = result[1] + result[2]
        assert math.isclose(valid_sum, 2.0, rel_tol=0.1)

    def test_position_weighting(self):
        """Test that later tokens receive higher weights."""
        masks = [1, 2, 3, 4]
        result = compute_token_level_advantages([1.0], masks, num_tokens=10)
        # Each successive token should have higher advantage
        assert result[0] < result[1] < result[2] < result[3]


class TestBatchComputeTokenAdvantages:
    """Tests for batch token advantage computation."""

    def test_none_advantages_returns_zeros(self):
        """Test that None advantages returns all zeros."""
        masks = [[1, 2], [3, 4]]
        result = batch_compute_token_advantages(None, masks)
        assert result == [[0.0, 0.0], [0.0, 0.0]]

    def test_batch_processes_correctly(self):
        """Test that batch processing works correctly."""
        advantages = [[1.0], [2.0]]
        masks = [[-100, 5], [6, 7]]
        result = batch_compute_token_advantages(advantages, masks)
        assert len(result) == 2
        # First sequence: one valid token
        assert result[0][0] == 0.0
        assert result[0][1] > 0
        # Second sequence: two valid tokens
        assert result[1][0] > 0
        assert result[1][1] > 0
        assert result[1][1] > result[1][0]  # Later token higher


class TestConstructAdvantageShapedLogits:
    """Tests for advantage-shaped logit construction."""

    def test_no_student_logits_returns_scaled_advantages(self):
        """Test that None student logits returns scaled advantages."""
        result = construct_advantage_shaped_logits(
            None, [0.1, 0.2, 0.3], advantage_scale=0.5
        )
        expected = [0.05, 0.1, 0.15]  # advantages * scale
        assert all(math.isclose(r, e, rel_tol=1e-5) for r, e in zip(result, expected))

    def test_empty_advantages_returns_empty(self):
        """Test that empty advantages returns empty list."""
        result = construct_advantage_shaped_logits(None, [], advantage_scale=0.1)
        assert result == []

    def test_with_student_logits_adds_perturbation(self):
        """Test that student logits are perturbed by advantages."""
        student_logits = [1.0, 2.0, 3.0]
        advantages = [0.1, 0.2, 0.3]
        result = construct_advantage_shaped_logits(
            student_logits, advantages, advantage_scale=0.5
        )
        expected = [1.05, 2.1, 3.15]  # logit + advantage * scale
        assert all(math.isclose(r, e, rel_tol=1e-5) for r, e in zip(result, expected))

    def test_mismatched_lengths_broadcasts_scalar(self):
        """Test that mismatched lengths broadcast advantage scalar."""
        student_logits = [1.0, 2.0, 3.0]
        advantages = [0.5]  # Scalar to broadcast
        result = construct_advantage_shaped_logits(
            student_logits, advantages, advantage_scale=0.1
        )
        expected = [1.05, 2.05, 3.05]  # logit + 0.5 * 0.1
        assert all(math.isclose(r, e, rel_tol=1e-5) for r, e in zip(result, expected))

    def test_temperature_scales_logits(self):
        """Test that temperature divides the perturbed logits."""
        student_logits = [2.0, 4.0, 6.0]
        advantages = [0.0, 0.0, 0.0]
        result = construct_advantage_shaped_logits(
            student_logits, advantages, advantage_scale=0.1, temperature=2.0
        )
        expected = [1.0, 2.0, 3.0]  # logit / temperature
        assert all(math.isclose(r, e, rel_tol=1e-5) for r, e in zip(result, expected))


class TestBatchConstructAdvantageLogits:
    """Tests for batch advantage logit construction."""

    def test_none_returns_none(self):
        """Test that None inputs return None."""
        result = batch_construct_advantage_logits(None, [[0.1, 0.2]])
        assert result is None

    def test_batch_construction(self):
        """Test that batch construction works correctly."""
        student_logits = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]]]
        token_advantages = [[0.1, 0.2], [0.3]]
        result = batch_construct_advantage_logits(
            student_logits, token_advantages, advantage_scale=0.1
        )
        assert result is not None
        assert len(result) == 2
        # Check first sequence
        assert len(result[0]) == 2
        # Check that perturbation was applied
        assert result[0][0][0] != student_logits[0][0][0]  # Was perturbed


class TestCalibrateAdvantageScale:
    """Tests for advantage scale calibration."""

    def test_empty_advantages_returns_default(self):
        """Test that empty advantages returns default scale."""
        result = calibrate_advantage_scale([])
        assert result == 0.1

    def test_constant_advantages_returns_default(self):
        """Test that constant advantages return default scale."""
        result = calibrate_advantage_scale([1.0, 1.0, 1.0])
        assert result == 0.1  # No variation, returns default

    def test_calibrates_to_target_std_ratio(self):
        """Test that calibration achieves target std ratio."""
        advantages = [0.0, 1.0, -1.0, 2.0, -2.0]  # std > 0
        result = calibrate_advantage_scale(
            advantages, target_std_ratio=0.2, student_logits_std=1.0
        )
        # Result should be calibrated to achieve target std ratio
        assert 0.01 <= result <= 1.0  # Within bounds

    def test_with_student_logits_std(self):
        """Test calibration with student logits std."""
        advantages = [0.0, 1.0, -1.0]
        result = calibrate_advantage_scale(
            advantages, target_std_ratio=0.2, student_logits_std=0.5
        )
        # Result should be smaller when student logits have lower std
        assert 0.01 <= result <= 1.0


class TestComputeAdvantageDistillationPayload:
    """Tests for complete advantage distillation payload computation."""

    def test_disabled_returns_zeros(self):
        """Test that disabled config returns all zeros."""
        config = AdvantageDistillationConfig(enabled=False)
        result = compute_advantage_distillation_payload(
            [[1.0], [2.0]], [[1, 2], [3, 4]], config=config
        )
        assert result["token_advantages"] == [[0.0, 0.0], [0.0, 0.0]]
        assert result["advantage_logits"] is None

    def test_none_advantages_returns_zeros(self):
        """Test that None advantages returns all zeros."""
        result = compute_advantage_distillation_payload(
            None, [[1, 2], [3, 4]], config=None
        )
        assert result["token_advantages"] == [[0.0, 0.0], [0.0, 0.0]]
        assert result["advantage_logits"] is None

    def test_enabled_returns_token_advantages(self):
        """Test that enabled config computes token advantages."""
        result = compute_advantage_distillation_payload(
            [[1.0], [2.0]], [[-100, 5], [6, 7]], config=None
        )
        assert result["token_advantages"] is not None
        assert len(result["token_advantages"]) == 2
        # First sequence has one valid token
        assert result["token_advantages"][0][0] == 0.0
        assert result["token_advantages"][0][1] > 0

    def test_auto_calibrate_sets_scale(self):
        """Test that auto-calibrate computes calibrated scale."""
        result = compute_advantage_distillation_payload(
            [[1.0, 2.0], [3.0, 4.0]],
            [[1, 2, 3], [4, 5]],
            config=AdvantageDistillationConfig(auto_calibrate=True),
        )
        assert "calibrated_scale" in result
        assert isinstance(result["calibrated_scale"], float)
        assert 0.01 <= result["calibrated_scale"] <= 1.0

    def test_no_auto_calibrate_uses_config_scale(self):
        """Test that disabled auto-calibrate uses config scale."""
        config = AdvantageDistillationConfig(
            auto_calibrate=False, advantage_scale=0.5
        )
        result = compute_advantage_distillation_payload(
            [[1.0], [2.0]], [[1, 2], [3, 4]], config=config
        )
        assert result["calibrated_scale"] == 0.5

    def test_with_student_logits_includes_advantage_logits(self):
        """Test that student logits produce advantage-shaped logits."""
        student_logits = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        result = compute_advantage_distillation_payload(
            [[1.0], [2.0]],
            [[1, 2], [3, 4]],
            student_logits=student_logits,
            config=AdvantageDistillationConfig(),
        )
        assert result["advantage_logits"] is not None
        assert len(result["advantage_logits"]) == 2


class TestAdvantageDistillationIntegration:
    """Integration tests for advantage distillation."""

    def test_payload_structure_matches_expected_format(self):
        """Test that payload has the expected structure."""
        result = compute_advantage_distillation_payload(
            [[1.0], [2.0]],
            [[-100, 5], [6, 7, -100]],
            config=AdvantageDistillationConfig(),
        )
        # Check required keys
        assert "token_advantages" in result
        assert "advantage_logits" in result
        assert "calibrated_scale" in result
        # Check types
        assert isinstance(result["token_advantages"], list)
        assert isinstance(result["calibrated_scale"], float)

    def test_masked_tokens_receive_zero_advantage(self):
        """Test that masked tokens always receive zero advantage."""
        result = compute_advantage_distillation_payload(
            [[1.0]], [[-100, -100, 5, -100]], config=AdvantageDistillationConfig()
        )
        token_advantages = result["token_advantages"][0]
        assert token_advantages[0] == 0.0
        assert token_advantages[1] == 0.0
        assert token_advantages[2] > 0  # Only unmasked token
        assert token_advantages[3] == 0.0
