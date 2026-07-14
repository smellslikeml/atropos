"""
Tests for ROAD-VLA advantage-guided self-distillation.

This test module verifies the advantage distillation functionality that
implements the core insight from ROAD-VLA: converting sparse rewards into
dense token-level supervision through advantage-shaped teacher logits.
"""

import math

from atroposlib.utils.advantage_calibration import (
    DEFAULT_ADVANTAGE_CLIP,
    calibrate_signed_advantage_weights,
)
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
        """Valid tokens all receive the sequence's signed weight; masked get 0."""
        masks = [-100, 5, -100, 7]
        result = compute_token_level_advantages([1.0], masks)
        # Only positions 1 and 3 should have non-zero advantages
        assert result[0] == 0.0  # Masked
        assert result[1] == 1.0  # Valid
        assert result[2] == 0.0  # Masked
        assert result[3] == 1.0  # Valid
        # ROAD-VLA applies a single signed per-timestep weight (no position ramp)
        assert result[3] == result[1]

    def test_multiple_advantages_reduced_to_scalar(self):
        """Multiple advantages reduce to their mean, broadcast to valid tokens."""
        masks = [-100, 5, 7]
        result = compute_token_level_advantages([1.0, 2.0, 3.0], masks)
        # Mean is 2.0, applied uniformly to each valid token (no dilution)
        assert result[0] == 0.0  # Masked
        assert math.isclose(result[1], 2.0)
        assert math.isclose(result[2], 2.0)

    def test_uniform_weighting(self):
        """All valid tokens receive equal weight (no positional ramp)."""
        masks = [1, 2, 3, 4]
        result = compute_token_level_advantages([1.0], masks)
        assert result[0] == result[1] == result[2] == result[3] == 1.0

    def test_negative_advantage_is_signed(self):
        """Negative advantages are preserved (no ReLU)."""
        masks = [1, 2, 3]
        result = compute_token_level_advantages([-1.5], masks)
        assert all(math.isclose(r, -1.5) for r in result)

    def test_symmetric_clip(self):
        """A large-magnitude advantage is clipped symmetrically to [-clip, clip]."""
        masks = [1, 2]
        assert compute_token_level_advantages([9.0], masks, clip=2.0) == [2.0, 2.0]
        assert compute_token_level_advantages([-9.0], masks, clip=2.0) == [-2.0, -2.0]


class TestBatchComputeTokenAdvantages:
    """Tests for batch token advantage computation."""

    def test_none_advantages_returns_zeros(self):
        """Test that None advantages returns all zeros."""
        masks = [[1, 2], [3, 4]]
        result = batch_compute_token_advantages(None, masks)
        assert result == [[0.0, 0.0], [0.0, 0.0]]

    def test_batch_standardizes_across_group(self):
        """Batch standardizes advantages across the group, then broadcasts."""
        advantages = [[1.0], [2.0]]
        masks = [[-100, 5], [6, 7]]
        result = batch_compute_token_advantages(advantages, masks)
        assert len(result) == 2
        # Group mean 1.5, std 0.5 -> z-scores -1 and +1 (within the clip range).
        # First sequence: masked position stays 0, the lower-advantage sequence
        # gets a negative signed weight.
        assert result[0][0] == 0.0
        assert math.isclose(result[0][1], -1.0)
        # Second (higher-advantage) sequence gets a positive weight, applied
        # uniformly to both valid tokens.
        assert math.isclose(result[1][0], 1.0)
        assert math.isclose(result[1][1], 1.0)

    def test_batch_zero_variance_group_emits_no_signal(self):
        """A group with identical advantages produces no perturbation (stable)."""
        advantages = [[1.0], [1.0]]
        masks = [[5, 6], [7, 8]]
        result = batch_compute_token_advantages(advantages, masks)
        assert result == [[0.0, 0.0], [0.0, 0.0]]


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
        """Test that enabled config computes signed, calibrated token advantages."""
        result = compute_advantage_distillation_payload(
            [[1.0], [2.0]], [[-100, 5], [6, 7]], config=None
        )
        assert result["token_advantages"] is not None
        assert len(result["token_advantages"]) == 2
        # Masked position stays 0; the lower-advantage sequence is signed negative
        assert result["token_advantages"][0][0] == 0.0
        assert result["token_advantages"][0][1] < 0
        # Higher-advantage sequence is signed positive and applied to every token
        assert all(v > 0 for v in result["token_advantages"][1])

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
        config = AdvantageDistillationConfig(auto_calibrate=False, advantage_scale=0.5)
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
        """Masked tokens always receive zero advantage; valid tokens get omega_t."""
        # Two-sequence group so the advantage standardization is well-defined.
        result = compute_advantage_distillation_payload(
            [[2.0], [1.0]],
            [[-100, -100, 5, -100], [6, 7, 8, 9]],
            config=AdvantageDistillationConfig(),
        )
        token_advantages = result["token_advantages"][0]
        assert token_advantages[0] == 0.0
        assert token_advantages[1] == 0.0
        assert token_advantages[2] != 0.0  # Only unmasked token carries omega_t
        assert token_advantages[3] == 0.0


class TestSignedAdvantageCalibration:
    """Paper-faithful calibration: signed, standardized, symmetrically clipped."""

    def test_standardizes_to_zero_mean_unit_std(self):
        """Weights are the group's z-scores when within the clip range."""
        weights = calibrate_signed_advantage_weights([[1.0], [2.0], [3.0]])
        # Mean 2.0, std ~0.8165 -> z-scores approx [-1.2247, 0, 1.2247]
        assert math.isclose(weights[1], 0.0, abs_tol=1e-9)
        assert math.isclose(weights[0], -weights[2], rel_tol=1e-9)
        assert weights[0] < 0 < weights[2]

    def test_no_relu_negative_preserved(self):
        """Below-average sequences keep a negative sign (no ReLU rectification)."""
        weights = calibrate_signed_advantage_weights([[0.0], [10.0]])
        assert weights[0] < 0
        assert weights[1] > 0

    def test_symmetric_clip_bounds_perturbation(self):
        """Outliers are clipped to +/- clip, keeping the teacher proximal."""
        # Six zeros + one large value: the outlier's z-score (~2.449) exceeds 2.
        weights = calibrate_signed_advantage_weights([[0.0]] * 6 + [[100.0]], clip=2.0)
        assert all(abs(w) <= 2.0 + 1e-9 for w in weights)
        assert max(weights) == 2.0

    def test_zero_variance_group_is_all_zero(self):
        """Zero-variance groups emit no signal (mitigates issue #457 spikes)."""
        assert calibrate_signed_advantage_weights([[1.0], [1.0], [1.0]]) == [
            0.0,
            0.0,
            0.0,
        ]

    def test_empty_group_returns_empty(self):
        assert calibrate_signed_advantage_weights([]) == []

    def test_default_clip_is_two(self):
        assert DEFAULT_ADVANTAGE_CLIP == 2.0


class TestPayloadWiresCalibration:
    """The payload entry point (call site) must honor the calibration + clip."""

    def test_payload_token_advantages_respect_clip(self):
        """token_advantages come out signed and bounded by the configured clip."""
        config = AdvantageDistillationConfig(advantage_clip=1.0)
        result = compute_advantage_distillation_payload(
            [[-50.0], [0.0], [50.0]],
            [[5], [6], [7]],
            config=config,
        )
        flat = [v for seq in result["token_advantages"] for v in seq]
        assert all(abs(v) <= 1.0 + 1e-9 for v in flat)
        # Lowest-advantage sequence pinned to -clip, highest to +clip
        assert math.isclose(result["token_advantages"][0][0], -1.0)
        assert math.isclose(result["token_advantages"][2][0], 1.0)

    def test_payload_matches_direct_calibration(self):
        """Token advantages equal the calibrated omega broadcast over valid tokens."""
        advantages = [[1.0], [4.0]]
        masks = [[5, 6], [7, -100]]
        result = compute_advantage_distillation_payload(
            advantages, masks, config=AdvantageDistillationConfig()
        )
        omegas = calibrate_signed_advantage_weights(
            advantages, clip=DEFAULT_ADVANTAGE_CLIP
        )
        assert result["token_advantages"][0] == [omegas[0], omegas[0]]
        assert result["token_advantages"][1] == [omegas[1], 0.0]
