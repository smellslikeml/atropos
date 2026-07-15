"""
Tests for DPH-RL f-divergence loss integration in GRPO training.

Tests that the f-divergence loss computation works correctly and produces
valid outputs for different divergence types.

Reference: "The Choice of Divergence: A Neglected Key to Mitigating Diversity
           Collapse in Reinforcement Learning with Verifiable Reward"
           https://arxiv.org/abs/2509.07430v1
"""

import math

import pytest
import torch

# Test imports from the training module (the integration point)
from example_trainer.training import _compute_f_divergence_loss, compute_grpo_loss
from example_trainer.config import TrainingConfig


class TestFDivergenceLoss:
    """Test the f-divergence loss computation for DPH-RL."""

    @pytest.mark.parametrize(
        "divergence_type",
        ["importance", "kl", "reverse_kl", "js", "chi_squared"],
    )
    def test_f_divergence_loss_shapes(self, divergence_type: str):
        """Test that f-divergence loss produces correctly shaped outputs."""
        batch_size = 4
        seq_len = 16

        # Create dummy logprobs
        logp_current = torch.randn(batch_size, seq_len) * 0.5 - 2.0
        logp_ref = torch.randn(batch_size, seq_len) * 0.5 - 2.0

        # Create dummy advantages
        advantages = torch.randn(batch_size, seq_len)

        # Create mask (all positions valid for this test)
        mask = torch.ones(batch_size, seq_len)

        loss, metrics = _compute_f_divergence_loss(
            logp_current=logp_current,
            logp_ref=logp_ref,
            advantages=advantages,
            mask=mask,
            divergence_type=divergence_type,
            clip_eps=0.2,
        )

        # Check loss is a scalar tensor
        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"

        # Check loss is finite
        assert torch.isfinite(loss).all(), f"Loss should be finite, got {loss}"

        # Check metrics contain expected keys
        assert "mean_divergence_weight" in metrics
        assert "clipped_fraction" in metrics
        assert "raw_divergence" in metrics

    @pytest.mark.parametrize(
        "divergence_type",
        ["kl", "reverse_kl", "js", "chi_squared"],
    )
    def test_f_divergence_loss_is_finite(self, divergence_type: str):
        """Test that f-divergence loss produces finite values with realistic inputs."""
        batch_size = 2
        vocab_size = 1000
        seq_len = 8

        # Create realistic logprobs (from softmax of random logits)
        logits_current = torch.randn(batch_size, seq_len, vocab_size)
        logits_ref = torch.randn(batch_size, seq_len, vocab_size)

        # Convert to logprobs (log_softmax)
        logp_current = torch.log_softmax(logits_current, dim=-1)
        logp_ref = torch.log_softmax(logits_ref, dim=-1)

        # Reduce to per-token logprobs (sum over vocab for simplicity)
        logp_current = logp_current.sum(dim=-1)
        logp_ref = logp_ref.sum(dim=-1)

        # Create advantages (positive for some tokens, negative for others)
        advantages = torch.randn(batch_size, seq_len)

        # Create mask
        mask = torch.ones(batch_size, seq_len)

        loss, metrics = _compute_f_divergence_loss(
            logp_current=logp_current,
            logp_ref=logp_ref,
            advantages=advantages,
            mask=mask,
            divergence_type=divergence_type,
            clip_eps=0.2,
        )

        # Check all values are finite
        assert torch.isfinite(
            loss
        ).all(), f"Loss should be finite for {divergence_type}"
        assert torch.isfinite(metrics["mean_divergence_weight"]).all()
        assert math.isfinite(metrics["raw_divergence"])

    def test_f_divergence_importance_mode_matches_standard(self):
        """Test that 'importance' mode behaves like standard GRPO ratio."""
        batch_size = 4
        seq_len = 16

        logp_current = torch.randn(batch_size, seq_len) * 0.5 - 2.0
        logp_ref = torch.randn(batch_size, seq_len) * 0.5 - 2.0
        advantages = torch.randn(batch_size, seq_len)
        mask = torch.ones(batch_size, seq_len)

        loss, metrics = _compute_f_divergence_loss(
            logp_current=logp_current,
            logp_ref=logp_ref,
            advantages=advantages,
            mask=mask,
            divergence_type="importance",
            clip_eps=0.2,
        )

        # In importance mode, mean_divergence_weight should match exp(logp_current - logp_ref)
        expected_ratio = torch.exp(logp_current - logp_ref)
        expected_mean = (expected_ratio * mask).sum() / mask.sum()

        # Check that the divergence weight matches the importance ratio
        assert torch.allclose(
            metrics["mean_divergence_weight"],
            expected_mean,
            rtol=1e-4,
            atol=1e-5,
        ), "Importance mode should match standard ratio computation"

    def test_f_divergence_unknown_type_raises(self):
        """Test that unknown divergence types raise ValueError."""
        with pytest.raises(ValueError, match="Unknown divergence_type"):
            _compute_f_divergence_loss(
                logp_current=torch.randn(2, 8),
                logp_ref=torch.randn(2, 8),
                advantages=torch.randn(2, 8),
                mask=torch.ones(2, 8),
                divergence_type="unknown_type",  # type: ignore
                clip_eps=0.2,
            )


class TestDivergenceConfig:
    """Test that the TrainingConfig properly handles divergence_type configuration."""

    def test_config_accepts_divergence_type(self):
        """Test that TrainingConfig accepts all valid divergence types."""
        valid_types = ["importance", "kl", "reverse_kl", "js", "chi_squared"]

        for div_type in valid_types:
            config = TrainingConfig(
                model_name="test_model",
                divergence_type=div_type,  # type: ignore
            )
            assert config.divergence_type == div_type

    def test_config_default_divergence_type(self):
        """Test that TrainingConfig defaults to 'importance' mode."""
        config = TrainingConfig(model_name="test_model")
        assert config.divergence_type == "importance"


class TestGrpoLossWithDivergence:
    """Test compute_grpo_loss integration with different divergence types."""

    def test_grpo_loss_divergence_type_parameter(self):
        """Test that compute_grpo_loss accepts divergence_type parameter."""
        # This is a minimal integration test that verifies the parameter
        # is accepted without creating a full model

        # Just verify the function signature accepts the parameter
        import inspect

        sig = inspect.signature(compute_grpo_loss)
        assert "divergence_type" in sig.parameters
        assert sig.parameters["divergence_type"].default == "importance"


@pytest.mark.parametrize(
    "divergence_type",
    ["importance", "kl", "reverse_kl", "js", "chi_squared"],
)
def test_divergence_types_exist_in_config(divergence_type: str):
    """Test that all divergence types are valid config options."""
    # This test verifies the integration between config and training
    config = TrainingConfig(
        model_name="test",
        divergence_type=divergence_type,  # type: ignore
    )

    # Verify the divergence_type is set correctly
    assert config.divergence_type == divergence_type

    # Verify it's in the allowed types (from the Literal type hint)
    # If we got here without ValidationError, the type is valid
    assert config.divergence_type in [
        "importance",
        "kl",
        "reverse_kl",
        "js",
        "chi_squared",
    ]
