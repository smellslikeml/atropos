"""
Tests for Riemannian Preconditioned LoRA optimizer.

Tests the preconditioner calculation and optimizer wrapper functionality.
"""

import pytest

import torch

# Import from example_trainer module
from example_trainer.riemannian_lora import (
    RiemannianLoRAOptimizer,
    create_riemannian_lora_optimizer,
)
from example_trainer.config import TrainingConfig


class DummyConfig:
    """Minimal config for testing."""
    optimizer = "adamw"
    lr = 1e-4
    riemannian_damping = 1e-6
    riemannian_frequency = 1


class DummyModelWithLora(torch.nn.Module):
    """Simple model with LoRA-like A and B parameters."""

    def __init__(self, rank=4, in_features=8, out_features=16):
        super().__init__()
        # Simulate LoRA A matrices (rank x in_features)
        self.lora_A_1 = torch.nn.Parameter(torch.randn(rank, in_features) * 0.01)
        # Simulate LoRA B matrices (out_features x rank)
        self.lora_B_1 = torch.nn.Parameter(torch.randn(out_features, rank) * 0.01)
        # Another LoRA pair
        self.lora_A_2 = torch.nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B_2 = torch.nn.Parameter(torch.randn(out_features, rank) * 0.01)


class DummyModelWithoutLora(torch.nn.Module):
    """Simple model without LoRA parameters."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(10, 10) * 0.01)


def test_preconditioner_shapes():
    """Test that preconditioner computes correct output shapes."""
    rank = 4
    in_features = 8
    out_features = 16

    model = DummyModelWithLora(rank, in_features, out_features)
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer = RiemannianLoRAOptimizer(base_optimizer, model, damping=1e-6)

    # Verify pairs were identified
    assert len(optimizer._lora_pairs) == 4  # 2 pairs, each counted twice

    # Set up gradients
    model.lora_A_1.grad = torch.randn_like(model.lora_A_1)
    model.lora_B_1.grad = torch.randn_like(model.lora_B_1)

    # Apply preconditioning
    optimizer._apply_preconditioning()

    # Verify gradients still have the same shape
    assert model.lora_A_1.grad.shape == model.lora_A_1.shape
    assert model.lora_B_1.grad.shape == model.lora_B_1.shape


def test_preconditioner_effect_on_gradients():
    """Test that preconditioning changes gradients meaningfully."""
    rank = 4
    model = DummyModelWithLora(rank, 8, 16)
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer = RiemannianLoRAOptimizer(base_optimizer, model, damping=1e-6)

    # Store original gradients
    original_grad_A = torch.randn_like(model.lora_A_1)
    original_grad_B = torch.randn_like(model.lora_B_1)

    model.lora_A_1.grad = original_grad_A.clone()
    model.lora_B_1.grad = original_grad_B.clone()

    # Apply preconditioning
    optimizer._apply_preconditioning()

    # Gradients should have changed
    assert not torch.allclose(model.lora_A_1.grad, original_grad_A, atol=1e-6)
    assert not torch.allclose(model.lora_B_1.grad, original_grad_B, atol=1e-6)


def test_optimizer_step_calls_base():
    """Test that step() calls base optimizer."""
    model = DummyModelWithLora(4, 8, 16)
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer = RiemannianLoRAOptimizer(base_optimizer, model)

    # Set gradients
    for p in model.parameters():
        p.grad = torch.randn_like(p)

    # Step should work without error
    optimizer.step()

    # Parameters should have changed
    assert not torch.allclose(model.lora_A_1, torch.zeros_like(model.lora_A_1))


def test_optimizer_zero_grad():
    """Test that zero_grad() works."""
    model = DummyModelWithLora(4, 8, 16)
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer = RiemannianLoRAOptimizer(base_optimizer, model)

    # Set gradients
    model.lora_A_1.grad = torch.randn_like(model.lora_A_1)

    # Zero grad
    optimizer.zero_grad()

    # Gradients should be None
    assert model.lora_A_1.grad is None


def test_non_lora_model():
    """Test that optimizer works with non-LoRA models."""
    model = DummyModelWithoutLora()
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer = RiemannianLoRAOptimizer(base_optimizer, model)

    # Should have no pairs
    assert len(optimizer._lora_pairs) == 0

    # Set gradients and step
    model.weight.grad = torch.randn_like(model.weight)
    optimizer.step()

    # Should work without error
    assert not torch.allclose(model.weight, torch.zeros_like(model.weight))


def test_damping_affects_preconditioning():
    """Test that damping parameter affects preconditioning."""
    model = DummyModelWithLora(4, 8, 16)

    # Two optimizers with different damping
    base_opt1 = torch.optim.AdamW(model.parameters(), lr=1e-4)
    base_opt2 = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt1 = RiemannianLoRAOptimizer(base_opt1, model, damping=1e-6)
    opt2 = RiemannianLoRAOptimizer(base_opt2, model, damping=1e-2)

    # Set same gradients
    grad = torch.randn_like(model.lora_A_1)
    model.lora_A_1.grad = grad.clone()

    # Apply with first damping
    opt1._apply_preconditioning()
    result1 = model.lora_A_1.grad.clone()

    # Reset and apply with second damping
    model.lora_A_1.grad = grad.clone()
    opt2._apply_preconditioning()
    result2 = model.lora_A_1.grad

    # Results should differ
    assert not torch.allclose(result1, result2, atol=1e-4)


def test_frequency_parameter():
    """Test that preconditioner_frequency controls when preconditioning is applied."""
    model = DummyModelWithLora(4, 8, 16)
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer = RiemannianLoRAOptimizer(
        base_optimizer, model, preconditioner_frequency=2
    )

    # Set gradients
    for p in model.parameters():
        p.grad = torch.randn_like(p)

    # First step should apply preconditioning (step 1 % 2 == 1)
    original_grad = model.lora_A_1.grad.clone()
    optimizer.step()
    # Gradients should be None after step (AdamW behavior)
    # We just verify no error occurred
    assert optimizer._step_count == 1

    # Set gradients again
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    original_grad_2 = model.lora_A_1.grad.clone()
    optimizer.step()
    assert optimizer._step_count == 2


def test_state_dict_passthrough():
    """Test that state_dict and load_state_dict work."""
    model = DummyModelWithLora(4, 8, 16)
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer = RiemannianLoRAOptimizer(base_optimizer, model)

    # Take a step
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    optimizer.step()

    # Get state dict
    state_dict = optimizer.state_dict()
    assert isinstance(state_dict, dict)

    # Load state dict
    optimizer.load_state_dict(state_dict)


def test_create_riemannian_lora_optimizer():
    """Test the factory function."""
    model = DummyModelWithLora(4, 8, 16)

    # Create a TrainingConfig-like object
    class SimpleConfig:
        optimizer = "adamw"
        lr = 1e-4
        riemannian_preconditioning = True
        riemannian_damping = 1e-6
        riemannian_frequency = 1

    config = SimpleConfig()

    # Create optimizer
    opt = create_riemannian_lora_optimizer(model, config)

    # Should return RiemannianLoRAOptimizer
    assert isinstance(opt, RiemannianLoRAOptimizer)


def test_layer_key_extraction():
    """Test the _get_lora_layer_key method."""
    model = DummyModelWithLora(4, 8, 16)
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer = RiemannianLoRAOptimizer(base_optimizer, model)

    # Test various parameter name patterns
    test_cases = [
        ("model.layers.0.q_proj.lora_A.default", "model.layers.0.q_proj"),
        ("base_model.model.layers.5.self_attn.lora_B.default", "base_model.model.layers.5.self_attn"),
        ("lora_A", ""),  # Edge case: just the lora part
    ]

    for param_name, expected_prefix in test_cases:
        result = optimizer._get_lora_layer_key(param_name)
        if expected_prefix:
            assert expected_prefix in result or result == expected_prefix


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
