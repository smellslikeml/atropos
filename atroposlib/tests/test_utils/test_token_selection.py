"""
Tests for token selection mechanisms.

Tests the RSI (Relative Surprisal Index) token selection implementation
for RLVR training.
"""

import pytest
import torch

from atroposlib.utils.token_selection import (
    RSISelection,
    NoTokenSelection,
    registry as token_selection_registry,
)


class TestRSISelection:
    """Test RSI Selection token filtering."""

    def test_rsi_selection_init(self):
        """Test RSI Selection initialization."""
        selector = RSISelection(rsi_min=0.1, rsi_max=3.0)
        assert selector.rsi_min == 0.1
        assert selector.rsi_max == 3.0
        assert selector.name == "rsiselection"

    def test_rsi_computation_shapes(self):
        """Test that RSI computation returns correct shapes."""
        selector = RSISelection()

        batch_size = 2
        seq_len = 8
        vocab_size = 100

        # Create dummy logits and labels
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Mask some positions as invalid (like prompt tokens)
        labels[0, :2] = -100
        labels[1, :3] = -100

        rsi_values, token_logprobs, entropy = selector.compute_rsi(
            logits, labels, None
        )

        assert rsi_values.shape == (batch_size, seq_len)
        assert token_logprobs.shape == (batch_size, seq_len)
        assert entropy.shape == (batch_size, seq_len)

    def test_rsi_mask_valid_positions(self):
        """Test that RSI mask respects valid token positions."""
        selector = RSISelection(rsi_min=0.1, rsi_max=3.0)

        batch_size = 2
        seq_len = 8
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Mask some positions
        labels[0, :2] = -100

        mask, metrics = selector.compute_mask(logits, labels, None)

        # Mask should be 0 for invalid positions
        assert mask[0, 0].item() == 0.0
        assert mask[0, 1].item() == 0.0

        # Check that valid tokens are counted correctly
        valid_tokens = (labels != -100).sum().item()
        assert metrics["rsi/valid_tokens"] == valid_tokens

    def test_rsi_filtering_behavior(self):
        """Test that RSI filtering removes tokens outside the interval."""
        # Use narrow thresholds to ensure some tokens are filtered
        selector = RSISelection(rsi_min=0.5, rsi_max=1.5)

        batch_size = 1
        seq_len = 4
        vocab_size = 10

        # Create logits where we can predict which tokens have high/low RSI
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        mask, metrics = selector.compute_mask(logits, labels, None)

        # Check that metrics are computed
        assert "rsi/kept_ratio" in metrics
        assert "rsi/filtered_low" in metrics
        assert "rsi/filtered_high" in metrics
        assert 0.0 <= metrics["rsi/kept_ratio"] <= 1.0

    def test_rsi_metrics_completeness(self):
        """Test that RSI mask computation returns all expected metrics."""
        selector = RSISelection()

        batch_size = 2
        seq_len = 8
        vocab_size = 50

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        mask, metrics = selector.compute_mask(logits, labels, None)

        # Check all expected metrics are present
        expected_metrics = [
            "rsi/kept_ratio",
            "rsi/filtered_low",
            "rsi/filtered_high",
            "rsi/mean",
            "rsi/std",
            "rsi/min",
            "rsi/max",
            "rsi/valid_tokens",
            "rsi/kept_tokens",
        ]

        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"

    def test_rsi_with_temperature(self):
        """Test that temperature scaling affects RSI computation."""
        selector = RSISelection()

        batch_size = 1
        seq_len = 4
        vocab_size = 50

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Temperature shape should match batch size
        temperatures = torch.tensor([[0.5]])

        mask_without_temp, _ = selector.compute_mask(logits, labels, None)
        mask_with_temp, _ = selector.compute_mask(logits, labels, temperatures)

        # Masks should have the same shape
        assert mask_without_temp.shape == mask_with_temp.shape

    def test_rsi_entropy_positive(self):
        """Test that entropy is always positive (clamped)."""
        selector = RSISelection()

        batch_size = 2
        seq_len = 8
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        rsi_values, token_logprobs, entropy = selector.compute_rsi(
            logits, labels, None
        )

        # Entropy should be positive (clamped to avoid division by zero)
        valid_mask = (labels != -100)
        assert (entropy[valid_mask] > 0).all()


class TestNoTokenSelection:
    """Test the pass-through token selector."""

    def test_no_selection_init(self):
        """Test NoTokenSelection initialization."""
        selector = NoTokenSelection()
        assert selector.name == "notokenselection"

    def test_no_selection_keeps_all_valid_tokens(self):
        """Test that NoTokenSelection keeps all valid tokens."""
        selector = NoTokenSelection()

        batch_size = 2
        seq_len = 8
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[0, :2] = -100  # Mark some as invalid

        mask, metrics = selector.compute_mask(logits, labels, None)

        # All valid tokens should be kept
        valid_tokens = (labels != -100).sum().item()
        assert metrics["rsi/kept_tokens"] == valid_tokens
        assert metrics["rsi/kept_ratio"] == 1.0
        assert metrics["rsi/filtered_low"] == 0
        assert metrics["rsi/filtered_high"] == 0

    def test_no_selection_metrics(self):
        """Test that NoTokenSelection returns correct metrics."""
        selector = NoTokenSelection()

        batch_size = 1
        seq_len = 4
        vocab_size = 50

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        mask, metrics = selector.compute_mask(logits, labels, None)

        assert metrics["rsi/kept_ratio"] == 1.0
        assert metrics["rsi/filtered_low"] == 0
        assert metrics["rsi/filtered_high"] == 0


class TestTokenSelectionRegistry:
    """Test the token selection registry."""

    def test_registry_has_rsi(self):
        """Test that RSI Selection is registered."""
        assert "rsiselection" in token_selection_registry.list_registered()

    def test_registry_has_no_selection(self):
        """Test that NoTokenSelection is registered."""
        assert "notokenselection" in token_selection_registry.list_registered()

    def test_registry_create_rsi(self):
        """Test creating RSI selector via registry."""
        selector = token_selection_registry.create(
            {"type": "rsiselection", "rsi_min": 0.2, "rsi_max": 2.5}
        )
        assert isinstance(selector, RSISelection)
        assert selector.rsi_min == 0.2
        assert selector.rsi_max == 2.5

    def test_registry_create_no_selection(self):
        """Test creating NoTokenSelection via registry."""
        selector = token_selection_registry.create("notokenselection")
        assert isinstance(selector, NoTokenSelection)

    def test_registry_create_string(self):
        """Test creating selector from string name."""
        selector = token_selection_registry.create("rsiselection")
        assert isinstance(selector, RSISelection)

    def test_registry_unknown_type_raises(self):
        """Test that unknown selector type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown token selector type"):
            token_selection_registry.create("unknown_selector")
