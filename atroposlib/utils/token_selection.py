"""
Token selection mechanisms for RLVR training.

This module provides token selection methods that can be used during
RLVR (Reinforcement Learning with Verifiable Rewards) training to filter
which tokens contribute to gradient updates.

The key concept is RSI (Relative Surprisal Index), an information-theoretic
metric that couples token entropy with probability:
    RSI = -log p(x_t) / H(X|context)

RSI Selection (RSI-S) retains tokens within a stable RSI interval, filtering
out both redundant low-surprisal tokens and unstable high-surprisal tail tokens.

Reference:
    "Which Tokens Matter? Adaptive Token Selection for RLVR with the Relative
    Surprisal Index" (arXiv:2606.31575v1)
"""

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TokenSelector(ABC):
    """Abstract base class for token selection methods."""

    def __init__(self, rsi_min: float = 0.1, rsi_max: float = 3.0, **kwargs):
        """
        Initialize the token selector.

        Args:
            rsi_min: Minimum RSI threshold (filters low-surprisal tokens)
            rsi_max: Maximum RSI threshold (filters high-surprisal tokens)
            **kwargs: Additional configuration parameters
        """
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.config = kwargs

    @property
    def name(self) -> str:
        """Unique identifier for this token selector."""
        return self.__class__.__name__.lower()

    @abstractmethod
    def compute_mask(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        temperatures: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute a token mask based on the selection criteria.

        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len], -100 for masked positions
            temperatures: Temperature values [batch, 1] for scaling logits

        Returns:
            Tuple of (mask tensor [batch, seq_len], metrics dict)
            The mask is 1.0 for tokens to keep, 0.0 for tokens to filter out.
        """
        pass


class TokenSelectionRegistry:
    """Registry for token selection methods with factory pattern."""

    def __init__(self):
        self._registry: Dict[str, Type[TokenSelector]] = {}

    def register(self, cls=None, name=None):
        """
        Register a token selector class.

        Can be used as a decorator:

        @registry.register
        class MySelector(TokenSelector):
            ...

        or with a custom name:

        @registry.register(name="custom_name")
        class MySelector(TokenSelector):
            ...

        Args:
            cls: The token selector class to register
            name: Optional custom name to register the class under

        Returns:
            The registered class (for decorator use)
        """

        def _register(cls):
            # Validate that it's a subclass of TokenSelector
            if not inspect.isclass(cls) or not issubclass(cls, TokenSelector):
                raise TypeError(
                    f"Class {cls.__name__} is not a subclass of TokenSelector"
                )

            registered_name = name or cls.__name__.lower()
            self._registry[registered_name] = cls
            logger.debug(f"Registered token selector: {registered_name}")
            return cls

        if cls is None:
            return _register
        return _register(cls)

    def create(self, name_or_config, **kwargs) -> TokenSelector:
        """
        Create a token selector from name or config dict.

        Args:
            name_or_config: Either a string name or a dict with 'type' key
            **kwargs: Default parameters that can be overridden by config

        Returns:
            Instantiated TokenSelector object
        """
        if isinstance(name_or_config, str):
            selector_type = name_or_config
            selector_params = kwargs
        else:
            selector_config = name_or_config.copy()
            selector_type = selector_config.pop("type", "rsi")

            # Handle params dictionary if present
            if "params" in selector_config:
                params = selector_config.pop("params")
                selector_config.update(params)

            # Start with kwargs as defaults, override with config
            selector_params = {**kwargs}
            selector_params.update(selector_config)

        if selector_type not in self._registry:
            raise ValueError(
                f"Unknown token selector type: {selector_type}. "
                f"Available: {list(self._registry.keys())}"
            )

        selector_class = self._registry[selector_type]
        return selector_class(**selector_params)

    def list_registered(self) -> List[str]:
        """Return list of all registered token selector names."""
        return list(self._registry.keys())


# Global registry instance
registry = TokenSelectionRegistry()


@registry.register
class RSISelection(TokenSelector):
    """
    Relative Surprisal Index (RSI) token selection method.

    RSI is an information-theoretic metric that couples token entropy
    with probability:
        RSI = -log p(x_t) / H(X|context)

    where:
        p(x_t) is the probability of the selected token
        H(X|context) is the conditional entropy of the prediction distribution

    RSI Selection (RSI-S) retains tokens within a stable RSI interval,
    filtering out both redundant low-surprisal tokens and unstable
    high-surprisal tail tokens.

    This reconciles contradictory paradigms in RLVR:
    - High-entropy token prioritization vs. low-probability token caution
    - By using RSI, we get the best of both: tokens that are both
      uncertain (high entropy) and reasonably probable (not too low prob)

    Args:
        rsi_min: Minimum RSI threshold (default: 0.1)
                 Tokens with RSI < rsi_min are filtered as "too predictable"
        rsi_max: Maximum RSI threshold (default: 3.0)
                 Tokens with RSI > rsi_max are filtered as "too unstable"
        temperature: Temperature for scaling logits (default: 1.0)
        normalize_rsi: Whether to normalize RSI by dividing by entropy (default: True)
    """

    def __init__(
        self,
        rsi_min: float = 0.1,
        rsi_max: float = 3.0,
        temperature: float = 1.0,
        normalize_rsi: bool = True,
        **kwargs,
    ):
        super().__init__(rsi_min=rsi_min, rsi_max=rsi_max, **kwargs)
        self.temperature = temperature
        self.normalize_rsi = normalize_rsi

    def compute_rsi(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        temperatures: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Relative Surprisal Index (RSI) for each token.

        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len]
            temperatures: Temperature values [batch, 1] for scaling logits

        Returns:
            Tuple of (rsi_values, token_logprobs, entropy)
            - rsi_values: RSI for each token position [batch, seq_len]
            - token_logprobs: Log probability of selected token [batch, seq_len]
            - entropy: Predictive entropy at each position [batch, seq_len]
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Apply temperature scaling
        if temperatures is not None:
            temps = temperatures.unsqueeze(-1)  # [batch, 1, 1]
            temps = torch.where(temps <= 0, torch.ones_like(temps), temps)
            scaled_logits = logits / temps.to(logits.device, logits.dtype)
        else:
            scaled_logits = logits / self.temperature

        # Compute log probabilities for all tokens
        log_probs = F.log_softmax(scaled_logits, dim=-1)  # [batch, seq_len, vocab_size]

        # Compute predictive entropy: H = -sum(p * log(p))
        probs = F.softmax(scaled_logits, dim=-1)  # [batch, seq_len, vocab_size]
        entropy = -(probs * log_probs).sum(dim=-1)  # [batch, seq_len]
        entropy = entropy.clamp(min=1e-8)  # Avoid division by zero

        # Get log probability of the selected token (from labels)
        # Create one-hot encoding for selected tokens
        label_mask = (labels != -100).unsqueeze(-1)  # [batch, seq_len, 1]
        valid_labels = labels.clamp(min=0)  # Replace -100 with 0 (will be masked later)

        # Gather log probs for selected tokens
        token_indices = valid_labels.unsqueeze(-1)  # [batch, seq_len, 1]
        token_logprobs = torch.gather(log_probs, -1, token_indices).squeeze(-1)  # [batch, seq_len]

        # Mask out invalid positions (where labels == -100)
        token_logprobs = token_logprobs * label_mask.squeeze(-1).float()

        # Compute RSI: RSI = -log p(x_t) / H(X|context)
        # -log p(x_t) is the surprisal
        surprisal = -token_logprobs  # [batch, seq_len]

        if self.normalize_rsi:
            rsi_values = surprisal / entropy  # [batch, seq_len]
        else:
            rsi_values = surprisal

        # Mask out invalid positions
        mask = (labels != -100).float()
        rsi_values = rsi_values * mask

        return rsi_values, token_logprobs, entropy

    def compute_mask(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        temperatures: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute RSI-based token mask.

        Tokens with RSI within [rsi_min, rsi_max] are kept (mask = 1.0).
        Tokens outside this interval are filtered (mask = 0.0).

        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len], -100 for masked positions
            temperatures: Temperature values [batch, 1] for scaling logits

        Returns:
            Tuple of (mask tensor [batch, seq_len], metrics dict)
        """
        rsi_values, token_logprobs, entropy = self.compute_rsi(
            logits, labels, temperatures
        )

        # Create RSI-based mask
        rsi_mask = (rsi_values >= self.rsi_min) & (rsi_values <= self.rsi_max)
        rsi_mask = rsi_mask.float()

        # Also respect the original label mask (where labels != -100)
        label_mask = (labels != -100).float()
        final_mask = rsi_mask * label_mask

        # Compute metrics
        with torch.no_grad():
            valid_tokens = label_mask.sum().item()
            kept_tokens = final_mask.sum().item()
            filtered_low = ((rsi_values < self.rsi_min) & (label_mask > 0)).sum().item()
            filtered_high = ((rsi_values > self.rsi_max) & (label_mask > 0)).sum().item()

            # Mean RSI for valid tokens
            valid_rsi = rsi_values[label_mask > 0]
            mean_rsi = valid_rsi.mean().item() if valid_rsi.numel() > 0 else 0.0
            std_rsi = valid_rsi.std().item() if valid_rsi.numel() > 0 else 0.0

            metrics = {
                "rsi/kept_ratio": kept_tokens / valid_tokens if valid_tokens > 0 else 0.0,
                "rsi/filtered_low": filtered_low,
                "rsi/filtered_high": filtered_high,
                "rsi/mean": mean_rsi,
                "rsi/std": std_rsi,
                "rsi/min": valid_rsi.min().item() if valid_rsi.numel() > 0 else 0.0,
                "rsi/max": valid_rsi.max().item() if valid_rsi.numel() > 0 else 0.0,
                "rsi/valid_tokens": valid_tokens,
                "rsi/kept_tokens": kept_tokens,
            }

        return final_mask, metrics


@registry.register
class NoTokenSelection(TokenSelector):
    """
    Pass-through token selector that keeps all valid tokens.

    This is the default/no-op selector for baseline comparison.
    """

    def compute_mask(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        temperatures: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Return a mask that keeps all valid tokens (where labels != -100).

        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            labels: Target labels [batch, seq_len]
            temperatures: Temperature values (ignored)

        Returns:
            Tuple of (mask tensor [batch, seq_len], metrics dict)
        """
        mask = (labels != -100).float()
        metrics = {
            "rsi/kept_ratio": 1.0,
            "rsi/filtered_low": 0,
            "rsi/filtered_high": 0,
            "rsi/valid_tokens": mask.sum().item(),
            "rsi/kept_tokens": mask.sum().item(),
        }
        return mask, metrics


# Export registry and key classes
__all__ = [
    "TokenSelector",
    "TokenSelectionRegistry",
    "RSISelection",
    "NoTokenSelection",
    "registry",
]
