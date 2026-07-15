"""
Riemannian Preconditioned LoRA optimizer.

Implements the preconditioned gradient updates from "Riemannian Preconditioned LoRA
for Fine-Tuning Foundation Models" (arXiv:2402.02347v3).

The key insight is that LoRA's low-rank structure A·B enables efficient r×r
preconditioning, where r is the LoRA rank. The preconditioned updates are:

    A ← A - η (B^T B)^(-1) ∇_A L
    B ← B - η ∇_B L (A A^T)^(-1)

This is an optimizer wrapper that:
1. Identifies LoRA A and B parameters by name pattern
2. Computes the r×r preconditioner matrices efficiently
3. Applies preconditioned gradients before the base optimizer step

The preconditioning cost is O(r³) per LoRA layer, which is negligible since
r is typically small (8-64), while the full parameter dimension is thousands.

Based on: https://arxiv.org/abs/2402.02347v3
"""

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class RiemannianLoRAOptimizer:
    """
    Optimizer wrapper that applies Riemannian preconditioning to LoRA parameters.

    This wraps any PyTorch optimizer and applies the r×r preconditioner to
    LoRA A and B matrices before each optimizer step.

    Usage:
        base_opt = AdamW(params, lr=1e-4)
        opt = RiemannianLoRAOptimizer(base_opt, model)
        opt.step()  # Applies preconditioning then base optimizer step
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        model: Optional[torch.nn.Module] = None,
        damping: float = 1e-6,
        preconditioner_frequency: int = 1,
    ):
        """
        Initialize the Riemannian preconditioned LoRA optimizer.

        Args:
            base_optimizer: The underlying optimizer (AdamW, AdamW8bit, etc.)
            model: Optional model to auto-detect LoRA parameters. If None,
                   parameters are detected by name pattern during step().
            damping: Small value added to diagonal for numerical stability
            preconditioner_frequency: Apply preconditioning every N steps
        """
        self.base_optimizer = base_optimizer
        self.damping = damping
        self.preconditioner_frequency = preconditioner_frequency
        self._step_count = 0

        # Map parameters to their A/B pairs
        self._lora_pairs: dict[
            torch.nn.Parameter, Tuple[str, Optional[torch.nn.Parameter]]
        ] = {}

        if model is not None:
            self._identify_lora_parameters(model)

    def _identify_lora_parameters(self, model: torch.nn.Module) -> None:
        """
        Identify LoRA A and B parameter pairs from the model.

        PEFT/LoRA convention uses names like:
        - base_model.model.model.layers.N.{module}.lora_A.default
        - base_model.model.model.layers.N.{module}.lora_B.default
        """
        lora_a_params = {}
        lora_b_params = {}

        # Group parameters by their base layer/module
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if "lora_A" in name or "lora_a" in name:
                # Extract the key that identifies the LoRA layer
                key = self._get_lora_layer_key(name)
                lora_a_params[key] = (name, param)
            elif "lora_B" in name or "lora_b" in name:
                key = self._get_lora_layer_key(name)
                lora_b_params[key] = (name, param)

        # Pair up A and B parameters
        matched = 0
        for key, (name_a, param_a) in lora_a_params.items():
            if key in lora_b_params:
                name_b, param_b = lora_b_params[key]
                self._lora_pairs[param_a] = (name_a, param_b)
                self._lora_pairs[param_b] = (name_b, param_a)
                matched += 1

        logger.info(f"[RiemannianLoRA] Identified {matched} LoRA A-B pairs")

    def _get_lora_layer_key(self, param_name: str) -> str:
        """
        Extract a key that identifies the LoRA layer/module from parameter name.

        For example, from:
          "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default"
        Extract:
          "layers.0.self_attn.q_proj"
        """
        parts = param_name.split(".")
        key_parts = []
        for i, part in enumerate(parts):
            if "lora" in part.lower():
                # Take everything up to (but not including) the lora part
                key_parts = parts[:i]
                break
        return ".".join(key_parts)

    def _compute_preconditioner(
        self, grad: torch.Tensor, partner: torch.nn.Parameter
    ) -> torch.Tensor:
        """
        Compute the r×r preconditioner matrix.

        For A parameter (shape r x k):
            P_A = (B^T @ B + λI)^(-1)

        For B parameter (shape m x r):
            P_B = (A @ A^T + λI)^(-1)

        Args:
            grad: Gradient tensor for the parameter
            partner: The paired parameter (A's partner is B, B's partner is A)

        Returns:
            Preconditioned gradient (same shape as input grad)
        """
        if partner is None:
            # No partner found, return gradient as-is
            return grad

        # Get the shapes to determine which is A (r x k) and which is B (m x r)
        grad_shape = grad.shape
        partner_shape = partner.shape

        # r (rank) is the smaller dimension that connects A and B
        # A is typically (rank, in_features) or (in_features, rank)
        # B is typically (out_features, rank)
        # We need to identify which dimension is the rank dimension

        if grad_shape[0] < grad_shape[1] and grad_shape[0] == partner_shape[-1]:
            # grad is A-like (rank, ...) and partner's last dim is rank
            rank = grad_shape[0]
            # Compute (B^T @ B)^(-1) - B is (out_features, rank)
            B = partner.data
            # B^T @ B is (rank, rank)
            gram = torch.mm(B.t(), B)
        elif grad_shape[-1] < grad_shape[0] and grad_shape[-1] == partner_shape[0]:
            # grad is A-like (..., rank) and partner's first dim is rank
            rank = grad_shape[-1]
            # Compute (B @ B^T)^(-1) - B is (rank, out_features)
            B = partner.data
            gram = torch.mm(B, B.t())
        elif grad_shape[-1] == partner_shape[0]:
            # grad is B-like (..., rank), partner is A (rank, ...)
            rank = grad_shape[-1]
            # Compute (A @ A^T)^(-1)
            A = partner.data
            gram = torch.mm(A, A.t())
        elif grad_shape[0] == partner_shape[-1]:
            # grad is B-like (rank, ...), partner is A (..., rank)
            rank = grad_shape[0]
            # Compute (A^T @ A)^(-1)
            A = partner.data
            gram = torch.mm(A.t(), A)
        else:
            # Cannot determine structure, return gradient as-is
            return grad

        # Add damping for numerical stability
        gram = gram + self.damping * torch.eye(
            rank, device=gram.device, dtype=gram.dtype
        )

        # Compute inverse (rank is small, so this is cheap)
        try:
            preconditioner = torch.linalg.solve(
                gram, torch.eye(rank, device=gram.device, dtype=gram.dtype)
            )
        except RuntimeError:
            # Fallback to pseudo-inverse if singular
            preconditioner = torch.linalg.pinv(gram)

        # Apply preconditioner to gradient
        if grad.ndim == 2:
            if grad_shape[0] == rank and grad_shape[-1] == rank:
                # Both dimensions are rank - apply from left and right
                grad_new = preconditioner @ grad @ preconditioner
            elif grad_shape[0] == rank:
                # Left multiplication
                grad_new = preconditioner @ grad
            elif grad_shape[-1] == rank:
                # Right multiplication
                grad_new = grad @ preconditioner
            else:
                grad_new = grad
        else:
            # For 1D or higher-dimensional tensors, handle accordingly
            if grad_shape[0] == rank:
                grad_new = preconditioner @ grad
            elif grad_shape[-1] == rank:
                grad_new = (
                    grad @ preconditioner.t()
                    if grad.ndim > 1
                    else preconditioner @ grad
                )
            else:
                grad_new = grad

        return grad_new

    def _apply_preconditioning(self) -> None:
        """Apply Riemannian preconditioning to LoRA parameter gradients."""
        for group in self.base_optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                # Check if this is a LoRA parameter
                if param in self._lora_pairs:
                    _, partner = self._lora_pairs[param]
                    param.grad = self._compute_preconditioner(param.grad, partner)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step with preconditioning.

        Args:
            closure: Optional closure for re-evaluating the model

        Returns:
            The result of the base optimizer's step() call
        """
        self._step_count += 1

        # Apply preconditioning if it's time
        if self._step_count % self.preconditioner_frequency == 0:
            self._apply_preconditioning()

        # Call base optimizer step
        return self.base_optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        """Clear gradients."""
        return self.base_optimizer.zero_grad(set_to_none)

    def state_dict(self):
        """Return state dict."""
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict."""
        return self.base_optimizer.load_state_dict(state_dict)

    def __getattr__(self, name):
        """Delegate other attributes to base optimizer."""
        return getattr(self.base_optimizer, name)


def create_riemannian_lora_optimizer(
    model: torch.nn.Module,
    config,
) -> torch.optim.Optimizer:
    """
    Create an optimizer with Riemannian preconditioning for LoRA parameters.

    This is a drop-in replacement for create_optimizer_for_params that
    wraps the base optimizer with Riemannian preconditioning.

    Args:
        model: The model with LoRA parameters
        config: TrainingConfig with optimizer settings

    Returns:
        RiemannianLoRAOptimizer wrapping the base optimizer
    """
    from .trainers import create_optimizer_for_params

    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for optimizer creation.")

    # Create base optimizer
    base_optimizer = create_optimizer_for_params(trainable_params, config)

    # Check if any parameters are LoRA parameters
    has_lora = any(
        "lora" in name for name, _ in model.named_parameters() if _.requires_grad
    )

    if not has_lora:
        logger.info(
            "[RiemannianLoRA] No LoRA parameters detected, using base optimizer"
        )
        return base_optimizer

    # Wrap with Riemannian preconditioning
    damping = getattr(config, "riemannian_damping", 1e-6)
    frequency = getattr(config, "riemannian_frequency", 1)

    logger.info(
        f"[RiemannianLoRA] Wrapping optimizer with preconditioning (damping={damping})"
    )

    return RiemannianLoRAOptimizer(
        base_optimizer=base_optimizer,
        model=model,
        damping=damping,
        preconditioner_frequency=frequency,
    )
