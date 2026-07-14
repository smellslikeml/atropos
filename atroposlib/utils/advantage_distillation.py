"""
Advantage-guided self-distillation for token-level supervision.

This module implements the core insight from ROAD-VLA (Robust Online Adaptation
via Self-Distillation for Vision-Language-Action Models), which constructs a
proximal teacher directly in action space by perturbing action-token logits with
calibrated advantage estimates.

The key contribution is converting sparse rewards into dense token-level
supervision, providing a stronger training signal for student models.

Reference:
    ROAD-VLA: Robust Online Adaptation via Self-Distillation for
    Vision-Language-Action Models. https://arxiv.org/abs/2606.25800v1
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def compute_token_level_advantages(
    sequence_advantages: List[float],
    masks: List[int],
    num_tokens: int,
) -> List[float]:
    """
    Distribute sequence-level advantages across tokens.

    This implements ROAD-VLA's token-level advantage construction. Each token
    in the sequence receives a share of the sequence's advantage, weighted by
    its position (later tokens receive slightly higher weight as they are more
    directly responsible for the outcome).

    Args:
        sequence_advantages: Sequence-level advantage scores.
        masks: Token masks (-100 for padding/ignore, token ID otherwise).
        num_tokens: Total vocabulary size (for normalization).

    Returns:
        Token-level advantages as a list of floats, same length as masks.
        Non-masked positions receive distributed advantages, masked positions
        receive 0.0.
    """
    if not sequence_advantages or not masks:
        return []

    # Find valid (non-padding) token positions
    valid_positions = [i for i, mask in enumerate(masks) if mask != -100]
    if not valid_positions:
        return [0.0] * len(masks)

    # Average the sequence advantages if multiple are provided
    avg_advantage = float(np.mean(sequence_advantages))

    # Position-weighted distribution: later tokens get slightly higher weight
    # This reflects that later tokens have more causal influence on outcomes
    num_valid = len(valid_positions)
    position_weights = np.linspace(0.5, 1.5, num_valid)  # Linear ramp
    position_weights /= position_weights.sum()  # Normalize

    # Initialize with zeros (padding tokens get no advantage signal)
    token_advantages = [0.0] * len(masks)

    # Distribute advantage across valid tokens
    for idx, pos in enumerate(valid_positions):
        token_advantages[pos] = avg_advantage * position_weights[idx]

    return token_advantages


def batch_compute_token_advantages(
    advantages: Optional[List[List[float]]],
    masks: List[List[int]],
) -> List[List[float]]:
    """
    Compute token-level advantages for a batch of sequences.

    Args:
        advantages: Sequence-level advantages [batch_size][num_sequences] or None.
        masks: Token masks [batch_size][seq_len].

    Returns:
        Token-level advantages [batch_size][seq_len]. Returns all zeros if
        advantages is None.
    """
    if advantages is None:
        # No advantages provided, return zeros
        return [[0.0] * len(mask) for mask in masks]

    token_advantages = []
    for seq_advantages, mask in zip(advantages, masks):
        token_adv = compute_token_level_advantages(
            seq_advantages, mask, num_tokens=0
        )
        token_advantages.append(token_adv)

    return token_advantages


def construct_advantage_shaped_logits(
    student_logits: Optional[List[List[float]]],
    token_advantages: List[float],
    advantage_scale: float = 0.1,
    temperature: float = 1.0,
) -> List[float]:
    """
    Construct advantage-shaped teacher logits.

    ROAD-VLA's core mechanism: perturb student logits with calibrated advantage
    estimates to create a proximal teacher. This keeps the teacher close to the
    current policy while incorporating advantage information.

    The teacher logits are:
        teacher_logits = student_logits + advantage_scale * advantages

    Args:
        student_logits: Student model logits [vocab_size] or None.
        token_advantages: Token-level advantage values (must match vocab size
            when logits are provided, or be scalar to broadcast).
        advantage_scale: Scaling factor for advantage perturbation (default 0.1).
            Smaller values keep the teacher closer to the student policy.
        temperature: Temperature for softening the distribution (default 1.0).

    Returns:
        Advantage-shaped teacher logits [vocab_size].
    """
    if student_logits is None:
        # No student logits available, return advantage-shaped uniform prior
        vocab_size = len(token_advantages) if isinstance(token_advantages, list) else 1
        if vocab_size == 0:
            return []

        # Scale advantages and apply temperature
        scaled_advantages = [
            adv * advantage_scale / temperature for adv in token_advantages
        ]
        return scaled_advantages

    # Student logits available, perturb with advantages
    if len(student_logits) != len(token_advantages):
        # Assume token_advantages is a scalar to broadcast
        token_advantages_list = [float(np.mean(token_advantages))] * len(student_logits)
    else:
        token_advantages_list = token_advantages

    # Apply advantage perturbation and temperature
    teacher_logits = []
    for logit, advantage in zip(student_logits, token_advantages_list):
        perturbed = (logit + advantage_scale * advantage) / temperature
        teacher_logits.append(perturbed)

    return teacher_logits


def batch_construct_advantage_logits(
    student_logits: Optional[List[List[List[float]]]],
    token_advantages: List[List[float]],
    advantage_scale: float = 0.1,
    temperature: float = 1.0,
) -> Optional[List[List[List[float]]]]:
    """
    Construct advantage-shaped teacher logits for a batch of sequences.

    Args:
        student_logits: Student logits [batch][seq][vocab] or None.
        token_advantages: Token-level advantages [batch][seq].
        advantage_scale: Scaling factor for advantage perturbation.
        temperature: Temperature for softening.

    Returns:
        Advantage-shaped teacher logits [batch][seq][vocab], or None if input
        is None.
    """
    if student_logits is None or token_advantages is None:
        return None

    teacher_logits = []
    for seq_logits, seq_advantages in zip(student_logits, token_advantages):
        seq_teacher = []
        for pos_logits, pos_advantages in zip(seq_logits, seq_advantages):
            pos_teacher = construct_advantage_shaped_logits(
                pos_logits,
                [pos_advantages],  # Wrap as list for scalar broadcasting
                advantage_scale,
                temperature,
            )
            seq_teacher.append(pos_teacher)
        teacher_logits.append(seq_teacher)

    return teacher_logits


def calibrate_advantage_scale(
    advantages: List[float],
    target_std_ratio: float = 0.2,
    student_logits_std: Optional[float] = None,
) -> float:
    """
    Calibrate advantage scale to maintain policy proximity.

    ROAD-VLA requires calibrated advantages to ensure the teacher remains
    close to the current policy. This computes an appropriate scale factor
    based on the variability of advantages.

    Args:
        advantages: Advantage values to calibrate.
        target_std_ratio: Target ratio of advantage std to logit std (default 0.2).
        student_logits_std: Standard deviation of student logits for normalization.

    Returns:
        Calibrated advantage scale factor.
    """
    if not advantages:
        return 0.1  # Default scale

    adv_std = float(np.std(advantages))
    if adv_std < 1e-8:
        return 0.1  # No variation, use default

    # Scale to achieve target std ratio
    if student_logits_std is not None and student_logits_std > 0:
        calibrated_scale = (target_std_ratio * student_logits_std) / adv_std
    else:
        # Use normalized scale
        calibrated_scale = target_std_ratio / adv_std

    # Clamp to reasonable bounds
    calibrated_scale = max(0.01, min(calibrated_scale, 1.0))

    return calibrated_scale


class AdvantageDistillationConfig:
    """
    Configuration for advantage-guided self-distillation.

    Attributes:
        enabled: Whether to enable advantage distillation.
        advantage_scale: Base scaling factor for advantage perturbation.
        auto_calibrate: Whether to auto-calibrate advantage scale.
        target_std_ratio: Target std ratio for auto-calibration.
        temperature: Temperature for softening the distribution.
    """

    def __init__(
        self,
        enabled: bool = True,
        advantage_scale: float = 0.1,
        auto_calibrate: bool = True,
        target_std_ratio: float = 0.2,
        temperature: float = 1.0,
    ):
        self.enabled = enabled
        self.advantage_scale = advantage_scale
        self.auto_calibrate = auto_calibrate
        self.target_std_ratio = target_std_ratio
        self.temperature = temperature


def compute_advantage_distillation_payload(
    advantages: Optional[List[List[float]]],
    masks: List[List[int]],
    student_logits: Optional[List[List[List[float]]]] = None,
    config: Optional[AdvantageDistillationConfig] = None,
) -> dict:
    """
    Compute the complete advantage distillation payload.

    This is the main entry point for ROAD-VLA integration. It computes:
    1. Token-level advantages from sequence-level advantages
    2. Advantage-shaped teacher logits
    3. Calibrated advantage scale (if auto-calibrate is enabled)

    Args:
        advantages: Sequence-level advantages [batch][num_sequences].
        masks: Token masks [batch][seq_len].
        student_logits: Optional student logits [batch][seq][vocab].
        config: AdvantageDistillationConfig or None (uses defaults).

    Returns:
        Dictionary with:
            - token_advantages: [batch][seq_len] token-level advantages
            - advantage_logits: [batch][seq][vocab] advantage-shaped logits (or None)
            - calibrated_scale: float, the calibrated scale factor used
    """
    if config is None:
        config = AdvantageDistillationConfig()

    if not config.enabled or advantages is None:
        return {
            "token_advantages": [[0.0] * len(mask) for mask in masks],
            "advantage_logits": None,
            "calibrated_scale": config.advantage_scale,
        }

    # Step 1: Compute token-level advantages
    token_advantages = batch_compute_token_advantages(advantages, masks)

    # Step 2: Calibrate advantage scale if requested
    if config.auto_calibrate:
        flat_advantages = [a for seq in token_advantages for a in seq]
        student_logits_std = None
        if student_logits is not None:
            flat_logits = [l for seq in student_logits for pos in seq for l in pos]
            if flat_logits:
                student_logits_std = float(np.std(flat_logits))

        calibrated_scale = calibrate_advantage_scale(
            flat_advantages,
            config.target_std_ratio,
            student_logits_std,
        )
    else:
        calibrated_scale = config.advantage_scale

    # Step 3: Construct advantage-shaped teacher logits
    advantage_logits = batch_construct_advantage_logits(
        student_logits,
        token_advantages,
        calibrated_scale,
        config.temperature,
    )

    return {
        "token_advantages": token_advantages,
        "advantage_logits": advantage_logits,
        "calibrated_scale": calibrated_scale,
    }
