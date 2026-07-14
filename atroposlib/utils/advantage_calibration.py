"""Calibrated signed advantage weights for advantage-guided self-distillation.

ROAD-VLA builds its proximal teacher by perturbing the sampled action token's
logit with a *calibrated signed advantage weight* ``omega_t`` -- the teacher for
token ``k`` is ``softmax(z_{t,k} + eta * omega_t * e_{a_hat_{t,k}})``. This module
produces that ``omega_t``.

The calibration is deliberately faithful to the paper (Section 4.2) on two points
that a naive "advantage * scale" perturbation gets wrong:

  * **Signed, no ReLU.** ``omega_t`` keeps its sign. A negative advantage pushes
    probability mass *away* from the sampled token; a positive advantage pushes it
    *toward* the token. Rectifying to non-negative would discard half the signal.

  * **Standardize, then clip symmetrically.** Advantages are standardized to zero
    mean / unit variance across the group and then clipped to ``[-clip, clip]``
    (default ``[-2, 2]``). The clip is what keeps the teacher *proximal*: the
    perturbation cannot grow without bound, so the teacher stays close to the
    current policy. It also degrades gracefully on low-variance groups -- when the
    group std is ~0 every weight is 0.0 rather than the exploding
    ``advantage / std`` ratio that unbounded normalization produces (the failure
    mode tracked in atropos issue #457).

Reference:
    ROAD-VLA: Robust Online Adaptation via Self-Distillation for
    Vision-Language-Action Models. https://arxiv.org/abs/2606.25800v1
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

# Symmetric clip range applied after standardization (paper uses [-2, 2]).
DEFAULT_ADVANTAGE_CLIP = 2.0


def _reduce_sequence_advantage(sequence_advantages: Sequence[float]) -> float:
    """Reduce a sequence's (possibly per-token) advantages to one scalar.

    GRPO supplies group-normalized advantages that are typically constant across a
    sequence's tokens; the mean recovers that scalar and also handles the general
    per-token case without biasing the sign.
    """
    if sequence_advantages is None or len(sequence_advantages) == 0:
        return 0.0
    return float(np.mean(np.asarray(sequence_advantages, dtype=np.float64)))


def calibrate_signed_advantage_weights(
    advantages: Sequence[Sequence[float]],
    clip: float = DEFAULT_ADVANTAGE_CLIP,
    eps: float = 1e-8,
) -> List[float]:
    """Compute one signed, calibrated advantage weight ``omega_t`` per sequence.

    Each sequence is reduced to a scalar advantage, the scalars are standardized
    across the group (zero mean / unit std), and the result is clipped to
    ``[-clip, clip]``.

    Args:
        advantages: Group of per-sequence advantages ``[num_sequences][*]``. Each
            entry may be a scalar-in-a-list or a per-token list.
        clip: Symmetric clip bound applied after standardization.
        eps: Variance floor; groups with std below this get all-zero weights
            (no perturbation) instead of an exploding normalization ratio.

    Returns:
        One signed weight ``omega_t`` per sequence, each in ``[-clip, clip]``.
    """
    if advantages is None or len(advantages) == 0:
        return []

    scalars = np.asarray(
        [_reduce_sequence_advantage(seq) for seq in advantages], dtype=np.float64
    )

    std = float(scalars.std())
    if std < eps:
        # Low/zero-variance group: standardization is undefined, so emit no signal.
        return [0.0] * len(scalars)

    standardized = (scalars - float(scalars.mean())) / std
    bound = abs(float(clip))
    clipped = np.clip(standardized, -bound, bound)
    return [float(v) for v in clipped]
