"""Integration tests for the metacognitive-accuracy (RLMF Z_g) reward.

These exercise the reward through the same public surface ``DatasetEnv.score``
uses -- the reward ``registry`` and ``CombinedReward`` -- and check that it
correctly reuses the existing first-order ``MetacognitiveReward`` to source
``F_gold``. The integration point at
``environments/dataset_environment/dataset_env.py`` is
``registry.create(<name>)``.
"""

import pytest

from atroposlib.envs.reward_fns import CombinedReward, registry
from atroposlib.envs.reward_fns.metacognitive_accuracy import (
    MetacognitiveAccuracyReward,
)
from atroposlib.envs.reward_fns.metacognitive_reward import MetacognitiveReward


def test_registry_creates_metacognitive_accuracy_reward():
    # This is the precise call DatasetEnv._initialize_reward_function makes.
    reward = registry.create("metacognitiveaccuracy")
    assert isinstance(reward, MetacognitiveAccuracyReward)


def test_z_g_rewards_accurate_self_assessment():
    reward = registry.create("metacognitiveaccuracy")
    completions = [
        "<self_assessment>1.0</self_assessment>",  # claims fully calibrated
        "<self_assessment>1.0</self_assessment>",  # claims fully calibrated
    ]
    # F_gold supplied directly: item 0 is truly faithful, item 1 is not.
    scores = reward.compute(completions, faithfulness=[1.0, 0.0])
    assert scores[0] == pytest.approx(1.0)  # accurate self-assessment -> 1
    assert scores[1] == pytest.approx(0.0)  # overconfident self-assessment -> 0


def test_correctness_signal_drives_gold_level():
    reward = registry.create("metacognitiveaccuracy")
    completions = ["<self_assessment>0.5</self_assessment>"]
    scores = reward.compute(completions, correctness=[True])
    # F_gold = 1.0, F_pred = 0.5 -> Z_g = 1 - (0.5 - 1.0)^2 = 0.75
    assert scores[0] == pytest.approx(0.75)


def test_missing_self_assessment_returns_default():
    reward = registry.create({"type": "metacognitiveaccuracy", "default_reward": 0.3})
    scores = reward.compute(["<answer>4</answer>"], faithfulness=[1.0])
    assert scores[0] == pytest.approx(0.3)


def test_gold_falls_back_to_first_order_metacognitive_reward():
    # No faithfulness/correctness supplied: F_gold must be computed by the
    # existing (non-new) MetacognitiveReward from ground_truth, proving the
    # two rewards are wired together.
    reward = registry.create("metacognitiveaccuracy")
    completion = (
        "<answer>42</answer>"
        "<confidence>0.95</confidence>"
        "<self_assessment>0.95</self_assessment>"
    )
    scores = reward.compute([completion], ground_truth="42")

    # Cross-check against the first-order reward's own F_gold.
    f_gold = MetacognitiveReward().compute([completion], ground_truth="42")[0]
    assert f_gold == pytest.approx(0.9975)  # 1 - (0.95 - 1)^2
    assert scores[0] == pytest.approx(1.0 - (0.95 - f_gold) ** 2)


def test_composes_with_first_order_reward_via_combined_reward():
    # The second-order Z_g signal is meant to sit alongside the first-order
    # faithfulness reward; CombinedReward is the composition primitive
    # DatasetEnv uses when >1 reward is named.
    combined = CombinedReward(
        rewards=[{"type": "metacognitive"}, {"type": "metacognitiveaccuracy"}],
        normalization="none",
    )
    completion = (
        "<answer>4</answer>"
        "<confidence>1.0</confidence>"
        "<self_assessment>1.0</self_assessment>"
    )
    scores = combined.compute([completion], correctness=[True])
    # first-order faithfulness 1-(1-1)^2 = 1.0 plus Z_g 1-(1-1)^2 = 1.0
    assert scores[0] == pytest.approx(2.0)
