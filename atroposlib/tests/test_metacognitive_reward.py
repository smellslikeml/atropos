"""Integration tests for the metacognitive (RLMF-style) reward function.

These exercise the reward through the existing public surface that
``DatasetEnv.score`` relies on -- the reward ``registry`` and
``CombinedReward`` -- so they prove the new reward plugs into the
dataset_env reward path rather than only self-testing the new module.
The exact integration point used at
``environments/dataset_environment/dataset_env.py`` is
``registry.create(<name>)``.
"""

import pytest

from atroposlib.envs.reward_fns import CombinedReward, registry
from atroposlib.envs.reward_fns.metacognitive_reward import (
    MetacognitiveReward,
    parse_confidence,
)


def test_parse_confidence_handles_tags_percents_and_absence():
    assert parse_confidence("<confidence>0.8</confidence>") == pytest.approx(0.8)
    assert parse_confidence("<confidence>80%</confidence>") == pytest.approx(0.8)
    assert parse_confidence("Confidence: 25%") == pytest.approx(0.25)
    assert parse_confidence("[confidence: 0.1]") == pytest.approx(0.1)
    assert parse_confidence("no confidence stated here") is None


def test_registry_creates_metacognitive_reward():
    # This is the precise call DatasetEnv._initialize_reward_function makes.
    reward = registry.create("metacognitive")
    assert isinstance(reward, MetacognitiveReward)


def test_faithfulness_rewards_calibrated_confidence():
    reward = registry.create("metacognitive")
    completions = [
        "<answer>4</answer><confidence>0.9</confidence>",  # correct + confident
        "<answer>5</answer><confidence>0.9</confidence>",  # wrong + overconfident
        "<answer>5</answer><confidence>0.1</confidence>",  # wrong + appropriately doubtful
        "<answer>4</answer>",  # no confidence stated
    ]
    scores = reward.compute(completions, correctness=[True, False, False, True])

    assert scores[0] == pytest.approx(0.99)  # confidently right (1-(0.9-1)^2)
    assert scores[1] < 0.25  # confidently wrong -> near 0 (1-(0.9-0)^2=0.19)
    assert scores[2] > 0.95  # humbly wrong -> ~1 (1-(0.1-0)^2=0.99)
    assert scores[3] == pytest.approx(0.5)  # no confidence -> neutral default


def test_ground_truth_drives_proxy_correctness():
    reward = registry.create({"type": "metacognitive", "default_reward": 0.0})
    completions = [
        "<answer>42</answer><confidence>0.95</confidence>",  # correct
        "<answer>13</answer><confidence>0.95</confidence>",  # wrong
    ]
    scores = reward.compute(completions, ground_truth="42")
    assert scores[0] == pytest.approx(0.9975)  # 1 - (0.95 - 1)^2
    assert scores[1] == pytest.approx(0.0975)  # 1 - (0.95 - 0)^2


def test_composes_with_existing_reward_via_combined_reward():
    # RLMF signal is meant to combine with task rewards; CombinedReward is the
    # existing composition primitive DatasetEnv uses when >1 reward is named.
    combined = CombinedReward(
        rewards=[{"type": "metacognitive"}, {"type": "format"}],
        normalization="none",
    )
    completion = "<answer>4</answer><confidence>0.9</confidence>"
    scores = combined.compute([completion], correctness=[True])
    # metacognitive faithfulness (correct + confident) ~0.99 plus format
    # reward 1.0 for the <answer> tag.
    assert scores[0] == pytest.approx(1.99, abs=1e-2)
