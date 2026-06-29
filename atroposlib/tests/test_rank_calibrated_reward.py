"""Tests for the RiVER-inspired rank-calibrated reward function.

These exercise the reward through the existing (non-new) reward-function
framework: the package ``atroposlib.envs.reward_fns`` (whose ``__init__`` eagerly
registers the reward) and the ``RewardFunction`` base class ``__call__`` path.
"""

import math

from atroposlib.envs.reward_fns import RankCalibratedReward, registry


def test_registry_creates_rank_calibrated():
    """The reward is registered and creatable through the existing registry."""
    reward_fn = registry.create("rankcalibrated")
    assert isinstance(reward_fn, RankCalibratedReward)
    # Importable from the package proves the __init__ wiring runs registration.
    assert RankCalibratedReward.__name__ == "RankCalibratedReward"


def test_top_ranked_gets_full_reward_with_emphasis():
    """Top solver gets max_reward; emphasis sharpens the gap for the rest."""
    reward_fn = RankCalibratedReward(emphasis=2.0)
    rewards = reward_fn.compute(["a", "b", "c"], scores=[0.0, 50.0, 100.0])
    # calibrated = [0, 0.5, 1.0]; emphasized (**)2 = [0, 0.25, 1.0]
    assert math.isclose(rewards[0], 0.0)
    assert math.isclose(rewards[1], 0.25)
    assert math.isclose(rewards[2], 1.0)


def test_linear_calibration_when_emphasis_is_one():
    """emphasis=1.0 yields plain min-max calibration with no extra emphasis."""
    reward_fn = RankCalibratedReward(emphasis=1.0)
    rewards = reward_fn.compute(["a", "b", "c"], scores=[0.0, 50.0, 100.0])
    assert math.isclose(rewards[0], 0.0)
    assert math.isclose(rewards[1], 0.5)
    assert math.isclose(rewards[2], 1.0)


def test_scale_invariance_across_instances():
    """Scale dominance is removed: equal relative structure -> equal rewards.

    Instance A scores span [0, 100]; instance B is the *same* relative ordering
    scaled down 100x to [0.0, 1.0]. A naive reward using raw magnitudes would let
    instance A dominate policy updates; calibration makes both identical.
    """
    reward_fn = RankCalibratedReward(emphasis=2.0)
    big = reward_fn.compute(["a", "b", "c"], scores=[0.0, 50.0, 100.0])
    small = reward_fn.compute(["a", "b", "c"], scores=[0.0, 0.5, 1.0])
    for got, expected in zip(small, big):
        assert math.isclose(got, expected)


def test_frequency_dominance_keeps_others_bounded():
    """Rare strong solver is not crowded out by many mediocre, equal ones.

    Three mediocre completions share the group minimum while one strong solver
    tops the group. Each mediocre completion receives the bounded min_reward and
    only the strong solver receives the full reward.
    """
    reward_fn = RankCalibratedReward(emphasis=2.0, min_reward=-1.0, max_reward=1.0)
    rewards = reward_fn.compute(
        ["m1", "m2", "m3", "best"], scores=[40.0, 40.0, 40.0, 100.0]
    )
    assert math.isclose(rewards[0], -1.0)
    assert math.isclose(rewards[1], -1.0)
    assert math.isclose(rewards[2], -1.0)
    assert math.isclose(rewards[3], 1.0)


def test_higher_is_better_false_inverts_ranking():
    """Cost-style scores (lower is better) rank the smallest score on top."""
    reward_fn = RankCalibratedReward(higher_is_better=False, emphasis=2.0)
    rewards = reward_fn.compute(["cheap", "pricey"], scores=[10.0, 90.0])
    assert math.isclose(rewards[0], 1.0)
    assert math.isclose(rewards[1], 0.0)


def test_constant_scores_return_neutral_feedback():
    """An indistinguishable group yields bounded, equal feedback."""
    reward_fn = RankCalibratedReward(min_reward=0.0, max_reward=1.0)
    rewards = reward_fn.compute(["a", "b", "c"], scores=[42.0, 42.0, 42.0])
    assert rewards == [0.0, 0.0, 0.0]


def test_scalar_score_is_broadcast():
    """A scalar ``scores`` value applies to every completion."""
    reward_fn = RankCalibratedReward()
    rewards = reward_fn.compute(["a", "b"], scores=5.0)
    assert rewards == [0.0, 0.0]


def test_scores_parsed_from_content_when_omitted():
    """Without explicit scores, a numeric token is parsed from each completion."""
    reward_fn = RankCalibratedReward(emphasis=1.0)
    rewards = reward_fn.compute(["final score 10", "final score 90"])
    assert math.isclose(rewards[0], 0.0)
    assert math.isclose(rewards[1], 1.0)


def test_weight_applied_through_base_call():
    """The RewardFunction.__call__ path applies the configured weight."""
    reward_fn = RankCalibratedReward(emphasis=1.0, weight=2.0)
    rewards = reward_fn(["a", "b", "c"], scores=[0.0, 50.0, 100.0])
    for got, expected in zip(rewards, [0.0, 1.0, 2.0]):
        assert math.isclose(got, expected)


def test_empty_completions_returns_empty():
    reward_fn = RankCalibratedReward()
    assert reward_fn.compute([], scores=None) == []


def test_misaligned_scores_raises():
    reward_fn = RankCalibratedReward()
    try:
        reward_fn.compute(["a", "b"], scores=[1.0, 2.0, 3.0])
    except ValueError:
        return
    raise AssertionError("expected ValueError for misaligned scores")
