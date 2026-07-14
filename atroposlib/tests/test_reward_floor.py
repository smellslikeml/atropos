"""
Tests for the deterministic reward floor (arXiv:2606.27291's S_r).

Exercises the two paper-faithful degeneracy rules (6-gram verbatim overlap and
lifted date-range fragments), the -1.0 hard cap, and integration with the
reward-function registry and package exports.
"""

# Package-level export added by the integration edit in __init__.py.
from atroposlib.envs.reward_fns import RewardFloor as ExportedRewardFloor
from atroposlib.envs.reward_fns.registry import registry
from atroposlib.envs.reward_fns.reward_floor import (
    DegeneracyDetector,
    RewardFloor,
    reward_floor,
)
from atroposlib.envs.reward_fns.reward_function import RewardFunction


class TestSixGramVerbatimOverlap:
    """The paper's first S_r rule: exact 6-gram overlap with the profile."""

    def test_no_overlap(self):
        profile = "Senior software engineer at Acme Corporation in San Francisco"
        completion = "Experienced backend developer seeking new remote opportunities"
        assert (
            DegeneracyDetector.six_gram_verbatim_overlap(completion, profile) is False
        )

    def test_verbatim_six_gram_triggers(self):
        profile = "Senior software engineer at Acme Corporation in San Francisco"
        completion = "candidate: senior software engineer at Acme Corporation"
        assert DegeneracyDetector.six_gram_verbatim_overlap(completion, profile) is True

    def test_partial_overlap_below_six_does_not_trigger(self):
        # Shares a 5-token run ("the quick brown fox jumps"), then diverges.
        profile = "the quick brown fox jumps over the lazy dog"
        completion = "the quick brown fox jumps swiftly across the town"
        assert (
            DegeneracyDetector.six_gram_verbatim_overlap(completion, profile) is False
        )

    def test_completion_shorter_than_ngram(self):
        profile = "the quick brown fox jumps over the lazy dog"
        completion = "quick brown fox"
        assert (
            DegeneracyDetector.six_gram_verbatim_overlap(completion, profile) is False
        )

    def test_empty_inputs(self):
        assert DegeneracyDetector.six_gram_verbatim_overlap("", "profile") is False
        assert DegeneracyDetector.six_gram_verbatim_overlap("completion", "") is False


class TestDateRangeLifted:
    """The paper's second S_r rule: date-range fragment lifted from the profile."""

    def test_year_range_lifted(self):
        profile = "Worked as an analyst from 2019-2021 before moving on."
        completion = "portable query for analyst 2019-2021"
        assert DegeneracyDetector.date_range_lifted(completion, profile) is True

    def test_month_year_range_lifted(self):
        profile = "Employment: Jan 2020 - Mar 2021 at BigCo."
        completion = "role held Jan 2020 - Mar 2021"
        assert DegeneracyDetector.date_range_lifted(completion, profile) is True

    def test_endash_normalized(self):
        # Completion uses an en-dash, profile uses a hyphen: still a lift.
        profile = "tenure 2019-2021"
        completion = "experience 2019–2021"
        assert DegeneracyDetector.date_range_lifted(completion, profile) is True

    def test_open_range_present(self):
        profile = "Current role 2019 - present"
        completion = "candidate 2019 - present"
        assert DegeneracyDetector.date_range_lifted(completion, profile) is True

    def test_date_range_not_in_profile(self):
        profile = "No dates mentioned anywhere in this profile text."
        completion = "some query 2019-2021"
        assert DegeneracyDetector.date_range_lifted(completion, profile) is False

    def test_no_date_range(self):
        profile = "Senior engineer with cloud experience"
        completion = "portable query for senior engineer"
        assert DegeneracyDetector.date_range_lifted(completion, profile) is False


class TestRewardFloor:
    """The RewardFloor reward function and its -1.0 hard cap."""

    def test_is_reward_function_subclass(self):
        assert isinstance(RewardFloor(), RewardFunction)

    def test_defaults_are_paper_faithful(self):
        floor = RewardFloor()
        assert floor.floor_value == -1.0
        assert floor.pass_value == 0.0
        assert floor.ngram_size == 6
        assert set(floor.rules) == {"six_gram_verbatim", "date_range"}

    def test_registry_integration(self):
        floor = registry.create("reward_floor")
        assert isinstance(floor, RewardFunction)
        assert floor.name == "rewardfloor"

    def test_registry_integration_with_params(self):
        floor = registry.create("reward_floor", floor_value=-1.0, ngram_size=4)
        assert isinstance(floor, RewardFunction)
        assert floor.ngram_size == 4

    def test_empty_completions(self):
        assert RewardFloor().compute([]) == []

    def test_six_gram_overlap_floors_to_minus_one(self):
        floor = RewardFloor()
        profile = "Senior software engineer at Acme Corporation in San Francisco"
        completions = [
            {
                "role": "assistant",
                "content": "senior software engineer at Acme Corporation",
            }
        ]
        rewards = floor.compute(completions, profile=profile)
        assert rewards[0] == -1.0

    def test_date_range_floors_to_minus_one(self):
        floor = RewardFloor()
        profile = "Analyst from 2019-2021 at BigCo."
        completions = [{"role": "assistant", "content": "analyst query 2019-2021"}]
        rewards = floor.compute(completions, profile=profile)
        assert rewards[0] == -1.0

    def test_clean_completion_passes(self):
        floor = RewardFloor()
        profile = "Senior software engineer at Acme Corporation in San Francisco"
        completions = [
            {
                "role": "assistant",
                "content": "backend engineer with distributed systems focus",
            }
        ]
        rewards = floor.compute(completions, profile=profile)
        assert rewards[0] == 0.0

    def test_rule_subset_disables_ngram(self):
        # Only the date-range rule is active, so a 6-gram lift is not penalized.
        floor = RewardFloor(rules=["date_range"])
        profile = "Senior software engineer at Acme Corporation in San Francisco"
        completions = [
            {
                "role": "assistant",
                "content": "senior software engineer at Acme Corporation",
            }
        ]
        rewards = floor.compute(completions, profile=profile)
        assert rewards[0] == 0.0

    def test_profile_from_prompt_kwarg(self):
        floor = RewardFloor()
        prompt = "Profile: staff data scientist at Globex working on forecasting"
        completions = [
            {"role": "assistant", "content": "staff data scientist at Globex working"}
        ]
        rewards = floor.compute(completions, prompt=prompt)
        assert rewards[0] == -1.0

    def test_per_completion_profile_list(self):
        floor = RewardFloor()
        profiles = [
            "Analyst from 2019-2021 at BigCo.",
            "Marketing lead with brand strategy background.",
        ]
        completions = [
            {"role": "assistant", "content": "analyst 2019-2021"},
            {"role": "assistant", "content": "senior marketing generalist"},
        ]
        rewards = floor.compute(completions, profile=profiles)
        assert rewards[0] == -1.0
        assert rewards[1] == 0.0

    def test_weight_application(self):
        floor = RewardFloor(weight=2.0)
        profile = "Analyst from 2019-2021 at BigCo."
        completions = [{"role": "assistant", "content": "analyst 2019-2021"}]
        rewards = floor(completions, profile=profile)  # __call__ applies weight
        assert rewards[0] == -2.0

    def test_completion_formats(self):
        floor = RewardFloor()
        profile = "unrelated profile text with no overlap"
        # String, role/content dict, and message-wrapper dict all pass cleanly.
        assert floor.compute(["clean portable query"], profile=profile)[0] == 0.0
        assert (
            floor.compute(
                [{"role": "assistant", "content": "clean portable query"}],
                profile=profile,
            )[0]
            == 0.0
        )
        assert (
            floor.compute(
                [{"message": {"role": "assistant", "content": "clean portable query"}}],
                profile=profile,
            )[0]
            == 0.0
        )

    def test_no_profile_context_passes(self):
        # With no source profile, neither rule can fire.
        floor = RewardFloor()
        completions = [{"role": "assistant", "content": "any query 2019-2021"}]
        assert floor.compute(completions)[0] == 0.0


class TestPackageExport:
    """The integration edit exposes RewardFloor from the package __init__."""

    def test_exported_symbol_is_reward_floor(self):
        assert ExportedRewardFloor is RewardFloor
        assert issubclass(ExportedRewardFloor, RewardFunction)

    def test_import_eagerly_registers(self):
        # Because __init__ imports reward_floor, @registry.register runs at
        # import time and the class is registered (as "rewardfloor", from the
        # class name) without needing lazy file discovery.
        assert "rewardfloor" in registry.list_registered()


class TestLegacyFunction:
    """The functional wrapper mirrors the other reward modules' convention."""

    def test_legacy_function_floors(self):
        profile = "Analyst from 2019-2021 at BigCo."
        completions = [{"role": "assistant", "content": "analyst 2019-2021"}]
        rewards = reward_floor(completions, profile=profile)
        assert rewards[0] == -1.0

    def test_legacy_function_passes_clean(self):
        profile = "unrelated profile text"
        completions = [{"role": "assistant", "content": "clean portable query"}]
        rewards = reward_floor(completions, profile=profile)
        assert rewards[0] == 0.0
