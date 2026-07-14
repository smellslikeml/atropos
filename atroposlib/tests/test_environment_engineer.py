"""
Tests for the environment engineer module.

These tests verify the LLM-as-Environment-Engineer framework integration,
following the paper: https://arxiv.org/abs/2606.17682v1
"""

from types import SimpleNamespace

import pytest

from atroposlib.envs.base import BaseEnv
from atroposlib.envs.environment_engineer import (
    BaseEnvironmentEngineer,
    ConfigChangeType,
    ConfigProposal,
    EngineerAnalysis,
    EnvironmentConfig,
    EnvironmentEngineerConfig,
    FailureCase,
    HeuristicEnvironmentEngineer,
    create_environment_engineer,
)


class ConcreteTestEnv(BaseEnv):
    """Minimal concrete BaseEnv for testing integration."""

    name = "test_env"

    async def get_next_item(self):
        return None

    async def evaluate(self, *args, **kwargs):
        return None

    async def setup(self):
        pass


@pytest.mark.asyncio
async def test_heuristic_engineer_proposes_changes_on_accuracy_failures():
    """Test that the heuristic engineer proposes difficulty reduction."""
    config = EnvironmentEngineerConfig(
        enabled=True,
        confidence_threshold=0.5,
        max_proposals_per_run=5,
    )
    engineer = HeuristicEnvironmentEngineer(config)

    failure_cases = [
        FailureCase(
            item_id="1",
            input_text="test input",
            model_output="wrong output",
            expected_output="correct output",
            score=0.1,
            failure_reason=(
                "low accuracy: output does not match expected"
            ),
        ),
        FailureCase(
            item_id="2",
            input_text="another test",
            model_output="also wrong",
            expected_output="correct again",
            score=0.2,
            failure_reason="accuracy failure: incorrect response",
        ),
    ]

    current_configs = {
        "difficulty_level": EnvironmentConfig(
            config_name="difficulty_level",
            current_value=0.8,
            min_value=0.0,
            max_value=1.0,
            description="Task difficulty level",
            change_type=ConfigChangeType.INCREASE_DIFFICULTY,
        ),
    }

    eval_metrics = {"overall_score": 0.3}

    analysis = await engineer.analyze(
        failure_cases, current_configs, eval_metrics
    )

    assert len(analysis.proposals) > 0
    assert any(p.config_name == "difficulty_level" for p in analysis.proposals)
    assert analysis.summary is not None
    assert len(analysis.failure_evidence) > 0


@pytest.mark.asyncio
async def test_heuristic_engineer_preserves_working_configs():
    """Test that the engineer preserves configs when performance is good."""
    config = EnvironmentEngineerConfig(
        enabled=True,
        confidence_threshold=0.5,
        preserve_working_configs=True,
    )
    engineer = HeuristicEnvironmentEngineer(config)

    failure_cases = [
        FailureCase(
            item_id="1",
            input_text="test",
            model_output="output",
            expected_output="output",
            score=0.95,
            failure_reason="minor formatting issue",
        ),
    ]

    current_configs = {
        "difficulty_level": EnvironmentConfig(
            config_name="difficulty_level",
            current_value=0.5,
            min_value=0.0,
            max_value=1.0,
            description="Task difficulty",
            change_type=ConfigChangeType.INCREASE_DIFFICULTY,
        ),
    }

    eval_metrics = {"overall_score": 0.85}

    analysis = await engineer.analyze(
        failure_cases, current_configs, eval_metrics
    )

    # Should preserve the config since performance is good
    assert "difficulty_level" in analysis.preserved_configs
    assert len(analysis.proposals) == 0


@pytest.mark.asyncio
async def test_heuristic_engineer_respects_confidence_threshold():
    """Test that low-confidence proposals are filtered out."""
    config = EnvironmentEngineerConfig(
        enabled=True,
        confidence_threshold=0.8,  # High threshold
    )
    engineer = HeuristicEnvironmentEngineer(config)

    failure_cases = [
        FailureCase(
            item_id="1",
            input_text="test",
            model_output="output",
            expected_output="expected",
            score=0.4,
            failure_reason="unclear failure",
        ),
    ]

    current_configs = {
        "param_x": EnvironmentConfig(
            config_name="param_x",
            current_value=0.5,
            min_value=0.0,
            max_value=1.0,
            description="Test parameter",
            change_type=ConfigChangeType.ADAPTER_PARAMETER,
        ),
    }

    eval_metrics = {"overall_score": 0.5}

    analysis = await engineer.analyze(
        failure_cases, current_configs, eval_metrics
    )

    # With high confidence threshold, expect no proposals
    assert len(analysis.proposals) == 0 or all(
        p.confidence >= 0.8 for p in analysis.proposals
    )


@pytest.mark.asyncio
async def test_heuristic_engineer_limits_max_proposals():
    """Test that the engineer respects max_proposals_per_run."""
    config = EnvironmentEngineerConfig(
        enabled=True,
        max_proposals_per_run=2,
    )
    engineer = HeuristicEnvironmentEngineer(config)

    # Create many failure cases and configs
    failure_cases = [
        FailureCase(
            item_id=str(i),
            input_text=f"test {i}",
            model_output=f"output {i}",
            expected_output=f"expected {i}",
            score=0.1,
            failure_reason="low accuracy",
        )
        for i in range(10)
    ]

    current_configs = {
        f"param_{i}": EnvironmentConfig(
            config_name=f"param_{i}",
            current_value=0.5,
            min_value=0.0,
            max_value=1.0,
            description=f"Parameter {i}",
            change_type=ConfigChangeType.ADAPTER_PARAMETER,
        )
        for i in range(5)
    }

    eval_metrics = {"overall_score": 0.2}

    analysis = await engineer.analyze(
        failure_cases, current_configs, eval_metrics
    )

    # Should not exceed max_proposals_per_run
    assert len(analysis.proposals) <= 2


@pytest.mark.asyncio
async def test_heuristic_engineer_proposes_context_window_on_incomplete():
    """Test that incomplete outputs trigger context window proposals."""
    config = EnvironmentEngineerConfig(enabled=True)
    engineer = HeuristicEnvironmentEngineer(config)

    failure_cases = [
        FailureCase(
            item_id="1",
            input_text="long input that requires long output",
            model_output="short",
            expected_output="expected long output",
            score=0.3,
            failure_reason="incomplete output: response was truncated",
        ),
        FailureCase(
            item_id="2",
            input_text="another long input",
            model_output="another short",
            expected_output="another long expected",
            score=0.4,
            failure_reason="output truncated: incomplete response",
        ),
    ]

    current_configs = {
        "context_window": EnvironmentConfig(
            config_name="context_window",
            current_value=1024,
            min_value=512,
            max_value=4096,
            description="Context window size",
            change_type=ConfigChangeType.CONTEXT_WINDOW,
        ),
    }

    eval_metrics = {"overall_score": 0.4}

    analysis = await engineer.analyze(
        failure_cases, current_configs, eval_metrics
    )

    # Should propose increasing context window
    proposals = [
        p for p in analysis.proposals
        if p.config_name == "context_window"
    ]
    assert len(proposals) > 0
    assert proposals[0].new_value > 1024  # Should increase


@pytest.mark.asyncio
async def test_heuristic_engineer_extracts_failure_evidence():
    """Test that failure evidence is correctly extracted and formatted."""
    config = EnvironmentEngineerConfig(enabled=True)
    engineer = HeuristicEnvironmentEngineer(config)

    failure_cases = [
        FailureCase(
            item_id="1",
            input_text="a" * 200,  # Long input
            model_output="b" * 200,  # Long output
            expected_output="c" * 200,  # Long expected
            score=0.1,
            failure_reason="complete failure",
            metadata={"extra": "info"},
        ),
    ]

    evidence = engineer._extract_failure_evidence(failure_cases)

    assert len(evidence) > 0
    assert "complete failure" in evidence[0]
    # Should truncate long values
    assert "..." in evidence[0]


def test_create_environment_engineer_factory():
    """Test the factory function for creating engineers."""
    config = EnvironmentEngineerConfig(enabled=True)

    # Create heuristic engineer
    engineer = create_environment_engineer(config, use_llm=False)
    assert isinstance(engineer, HeuristicEnvironmentEngineer)


def test_create_environment_engineer_disabled_raises():
    """Test that creating engineer when disabled raises."""
    config = EnvironmentEngineerConfig(enabled=False)

    with pytest.raises(ValueError, match="not enabled"):
        create_environment_engineer(config, use_llm=False)


def test_create_environment_engineer_llm_requires_server():
    """Test that LLM engineer requires server."""
    config = EnvironmentEngineerConfig(enabled=True)

    with pytest.raises(ValueError, match="Server must be provided"):
        create_environment_engineer(config, use_llm=True, server=None)


@pytest.mark.asyncio
async def test_base_env_integration_hook():
    """Test that BaseEnv.run_environment_engineer works correctly."""
    env = object.__new__(ConcreteTestEnv)
    env.config = SimpleNamespace(
        environment_engineer={
            "enabled": True,
            "confidence_threshold": 0.5,
        }
    )

    # Create failure cases as dicts (to test normalization)
    failure_cases = [
        {
            "item_id": "1",
            "input": "test input",
            "output": "wrong output",
            "expected": "correct",
            "score": 0.2,
            "reason": "accuracy failure",
        }
    ]

    # Create configs as dicts (to test normalization)
    current_configs = {
        "difficulty": {
            "current_value": 0.8,
            "min_value": 0.0,
            "max_value": 1.0,
            "description": "Difficulty",
            "change_type": "increase_difficulty",
        }
    }

    eval_metrics = {"overall_score": 0.3}

    analysis = await env.run_environment_engineer(
        failure_cases, current_configs, eval_metrics
    )

    assert analysis is not None
    assert isinstance(analysis, EngineerAnalysis)
    assert analysis.summary is not None


@pytest.mark.asyncio
async def test_base_env_integration_hook_disabled_config_returns_none():
    """Test that disabled engineer config returns None."""
    env = object.__new__(ConcreteTestEnv)
    env.config = SimpleNamespace(
        environment_engineer={
            "enabled": False,
        }
    )

    failure_cases = []
    current_configs = {}
    eval_metrics = {}

    analysis = await env.run_environment_engineer(
        failure_cases, current_configs, eval_metrics
    )

    assert analysis is None


@pytest.mark.asyncio
async def test_base_env_integration_hook_no_config_returns_none():
    """Test that missing engineer config returns None."""
    env = object.__new__(ConcreteTestEnv)
    env.config = SimpleNamespace()  # No environment_engineer field

    failure_cases = []
    current_configs = {}
    eval_metrics = {}

    analysis = await env.run_environment_engineer(
        failure_cases, current_configs, eval_metrics
    )

    assert analysis is None


@pytest.mark.asyncio
async def test_base_env_integration_hook_empty_failures_returns_none():
    """Test that empty failure cases returns None with log."""
    env = object.__new__(ConcreteTestEnv)
    env.config = SimpleNamespace(
        environment_engineer={
            "enabled": True,
        }
    )

    failure_cases = []  # Empty
    current_configs = {}
    eval_metrics = {}

    analysis = await env.run_environment_engineer(
        failure_cases, current_configs, eval_metrics
    )

    # Should return None when no failures to analyze
    assert analysis is None
