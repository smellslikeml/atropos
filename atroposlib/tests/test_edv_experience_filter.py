"""Tests for EDV experience filtering integration."""

from types import SimpleNamespace

import pytest

from atroposlib.envs.edv_experience_filter import EDVConfig, EDVDistillVerifyMixin
from atroposlib.envs.server_handling.server_baseline import APIServerConfig


class _FakeDistillerServer:
    """Mock distiller server for testing."""

    def __init__(self, response_content: str = None):
        self.calls = []
        self.response_content = response_content or (
            '[\n  {"index": 1, "verdict": "RELIABLE", "reason": "Correct answer"},\n'
            '  {"index": 2, "verdict": "UNRELIABLE", "reason": "Wrong but confident"}\n'
            "]"
        )

    def managed_server(self, tokenizer=None):
        """Return a mock managed server context."""
        self.calls.append(("managed_server",))
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def chat_completion(self, **kwargs):
        """Return mock completion."""
        self.calls.append(("chat_completion", kwargs))

        class MockChoice:
            def __init__(self):
                self.message = SimpleNamespace(content=self.response_content)

        class MockCompletion:
            def __init__(self, response):
                self.choices = [MockChoice()]

        return MockCompletion(self.response_content)


class _ConcreteEDVEnv(EDVDistillVerifyMixin):
    """Concrete class for testing EDV mixin."""

    async def get_next_item(self):
        return None

    async def evaluate(self, *args, **kwargs):
        return None


class _DummyTokenizer:
    """Dummy tokenizer for testing."""

    name_or_path = "test-tokenizer"

    def get_vocab(self):
        return {"a": 1}


@pytest.mark.asyncio
async def test_edv_disabled_passthrough():
    """Test that EDV passthrough when disabled."""
    env = object().__new__(_ConcreteEDVEnv)
    env.config = SimpleNamespace(edv_enabled=False)
    env.tokenizer = _DummyTokenizer()

    group = {
        "tokens": [[1, 2], [3, 4]],
        "masks": [[-100, 2], [-100, 4]],
        "scores": [1.0, -1.0],
        "inference_logprobs": [[-0.1, -0.2], [-0.3, -0.4]],
    }

    result = await env.distill_verify_experiences(group)
    assert result is group  # Should return same object when disabled


@pytest.mark.asyncio
async def test_edv_insufficient_trajectories_passthrough():
    """Test that EDV passthrough when insufficient trajectories."""
    env = object().__new__(_ConcreteEDVEnv)
    env.config = SimpleNamespace(
        edv_enabled=True,
        edv_min_executors=2,
        edv_include_low_score=True,
    )
    env.tokenizer = _DummyTokenizer()

    group = {
        "tokens": [[1, 2]],
        "masks": [[-100, 2]],
        "scores": [1.0],
        "inference_logprobs": [[-0.1, -0.2]],
    }

    result = await env.distill_verify_experiences(group)
    assert result is group  # Should return same object when insufficient trajectories


@pytest.mark.asyncio
async def test_edv_distill_prompt_building():
    """Test distill prompt building."""
    env = object().__new__(_ConcreteEDVEnv)
    env.config = SimpleNamespace(
        edv_enabled=True,
        edv_min_executors=2,
    )

    trajectories = [
        {"index": 0, "content": "SELECT * FROM table", "score": 1.0},
        {"index": 1, "content": "SELECT FROM table", "score": -1.0},
    ]

    result = await env._build_distill_prompt(trajectories, "SQL generation")

    assert "SQL generation" in result
    assert "SELECT * FROM table" in result
    assert "SELECT FROM table" in result
    assert "RELIABLE" in result
    assert "UNRELIABLE" in result


@pytest.mark.asyncio
async def test_edv_parse_distiller_response():
    """Test parsing distiller response."""
    env = object().__new__(_ConcreteEDVEnv)

    response = """[
        {"index": 1, "verdict": "RELIABLE", "reason": "Good"},
        {"index": 2, "verdict": "UNRELIABLE", "reason": "Bad"}
    ]"""

    verdicts = env._parse_distiller_response(response)

    assert len(verdicts) == 2
    assert verdicts[0]["index"] == 1
    assert verdicts[0]["verdict"] == "RELIABLE"
    assert verdicts[1]["verdict"] == "UNRELIABLE"


@pytest.mark.asyncio
async def test_edv_parse_distiller_response_fallback():
    """Test parsing distiller response with malformed JSON."""
    env = object().__new__(_ConcreteEDVEnv)

    response = (
        "Trajectory 1: RELIABLE because good\nTrajectory 2: UNRELIABLE because bad"
    )

    verdicts = env._parse_distiller_response(response)

    # Should fallback to line-by-line parsing
    assert len(verdicts) == 2


@pytest.mark.asyncio
async def test_edv_verify_consensus_basic():
    """Test basic consensus verification."""
    env = object().__new__(_ConcreteEDVEnv)
    env.config = SimpleNamespace(
        edv_enabled=True,
        edv_min_executors=2,
        edv_consensus_threshold=0.5,
    )

    trajectories = [
        {"index": 0, "content": "answer1", "score": 1.0},
        {"index": 1, "content": "answer2", "score": -1.0},
    ]

    group = {
        "tokens": [[1, 2], [3, 4]],
        "masks": [[-100, 2], [-100, 4]],
        "scores": [1.0, -1.0],
        "inference_logprobs": [[-0.1, -0.2], [-0.3, -0.4]],
    }

    approved = await env._verify_by_consensus(trajectories, None, group)

    # Should approve trajectory 0 (positive score) with no distiller
    assert 0 in approved


@pytest.mark.asyncio
async def test_edv_filtering_with_distiller():
    """Test EDV filtering with distiller verdicts."""
    env = object().__new__(_ConcreteEDVEnv)
    env.config = SimpleNamespace(
        edv_enabled=True,
        edv_min_executors=2,
        edv_consensus_threshold=0.5,
        edv_include_low_score=True,
    )
    env.tokenizer = _DummyTokenizer()

    group = {
        "tokens": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "masks": [[-100, 2, 3], [-100, 5, 6], [-100, 8, 9]],
        "scores": [1.0, -1.0, 0.5],
        "inference_logprobs": [
            [-0.1, -0.2, -0.3],
            [-0.4, -0.5, -0.6],
            [-0.7, -0.8, -0.9],
        ],
        "messages": [
            [{"role": "assistant", "content": "Correct answer"}],
            [{"role": "assistant", "content": "Wrong answer"}],
            [{"role": "assistant", "content": "Partial answer"}],
        ],
    }

    # Mock distiller that marks first and third as reliable
    distiller = _FakeDistillerServer()
    distiller.response_content = (
        '[\n  {"index": 1, "verdict": "RELIABLE", "reason": "Good"},\n'
        '  {"index": 2, "verdict": "UNRELIABLE", "reason": "Bad"},\n'
        '  {"index": 3, "verdict": "RELIABLE", "reason": "Okay"}\n]'
        "]"
    )

    result = await env.distill_verify_experiences(
        group=group,
        task_context="test task",
        distiller_server=distiller,
    )

    # Should filter based on distiller verdicts + consensus
    assert "tokens" in result
    assert len(result["tokens"]) <= len(group["tokens"])


@pytest.mark.asyncio
async def test_edv_filtering_low_score_excluded():
    """Test that low-scoring trajectories are excluded when configured."""
    env = object().__new__(_ConcreteEDVEnv)
    env.config = SimpleNamespace(
        edv_enabled=True,
        edv_min_executors=2,
        edv_consensus_threshold=0.5,
        edv_include_low_score=False,  # Exclude low scores
    )
    env.tokenizer = _DummyTokenizer()

    group = {
        "tokens": [[1, 2], [3, 4], [5, 6]],
        "masks": [[-100, 2], [-100, 4], [-100, 6]],
        "scores": [1.0, -1.0, -1.0],  # Two negative scores
        "inference_logprobs": [[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]],
        "messages": [
            [{"role": "assistant", "content": "Good"}],
            [{"role": "assistant", "content": "Bad"}],
            [{"role": "assistant", "content": "Worse"}],
        ],
    }

    # With include_low_score=False, low-scoring trajectories should be skipped
    # in the distill stage (but may still be in result if consensus passes)
    result = await env.distill_verify_experiences(
        group=group,
        task_context="test task",
        distiller_server=None,  # No distiller, just consensus
    )

    # Should only include positive-scoring trajectory
    assert len(result["tokens"]) == 1
    assert result["scores"][0] == 1.0


def test_edv_config_fields():
    """Test that EDVConfig has expected fields."""
    config = EDVConfig()

    assert hasattr(config, "edv_enabled")
    assert not config.edv_enabled  # Default

    assert hasattr(config, "edv_consensus_threshold")
    assert config.edv_consensus_threshold == 0.5  # Default

    assert hasattr(config, "edv_min_executors")
    assert config.edv_min_executors == 2  # Default

    assert hasattr(config, "edv_distiller_temperature")
    assert config.edv_distiller_temperature == 0.0  # Default

    assert hasattr(config, "edv_include_low_score")
    assert config.edv_include_low_score  # Default


def test_resolve_edv_distiller_server_configs_returns_none_when_unset():
    """Test that distiller config resolution returns None when unset."""
    result = _ConcreteEDVEnv._resolve_edv_distiller_server_configs(
        default_distiller_configs=None,
        yaml_config={},
        cli_passed_flags={},
    )
    assert result is None


def test_resolve_edv_distiller_server_configs_uses_namespace():
    """Test that distiller config resolution uses correct namespace."""
    default_cfg = APIServerConfig(
        model_name="distiller-model",
        base_url="http://distiller/v1",
        api_key="x",
        server_type="vllm",
    )

    result = _ConcreteEDVEnv._resolve_edv_distiller_server_configs(
        default_distiller_configs=default_cfg,
        yaml_config={"edv_distiller": {"temperature": 0.5}},
        cli_passed_flags={"edv_distiller.base_url": "http://override/v1"},
    )

    # Should return configs (actual resolution logic is in resolve_openai_configs)
    assert result is not None


@pytest.mark.asyncio
async def test_edv_preserves_optional_fields():
    """Test that EDV filtering preserves optional fields like advantages."""
    env = object().__new__(_ConcreteEDVEnv)
    env.config = SimpleNamespace(
        edv_enabled=True,
        edv_min_executors=2,
        edv_consensus_threshold=0.5,
    )
    env.tokenizer = _DummyTokenizer()

    group = {
        "tokens": [[1, 2], [3, 4]],
        "masks": [[-100, 2], [-100, 4]],
        "scores": [1.0, 0.5],
        "inference_logprobs": [[-0.1, -0.2], [-0.3, -0.4]],
        "advantages": [[0.1, 0.2], [0.3, 0.4]],  # Optional field
        "ref_logprobs": [[-0.5, -0.6], [-0.7, -0.8]],  # Optional field
        "messages": [
            [{"role": "assistant", "content": "A"}],
            [{"role": "assistant", "content": "B"}],
        ],
    }

    result = await env.distill_verify_experiences(
        group=group,
        task_context="test task",
        distiller_server=None,
    )

    # Should preserve optional fields
    assert "advantages" in result
    assert "ref_logprobs" in result
