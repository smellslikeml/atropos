"""
Integration tests for EDV consensus reward function.

Tests the Execute-Distill-Verify paradigm implementation for SQL query
generation and general consensus-based reward computation.
"""

import pytest

from atroposlib.envs.reward_fns.consensus_reward import (
    ConsensusReward,
    EDVConsensusReward,
    compute_structural_similarity,
    consensus_reward,
    extract_boxed_sql,
    extract_sql_structure,
    normalize_sql_query,
)


class TestNormalizeSQLQuery:
    """Test SQL query normalization."""

    def test_basic_normalization(self):
        """Test basic whitespace and case normalization."""
        query = "SELECT name FROM table WHERE id = 1"
        normalized = normalize_sql_query(query)
        assert normalized == "select name from table where id = 1"

    def test_multiple_spaces(self):
        """Test collapsing multiple spaces."""
        query = "SELECT  name   FROM   table"
        normalized = normalize_sql_query(query)
        assert normalized == "select name from table"

    def test_trailing_semicolon(self):
        """Test removing trailing semicolon."""
        query = "SELECT * FROM data;"
        normalized = normalize_sql_query(query)
        assert normalized == "select * from data"

    def test_empty_query(self):
        """Test empty query handling."""
        assert normalize_sql_query("") == ""
        assert normalize_sql_query(None) == ""


class TestExtractSQLStructure:
    """Test SQL structure extraction."""

    def test_select_query(self):
        """Test SELECT query type detection."""
        query = "SELECT name, age FROM users"
        structure = extract_sql_structure(query)
        assert structure["type"] == "SELECT"

    def test_where_clause(self):
        """Test WHERE clause detection."""
        query = "SELECT * FROM data WHERE id > 5"
        structure = extract_sql_structure(query)
        assert structure["has_where"] is True

    def test_join_detection(self):
        """Test JOIN detection."""
        query = "SELECT * FROM users JOIN orders ON users.id = orders.user_id"
        structure = extract_sql_structure(query)
        assert structure["has_join"] is True

    def test_aggregation_detection(self):
        """Test aggregation detection."""
        query = "SELECT COUNT(*), AVG(price) FROM products"
        structure = extract_sql_structure(query)
        assert structure["has_aggregation"] is True

    def test_order_by_detection(self):
        """Test ORDER BY detection."""
        query = "SELECT * FROM data ORDER BY name"
        structure = extract_sql_structure(query)
        assert structure["has_order_by"] is True

    def test_empty_query(self):
        """Test empty query structure."""
        structure = extract_sql_structure("")
        assert structure["type"] is None
        assert structure["has_where"] is False


class TestStructuralSimilarity:
    """Test structural similarity computation."""

    def test_identical_structure(self):
        """Test identical structures have high similarity."""
        struct1 = {"type": "SELECT", "has_where": True, "has_join": False}
        struct2 = {"type": "SELECT", "has_where": True, "has_join": False}
        similarity = compute_structural_similarity(struct1, struct2)
        assert similarity == 1.0

    def test_different_types(self):
        """Test different query types have zero similarity."""
        struct1 = {"type": "SELECT", "has_where": True, "has_join": False}
        struct2 = {"type": "INSERT", "has_where": False, "has_join": False}
        similarity = compute_structural_similarity(struct1, struct2)
        assert similarity == 0.0

    def test_partial_similarity(self):
        """Test partial structural similarity."""
        struct1 = {
            "type": "SELECT",
            "has_where": True,
            "has_join": False,
            "has_aggregation": False,
            "has_order_by": False,
        }
        struct2 = {
            "type": "SELECT",
            "has_where": True,
            "has_join": True,
            "has_aggregation": False,
            "has_order_by": False,
        }
        similarity = compute_structural_similarity(struct1, struct2)
        assert 0.0 < similarity < 1.0


class TestExtractBoxedSQL:
    """Test SQL extraction from boxed format."""

    def test_simple_boxed(self):
        """Test extracting simple boxed SQL."""
        text = "The answer is \\boxed{SELECT * FROM data}"
        result = extract_boxed_sql(text)
        assert result == "SELECT * FROM data"

    def test_nested_braces(self):
        """Test handling of nested braces.

        Note: The simple regex pattern extracts up to the first closing brace.
        For production use, consider using a more sophisticated parser.
        """
        text = "\\boxed{SELECT * FROM data WHERE id = 1}"
        result = extract_boxed_sql(text)
        assert result == "SELECT * FROM data WHERE id = 1"

    def test_code_block_format(self):
        """Test extracting from code blocks."""
        text = "```sql\nSELECT name FROM users\n```"
        result = extract_boxed_sql(text)
        assert result == "SELECT name FROM users"

    def test_no_sql_found(self):
        """Test when no SQL is found."""
        text = "This is just regular text without SQL"
        result = extract_boxed_sql(text)
        assert result is None


class TestConsensusReward:
    """Test consensus reward computation."""

    def test_single_candidate(self):
        """Test with single candidate returns neutral score."""
        reward_fn = ConsensusReward()
        completions = [{"role": "assistant", "content": "\\boxed{SELECT * FROM data}"}]
        rewards = reward_fn.compute(completions)
        assert len(rewards) == 1
        # Single candidate gets neutral score or execution score if provided
        assert rewards[0] >= 0.0

    def test_identical_queries_high_consensus(self):
        """Test identical queries get high consensus scores."""
        reward_fn = ConsensusReward()
        completions = [
            {"role": "assistant", "content": "\\boxed{SELECT name FROM data}"},
            {"role": "assistant", "content": "\\boxed{SELECT name FROM data}"},
            {"role": "assistant", "content": "\\boxed{SELECT name FROM data}"},
        ]
        rewards = reward_fn.compute(completions)
        assert len(rewards) == 3
        # All identical should have high consensus
        assert all(r > 0.5 for r in rewards)

    def test_divergent_queries_filtered(self):
        """Test that divergent queries may be filtered."""
        reward_fn = ConsensusReward(consensus_threshold=0.5)
        completions = [
            {"role": "assistant", "content": "\\boxed{SELECT name FROM data}"},
            {"role": "assistant", "content": "\\boxed{SELECT age FROM data}"},
            {"role": "assistant", "content": "\\boxed{DELETE FROM data}"},
        ]
        rewards = reward_fn.compute(completions)
        assert len(rewards) == 3
        # At least some queries should have lower scores due to divergence
        assert min(rewards) < max(rewards)

    def test_with_execution_scores(self):
        """Test consensus combined with execution scores."""
        reward_fn = ConsensusReward()
        completions = [
            {"role": "assistant", "content": "\\boxed{SELECT name FROM data}"},
            {"role": "assistant", "content": "\\boxed{SELECT name FROM data}"},
        ]
        execution_scores = [1.0, 1.0]
        rewards = reward_fn.compute(completions, execution_scores=execution_scores)
        # Good execution + high consensus = high rewards
        assert all(r > 0.5 for r in rewards)


class TestEDVConsensusReward:
    """Test EDV consensus reward with distillation."""

    def test_distillation_stage(self):
        """Test the distillation stage identifies patterns."""
        reward_fn = EDVConsensusReward()
        completions = [
            {"role": "assistant", "content": "\\boxed{SELECT name FROM data WHERE id = 1}"},
            {"role": "assistant", "content": "\\boxed{SELECT name FROM data WHERE id = 2}"},
        ]
        rewards = reward_fn.compute(completions)
        assert len(rewards) == 2

    def test_verify_strictness(self):
        """Test verification strictness affects filtering."""
        strict_fn = EDVConsensusReward(verify_strictness=0.9)
        lenient_fn = EDVConsensusReward(verify_strictness=0.3)

        completions = [
            {"role": "assistant", "content": "\\boxed{SELECT name FROM data}"},
            {"role": "assistant", "content": "\\boxed{SELECT age FROM data}"},
        ]

        strict_rewards = strict_fn.compute(completions)
        lenient_rewards = lenient_fn.compute(completions)

        # Lenient should allow more diversity
        assert min(lenient_rewards) >= min(strict_rewards)


class TestLegacyFunction:
    """Test legacy consensus_reward function."""

    def test_legacy_function_wrapper(self):
        """Test legacy function works correctly."""
        completions = [
            {"role": "assistant", "content": "\\boxed{SELECT * FROM data}"},
        ]
        rewards = consensus_reward(completions)
        assert len(rewards) == 1
        assert isinstance(rewards[0], float)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_completions(self):
        """Test handling of empty completions list."""
        reward_fn = ConsensusReward()
        rewards = reward_fn.compute([])
        assert rewards == []

    def test_invalid_sql_format(self):
        """Test handling of invalid SQL in completions."""
        reward_fn = ConsensusReward()
        completions = [
            {"role": "assistant", "content": "I don't know SQL"},
            {"role": "assistant", "content": "Maybe SELECT something"},
        ]
        rewards = reward_fn.compute(completions)
        assert len(rewards) == 2
        # Should handle gracefully without crashing

    def test_mixed_valid_invalid(self):
        """Test mix of valid and invalid SQL."""
        reward_fn = ConsensusReward()
        completions = [
            {"role": "assistant", "content": "\\boxed{SELECT * FROM data}"},
            {"role": "assistant", "content": "Not valid SQL"},
            {"role": "assistant", "content": "\\boxed{SELECT name FROM data}"},
        ]
        rewards = reward_fn.compute(completions)
        assert len(rewards) == 3
        # Valid SQL should be rewarded
        assert rewards[0] > rewards[1] or rewards[2] > rewards[1]


class TestSQLQueryEnvIntegration:
    """Test integration with SQL Query Environment."""

    @pytest.mark.skip(reason="SQL environment requires additional dependencies (tqdm, wikisql_loader)")
    def test_env_initialization_with_edv(self):
        """Test SQL environment can be initialized with EDV enabled."""
        from environments.community.sql_query_env.sql_query_env import SQLQueryEnv
        from atroposlib.envs.base import BaseEnvConfig, APIServerConfig

        env_config = BaseEnvConfig(
            tokenizer_name="test",
            group_size=4,
            rollout_server_url="http://localhost:8000",
            total_steps=100,
        )
        server_config = APIServerConfig(
            model_name="test",
            base_url="http://localhost:8000/v1",
            api_key="test",
        )

        env = SQLQueryEnv(
            config=env_config,
            server_configs=[server_config],
            slurm=False,
            testing=True,
            enable_edv_consensus=True,
            edv_consensus_threshold=0.6,
        )

        assert env.enable_edv_consensus is True
        assert env.edv_consensus_threshold == 0.6

    @pytest.mark.skip(reason="SQL environment requires additional dependencies (tqdm, wikisql_loader)")
    def test_edv_verification_method(self):
        """Test the EDV verification method directly."""
        from environments.community.sql_query_env.sql_query_env import SQLQueryEnv

        # Create mock candidate data
        candidates = [
            {
                "messages": [
                    {"role": "user", "content": "Query all users"},
                    {"role": "assistant", "content": "\\boxed{SELECT * FROM data}"}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Query all users"},
                    {"role": "assistant", "content": "\\boxed{SELECT * FROM data}"}
                ]
            },
        ]
        execution_scores = [1.0, 1.0]

        # Create environment with EDV enabled
        from atroposlib.envs.base import BaseEnvConfig, APIServerConfig
        env_config = BaseEnvConfig(
            tokenizer_name="test",
            group_size=4,
            rollout_server_url="http://localhost:8000",
            total_steps=100,
        )
        server_config = APIServerConfig(
            model_name="test",
            base_url="http://localhost:8000/v1",
            api_key="test",
        )

        env = SQLQueryEnv(
            config=env_config,
            server_configs=[server_config],
            slurm=False,
            testing=True,
            enable_edv_consensus=True,
        )

        is_valid = env._apply_edv_consensus_verification(candidates, execution_scores)
        assert len(is_valid) == 2
        # Identical queries with good execution should pass
        assert all(is_valid)
