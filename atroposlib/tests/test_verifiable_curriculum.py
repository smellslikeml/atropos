"""
Tests for verifiable curriculum and multi-teacher RLVR distillation.

These tests verify the compete-then-collaborate framework integration.
"""

import pytest

from atroposlib.envs.verifiable_curriculum import (
    CompetitionMode,
    CurriculumMode,
    CollaborativeSample,
    ExecutionJudge,
    TeacherRanking,
    TeacherSolution,
    VerifiableCurriculum,
    create_compete_collaborate_hook,
)


class DummyExecutionJudge(ExecutionJudge):
    """A simple execution judge for testing."""

    def __init__(self, verified_solutions=None):
        """
        Initialize the dummy judge.

        Args:
            verified_solutions: Set of solutions that should be verified.
        """
        self.verified_solutions = verified_solutions or set()
        self.verify_calls = []

    async def verify_solution(self, solution, item):
        """Verify a solution by checking if it's in the verified set."""
        self.verify_calls.append((solution, item))
        is_verified = solution in self.verified_solutions
        return is_verified, None, None


class DummyServerManager:
    """A dummy server manager for testing."""

    def __init__(self, name, solutions=None):
        self.name = name
        self.solutions = solutions or []
        self.get_logprobs_calls = []

    async def get_logprobs(self, **kwargs):
        """Return a dummy logprobs response."""
        self.get_logprobs_calls.append(kwargs)
        solution = self.solutions.pop(0) if self.solutions else f"solution_{self.name}"
        return {
            "generated_text": solution,
            "prompt_tokens": [1, 2, 3],
        }


@pytest.mark.asyncio
async def test_execution_judge_verify_solution():
    """Test ExecutionJudge.verify_solution interface."""
    judge = DummyExecutionJudge(verified_solutions={"correct_solution"})

    verified, output, error = await judge.verify_solution("correct_solution", {})
    assert verified is True
    assert output is None
    assert error is None

    verified, output, error = await judge.verify_solution("wrong_solution", {})
    assert verified is False


@pytest.mark.asyncio
async def test_verifiable_curriculum_initialization():
    """Test VerifiableCurriculum initialization with multiple teachers."""
    teacher1 = DummyServerManager("teacher1")
    teacher2 = DummyServerManager("teacher2")
    judge = DummyExecutionJudge()

    curriculum = VerifiableCurriculum(
        teacher_servers=[teacher1, teacher2],
        judge=judge,
        competition_mode=CompetitionMode.EXECUTION_BASED,
        curriculum_mode=CurriculumMode.COLLABORATIVE_POOL,
    )

    assert len(curriculum.teacher_servers) == 2
    assert len(curriculum.teacher_rankings) == 2
    assert "teacher_0" in curriculum.teacher_rankings
    assert "teacher_1" in curriculum.teacher_rankings


@pytest.mark.asyncio
async def test_verifiable_curriculum_teacher_rankings_update():
    """Test teacher rankings are updated correctly after competition."""
    teacher1 = DummyServerManager("teacher1")
    teacher2 = DummyServerManager("teacher2")
    judge = DummyExecutionJudge(verified_solutions={"solution1"})

    curriculum = VerifiableCurriculum(
        teacher_servers=[teacher1, teacher2],
        judge=judge,
        competition_mode=CompetitionMode.EXECUTION_BASED,
    )

    # Simulate competition results
    solutions = {
        "teacher_0": TeacherSolution("teacher_0", "solution1", True),
        "teacher_1": TeacherSolution("teacher_1", "solution2", False),
    }

    curriculum.update_rankings(solutions)

    # Check rankings
    ranking0 = curriculum.teacher_rankings["teacher_0"]
    ranking1 = curriculum.teacher_rankings["teacher_1"]

    assert ranking0.num_verified == 1
    assert ranking0.num_attempted == 1
    assert ranking0.score == 1.0

    assert ranking1.num_verified == 0
    assert ranking1.num_attempted == 1
    assert ranking1.score == 0.0


@pytest.mark.asyncio
async def test_verifiable_curriculum_build_collaborative_sample():
    """Test collaborative sample building from teacher solutions."""
    teacher1 = DummyServerManager("teacher1")
    teacher2 = DummyServerManager("teacher2")
    judge = DummyExecutionJudge()

    curriculum = VerifiableCurriculum(
        teacher_servers=[teacher1, teacher2],
        judge=judge,
        curriculum_mode=CurriculumMode.COLLABORATIVE_POOL,
    )

    solutions = {
        "teacher_0": TeacherSolution("teacher_0", "solution1", True),
        "teacher_1": TeacherSolution("teacher_1", "solution2", True),
    }

    sample = curriculum.build_collaborative_sample(solutions, {"item": "test"})

    assert isinstance(sample, CollaborativeSample)
    assert sample.num_verified == 2
    assert sample.has_collaboration is True
    assert len(sample.solutions) == 2


@pytest.mark.asyncio
async def test_verifiable_curriculum_best_teacher_only_mode():
    """Test curriculum mode that uses only the best teacher's solutions."""
    teacher1 = DummyServerManager("teacher1")
    teacher2 = DummyServerManager("teacher2")
    judge = DummyExecutionJudge()

    curriculum = VerifiableCurriculum(
        teacher_servers=[teacher1, teacher2],
        judge=judge,
        curriculum_mode=CurriculumMode.BEST_TEACHER_ONLY,
    )

    # Set up rankings so teacher_0 is best
    curriculum.teacher_rankings["teacher_0"].rank = 0
    curriculum.teacher_rankings["teacher_1"].rank = 1

    solutions = {
        "teacher_0": TeacherSolution("teacher_0", "solution1", True),
        "teacher_1": TeacherSolution("teacher_1", "solution2", True),
    }

    sample = curriculum.build_collaborative_sample(solutions, {"item": "test"})

    # Should only have teacher_0's solution
    assert len(sample.solutions) == 1
    assert sample.solutions[0].teacher_name == "teacher_0"


@pytest.mark.asyncio
async def test_verifiable_curriculum_verified_only_mode():
    """Test curriculum mode that uses only verified solutions."""
    teacher1 = DummyServerManager("teacher1")
    teacher2 = DummyServerManager("teacher2")
    judge = DummyExecutionJudge()

    curriculum = VerifiableCurriculum(
        teacher_servers=[teacher1, teacher2],
        judge=judge,
        curriculum_mode=CurriculumMode.VERIFIED_ONLY,
    )

    solutions = {
        "teacher_0": TeacherSolution("teacher_0", "solution1", True),
        "teacher_1": TeacherSolution("teacher_1", "solution2", False),
    }

    sample = curriculum.build_collaborative_sample(solutions, {"item": "test"})

    # Should only have verified solution
    assert len(sample.solutions) == 1
    assert sample.solutions[0].verified is True


def test_verifiable_curriculum_compute_rlvr_reward():
    """Test RLVR reward computation for student solutions."""
    teacher1 = DummyServerManager("teacher1")
    teacher2 = DummyServerManager("teacher2")
    judge = DummyExecutionJudge()

    curriculum = VerifiableCurriculum(
        teacher_servers=[teacher1, teacher2],
        judge=judge,
    )

    # Create a collaborative sample with verified solutions
    solutions = [
        TeacherSolution("teacher_0", "correct_solution", True),
        TeacherSolution("teacher_1", "another_solution", True),
    ]
    sample = CollaborativeSample(
        solutions=solutions,
        item={"item": "test"},
        num_verified=2,
        has_collaboration=True,
    )

    # Student solution matching a verified one should get positive reward
    reward = curriculum.compute_rlvr_reward("correct_solution", sample)
    assert reward == 1.0

    # Non-matching solution should get negative reward
    reward = curriculum.compute_rlvr_reward("wrong_solution", sample)
    assert reward == -1.0


def test_verifiable_curriculum_get_stats():
    """Test getting curriculum statistics."""
    teacher1 = DummyServerManager("teacher1")
    teacher2 = DummyServerManager("teacher2")
    judge = DummyExecutionJudge()

    curriculum = VerifiableCurriculum(
        teacher_servers=[teacher1, teacher2],
        judge=judge,
    )

    # Set up some stats
    curriculum.teacher_rankings["teacher_0"].num_verified = 8
    curriculum.teacher_rankings["teacher_0"].num_attempted = 10
    curriculum.teacher_rankings["teacher_1"].num_verified = 6
    curriculum.teacher_rankings["teacher_1"].num_attempted = 10

    stats = curriculum.get_curriculum_stats()

    assert stats["total_teacher_attempts"] == 20
    assert stats["total_verified_solutions"] == 14
    assert stats["overall_verification_rate"] == 0.7
    assert "teacher_rankings" in stats


@pytest.mark.asyncio
async def test_create_compete_collaborate_hook():
    """Test creating the RLVR reward hook."""
    teacher1 = DummyServerManager("teacher1")
    teacher2 = DummyServerManager("teacher2")
    judge = DummyExecutionJudge(verified_solutions={"correct"})

    curriculum = VerifiableCurriculum(
        teacher_servers=[teacher1, teacher2],
        judge=judge,
    )
    curriculum.reward_scale = 2.0

    hook = create_compete_collaborate_hook(curriculum)

    # The hook should be a callable
    assert callable(hook)


def test_teacher_ranking_dataclass():
    """Test TeacherRanking dataclass."""
    ranking = TeacherRanking(
        teacher_name="test_teacher",
        rank=1,
        score=0.85,
        num_verified=85,
        num_attempted=100,
    )

    assert ranking.teacher_name == "test_teacher"
    assert ranking.rank == 1
    assert ranking.score == 0.85
    assert ranking.num_verified == 85
    assert ranking.num_attempted == 100


def test_teacher_solution_dataclass():
    """Test TeacherSolution dataclass."""
    solution = TeacherSolution(
        teacher_name="test_teacher",
        content="SELECT * FROM table",
        verified=True,
        execution_output=[(1, "row1")],
        error_message=None,
    )

    assert solution.teacher_name == "test_teacher"
    assert solution.content == "SELECT * FROM table"
    assert solution.verified is True
    assert solution.execution_output == [(1, "row1")]
    assert solution.error_message is None
