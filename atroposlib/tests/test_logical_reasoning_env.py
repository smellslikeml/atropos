"""Tests for LogicalReasoningEnv verifiable reasoning tasks."""

import pytest

from atroposlib.envs.logical_reasoning_env import (
    TaskType,
    LogicalTask,
    ArrowMazeGenerator,
    SequenceCompletionGenerator,
    PatternRecognitionGenerator,
    LogicGridGenerator,
)


class TestArrowMazeGenerator:
    """Tests for ArrowMazeGenerator."""

    def test_generate_creates_valid_task(self):
        """Test that generate creates a valid LogicalTask."""
        generator = ArrowMazeGenerator(seed=42)
        task = generator.generate(difficulty=0.5)

        assert isinstance(task, LogicalTask)
        assert task.task_type == TaskType.ARROW_MAZE
        assert isinstance(task.question, str)
        assert len(task.question) > 0
        assert task.answer in ["YES", "NO"]
        assert 0.0 <= task.difficulty <= 1.0

    def test_generate_difficulty_affects_grid_size(self):
        """Test that difficulty parameter affects grid size."""
        generator = ArrowMazeGenerator(seed=42)

        low_diff_task = generator.generate(difficulty=0.1)
        high_diff_task = generator.generate(difficulty=0.9)

        # Higher difficulty should result in larger grid
        low_grid_size = low_diff_task.metadata["grid_size"]
        high_grid_size = high_diff_task.metadata["grid_size"]

        assert high_grid_size >= low_grid_size

    def test_verify_correct_yes_answer(self):
        """Test verification of correct YES answer."""
        generator = ArrowMazeGenerator(seed=42)
        task = generator.generate(difficulty=0.5)

        # If answer is YES, verify returns 1.0 for YES
        if task.answer == "YES":
            reward = generator.verify("YES", task.answer)
            assert reward == 1.0

    def test_verify_correct_no_answer(self):
        """Test verification of correct NO answer."""
        generator = ArrowMazeGenerator(seed=42)
        task = generator.generate(difficulty=0.5)

        # If answer is NO, verify returns 1.0 for NO
        if task.answer == "NO":
            reward = generator.verify("NO", task.answer)
            assert reward == 1.0

    def test_verify_incorrect_answer(self):
        """Test verification of incorrect answer."""
        generator = ArrowMazeGenerator(seed=42)
        task = generator.generate(difficulty=0.5)

        # Verify returns 0.0 for wrong answer
        if task.answer == "YES":
            reward = generator.verify("NO", task.answer)
        else:
            reward = generator.verify("YES", task.answer)
        assert reward == 0.0

    def test_verify_case_insensitive(self):
        """Test that verification is case-insensitive."""
        generator = ArrowMazeGenerator(seed=42)
        task = generator.generate(difficulty=0.5)

        reward = generator.verify("yes", task.answer)
        if task.answer == "YES":
            assert reward == 1.0

    def test_maze_solution_correctness(self):
        """Test that the maze solver produces correct solutions."""
        generator = ArrowMazeGenerator(seed=42)

        # Create a simple test maze
        grid = [["→", "↓"], ["↑", "←"]]
        # Start at (0,0), end at (1,1)
        # Path: (0,0) -> → to (0,1) -> ↓ to (1,1) = YES
        result = generator._solve_maze(grid, 0, 0, 1, 1, 2)
        assert result == "YES"


class TestSequenceCompletionGenerator:
    """Tests for SequenceCompletionGenerator."""

    def test_generate_creates_valid_task(self):
        """Test that generate creates a valid LogicalTask."""
        generator = SequenceCompletionGenerator(seed=42)
        task = generator.generate(difficulty=0.5)

        assert isinstance(task, LogicalTask)
        assert task.task_type == TaskType.SEQUENCE_COMPLETION
        assert isinstance(task.question, str)
        assert len(task.question) > 0
        # Answer should be a valid integer (may be negative)
        try:
            int(task.answer)
        except ValueError:
            pytest.fail(f"Answer '{task.answer}' is not a valid integer")

    def test_verify_correct_numeric_answer(self):
        """Test verification of correct numeric answer."""
        generator = SequenceCompletionGenerator(seed=42)
        task = generator.generate(difficulty=0.5)

        reward = generator.verify(task.answer, task.answer)
        assert reward == 1.0

    def test_verify_incorrect_answer(self):
        """Test verification of incorrect answer."""
        generator = SequenceCompletionGenerator(seed=42)
        task = generator.generate(difficulty=0.5)

        wrong_answer = str(int(task.answer) + 1)
        reward = generator.verify(wrong_answer, task.answer)
        assert reward == 0.0

    def test_verify_non_numeric_answer(self):
        """Test verification of non-numeric answer."""
        generator = SequenceCompletionGenerator(seed=42)
        task = generator.generate(difficulty=0.5)

        reward = generator.verify("not a number", task.answer)
        assert reward == 0.0

    def test_arithmetic_sequence_generation(self):
        """Test arithmetic sequence pattern generation."""
        generator = SequenceCompletionGenerator(seed=42)

        # Generate multiple tasks and check they're valid
        for _ in range(10):
            task = generator.generate(difficulty=0.3)
            assert task.task_type == TaskType.SEQUENCE_COMPLETION
            assert "," in task.question  # Should have comma-separated numbers


class TestPatternRecognitionGenerator:
    """Tests for PatternRecognitionGenerator."""

    def test_generate_creates_valid_task(self):
        """Test that generate creates a valid LogicalTask."""
        generator = PatternRecognitionGenerator(seed=42)
        task = generator.generate(difficulty=0.5)

        assert isinstance(task, LogicalTask)
        assert task.task_type == TaskType.PATTERN_RECOGNITION
        assert isinstance(task.question, str)
        assert len(task.question) > 0

    def test_verify_exact_match(self):
        """Test verification requires exact match."""
        generator = PatternRecognitionGenerator(seed=42)
        task = generator.generate(difficulty=0.5)

        reward = generator.verify(task.answer, task.answer)
        assert reward == 1.0

    def test_verify_whitespace_tolerant(self):
        """Test that verification is whitespace-tolerant (strips whitespace)."""
        generator = PatternRecognitionGenerator(seed=42)
        task = generator.generate(difficulty=0.5)

        # Add whitespace - should still match due to strip()
        reward = generator.verify(f" {task.answer} ", task.answer)
        assert reward == 1.0

        # But different content should still fail
        wrong_answer = f"X{task.answer}"
        reward = generator.verify(wrong_answer, task.answer)
        assert reward == 0.0

    def test_pattern_type_in_metadata(self):
        """Test that pattern type is stored in metadata."""
        generator = PatternRecognitionGenerator(seed=42)
        task = generator.generate(difficulty=0.5)

        assert "pattern_type" in task.metadata
        assert task.metadata["pattern_type"] in ["repeating", "rotation", "alternating", "counting"]


class TestLogicGridGenerator:
    """Tests for LogicGridGenerator."""

    def test_generate_creates_valid_task(self):
        """Test that generate creates a valid LogicalTask."""
        generator = LogicGridGenerator(seed=42)
        task = generator.generate(difficulty=0.5)

        assert isinstance(task, LogicalTask)
        assert task.task_type == TaskType.LOGIC_GRID
        assert isinstance(task.question, str)
        assert len(task.question) > 0
        assert task.answer in ["red", "blue", "green", "yellow"]

    def test_verify_correct_color(self):
        """Test verification of correct color answer."""
        generator = LogicGridGenerator(seed=42)
        task = generator.generate(difficulty=0.5)

        reward = generator.verify(task.answer, task.answer)
        assert reward == 1.0

    def test_verify_incorrect_color(self):
        """Test verification of incorrect color answer."""
        generator = LogicGridGenerator(seed=42)
        task = generator.generate(difficulty=0.5)

        colors = ["red", "blue", "green", "yellow"]
        wrong_answer = next(c for c in colors if c != task.answer)
        reward = generator.verify(wrong_answer, task.answer)
        assert reward == 0.0

    def test_verify_case_insensitive(self):
        """Test that verification is case-insensitive for colors."""
        generator = LogicGridGenerator(seed=42)
        task = generator.generate(difficulty=0.5)

        reward = generator.verify(task.answer.upper(), task.answer)
        assert reward == 1.0

    def test_correct_order_in_metadata(self):
        """Test that correct order is stored in metadata."""
        generator = LogicGridGenerator(seed=42)
        task = generator.generate(difficulty=0.5)

        assert "correct_order" in task.metadata
        assert isinstance(task.metadata["correct_order"], list)


class TestTaskType:
    """Tests for TaskType enum."""

    def test_all_task_types_defined(self):
        """Test that all expected task types are defined."""
        expected_types = ["arrow_maze", "sequence_completion", "pattern_recognition", "logic_grid"]
        actual_types = [t.value for t in TaskType]

        for expected in expected_types:
            assert expected in actual_types

    def test_task_type_from_string(self):
        """Test creating TaskType from string value."""
        task_type = TaskType("arrow_maze")
        assert task_type == TaskType.ARROW_MAZE

    def test_invalid_task_type_raises(self):
        """Test that invalid task type raises ValueError."""
        with pytest.raises(ValueError):
            TaskType("invalid_task_type")


class TestLogicalTask:
    """Tests for LogicalTask dataclass."""

    def test_create_logical_task(self):
        """Test creating a LogicalTask."""
        task = LogicalTask(
            task_type=TaskType.ARROW_MAZE,
            question="What is 2+2?",
            answer="4",
            difficulty=0.5,
            metadata={"test": "value"},
        )

        assert task.task_type == TaskType.ARROW_MAZE
        assert task.question == "What is 2+2?"
        assert task.answer == "4"
        assert task.difficulty == 0.5
        assert task.metadata == {"test": "value"}

    def test_logical_task_metadata_default(self):
        """Test LogicalTask with default metadata."""
        task = LogicalTask(
            task_type=TaskType.SEQUENCE_COMPLETION,
            question="Test",
            answer="1",
            difficulty=0.0,
            metadata={},
        )

        assert task.metadata == {}
