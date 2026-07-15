"""
Logical Reasoning Environment for Atropos.

This module implements a logical reasoning environment inspired by the SynLogic paper
(arXiv:2505.19641). It provides verifiable logical reasoning tasks with rule-based
scoring, suitable for reinforcement learning with verifiable rewards.

The environment supports multiple logical reasoning task types with configurable
difficulty levels. All tasks are verified through simple rules, providing binary
or scalar rewards for RL training.

Core mechanism from SynLogic paper:
- Verifiable logical reasoning tasks with rule-based verification
- Controllable difficulty synthesis
- Diverse task types (grid puzzles, sequence reasoning, pattern recognition)
- Binary/reward-based scoring suitable for RL training

Adapted for Atropos BaseEnv architecture:
- Uses BaseEnv's collect_trajectories and ScoredDataGroup interface
- Integrates with Atropos's server management and tokenization
- Supports wandb logging and evaluation hooks
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from pydantic import Field

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    Item,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


class TaskType(Enum):
    """Types of logical reasoning tasks supported."""

    ARROW_MAZE = "arrow_maze"
    SEQUENCE_COMPLETION = "sequence_completion"
    PATTERN_RECOGNITION = "pattern_recognition"
    LOGIC_GRID = "logic_grid"


@dataclass
class LogicalTask:
    """A single logical reasoning task."""

    task_type: TaskType
    question: str
    answer: str
    difficulty: float  # 0.0 to 1.0
    metadata: Dict[str, Any]


class LogicalReasoningConfig(BaseEnvConfig):
    """Configuration for LogicalReasoningEnv."""

    # Task configuration
    task_types: List[str] = Field(
        default_factory=lambda: [t.value for t in TaskType],
        description="List of task types to use for training.",
    )

    difficulty_min: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum difficulty level for task generation.",
    )

    difficulty_max: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Maximum difficulty level for task generation.",
    )

    dataset_size: int = Field(
        default=1000,
        ge=1,
        description="Number of tasks to generate for the training dataset.",
    )

    eval_dataset_size: int = Field(
        default=200,
        ge=1,
        description="Number of tasks to generate for the evaluation dataset.",
    )

    seed: int = Field(
        default=42,
        description="Random seed for reproducible task generation.",
    )

    # Generation parameters
    max_tokens: int = Field(
        default=1024,
        ge=1,
        description="Maximum tokens for model responses.",
    )

    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation.",
    )


class BaseTaskGenerator(ABC):
    """Base class for task generators."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    @abstractmethod
    def generate(self, difficulty: float) -> LogicalTask:
        """Generate a task with specified difficulty."""
        pass

    @abstractmethod
    def verify(self, answer: str, expected: str) -> float:
        """Verify an answer and return a reward score."""
        pass


class ArrowMazeGenerator(BaseTaskGenerator):
    """
    Arrow Maze task generator.

    Tasks involve navigating a grid using arrows to reach a target.
    Difficulty controls grid size and number of turns required.
    """

    def generate(self, difficulty: float) -> LogicalTask:
        # Grid size based on difficulty (3x3 to 8x8)
        grid_size = int(3 + difficulty * 5)
        grid_size = min(max(grid_size, 3), 8)

        # Generate maze
        arrows = ["↑", "↓", "←", "→"]
        grid = []
        for _ in range(grid_size):
            row = [self.rng.choice(arrows) for _ in range(grid_size)]
            grid.append(row)

        # Pick start and end positions
        start_row = self.rng.randint(0, grid_size - 1)
        start_col = self.rng.randint(0, grid_size - 1)
        end_row = self.rng.randint(0, grid_size - 1)
        end_col = self.rng.randint(0, grid_size - 1)

        # Ensure start != end
        while start_row == end_row and start_col == end_col:
            end_row = self.rng.randint(0, grid_size - 1)
            end_col = self.rng.randint(0, grid_size - 1)

        # Format question
        grid_str = "\n".join([" ".join(row) for row in grid])
        question = f"""Navigate the arrow maze from START to END.

Rules:
- You start at the START position and follow the arrow at your current position.
- The arrow points to the next position you move to.
- Continue until you reach the END position or exit the grid.

Grid:
{grid_str}

START: Row {start_row + 1}, Column {start_col + 1}
END: Row {end_row + 1}, Column {end_col + 1}

Will you reach the END position? Answer "YES" if you will reach END, "NO" if you will not."""

        # Solve to get correct answer
        answer = self._solve_maze(grid, start_row, start_col, end_row, end_col, grid_size)

        return LogicalTask(
            task_type=TaskType.ARROW_MAZE,
            question=question,
            answer=answer,
            difficulty=difficulty,
            metadata={"grid_size": grid_size, "start": (start_row, start_col), "end": (end_row, end_col)},
        )

    def _solve_maze(self, grid, start_row, start_col, end_row, end_col, grid_size):
        """Simulate the maze navigation."""
        current_row, current_col = start_row, start_col
        visited = set()
        max_steps = grid_size * grid_size * 2  # Prevent infinite loops

        for _ in range(max_steps):
            if (current_row, current_col) in visited:
                return "NO"  # Loop detected
            visited.add((current_row, current_col))

            if current_row == end_row and current_col == end_col:
                return "YES"

            arrow = grid[current_row][current_col]
            if arrow == "↑":
                current_row -= 1
            elif arrow == "↓":
                current_row += 1
            elif arrow == "←":
                current_col -= 1
            elif arrow == "→":
                current_col += 1

            # Check if out of bounds
            if current_row < 0 or current_row >= grid_size or current_col < 0 or current_col >= grid_size:
                return "NO"

        return "NO"  # Exceeded max steps without reaching end

    def verify(self, answer: str, expected: str) -> float:
        """Verify arrow maze answer."""
        response = answer.strip().upper()
        if "YES" in response and expected == "YES":
            return 1.0
        elif "NO" in response and expected == "NO":
            return 1.0
        else:
            return 0.0


class SequenceCompletionGenerator(BaseTaskGenerator):
    """
    Sequence completion task generator.

    Tasks involve identifying the pattern in a sequence and predicting the next element.
    Difficulty controls pattern complexity and sequence length.
    """

    PATTERNS = {
        "arithmetic": lambda start, step, n: [start + i * step for i in range(n)],
        "geometric": lambda start, ratio, n: [int(start * (ratio ** i)) for i in range(n)],
        "fibonacci": lambda a, b, n: SequenceCompletionGenerator._fib_nth_static(a, b, n),
        "alternating": lambda a, b, n: [a if i % 2 == 0 else b for i in range(n)],
    }

    @staticmethod
    def _fib_nth_static(a, b, n):
        """Get nth element of Fibonacci-like sequence starting with a, b."""
        if n == 0:
            return a
        if n == 1:
            return b
        for _ in range(n - 1):
            a, b = b, a + b
        return b

    def generate(self, difficulty: float) -> LogicalTask:
        # Select pattern type based on difficulty
        pattern_types = list(self.PATTERNS.keys())
        num_patterns = int(1 + difficulty * (len(pattern_types) - 1))
        selected_pattern = self.rng.choice(pattern_types[:num_patterns])

        # Generate sequence parameters
        seq_length = int(4 + difficulty * 4)  # 4 to 8 elements shown

        if selected_pattern == "arithmetic":
            start = self.rng.randint(1, 20)
            step = self.rng.randint(1, 10) * (1 if self.rng.random() > 0.5 else -1)
            sequence = [start + i * step for i in range(seq_length + 1)]
        elif selected_pattern == "geometric":
            start = self.rng.randint(1, 5)
            ratio = self.rng.randint(2, 3)
            sequence = [int(start * (ratio ** i)) for i in range(seq_length + 1)]
        elif selected_pattern == "fibonacci":
            a = self.rng.randint(1, 10)
            b = self.rng.randint(1, 10)
            sequence = []
            x, y = a, b
            for i in range(seq_length + 1):
                sequence.append(x)
                x, y = y, x + y
        else:  # alternating
            a = self.rng.randint(1, 20)
            b = self.rng.randint(1, 20)
            sequence = [a if i % 2 == 0 else b for i in range(seq_length + 1)]

        # Format question
        shown_sequence = sequence[:-1]
        next_element = sequence[-1]

        question = f"""Identify the pattern in the following sequence and predict the next number.

Sequence: {", ".join(map(str, shown_sequence))}

What is the next number in the sequence? Provide only the number as your answer."""

        return LogicalTask(
            task_type=TaskType.SEQUENCE_COMPLETION,
            question=question,
            answer=str(next_element),
            difficulty=difficulty,
            metadata={"pattern_type": selected_pattern, "sequence": sequence},
        )

    def verify(self, answer: str, expected: str) -> float:
        """Verify sequence completion answer."""
        try:
            response_num = int(answer.strip())
            expected_num = int(expected)
            return 1.0 if response_num == expected_num else 0.0
        except ValueError:
            return 0.0


class PatternRecognitionGenerator(BaseTaskGenerator):
    """
    Pattern recognition task generator.

    Tasks involve identifying logical patterns in arrangements of symbols.
    Difficulty controls pattern complexity and number of rules.
    """

    SYMBOLS = ["●", "○", "■", "□", "▲", "△", "◆", "◇", "★", "☆"]

    def generate(self, difficulty: float) -> LogicalTask:
        # Pattern complexity based on difficulty
        num_symbols = int(2 + difficulty * 4)
        num_symbols = min(max(num_symbols, 2), len(self.SYMBOLS))

        selected_symbols = self.rng.sample(self.SYMBOLS, num_symbols)

        # Generate pattern rules
        pattern_type = self.rng.choice(["repeating", "rotation", "alternating", "counting"])

        if pattern_type == "repeating":
            pattern = selected_symbols[:3] * 3
            question = f"""Identify the next symbol in the repeating pattern.

Pattern: {" ".join(pattern[:9])}

What is the next symbol in the pattern? Choose from: {", ".join(selected_symbols[:3])}"""
            answer = pattern[9]

        elif pattern_type == "rotation":
            pattern = [selected_symbols[i % len(selected_symbols)] for i in range(8)]
            question = f"""The symbols cycle in a fixed order.

Sequence: {" ".join(pattern)}

What is the next symbol?"""
            answer = selected_symbols[8 % len(selected_symbols)]

        elif pattern_type == "alternating":
            pattern = []
            for i in range(8):
                if i % 2 == 0:
                    pattern.append(selected_symbols[0])
                else:
                    pattern.append(selected_symbols[1])
            question = f"""The pattern alternates between two symbols.

Sequence: {" ".join(pattern)}

What is the next symbol?"""
            answer = selected_symbols[0] if len(pattern) % 2 == 0 else selected_symbols[1]

        else:  # counting
            count = difficulty * 5 + 1
            pattern = [selected_symbols[0]] * int(count) + [selected_symbols[1]]
            question = f"""Count how many times a symbol appears before it changes.

Pattern: {" ".join(pattern)}

How many {selected_symbols[0]} appear before the {selected_symbols[1]}?"""
            answer = str(int(count))

        return LogicalTask(
            task_type=TaskType.PATTERN_RECOGNITION,
            question=question,
            answer=answer,
            difficulty=difficulty,
            metadata={"pattern_type": pattern_type},
        )

    def verify(self, answer: str, expected: str) -> float:
        """Verify pattern recognition answer."""
        return 1.0 if answer.strip() == expected.strip() else 0.0


class LogicGridGenerator(BaseTaskGenerator):
    """
    Logic grid puzzle generator.

    Tasks involve solving simple logic grid puzzles with constraints.
    Difficulty controls grid size and number of constraints.
    """

    def generate(self, difficulty: float) -> LogicalTask:
        # Grid complexity based on difficulty
        num_items = int(2 + difficulty * 2)
        num_items = min(max(num_items, 2), 4)

        # Generate simple logic puzzle
        colors = ["red", "blue", "green", "yellow"][:num_items]
        positions = list(range(1, num_items + 1))

        # Assign a hidden ordering
        self.rng.shuffle(positions)
        correct_order = list(zip(colors, positions))

        # Create clues
        clues = []
        for i, (color, pos) in enumerate(correct_order):
            if i == 0:
                clues.append(f"The {color} item is in position {pos}.")
            else:
                clues.append(f"The {color} item is NOT in position {pos}.")

        self.rng.shuffle(clues)
        clues_text = "\n".join(f"{i+1}. {clue}" for i, clue in enumerate(clues[:3]))

        question = f"""Solve this logic puzzle:

{clues_text}

Based on these clues, which color is in position 1?"""

        # Find color in position 1
        answer = next(color for color, pos in correct_order if pos == 1)

        return LogicalTask(
            task_type=TaskType.LOGIC_GRID,
            question=question,
            answer=answer,
            difficulty=difficulty,
            metadata={"correct_order": correct_order},
        )

    def verify(self, answer: str, expected: str) -> float:
        """Verify logic grid answer."""
        response = answer.strip().lower()
        return 1.0 if expected.lower() in response else 0.0


class LogicalReasoningEnv(BaseEnv):
    """
    Logical Reasoning Environment for verifiable reasoning tasks.

    This environment implements logical reasoning tasks inspired by the SynLogic paper.
    It provides rule-based verification for all tasks, making it suitable for
    reinforcement learning with verifiable rewards.

    Core features:
    - Multiple task types (arrow maze, sequence completion, pattern recognition, logic grid)
    - Configurable difficulty levels
    - Rule-based verification with binary rewards
    - Compatible with Atropos BaseEnv architecture
    """

    name = "logical_reasoning"
    env_config_cls = LogicalReasoningConfig

    # Task generator registry
    GENERATORS = {
        TaskType.ARROW_MAZE: ArrowMazeGenerator,
        TaskType.SEQUENCE_COMPLETION: SequenceCompletionGenerator,
        TaskType.PATTERN_RECOGNITION: PatternRecognitionGenerator,
        TaskType.LOGIC_GRID: LogicGridGenerator,
    }

    def __init__(
        self,
        config: LogicalReasoningConfig,
        server_configs: Union[APIServerConfig, List[APIServerConfig]],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config: LogicalReasoningConfig = config

        # Initialize task generators with seed for reproducibility
        self.generators: Dict[TaskType, BaseTaskGenerator] = {}
        for task_type_str in self.config.task_types:
            try:
                task_type = TaskType(task_type_str)
                generator_cls = self.GENERATORS.get(task_type)
                if generator_cls:
                    self.generators[task_type] = generator_cls(seed=self.config.seed)
            except ValueError:
                continue

        # Metrics tracking
        self.percent_correct_buffer = []
        self.eval_metrics = []
        self.total_attempts = 0
        self.successful_tasks = 0

        # Datasets
        self.train_tasks: List[LogicalTask] = []
        self.eval_tasks: List[LogicalTask] = []
        self.current_task_idx = 0

    @classmethod
    def config_init(cls) -> Tuple[LogicalReasoningConfig, List[APIServerConfig]]:
        """Initialize default configuration."""
        env_config = LogicalReasoningConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=8,
            use_wandb=True,
            max_num_workers_per_node=8,
            rollout_server_url="http://localhost:8000",
            total_steps=2000,
            batch_size=1024,
            steps_per_eval=25,
            inference_weight=1.0,
            wandb_name="logical_reasoning",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
            min_batch_allocation=0.1,
            # Task configuration
            task_types=[t.value for t in TaskType],
            difficulty_min=0.0,
            difficulty_max=0.7,
            dataset_size=1000,
            eval_dataset_size=200,
            seed=42,
            # Generation parameters
            max_tokens=1024,
            temperature=1.0,
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
            ),
        ]
        return env_config, server_configs

    async def setup(self) -> None:
        """Set up the environment by generating task datasets."""
        print("\nSetting up Logical Reasoning Environment...")

        # Generate training dataset
        print(f"Generating {self.config.dataset_size} training tasks...")
        self.train_tasks = await self._generate_tasks(self.config.dataset_size)
        print(f"Generated {len(self.train_tasks)} training tasks")

        # Generate evaluation dataset
        print(f"Generating {self.config.eval_dataset_size} evaluation tasks...")
        self.eval_tasks = await self._generate_tasks(self.config.eval_dataset_size)
        print(f"Generated {len(self.eval_tasks)} evaluation tasks")

        # Display task distribution
        task_dist = {}
        for task in self.train_tasks:
            task_type = task.task_type.value
            task_dist[task_type] = task_dist.get(task_type, 0) + 1

        print("\nTraining task distribution:")
        for task_type, count in sorted(task_dist.items()):
            print(f"  {task_type}: {count}")

        # Difficulty distribution
        difficulties = [task.difficulty for task in self.train_tasks]
        if difficulties:
            avg_diff = sum(difficulties) / len(difficulties)
            min_diff = min(difficulties)
            max_diff = max(difficulties)
            print(f"\nDifficulty range: {min_diff:.2f} - {max_diff:.2f} (avg: {avg_diff:.2f})")

        self.current_task_idx = 0
        print("\nLogical Reasoning Environment setup complete!")

    async def _generate_tasks(self, count: int) -> List[LogicalTask]:
        """Generate a dataset of logical reasoning tasks."""
        tasks = []
        task_types = list(self.generators.keys())

        if not task_types:
            raise ValueError("No valid task types configured")

        for i in range(count):
            # Sample task type
            task_type = self.rng.choice(task_types)
            generator = self.generators[task_type]

            # Sample difficulty
            difficulty = self.config.difficulty_min + self.rng.random() * (
                self.config.difficulty_max - self.config.difficulty_min
            )

            # Generate task
            task = generator.generate(difficulty)
            tasks.append(task)

        return tasks

    async def get_next_item(self) -> Item:
        """Get the next training task."""
        if not self.train_tasks:
            raise ValueError("No training tasks available. Ensure setup() was called.")

        task = self.train_tasks[self.current_task_idx % len(self.train_tasks)]
        self.current_task_idx += 1

        # Create prompt
        system_prompt = "You are a logical reasoning expert. Solve the given problem step by step and provide a clear, concise answer."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.question},
        ]

        # Return as item tuple: (messages, expected_answer, task_metadata)
        return (tuple([frozenset(m.items()) for m in messages]), task.answer, task.metadata)

    def _convert_messages_to_list(self, prompt_tuple: Tuple) -> List[Dict]:
        """Convert frozenset message format to list format."""
        messages = []
        for role_dict in prompt_tuple:
            messages.append(dict(role_dict))
        return messages

    async def collect_trajectories(self, item: Item) -> Tuple[Optional[ScoredDataGroup], List]:
        """Collect and score model trajectories for logical reasoning tasks."""
        messages = self._convert_messages_to_list(item[0])
        expected_answer = item[1]
        task_metadata = item[2]

        completion_params = {
            "n": self.config.group_size,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        try:
            completions = await self.server.chat_completion(messages=messages, **completion_params)
        except Exception as e:
            print(f"Error in collect_trajectories: {e}")
            return None, []

        if not completions.choices:
            return None, []

        # Score the completions
        scored_data = ScoredDataGroup()
        scored_data["tokens"] = []
        scored_data["masks"] = []
        scored_data["scores"] = []

        # Get task type from metadata for proper verification
        task_type_str = task_metadata.get("task_type", TaskType.ARROW_MAZE.value)
        try:
            task_type = TaskType(task_type_str)
            generator = self.generators.get(task_type)
        except ValueError:
            generator = None

        for completion_choice in completions.choices:
            model_answer = completion_choice.message.content

            # Verify answer
            if generator:
                reward = generator.verify(model_answer, expected_answer)
            else:
                # Fallback to string matching
                reward = 1.0 if model_answer.strip() == expected_answer.strip() else 0.0

            # Track metrics
            self.total_attempts += 1
            if reward == 1.0:
                self.successful_tasks += 1

            # Build full conversation for tokenization
            full_messages = messages + [
                {"role": "assistant", "content": model_answer}
            ]

            # Tokenize
            out_dict = tokenize_for_trainer(self.tokenizer, full_messages)
            scored_data["tokens"].append(out_dict["tokens"])
            scored_data["masks"].append(out_dict["masks"])
            scored_data["scores"].append(reward)

        if not scored_data["tokens"]:
            return None, []

        # Update percent correct buffer
        for score in scored_data["scores"]:
            self.percent_correct_buffer.append(max(score, 0))

        # Return None if all scores are the same (no learning signal)
        if len(set(scored_data["scores"])) == 1:
            return None, []

        return scored_data, []

    async def evaluate(self, *args, **kwargs) -> None:
        """Evaluate the model on the evaluation dataset."""
        start_time = time.time()

        if not self.eval_tasks:
            print("Warning: No evaluation tasks available")
            return

        print(f"Evaluating on {len(self.eval_tasks)} tasks...")

        # Run evaluation tasks
        eval_results = []
        for task in self.eval_tasks:
            result = await self._rollout_and_score_eval(task)
            if result is not None:
                eval_results.append(result)

        if not eval_results:
            print("Warning: No valid evaluation results obtained")
            return

        # Calculate metrics
        scores = [r["score"] for r in eval_results]
        valid_scores = [s for s in scores if s is not None]

        if not valid_scores:
            print("Warning: No valid scores found during evaluation")
            return

        percent_correct = sum(valid_scores) / len(valid_scores)
        self.eval_metrics.append(("eval/percent_correct", percent_correct))

        # Additional metrics
        task_type_accuracy = {}
        for result in eval_results:
            task_type = result.get("task_type", "unknown")
            if task_type not in task_type_accuracy:
                task_type_accuracy[task_type] = {"correct": 0, "total": 0}
            task_type_accuracy[task_type]["total"] += 1
            if result.get("score", 0) == 1.0:
                task_type_accuracy[task_type]["correct"] += 1

        # Log per-task-type accuracy
        for task_type, stats in task_type_accuracy.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
                self.eval_metrics.append((f"eval/accuracy_{task_type}", accuracy))

        end_time = time.time()

        # Log evaluation results
        eval_metrics_dict = {
            "eval/percent_correct": percent_correct,
            "eval/total_samples": len(eval_results),
            "eval/correct_samples": sum(valid_scores),
        }

        # Add per-task-type metrics
        for task_type, stats in task_type_accuracy.items():
            if stats["total"] > 0:
                eval_metrics_dict[f"eval/accuracy_{task_type}"] = (
                    stats["correct"] / stats["total"]
                )

        try:
            await self.evaluate_log(
                metrics=eval_metrics_dict,
                samples=[r.get("sample") for r in eval_results],
                start_time=start_time,
                end_time=end_time,
                generation_parameters={
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
            )
        except Exception as e:
            print(f"Error logging evaluation results: {e}")

        print(f"Evaluation complete: {percent_correct:.2%} accuracy")

    async def _rollout_and_score_eval(self, task: LogicalTask) -> Optional[Dict]:
        """Rollout and score a single evaluation task."""
        try:
            system_prompt = "You are a logical reasoning expert. Solve the given problem step by step and provide a clear, concise answer."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task.question},
            ]

            completion_params = {
                "n": 1,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "split": "eval",
            }

            completion = await self.server.chat_completion(messages=messages, **completion_params)

            if not completion.choices:
                return {"score": 0.0, "task_type": task.task_type.value, "sample": None}

            model_answer = completion.choices[0].message.content

            # Verify answer
            generator = self.generators.get(task.task_type)
            if generator:
                score = generator.verify(model_answer, task.answer)
            else:
                score = 1.0 if model_answer.strip() == task.answer.strip() else 0.0

            sample = {
                "messages": messages + [{"role": "assistant", "content": model_answer}],
                "question": task.question,
                "expected_answer": task.answer,
                "model_answer": model_answer,
                "score": int(score),
                "correct": bool(score),
                "task_type": task.task_type.value,
                "difficulty": task.difficulty,
            }

            return {"score": score, "task_type": task.task_type.value, "sample": sample}

        except Exception as e:
            print(f"Error in evaluation rollout: {e}")
            return None

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Basic accuracy metrics
        if self.percent_correct_buffer:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)

        # Task-specific metrics
        if self.total_attempts > 0:
            wandb_metrics["train/success_rate"] = (
                self.successful_tasks / self.total_attempts
            )

        # Configuration metrics
        wandb_metrics.update({
            "config/dataset_size": self.config.dataset_size,
            "config/eval_dataset_size": self.config.eval_dataset_size,
            "config/difficulty_min": self.config.difficulty_min,
            "config/difficulty_max": self.config.difficulty_max,
            "config/num_task_types": len(self.generators),
        })

        # Add evaluation metrics
        for metric_name, metric_value in self.eval_metrics:
            wandb_metrics[metric_name] = metric_value
        self.eval_metrics = []

        # Reset training metrics
        self.percent_correct_buffer = []
        self.total_attempts = 0
        self.successful_tasks = 0

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    LogicalReasoningEnv.cli()
