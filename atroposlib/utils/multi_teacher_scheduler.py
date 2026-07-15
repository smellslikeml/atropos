"""
Multi-teacher scheduling for RL training with dynamic teacher selection.

This module implements a simplified version of the multi-teacher scheduling mechanism
described in "Mach-Mind-4-Flash Technical Report" (arXiv:2607.09375v1). The original
paper uses a routed reverse-KL objective for teacher selection; this implementation
adapts the core insight to use performance-based teacher selection with existing
infrastructure.

The scheduler tracks teacher performance over a rolling window and dynamically selects
the best teacher for each training step, enabling the model to benefit from diverse
teacher strengths without complex routed objectives.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from typing_extensions import Literal


@dataclass
class TeacherMetrics:
    """Performance metrics for a single teacher."""

    success_count: int = 0
    total_samples: int = 0
    recent_rewards: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=50))

    @property
    def success_rate(self) -> float:
        """Compute success rate from metrics."""
        if self.total_samples == 0:
            return 0.0
        return self.success_count / self.total_samples

    @property
    def avg_reward(self) -> float:
        """Compute average reward from recent window."""
        if not self.recent_rewards:
            return 0.0
        return sum(self.recent_rewards) / len(self.recent_rewards)

    @property
    def avg_latency(self) -> float:
        """Compute average latency from recent window."""
        if not self.recent_latencies:
            return 0.0
        return sum(self.recent_latencies) / len(self.recent_latencies)


@dataclass
class TeacherConfig:
    """Configuration for a single teacher endpoint."""

    name: str
    endpoint: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    weight: float = 1.0  # Base weight for selection probability


@dataclass
class SchedulerConfig:
    """Configuration for the multi-teacher scheduler."""

    selection_strategy: Literal[
        "performance", "round_robin", "random", "epsilon_greedy"
    ] = "performance"
    epsilon: float = 0.1  # For epsilon_greedy: exploration rate
    performance_window: int = 100  # Window size for reward tracking
    min_samples_before_selection: int = (
        5  # Minimum samples before using performance-based selection
    )
    load_balance_factor: float = 0.2  # Factor to balance load vs performance (0-1)
    fallback_teacher: Optional[str] = None  # Fallback teacher name if all fail


class MultiTeacherScheduler:
    """
    Scheduler for dynamic multi-teacher selection during RL training.

    This scheduler tracks performance metrics for each teacher and selects
    the best teacher for each training step based on the configured strategy.

    Example:
        >>> teachers = [
        ...     TeacherConfig(name="teacher1", endpoint="http://localhost:8001"),
        ...     TeacherConfig(name="teacher2", endpoint="http://localhost:8002"),
        ... ]
        >>> config = SchedulerConfig(selection_strategy="performance")
        >>> scheduler = MultiTeacherScheduler(teachers, config)
        >>> selected = scheduler.select_teacher()
        >>> scheduler.record_result(selected.name, reward=0.8, latency=0.5, success=True)
    """

    def __init__(
        self,
        teachers: List[TeacherConfig],
        config: Optional[SchedulerConfig] = None,
    ):
        """
        Initialize the multi-teacher scheduler.

        Args:
            teachers: List of teacher configurations
            config: Scheduler configuration (uses defaults if None)
        """
        self.teachers = {t.name: t for t in teachers}
        self.config = config or SchedulerConfig()
        self.metrics: Dict[str, TeacherMetrics] = {
            t.name: TeacherMetrics() for t in teachers
        }
        self._round_robin_index = 0
        self._call_counts: Dict[str, int] = {t.name: 0 for t in teachers}

    def select_teacher(self, step: Optional[int] = None) -> TeacherConfig:
        """
        Select the best teacher for the current training step.

        Args:
            step: Current training step (for logging/debugging)

        Returns:
            The selected TeacherConfig

        Raises:
            RuntimeError: If no teachers are configured
        """
        if not self.teachers:
            raise RuntimeError("No teachers configured for selection")

        teacher_names = list(self.teachers.keys())

        # Round-robin strategy
        if self.config.selection_strategy == "round_robin":
            selected_name = teacher_names[self._round_robin_index % len(teacher_names)]
            self._round_robin_index += 1
            return self.teachers[selected_name]

        # Random strategy
        if self.config.selection_strategy == "random":
            return self.teachers[random.choice(teacher_names)]

        # Epsilon-greedy strategy
        if self.config.selection_strategy == "epsilon_greedy":
            if random.random() < self.config.epsilon:
                return self.teachers[random.choice(teacher_names)]
            # Fall through to performance-based selection

        # Performance-based strategy (or epsilon-greedy exploitation)
        return self._select_by_performance(teacher_names)

    def _select_by_performance(self, teacher_names: List[str]) -> TeacherConfig:
        """
        Select teacher based on performance metrics with load balancing.

        Uses a combined score of:
        - Average reward (higher is better)
        - Load balancing (lower call counts get a boost)

        Returns:
            The selected TeacherConfig
        """
        scores = []
        for name in teacher_names:
            metrics = self.metrics[name]
            teacher = self.teachers[name]

            # Use average reward if we have enough samples
            if metrics.total_samples >= self.config.min_samples_before_selection:
                reward_score = metrics.avg_reward
            else:
                # If not enough samples, use base weight
                reward_score = teacher.weight

            # Apply load balancing: boost score for under-utilized teachers
            call_count = self._call_counts[name]
            max_calls = max(self._call_counts.values()) if self._call_counts else 1
            load_factor = 1.0 - (call_count / max(1, max_calls))

            # Combined score: weighted mix of performance and load balance
            combined_score = (
                (1 - self.config.load_balance_factor) * reward_score
                + self.config.load_balance_factor * load_factor
            )
            scores.append((name, combined_score))

        # Select teacher with highest score
        scores.sort(key=lambda x: x[1], reverse=True)
        selected_name = scores[0][0]

        return self.teachers[selected_name]

    def record_result(
        self,
        teacher_name: str,
        reward: float,
        latency: Optional[float] = None,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record performance metrics for a teacher after use.

        Args:
            teacher_name: Name of the teacher that was used
            reward: Reward/score obtained from this teacher
            latency: Optional latency in seconds
            success: Whether the request was successful
            metadata: Optional additional metadata for logging
        """
        if teacher_name not in self.metrics:
            return

        metrics = self.metrics[teacher_name]
        self._call_counts[teacher_name] += 1

        if success:
            metrics.success_count += 1
            metrics.recent_rewards.append(reward)

        metrics.total_samples += 1

        if latency is not None:
            metrics.recent_latencies.append(latency)

    def get_teacher_stats(self, teacher_name: str) -> Dict[str, Any]:
        """
        Get current statistics for a specific teacher.

        Args:
            teacher_name: Name of the teacher

        Returns:
            Dictionary of teacher statistics
        """
        if teacher_name not in self.metrics:
            return {}

        metrics = self.metrics[teacher_name]
        return {
            "success_rate": metrics.success_rate,
            "avg_reward": metrics.avg_reward,
            "avg_latency": metrics.avg_latency,
            "total_samples": metrics.total_samples,
            "call_count": self._call_counts.get(teacher_name, 0),
        }

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all teachers.

        Returns:
            Dictionary mapping teacher names to their statistics
        """
        return {
            name: self.get_teacher_stats(name) for name in self.teachers.keys()
        }

    def reset_metrics(self, teacher_name: Optional[str] = None) -> None:
        """
        Reset metrics for a teacher or all teachers.

        Args:
            teacher_name: Specific teacher to reset, or None for all
        """
        if teacher_name is None:
            for name in self.teachers:
                self.metrics[name] = TeacherMetrics()
                self._call_counts[name] = 0
        elif teacher_name in self.metrics:
            self.metrics[teacher_name] = TeacherMetrics()
            self._call_counts[teacher_name] = 0

    def get_fallback_teacher(self) -> Optional[TeacherConfig]:
        """
        Get the fallback teacher if configured.

        Returns:
            Fallback TeacherConfig or None
        """
        if self.config.fallback_teacher and self.config.fallback_teacher in self.teachers:
            return self.teachers[self.config.fallback_teacher]
        return None


def create_scheduler_from_configs(
    teacher_configs: List[Dict[str, Any]],
    scheduler_config: Optional[Dict[str, Any]] = None,
) -> MultiTeacherScheduler:
    """
    Create a MultiTeacherScheduler from configuration dictionaries.

    Args:
        teacher_configs: List of teacher configuration dicts
        scheduler_config: Optional scheduler configuration dict

    Returns:
        Configured MultiTeacherScheduler instance

    Example:
        >>> teachers = [
        ...     {"name": "gpt4", "endpoint": "...", "weight": 1.0},
        ...     {"name": "claude", "endpoint": "...", "weight": 0.9},
        ... ]
        >>> scheduler = create_scheduler_from_configs(
        ...     teachers,
        ...     {"selection_strategy": "performance", "epsilon": 0.1}
        ... )
    """
    teachers = [
        TeacherConfig(
            name=config["name"],
            endpoint=config["endpoint"],
            api_key=config.get("api_key"),
            model=config.get("model"),
            weight=config.get("weight", 1.0),
        )
        for config in teacher_configs
    ]

    config = None
    if scheduler_config:
        config = SchedulerConfig(
            selection_strategy=scheduler_config.get("selection_strategy", "performance"),
            epsilon=scheduler_config.get("epsilon", 0.1),
            performance_window=scheduler_config.get("performance_window", 100),
            min_samples_before_selection=scheduler_config.get(
                "min_samples_before_selection", 5
            ),
            load_balance_factor=scheduler_config.get("load_balance_factor", 0.2),
            fallback_teacher=scheduler_config.get("fallback_teacher"),
        )

    return MultiTeacherScheduler(teachers, config)
