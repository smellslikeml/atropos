"""
Tests for multi-teacher scheduler utility.

This module tests the MultiTeacherScheduler which implements dynamic
multi-teacher selection for RL training, adapted from the mechanisms
described in "Mach-Mind-4-Flash Technical Report" (arXiv:2607.09375v1).
"""

import pytest

from atroposlib.utils.multi_teacher_scheduler import (
    MultiTeacherScheduler,
    TeacherConfig,
    SchedulerConfig,
    TeacherMetrics,
    create_scheduler_from_configs,
)


class TestTeacherConfig:
    """Tests for TeacherConfig dataclass."""

    def test_teacher_config_creation(self):
        """Test creating a teacher configuration."""
        config = TeacherConfig(
            name="teacher1",
            endpoint="http://localhost:8001",
            api_key="test_key",
            model="gpt-4",
            weight=1.5,
        )
        assert config.name == "teacher1"
        assert config.endpoint == "http://localhost:8001"
        assert config.api_key == "test_key"
        assert config.model == "gpt-4"
        assert config.weight == 1.5

    def test_teacher_config_defaults(self):
        """Test teacher configuration defaults."""
        config = TeacherConfig(name="teacher1", endpoint="http://localhost:8001")
        assert config.api_key is None
        assert config.model is None
        assert config.weight == 1.0


class TestTeacherMetrics:
    """Tests for TeacherMetrics dataclass."""

    def test_metrics_initialization(self):
        """Test metrics start at zero."""
        metrics = TeacherMetrics()
        assert metrics.success_count == 0
        assert metrics.total_samples == 0
        assert metrics.success_rate == 0.0
        assert len(metrics.recent_rewards) == 0
        assert len(metrics.recent_latencies) == 0

    def test_success_rate_calculation(self):
        """Test success rate is calculated correctly."""
        metrics = TeacherMetrics()
        metrics.success_count = 8
        metrics.total_samples = 10
        assert metrics.success_rate == 0.8

    def test_avg_reward_calculation(self):
        """Test average reward is calculated correctly."""
        metrics = TeacherMetrics()
        for reward in [1.0, 2.0, 3.0, 4.0, 5.0]:
            metrics.recent_rewards.append(reward)
        assert metrics.avg_reward == 3.0

    def test_avg_reward_empty(self):
        """Test average reward is 0 when no rewards."""
        metrics = TeacherMetrics()
        assert metrics.avg_reward == 0.0

    def test_avg_latency_calculation(self):
        """Test average latency is calculated correctly."""
        metrics = TeacherMetrics()
        for latency in [0.1, 0.2, 0.3]:
            metrics.recent_latencies.append(latency)
        assert metrics.avg_latency == pytest.approx(0.2)

    def test_recent_rewards_window(self):
        """Test recent rewards window respects maxlen."""
        metrics = TeacherMetrics()
        # Default maxlen is 100
        for i in range(150):
            metrics.recent_rewards.append(float(i))
        assert len(metrics.recent_rewards) == 100

    def test_recent_latencies_window(self):
        """Test recent latencies window respects maxlen."""
        metrics = TeacherMetrics()
        # Default maxlen is 50
        for i in range(100):
            metrics.recent_latencies.append(float(i))
        assert len(metrics.recent_latencies) == 50


class TestSchedulerConfig:
    """Tests for SchedulerConfig dataclass."""

    def test_config_defaults(self):
        """Test scheduler configuration defaults."""
        config = SchedulerConfig()
        assert config.selection_strategy == "performance"
        assert config.epsilon == 0.1
        assert config.performance_window == 100
        assert config.min_samples_before_selection == 5
        assert config.load_balance_factor == 0.2
        assert config.fallback_teacher is None

    def test_config_custom(self):
        """Test custom scheduler configuration."""
        config = SchedulerConfig(
            selection_strategy="epsilon_greedy",
            epsilon=0.2,
            performance_window=50,
            min_samples_before_selection=10,
            load_balance_factor=0.5,
            fallback_teacher="teacher0",
        )
        assert config.selection_strategy == "epsilon_greedy"
        assert config.epsilon == 0.2
        assert config.performance_window == 50
        assert config.min_samples_before_selection == 10
        assert config.load_balance_factor == 0.5
        assert config.fallback_teacher == "teacher0"


class TestMultiTeacherScheduler:
    """Tests for MultiTeacherScheduler class."""

    def test_scheduler_initialization(self):
        """Test scheduler initialization with teachers."""
        teachers = [
            TeacherConfig(name="teacher0", endpoint="http://localhost:8000"),
            TeacherConfig(name="teacher1", endpoint="http://localhost:8001"),
        ]
        scheduler = MultiTeacherScheduler(teachers)
        assert len(scheduler.teachers) == 2
        assert "teacher0" in scheduler.teachers
        assert "teacher1" in scheduler.teachers
        assert len(scheduler.metrics) == 2

    def test_scheduler_with_custom_config(self):
        """Test scheduler with custom configuration."""
        teachers = [
            TeacherConfig(name="teacher0", endpoint="http://localhost:8000"),
        ]
        config = SchedulerConfig(selection_strategy="random")
        scheduler = MultiTeacherScheduler(teachers, config)
        assert scheduler.config.selection_strategy == "random"

    def test_select_teacher_round_robin(self):
        """Test round-robin selection cycles through teachers."""
        teachers = [
            TeacherConfig(name="teacher0", endpoint="http://localhost:8000"),
            TeacherConfig(name="teacher1", endpoint="http://localhost:8001"),
            TeacherConfig(name="teacher2", endpoint="http://localhost:8002"),
        ]
        config = SchedulerConfig(selection_strategy="round_robin")
        scheduler = MultiTeacherScheduler(teachers, config)

        selected = [scheduler.select_teacher().name for _ in range(6)]
        assert selected == ["teacher0", "teacher1", "teacher2", "teacher0", "teacher1", "teacher2"]

    def test_select_teacher_random(self):
        """Test random selection."""
        teachers = [
            TeacherConfig(name="teacher0", endpoint="http://localhost:8000"),
            TeacherConfig(name="teacher1", endpoint="http://localhost:8001"),
        ]
        config = SchedulerConfig(selection_strategy="random", epsilon=0.0)
        scheduler = MultiTeacherScheduler(teachers, config)

        selected = [scheduler.select_teacher().name for _ in range(100)]
        # Both teachers should be selected
        assert "teacher0" in selected
        assert "teacher1" in selected

    def test_select_teacher_no_teachers_raises(self):
        """Test selecting from empty teacher list raises error."""
        scheduler = MultiTeacherScheduler([])
        with pytest.raises(RuntimeError, match="No teachers configured"):
            scheduler.select_teacher()

    def test_record_result_success(self):
        """Test recording successful result."""
        teachers = [TeacherConfig(name="teacher0", endpoint="http://localhost:8000")]
        scheduler = MultiTeacherScheduler(teachers)

        scheduler.record_result("teacher0", reward=1.0, latency=0.5, success=True)

        metrics = scheduler.metrics["teacher0"]
        assert metrics.success_count == 1
        assert metrics.total_samples == 1
        assert metrics.success_rate == 1.0
        assert metrics.avg_reward == 1.0
        assert metrics.avg_latency == 0.5

    def test_record_result_failure(self):
        """Test recording failed result."""
        teachers = [TeacherConfig(name="teacher0", endpoint="http://localhost:8000")]
        scheduler = MultiTeacherScheduler(teachers)

        scheduler.record_result("teacher0", reward=0.0, latency=0.1, success=False)

        metrics = scheduler.metrics["teacher0"]
        assert metrics.success_count == 0
        assert metrics.total_samples == 1
        assert metrics.success_rate == 0.0
        assert metrics.avg_reward == 0.0  # No reward recorded for failure

    def test_record_result_invalid_teacher(self):
        """Test recording result for invalid teacher does nothing."""
        teachers = [TeacherConfig(name="teacher0", endpoint="http://localhost:8000")]
        scheduler = MultiTeacherScheduler(teachers)

        # Should not raise, just silently ignore
        scheduler.record_result("invalid_teacher", reward=1.0, success=True)

        metrics = scheduler.metrics["teacher0"]
        assert metrics.total_samples == 0

    def test_get_teacher_stats(self):
        """Test getting statistics for a specific teacher."""
        teachers = [TeacherConfig(name="teacher0", endpoint="http://localhost:8000")]
        scheduler = MultiTeacherScheduler(teachers)

        scheduler.record_result("teacher0", reward=1.0, latency=0.5, success=True)
        scheduler.record_result("teacher0", reward=0.5, latency=0.3, success=True)
        scheduler.record_result("teacher0", reward=0.0, latency=0.2, success=False)

        stats = scheduler.get_teacher_stats("teacher0")
        assert stats["success_rate"] == 2/3
        assert stats["avg_reward"] == pytest.approx(0.75)  # (1.0 + 0.5) / 2
        # Latency is recorded for all calls: (0.5 + 0.3 + 0.2) / 3
        assert stats["avg_latency"] == pytest.approx(1.0 / 3)
        assert stats["total_samples"] == 3
        assert stats["call_count"] == 3

    def test_get_teacher_stats_invalid_teacher(self):
        """Test getting stats for invalid teacher returns empty dict."""
        teachers = [TeacherConfig(name="teacher0", endpoint="http://localhost:8000")]
        scheduler = MultiTeacherScheduler(teachers)

        stats = scheduler.get_teacher_stats("invalid_teacher")
        assert stats == {}

    def test_get_all_stats(self):
        """Test getting statistics for all teachers."""
        teachers = [
            TeacherConfig(name="teacher0", endpoint="http://localhost:8000"),
            TeacherConfig(name="teacher1", endpoint="http://localhost:8001"),
        ]
        scheduler = MultiTeacherScheduler(teachers)

        scheduler.record_result("teacher0", reward=1.0, success=True)
        scheduler.record_result("teacher1", reward=0.5, success=True)

        all_stats = scheduler.get_all_stats()
        assert "teacher0" in all_stats
        assert "teacher1" in all_stats
        assert all_stats["teacher0"]["total_samples"] == 1
        assert all_stats["teacher1"]["total_samples"] == 1

    def test_reset_metrics_single_teacher(self):
        """Test resetting metrics for a single teacher."""
        teachers = [TeacherConfig(name="teacher0", endpoint="http://localhost:8000")]
        scheduler = MultiTeacherScheduler(teachers)

        scheduler.record_result("teacher0", reward=1.0, success=True)
        assert scheduler.metrics["teacher0"].total_samples == 1

        scheduler.reset_metrics("teacher0")
        assert scheduler.metrics["teacher0"].total_samples == 0
        assert scheduler.metrics["teacher0"].success_count == 0

    def test_reset_metrics_all_teachers(self):
        """Test resetting metrics for all teachers."""
        teachers = [
            TeacherConfig(name="teacher0", endpoint="http://localhost:8000"),
            TeacherConfig(name="teacher1", endpoint="http://localhost:8001"),
        ]
        scheduler = MultiTeacherScheduler(teachers)

        scheduler.record_result("teacher0", reward=1.0, success=True)
        scheduler.record_result("teacher1", reward=0.5, success=True)
        assert scheduler.metrics["teacher0"].total_samples == 1
        assert scheduler.metrics["teacher1"].total_samples == 1

        scheduler.reset_metrics()  # No teacher name = reset all
        assert scheduler.metrics["teacher0"].total_samples == 0
        assert scheduler.metrics["teacher1"].total_samples == 0

    def test_performance_based_selection_after_warmup(self):
        """Test performance-based selection selects best teacher after warmup."""
        teachers = [
            TeacherConfig(name="teacher0", endpoint="http://localhost:8000", weight=1.0),
            TeacherConfig(name="teacher1", endpoint="http://localhost:8001", weight=1.0),
        ]
        config = SchedulerConfig(
            selection_strategy="performance",
            min_samples_before_selection=2,
            load_balance_factor=0.0,  # Pure performance-based
        )
        scheduler = MultiTeacherScheduler(teachers, config)

        # Record results: teacher0 consistently better
        for _ in range(5):
            scheduler.record_result("teacher0", reward=0.9, success=True)
        for _ in range(5):
            scheduler.record_result("teacher1", reward=0.5, success=True)

        # Now performance-based selection should prefer teacher0
        selections = [scheduler.select_teacher().name for _ in range(10)]
        # Teacher0 should be selected more often due to better performance
        assert selections.count("teacher0") > selections.count("teacher1")

    def test_get_fallback_teacher(self):
        """Test getting fallback teacher."""
        teachers = [
            TeacherConfig(name="teacher0", endpoint="http://localhost:8000"),
            TeacherConfig(name="teacher1", endpoint="http://localhost:8001"),
        ]
        config = SchedulerConfig(fallback_teacher="teacher1")
        scheduler = MultiTeacherScheduler(teachers, config)

        fallback = scheduler.get_fallback_teacher()
        assert fallback is not None
        assert fallback.name == "teacher1"

    def test_get_fallback_teacher_none(self):
        """Test getting fallback teacher when not configured."""
        teachers = [TeacherConfig(name="teacher0", endpoint="http://localhost:8000")]
        scheduler = MultiTeacherScheduler(teachers)

        fallback = scheduler.get_fallback_teacher()
        assert fallback is None


class TestCreateSchedulerFromConfigs:
    """Tests for create_scheduler_from_configs helper function."""

    def test_create_from_dicts(self):
        """Test creating scheduler from configuration dictionaries."""
        teacher_configs = [
            {"name": "teacher0", "endpoint": "http://localhost:8000", "weight": 1.0},
            {"name": "teacher1", "endpoint": "http://localhost:8001", "weight": 0.9},
        ]
        scheduler_config = {
            "selection_strategy": "performance",
            "epsilon": 0.15,
            "load_balance_factor": 0.3,
        }

        scheduler = create_scheduler_from_configs(teacher_configs, scheduler_config)

        assert len(scheduler.teachers) == 2
        assert scheduler.config.selection_strategy == "performance"
        assert scheduler.config.epsilon == 0.15
        assert scheduler.config.load_balance_factor == 0.3

    def test_create_from_dicts_default_config(self):
        """Test creating scheduler without scheduler config dict."""
        teacher_configs = [
            {"name": "teacher0", "endpoint": "http://localhost:8000"},
        ]

        scheduler = create_scheduler_from_configs(teacher_configs)

        assert len(scheduler.teachers) == 1
        assert scheduler.config.selection_strategy == "performance"  # Default
        assert scheduler.config.epsilon == 0.1  # Default

    def test_create_from_dicts_empty_teachers(self):
        """Test creating scheduler with empty teacher list."""
        scheduler = create_scheduler_from_configs([])
        assert len(scheduler.teachers) == 0

    def test_create_from_dicts_optional_fields(self):
        """Test creating scheduler with optional teacher fields."""
        teacher_configs = [
            {
                "name": "teacher0",
                "endpoint": "http://localhost:8000",
                "api_key": "test_key",
                "model": "gpt-4",
                "weight": 1.5,
            }
        ]

        scheduler = create_scheduler_from_configs(teacher_configs)

        teacher = scheduler.teachers["teacher0"]
        assert teacher.api_key == "test_key"
        assert teacher.model == "gpt-4"
        assert teacher.weight == 1.5
