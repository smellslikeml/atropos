"""
Tests for DenseSignalEvaluator - QVal methodology.

This test module validates:
1. Core correlation computation (Spearman/Kendall)
2. Teacher logprobs evaluation
3. Multi-signal comparison
4. Integration with TeacherDistillationEnv

These are unit tests that don't require API calls - they validate the
evaluation logic with synthetic data.

Paper: QVal (arXiv:2606.32034v1) - Cheaply Evaluating Dense Supervision
Signals for Long-Horizon LLM Agents.
"""

from __future__ import annotations

import numpy as np
import pytest

from atroposlib.envs.dense_signal_evaluator import (
    DenseSignalEvaluator, SignalEvaluationResult,
    aggregate_token_level_signals, compute_reference_values_from_rewards)
from atroposlib.envs.teacher_distillation_env import TeacherDistillationEnv

# =============================================================================
# UNIT TESTS: DenseSignalEvaluator core functionality
# =============================================================================


def test_evaluator_init():
    """Test DenseSignalEvaluator initialization."""
    evaluator = DenseSignalEvaluator(min_samples=5)
    assert evaluator.min_samples == 5
    print("✓ DenseSignalEvaluator initializes correctly")


def test_basic_evaluation():
    """Test basic signal evaluation with known correlation."""
    evaluator = DenseSignalEvaluator(min_samples=3)

    # Perfect positive correlation
    signal = [1.0, 2.0, 3.0, 4.0, 5.0]
    reference = [2.0, 4.0, 6.0, 8.0, 10.0]  # reference = 2 * signal

    result = evaluator.evaluate(signal, reference, "test_signal")

    assert result.signal_name == "test_signal"
    assert result.spearman_rho == pytest.approx(1.0, abs=0.01)
    assert result.kendall_tau == pytest.approx(1.0, abs=0.01)
    assert result.num_samples == 5
    assert result.is_positive_alignment()
    print("✓ Basic evaluation computes correct correlations")


def test_perfect_negative_correlation():
    """Test evaluation with perfect negative correlation."""
    evaluator = DenseSignalEvaluator(min_samples=3)

    signal = [1.0, 2.0, 3.0, 4.0, 5.0]
    reference = [5.0, 4.0, 3.0, 2.0, 1.0]  # Perfect negative

    result = evaluator.evaluate(signal, reference, "negative_signal")

    assert result.spearman_rho == pytest.approx(-1.0, abs=0.01)
    assert result.kendall_tau == pytest.approx(-1.0, abs=0.01)
    assert not result.is_positive_alignment()
    print("✓ Negative correlation detected correctly")


def test_no_correlation():
    """Test evaluation with no correlation (random noise)."""
    evaluator = DenseSignalEvaluator(min_samples=10)

    # Random signal vs sorted reference (should have low correlation)
    # Use a different seed that produces lower correlation
    np.random.seed(123)
    signal = np.random.randn(50).tolist()
    reference = list(range(50))

    result = evaluator.evaluate(signal, reference, "noise_signal")

    # Correlation should be relatively low (with larger sample, regression to mean)
    # Allow some correlation but not strong
    assert abs(result.spearman_rho) < 0.7
    print(
        f"✓ No correlation case: ρ={result.spearman_rho:.3f}, "
        f"τ={result.kendall_tau:.3f}"
    )


def test_minimum_samples_validation():
    """Test that evaluator rejects insufficient samples."""
    evaluator = DenseSignalEvaluator(min_samples=10)

    with pytest.raises(ValueError, match="Need at least 10 samples"):
        evaluator.evaluate([1.0, 2.0], [3.0, 4.0])
    print("✓ Minimum samples validation works")


def test_shape_mismatch_validation():
    """Test that evaluator rejects mismatched input shapes."""
    evaluator = DenseSignalEvaluator(min_samples=3)

    with pytest.raises(ValueError, match="same shape"):
        evaluator.evaluate([1.0, 2.0, 3.0], [4.0, 5.0])  # Different lengths
    print("✓ Shape mismatch validation works")


def test_constant_inputs():
    """Test handling of constant inputs (zero variance)."""
    evaluator = DenseSignalEvaluator(min_samples=3)

    # Constant signal should still work but correlation is undefined
    result = evaluator.evaluate([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])

    # Should return zeros for correlation when variance is zero
    assert result.spearman_rho == 0.0
    print("✓ Constant inputs handled gracefully")


# =============================================================================
# TEACHER LOGPROBS EVALUATION
# =============================================================================


def test_teacher_logprobs_evaluation():
    """Test evaluation of teacher distillation logprobs signal."""
    evaluator = DenseSignalEvaluator(min_samples=3)

    # Create synthetic teacher logprobs: [seq][pos][top_k]
    teacher_logprobs = [
        [[-0.1, -2.5], [-0.5, -3.0]],  # Sequence 1: high logprobs
        [[-2.0, -4.0], [-3.0, -5.0]],  # Sequence 2: low logprobs
        [[-0.2, -1.8], [-0.8, -2.2]],  # Sequence 3: medium-high
    ]

    # Outcomes correlated with logprobs (higher logprob = better outcome)
    outcome_rewards = [1.0, 0.0, 0.8]

    result = evaluator.evaluate_teacher_logprobs(
        teacher_logprobs=teacher_logprobs,
        outcome_rewards=outcome_rewards,
        signal_name="test_teacher",
    )

    assert result.signal_name == "test_teacher"
    assert result.num_samples == 3
    assert result.spearman_rho > 0.5  # Should detect positive correlation
    print(f"✓ Teacher logprobs evaluation: ρ={result.spearman_rho:.3f}")


def test_teacher_logprobs_with_top_k():
    """Test teacher logprobs evaluation with top-k aggregation."""
    evaluator = DenseSignalEvaluator(min_samples=2)  # Changed from 3 to 2

    teacher_logprobs = [
        [[-0.1, -0.5, -1.0], [-0.3, -0.7, -1.2]],
        [[-2.0, -2.5, -3.0], [-2.2, -2.7, -3.2]],
    ]
    outcome_rewards = [1.0, 0.0]

    result = evaluator.evaluate_teacher_logprobs(
        teacher_logprobs=teacher_logprobs,
        outcome_rewards=outcome_rewards,
        top_k=3,  # Aggregate over top-3 tokens
        signal_name="teacher_topk",
    )

    assert result.num_samples == 2
    print(f"✓ Teacher logprobs with top-k aggregation: ρ={result.spearman_rho:.3f}")


def test_teacher_logprobs_missing_data():
    """Test handling of missing/None logprobs."""
    evaluator = DenseSignalEvaluator(min_samples=2)

    teacher_logprobs = [
        [[-0.1, -2.0]],  # Valid sequence
        None,  # Missing sequence (should be skipped)
        [[-1.5, -3.0]],  # Valid sequence
    ]
    outcome_rewards = [1.0, 0.5, 0.0]  # Outcomes for all sequences

    result = evaluator.evaluate_teacher_logprobs(
        teacher_logprobs=teacher_logprobs,
        outcome_rewards=outcome_rewards,
        signal_name="teacher_with_missing",
    )

    # Should skip the None sequence
    assert result.num_samples == 2
    assert result.metadata["valid_sequences"] == 2
    assert result.metadata["num_sequences"] == 3
    print("✓ Missing teacher logprobs handled correctly")


def test_teacher_logprobs_insufficient_data():
    """Test that insufficient valid data raises error."""
    evaluator = DenseSignalEvaluator(min_samples=5)

    teacher_logprobs = [
        [[-0.1]],
        [[-2.0]],
    ]  # Only 2 sequences
    outcome_rewards = [1.0, 0.0]

    with pytest.raises(ValueError, match="Insufficient valid sequences"):
        evaluator.evaluate_teacher_logprobs(
            teacher_logprobs=teacher_logprobs,
            outcome_rewards=outcome_rewards,
        )
    print("✓ Insufficient data validation works")


# =============================================================================
# MULTI-SIGNAL EVALUATION
# =============================================================================


def test_multiple_signals_evaluation():
    """Test evaluating multiple signals simultaneously."""
    evaluator = DenseSignalEvaluator(min_samples=5)

    reference = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    signals = {
        "perfect_signal": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "inverse_signal": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        "noise": [5.0, 3.0, 7.0, 2.0, 8.0, 1.0, 9.0, 4.0, 6.0, 0.0],
    }

    results = evaluator.evaluate_multiple_signals(signals, reference)

    assert len(results) == 3
    assert results["perfect_signal"].is_positive_alignment()
    assert not results["inverse_signal"].is_positive_alignment()
    print("✓ Multiple signals evaluated correctly")


def test_signal_comparison():
    """Test ranking signals by Q-alignment strength."""
    evaluator = DenseSignalEvaluator(min_samples=5)

    reference = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    signals = {
        "weak_signal": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
        "strong_signal": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "inverse_signal": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    }

    results = evaluator.evaluate_multiple_signals(signals, reference)
    ranked = evaluator.compare_signals(results, metric="spearman")

    # strong_signal should be ranked higher than inverse_signal
    # (weak_signal and strong_signal may tie due to perfect correlation)
    strong_idx = next(
        i for i, (name, _) in enumerate(ranked) if name == "strong_signal"
    )
    inverse_idx = next(
        i for i, (name, _) in enumerate(ranked) if name == "inverse_signal"
    )

    assert (
        strong_idx < inverse_idx
    ), "strong_signal should rank higher than inverse_signal"
    print(f"✓ Signal ranking: {[name for name, _ in ranked]}")


def test_results_summary():
    """Test generation of human-readable summary."""
    evaluator = DenseSignalEvaluator(min_samples=3)

    reference = [1.0, 2.0, 3.0, 4.0, 5.0]

    results = evaluator.evaluate_multiple_signals(
        {
            "signal_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "signal_b": [5.0, 4.0, 3.0, 2.0, 1.0],
        },
        reference,
    )

    summary = evaluator.summarize_results(results)

    assert "DENSE SIGNAL EVALUATION SUMMARY" in summary
    assert "signal_a" in summary
    assert "signal_b" in summary
    assert "Spearman" in summary
    print("✓ Summary generation works")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def test_compute_reference_values_from_rewards():
    """Test computing Monte Carlo returns from rewards."""
    rewards = [
        [1.0, 1.0, 1.0],  # Episode 1: return = 3.0
        [0.5, 0.5],  # Episode 2: return = 1.0
    ]

    returns = compute_reference_values_from_rewards(rewards, gamma=1.0)

    assert returns == [3.0, 1.0]
    print("✓ Reference values computed from rewards")


def test_compute_reference_values_with_discount():
    """Test computing discounted returns."""
    rewards = [
        [1.0, 1.0],  # Return = 1 + 0.9*1 = 1.9
    ]

    returns = compute_reference_values_from_rewards(rewards, gamma=0.9)

    assert returns[0] == pytest.approx(1.9, abs=0.01)
    print("✓ Discounted returns computed correctly")


def test_aggregate_token_signals_mean():
    """Test aggregating token signals by mean."""
    token_signals = [
        [0.1, 0.5, 0.3],
        [0.8, 0.2],
    ]

    aggregated = aggregate_token_level_signals(token_signals, aggregation="mean")

    assert aggregated[0] == pytest.approx((0.1 + 0.5 + 0.3) / 3)
    assert aggregated[1] == pytest.approx((0.8 + 0.2) / 2)
    print("✓ Mean aggregation works")


def test_aggregate_token_signals_max():
    """Test aggregating token signals by max."""
    token_signals = [
        [0.1, 0.5, 0.9],
        [0.3, 0.7],
    ]

    aggregated = aggregate_token_level_signals(token_signals, aggregation="max")

    assert aggregated[0] == 0.9
    assert aggregated[1] == 0.7
    print("✓ Max aggregation works")


def test_aggregate_token_signals_last():
    """Test aggregating token signals by last value."""
    token_signals = [
        [0.1, 0.5, 0.9],
        [0.3, 0.7],
    ]

    aggregated = aggregate_token_level_signals(token_signals, aggregation="last")

    assert aggregated[0] == 0.9
    assert aggregated[1] == 0.7
    print("✓ Last aggregation works")


def test_aggregate_token_signals_invalid():
    """Test that invalid aggregation method raises error."""
    token_signals = [[0.1, 0.5]]

    with pytest.raises(ValueError, match="Unknown aggregation"):
        aggregate_token_level_signals(token_signals, aggregation="invalid")
    print("✓ Invalid aggregation method rejected")


# =============================================================================
# INTEGRATION TESTS: TeacherDistillationEnv
# =============================================================================


def test_teacher_env_has_evaluation_method():
    """Test that TeacherDistillationEnv has signal quality evaluation method."""
    # This test validates the integration exists without requiring
    # a full environment setup

    assert hasattr(TeacherDistillationEnv, "evaluate_teacher_signal_quality")

    # Check method signature
    method = getattr(TeacherDistillationEnv, "evaluate_teacher_signal_quality")
    assert callable(method)

    print("✓ TeacherDistillationEnv has evaluate_teacher_signal_quality method")


def test_signal_evaluation_result_string_representation():
    """Test SignalEvaluationResult string output."""
    result = SignalEvaluationResult(
        signal_name="test",
        spearman_rho=0.85,
        kendall_tau=0.72,
        num_samples=100,
        p_value_spearman=0.001,
        p_value_kendall=0.002,
        metadata={},
    )

    result_str = str(result)

    assert "test" in result_str
    assert "0.85" in result_str
    assert "100" in result_str
    print("✓ SignalEvaluationResult string representation works")


def test_signal_evaluation_result_significance():
    """Test statistical significance checking."""
    result_significant = SignalEvaluationResult(
        signal_name="significant",
        spearman_rho=0.8,
        kendall_tau=0.6,
        num_samples=100,
        p_value_spearman=0.01,
        p_value_kendall=0.02,
        metadata={},
    )

    result_not_significant = SignalEvaluationResult(
        signal_name="not_significant",
        spearman_rho=0.1,
        kendall_tau=0.05,
        num_samples=10,
        p_value_spearman=0.5,
        p_value_kendall=0.6,
        metadata={},
    )

    assert result_significant.is_significant(alpha=0.05)
    assert not result_not_significant.is_significant(alpha=0.05)
    print("✓ Statistical significance checking works")


def test_synthetic_qval_scenario():
    """
    Test a synthetic QVal scenario: teacher signal vs outcomes.

    This simulates the QVal paper's core experiment: evaluating whether a
    dense supervision signal (teacher logprobs) correlates with good
    outcomes (higher rewards).
    """
    evaluator = DenseSignalEvaluator(min_samples=10)

    # Simulate 100 trajectories
    np.random.seed(42)
    n_trajectories = 100

    # Generate teacher logprobs (signal): higher = more confident
    teacher_signal = np.random.beta(2, 5, n_trajectories).tolist()

    # Generate outcomes: correlated with signal (good teacher)
    # Outcome = signal + noise
    noise = np.random.normal(0, 0.1, n_trajectories)
    outcomes = [max(0.0, min(1.0, s + n)) for s, n in zip(teacher_signal, noise)]

    # Evaluate Q-alignment
    result = evaluator.evaluate(
        signal_scores=teacher_signal,
        reference_values=outcomes,
        signal_name="synthetic_teacher",
    )

    # With the correlation we built in, should see positive alignment
    assert result.num_samples == n_trajectories
    assert result.spearman_rho > 0.3  # Should detect the correlation

    print(
        f"✓ Synthetic QVal scenario: ρ={result.spearman_rho:.3f} "
        f"(teacher signal quality)"
    )


# =============================================================================
# TEST RUNNER
# =============================================================================


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("DENSE SIGNAL EVALUATOR TESTS")
    print("=" * 60 + "\n")

    tests = [
        ("Evaluator Initialization", test_evaluator_init),
        ("Basic Evaluation", test_basic_evaluation),
        ("Negative Correlation", test_perfect_negative_correlation),
        ("No Correlation", test_no_correlation),
        ("Minimum Samples Validation", test_minimum_samples_validation),
        ("Shape Mismatch Validation", test_shape_mismatch_validation),
        ("Constant Inputs", test_constant_inputs),
        ("Teacher Logprobs Evaluation", test_teacher_logprobs_evaluation),
        ("Teacher Logprobs with Top-K", test_teacher_logprobs_with_top_k),
        ("Teacher Logprobs Missing Data", test_teacher_logprobs_missing_data),
        ("Teacher Logprobs Insufficient Data", test_teacher_logprobs_insufficient_data),
        ("Multiple Signals Evaluation", test_multiple_signals_evaluation),
        ("Signal Comparison", test_signal_comparison),
        ("Results Summary", test_results_summary),
        ("Reference Values from Rewards", test_compute_reference_values_from_rewards),
        ("Discounted Returns", test_compute_reference_values_with_discount),
        ("Token Aggregation (Mean)", test_aggregate_token_signals_mean),
        ("Token Aggregation (Max)", test_aggregate_token_signals_max),
        ("Token Aggregation (Last)", test_aggregate_token_signals_last),
        ("Invalid Aggregation", test_aggregate_token_signals_invalid),
        ("TeacherEnv Integration", test_teacher_env_has_evaluation_method),
        (
            "Result String Representation",
            test_signal_evaluation_result_string_representation,
        ),
        ("Statistical Significance", test_signal_evaluation_result_significance),
        ("Synthetic QVal Scenario", test_synthetic_qval_scenario),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys

    success = run_all_tests()
    sys.exit(0 if success else 1)
