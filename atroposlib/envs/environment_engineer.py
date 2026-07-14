"""
Environment Engineer module.

This module implements the LLM-as-Environment-Engineer framework from
"From Trainee to Trainer: LLM-Designed Training Environment for RL"
https://arxiv.org/abs/2606.17682v1

The core insight: use the current policy model (or a teacher model) to
analyze failure trajectories and propose modifications to the next-stage
training environment configuration. This automates environment optimization,
reducing the need for manual environment redesign between training stages.

Implementation (Mode 2 - adapted port):
- Core mechanism: policy analyzes failures → proposes config changes
- Substituted components: structured summaries instead of learned MI
  estimators, simpler configurable environments instead of MAPF-FrozenLake
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConfigChangeType(Enum):
    """Types of environment configuration changes."""

    INCREASE_DIFFICULTY = "increase_difficulty"
    DECREASE_DIFFICULTY = "decrease_difficulty"
    ADAPTER_PARAMETER = "adapter_parameter"
    CONTEXT_WINDOW = "context_window"
    REWARD_WEIGHT = "reward_weight"
    TASK_DISTRIBUTION = "task_distribution"
    UNKNOWN = "unknown"


@dataclass
class FailureCase:
    """Represents a single failure case from evaluation."""

    item_id: str
    input_text: str
    model_output: str
    expected_output: str
    score: float
    failure_reason: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EnvironmentConfig:
    """Represents an environment configuration that can be modified."""

    config_name: str
    current_value: Any
    min_value: Any
    max_value: Any
    description: str
    change_type: ConfigChangeType


@dataclass
class ConfigProposal:
    """Represents a proposed configuration change."""

    config_name: str
    old_value: Any
    new_value: Any
    reason: str
    confidence: float
    change_type: ConfigChangeType


@dataclass
class EngineerAnalysis:
    """Result of an environment engineer analysis run."""

    proposals: List[ConfigProposal]
    summary: str
    failure_evidence: List[str]
    preserved_configs: List[str]
    metadata: Optional[Dict[str, Any]] = None


class EnvironmentEngineerConfig(BaseModel):
    """Configuration for the Environment Engineer."""

    enabled: bool = Field(
        default=False,
        description="Whether to enable the environment engineer.",
    )
    engineer_model: Optional[str] = Field(
        default=None,
        description=(
            "Model to use as the environment engineer. "
            "If None, uses the policy model."
        ),
    )
    max_proposals_per_run: int = Field(
        default=5,
        description=(
            "Maximum number of config changes to propose per analysis run."
        ),
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for a proposal to be included.",
    )
    preserve_working_configs: bool = Field(
        default=True,
        description="Whether to preserve configs working well.",
    )
    failure_score_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Score below which a case is considered a failure.",
    )


class BaseEnvironmentEngineer(ABC):
    """
    Abstract base class for environment engineers.

    Subclasses implement the specific logic for analyzing failure trajectories
    and proposing configuration changes.
    """

    config_cls = EnvironmentEngineerConfig

    def __init__(self, config: EnvironmentEngineerConfig):
        self.config = config

    @abstractmethod
    async def analyze(
        self,
        failure_cases: List[FailureCase],
        current_configs: Dict[str, EnvironmentConfig],
        eval_metrics: Dict[str, float],
    ) -> EngineerAnalysis:
        """
        Analyze failure cases and propose environment configuration changes.

        Args:
            failure_cases: List of failure cases from evaluation
            current_configs: Current environment configuration values
            eval_metrics: Overall evaluation metrics

        Returns:
            EngineerAnalysis with proposals and summary
        """
        raise NotImplementedError(
            "Analyze method must be implemented in subclass"
        )

    def _extract_failure_evidence(
        self, failure_cases: List[FailureCase]
    ) -> List[str]:
        """Extract key evidence from failure cases."""
        evidence = []
        for case in failure_cases[:10]:  # Limit to avoid context overflow
            evidence.append(
                f"Failure: {case.failure_reason} | "
                f"Input: {case.input_text[:100]}... | "
                f"Output: {case.model_output[:100]}... | "
                f"Score: {case.score:.2f}"
            )
        return evidence

    def _should_preserve_config(
        self,
        config_name: str,
        eval_metrics: Dict[str, float],
    ) -> bool:
        """Determine if a config should be preserved based on performance."""
        if not self.config.preserve_working_configs:
            return False

        # Simple heuristic: if overall performance is good, preserve
        overall_score = eval_metrics.get("overall_score", 0.0)
        return overall_score >= 0.7  # Presume 70%+ is "working well"


class HeuristicEnvironmentEngineer(BaseEnvironmentEngineer):
    """
    A heuristic-based environment engineer that doesn't require an LLM call.

    This implements Mode 2 (adapted port) of the paper:
    - Core mechanism: analyze failures → propose config changes
    - Substitution: uses rule-based heuristics instead of LLM analysis

    Useful for:
    - Quick iteration without LLM costs
    - Testing the environment engineer framework
    - Baseline comparisons
    """

    async def analyze(
        self,
        failure_cases: List[FailureCase],
        current_configs: Dict[str, EnvironmentConfig],
        eval_metrics: Dict[str, float],
    ) -> EngineerAnalysis:
        """Analyze failures using heuristics and propose config changes."""
        proposals: List[ConfigProposal] = []
        preserved_configs: List[str] = []

        # Extract failure evidence
        failure_evidence = self._extract_failure_evidence(failure_cases)

        # Analyze failure patterns
        failure_reasons = [case.failure_reason for case in failure_cases]
        n_failures = max(len(failure_reasons), 1)
        low_accuracy_rate = sum(
            "accuracy" in r.lower() for r in failure_reasons
        ) / n_failures
        high_refusal_rate = sum(
            "refusal" in r.lower() or "refused" in r.lower()
            for r in failure_reasons
        ) / n_failures
        incomplete_rate = sum(
            "incomplete" in r.lower() or "truncated" in r.lower()
            for r in failure_reasons
        ) / n_failures

        # Generate proposals based on patterns
        for config_name, config in current_configs.items():
            if self._should_preserve_config(config_name, eval_metrics):
                preserved_configs.append(config_name)
                continue

            proposal = self._propose_change_for_config(
                config_name,
                config,
                low_accuracy_rate,
                high_refusal_rate,
                incomplete_rate,
            )
            if (
                proposal is not None and
                proposal.confidence >= self.config.confidence_threshold
            ):
                proposals.append(proposal)

        # Sort by confidence and limit
        proposals.sort(key=lambda p: p.confidence, reverse=True)
        proposals = proposals[: self.config.max_proposals_per_run]

        # Generate summary
        summary = self._generate_summary(
            proposals, preserved_configs, failure_evidence, eval_metrics
        )

        return EngineerAnalysis(
            proposals=proposals,
            summary=summary,
            failure_evidence=failure_evidence,
            preserved_configs=preserved_configs,
            metadata={
                "low_accuracy_rate": low_accuracy_rate,
                "high_refusal_rate": high_refusal_rate,
                "incomplete_rate": incomplete_rate,
            },
        )

    def _propose_change_for_config(
        self,
        config_name: str,
        config: EnvironmentConfig,
        low_accuracy_rate: float,
        high_refusal_rate: float,
        incomplete_rate: float,
    ) -> Optional[ConfigProposal]:
        """Propose a change for a specific config based on failure patterns."""

        if config.change_type == ConfigChangeType.INCREASE_DIFFICULTY:
            if low_accuracy_rate > 0.5:
                # Accuracy issues - decrease difficulty
                new_value = self._adjust_value(
                    config.current_value, config.min_value, config.max_value,
                    -0.2
                )
                if new_value != config.current_value:
                    msg = (
                        f"High low-accuracy failure rate "
                        f"({low_accuracy_rate:.2%}) suggests "
                        f"reducing difficulty"
                    )
                    return ConfigProposal(
                        config_name=config_name,
                        old_value=config.current_value,
                        new_value=new_value,
                        reason=msg,
                        confidence=0.7,
                        change_type=ConfigChangeType.DECREASE_DIFFICULTY,
                    )

        elif config.change_type == ConfigChangeType.CONTEXT_WINDOW:
            if incomplete_rate > 0.3:
                # Incomplete outputs - increase context window
                new_value = self._adjust_value(
                    config.current_value, config.min_value, config.max_value,
                    0.3
                )
                if new_value != config.current_value:
                    msg = (
                        f"High incomplete output rate ({incomplete_rate:.2%}) "
                        f"suggests increasing context window"
                    )
                    return ConfigProposal(
                        config_name=config_name,
                        old_value=config.current_value,
                        new_value=new_value,
                        reason=msg,
                        confidence=0.6,
                        change_type=ConfigChangeType.CONTEXT_WINDOW,
                    )

        elif config.change_type == ConfigChangeType.REWARD_WEIGHT:
            if low_accuracy_rate > 0.4:
                # Adjust reward weights to encourage accuracy
                new_value = self._adjust_value(
                    config.current_value, config.min_value, config.max_value,
                    0.15
                )
                if new_value != config.current_value:
                    msg = (
                        "Accuracy issues detected, increasing reward weight "
                        "for correct outputs"
                    )
                    return ConfigProposal(
                        config_name=config_name,
                        old_value=config.current_value,
                        new_value=new_value,
                        reason=msg,
                        confidence=0.65,
                        change_type=ConfigChangeType.REWARD_WEIGHT,
                    )

        return None

    def _adjust_value(
        self,
        current: Any,
        min_val: Any,
        max_val: Any,
        relative_change: float,
    ) -> Any:
        """Adjust a value by a relative amount, clamping to min/max."""
        if isinstance(current, (int, float)):
            change = current * relative_change
            new_value = current + change
            return max(min_val, min(max_val, new_value))
        return current

    def _generate_summary(
        self,
        proposals: List[ConfigProposal],
        preserved_configs: List[str],
        failure_evidence: List[str],
        eval_metrics: Dict[str, float],
    ) -> str:
        """Generate a human-readable summary of the analysis."""
        lines = [
            "Environment Engineer Analysis Summary",
            "=" * 40,
            f"Overall evaluation score: "
            f"{eval_metrics.get('overall_score', 'N/A')}",
            f"Failure cases analyzed: {len(failure_evidence)}",
            "",
            f"Proposed changes ({len(proposals)}):",
        ]

        for proposal in proposals:
            change_str = (
                f"  - {proposal.config_name}: "
                f"{proposal.old_value} → {proposal.new_value} "
                f"(confidence: {proposal.confidence:.2f})"
            )
            lines.append(change_str)
            lines.append(f"    Reason: {proposal.reason}")

        if preserved_configs:
            lines.append("")
            lines.append(f"Preserved configs ({len(preserved_configs)}):")
            for name in preserved_configs:
                lines.append(f"  - {name} (performing well)")

        if failure_evidence:
            lines.append("")
            lines.append("Sample failure evidence:")
            for evidence in failure_evidence[:3]:
                lines.append(f"  - {evidence}")

        return "\n".join(lines)


class LLMEnvironmentEngineer(BaseEnvironmentEngineer):
    """
    An LLM-based environment engineer using the policy to analyze failures.

    This implements the core mechanism from the paper:
    - Uses the current policy checkpoint as the environment engineer
    - Conditions the engineer on structured summaries of failures
    - Outputs the next-stage environment configuration

    Requires an LLM server to be configured.
    """

    def __init__(
        self,
        config: EnvironmentEngineerConfig,
        server: Any,  # ServerManager or similar
    ):
        super().__init__(config)
        self.server = server

    async def analyze(
        self,
        failure_cases: List[FailureCase],
        current_configs: Dict[str, EnvironmentConfig],
        eval_metrics: Dict[str, float],
    ) -> EngineerAnalysis:
        """Use the LLM to analyze failures and propose config changes."""
        # Build the prompt for the LLM
        prompt = self._build_analysis_prompt(
            failure_cases, current_configs, eval_metrics
        )

        # Query the LLM
        response = await self._query_engineer(prompt)

        # Parse the response into proposals
        proposals = self._parse_proposals(response, current_configs)

        # Identify preserved configs
        preserved_configs = [
            name
            for name, config in current_configs.items()
            if self._should_preserve_config(name, eval_metrics)
        ]

        # Extract failure evidence
        failure_evidence = self._extract_failure_evidence(failure_cases)

        # Generate summary
        summary = self._generate_summary(
            proposals, preserved_configs, eval_metrics
        )

        return EngineerAnalysis(
            proposals=proposals,
            summary=summary,
            failure_evidence=failure_evidence,
            preserved_configs=preserved_configs,
        )

    def _build_analysis_prompt(
        self,
        failure_cases: List[FailureCase],
        current_configs: Dict[str, EnvironmentConfig],
        eval_metrics: Dict[str, float],
    ) -> str:
        """Build the analysis prompt for the LLM."""
        lines = [
            "You are an Environment Engineer. Analyze training failures and "
            "propose modifications to the environment config for the next "
            "training stage.",
            "",
            "Current Performance:",
        ]

        for key, value in eval_metrics.items():
            lines.append(f"  {key}: {value}")

        lines.extend([
            "",
            "Recent Failure Cases (sample):",
        ])

        for case in failure_cases[:5]:
            lines.extend([
                f"  - Input: {case.input_text[:150]}...",
                f"    Output: {case.model_output[:100]}...",
                f"    Expected: {case.expected_output[:100]}...",
                f"    Reason: {case.failure_reason}",
                f"    Score: {case.score:.2f}",
                ""
            ])

        lines.extend([
            "Current Environment Configuration:",
        ])

        for name, config in current_configs.items():
            lines.extend([
                f"  - {name}: {config.current_value}",
                f"    Description: {config.description}",
                f"    Range: [{config.min_value}, {config.max_value}]",
                f"    Type: {config.change_type.value}",
                ""
            ])

        lines.extend([
            "Instructions:",
            "1. Analyze failure patterns and identify which configs to "
            "change.",
            "2. Propose specific new values (within allowed ranges).",
            "3. For each proposal, provide reason and confidence (0-1).",
            "4. Preserve configs that are already working well.",
            "",
            "Output format (one per line):",
            "PROPOSAL: config_name | new_value | reason | confidence",
            "",
            "PROPOSALS:",
        ])

        return "\n".join(lines)

    async def _query_engineer(self, prompt: str) -> str:
        """Query the LLM server with the analysis prompt."""
        # This is a placeholder - in real implementation, use self.server
        # to make a chat completion request
        logger.warning(
            "LLMEnvironmentEngineer._query_engineer not fully implemented - "
            "requires server integration. Returning mock response."
        )
        return "PROPOSAL: mock_config | 0.5 | mock response | 0.5"

    def _parse_proposals(
        self,
        response: str,
        current_configs: Dict[str, EnvironmentConfig],
    ) -> List[ConfigProposal]:
        """Parse the LLM response into ConfigProposal objects."""
        proposals = []

        for line in response.split("\n"):
            line = line.strip()
            if not line.startswith("PROPOSAL:"):
                continue

            parts = line[len("PROPOSAL:"):].split("|")
            if len(parts) < 4:
                continue

            config_name = parts[0].strip()
            new_value_str = parts[1].strip()
            reason = parts[2].strip()
            confidence = float(parts[3].strip())

            if config_name not in current_configs:
                continue

            config = current_configs[config_name]

            # Try to convert new_value to the appropriate type
            try:
                if isinstance(config.current_value, int):
                    new_value = int(new_value_str)
                elif isinstance(config.current_value, float):
                    new_value = float(new_value_str)
                else:
                    new_value = new_value_str
            except (ValueError, TypeError):
                continue

            # Clamp to min/max
            if isinstance(new_value, (int, float)):
                new_value = max(
                    config.min_value, min(config.max_value, new_value)
                )

            if new_value == config.current_value:
                continue

            proposals.append(
                ConfigProposal(
                    config_name=config_name,
                    old_value=config.current_value,
                    new_value=new_value,
                    reason=reason,
                    confidence=confidence,
                    change_type=config.change_type,
                )
            )

        # Sort by confidence and limit
        proposals.sort(key=lambda p: p.confidence, reverse=True)
        return proposals[: self.config.max_proposals_per_run]

    def _generate_summary(
        self,
        proposals: List[ConfigProposal],
        preserved_configs: List[str],
        eval_metrics: Dict[str, float],
    ) -> str:
        """Generate a human-readable summary."""
        lines = [
            "LLM Environment Engineer Analysis",
            "=" * 40,
            f"Overall score: {eval_metrics.get('overall_score', 'N/A')}",
            "",
            f"Proposed changes ({len(proposals)}):",
        ]

        for proposal in proposals:
            change_str = (
                f"  - {proposal.config_name}: "
                f"{proposal.old_value} → {proposal.new_value} "
                f"(confidence: {proposal.confidence:.2f})"
            )
            lines.append(change_str)
            lines.append(f"    Reason: {proposal.reason}")

        return "\n".join(lines)


def create_environment_engineer(
    config: EnvironmentEngineerConfig,
    server: Optional[Any] = None,
    use_llm: bool = False,
) -> BaseEnvironmentEngineer:
    """
    Factory function to create an environment engineer instance.

    Args:
        config: Environment engineer configuration
        server: Optional server for LLM-based engineer
        use_llm: If True, use LLM-based engineer; otherwise use heuristic

    Returns:
        An instance of BaseEnvironmentEngineer
    """
    if not config.enabled:
        raise ValueError("Environment engineer is not enabled in config")

    if use_llm:
        if server is None:
            raise ValueError("Server must be provided for LLM-based engineer")
        return LLMEnvironmentEngineer(config, server)
    else:
        return HeuristicEnvironmentEngineer(config)
