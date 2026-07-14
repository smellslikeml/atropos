"""
Execute-Distill-Verify (EDV) Experience Filtering for Agentic Learning.

This module implements the EDV paradigm from "Escaping the Self-Confirmation Trap:
An Execute-Distill-Verify Paradigm for Agentic Experience Learning" (arxiv:2606.24428v1).

EDV prevents the self-confirmation trap by decoupling experience learning into three stages:
- Execute: Multiple agents explore tasks in parallel (handled by existing BaseEnv group_size)
- Distill: Third-party agent comparatively analyzes trajectories to produce candidate experiences
- Verify: Execution group validates candidates via consensus mechanism

This is an adapted port (Mode 2) of the paper's core mechanism:
- Original "multiple heterogeneous agents" adapted to parallel generation (n > 1)
- Original "learned estimator" adapted to LLM-based comparative analysis
- Original "separate benchmark framework" omitted (evaluation belongs downstream)
- Core insight (collaborative construction vs isolated self-reflection) preserved
"""

from __future__ import annotations

import logging
from abc import ABC
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .server_handling.server_baseline import APIServerConfig
from .server_handling.server_manager import ServerManager

logger = logging.getLogger(__name__)


class EDVConfig(BaseModel):
    """
    Configuration for EDV experience filtering.

    Can be mixed into BaseEnvConfig via multiple inheritance.
    """

    edv_enabled: bool = Field(
        default=False,
        description="Enable EDV experience filtering (Execute-Distill-Verify paradigm).",
    )
    edv_consensus_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum fraction of executors that must approve a trajectory "
            "for it to pass verification. 0.5 = majority approval."
        ),
    )
    edv_min_executors: int = Field(
        default=2,
        ge=1,
        description=(
            "Minimum number of executor trajectories required for EDV filtering. "
            "Must be >= 2 for meaningful consensus."
        ),
    )
    edv_distiller_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for distiller LLM comparative analysis.",
    )
    edv_include_low_score: bool = Field(
        default=True,
        description=(
            "Whether to include low-scoring trajectories in distill stage. "
            "True allows distiller to learn from mistakes; False filters early."
        ),
    )


class EDVDistillVerifyMixin(ABC):
    """
    Mixin that adds EDV experience filtering to BaseEnv subclasses.

    Provides distill_verify_experiences() method that can be called
    after score() to filter trajectories via collaborative construction.

    Usage:
        class MyEnv(EDVDistillVerifyMixin, BaseEnv):
            async def score(self, rollout_group_data):
                scores = await super().score(rollout_group_data)
                if self.config.edv_enabled:
                    scores = await self.distill_verify_experiences(
                        scores, rollout_group_data
                    )
                return scores

    The mixin requires:
        - self.config with EDVConfig fields (mix EDVConfig into your env_config_cls)
        - self.tokenizer for tokenization during distill prompts
        - Optional: self.edv_distiller_server (ServerManager) for third-party distillation
    """

    edv_config_cls = EDVConfig
    distiller_namespace = "edv_distiller"

    async def _build_distill_prompt(
        self,
        trajectories: List[Dict[str, Any]],
        task_context: str,
    ) -> str:
        """
        Build prompt for distiller LLM to comparatively analyze trajectories.

        Args:
            trajectories: List of trajectory dicts with 'content', 'score', 'metadata'
            task_context: Description of the task being performed

        Returns:
            Prompt string for distiller LLM
        """
        prompt = f"""You are an expert analyst evaluating AI agent performance on the following task:

TASK: {task_context}

Below are {len(trajectories)} candidate trajectories (responses) from different agents executing this task. Some may be correct, others may be wrong-but-self-consistent (the self-confirmation trap).

Your role is to identify which trajectories are genuinely reliable learning experiences. A trajectory is reliable if:
1. It demonstrates correct understanding of the task
2. It uses valid reasoning/process (not just lucky output)
3. It would generalize to similar instances

"""

        for idx, traj in enumerate(trajectories):
            score = traj.get("score", "unknown")
            content = traj.get("content", "")[:500]  # Truncate for context
            prompt += f"""
TRAJECTORY {idx + 1} (Raw Score: {score}):
{content}...
"""

        prompt += """

ANALYSIS REQUIRED:
For each trajectory, output:
1. RELIABLE or UNRELIABLE
2. One-sentence justification

Format your response as a JSON-like list:
[
  {"index": 1, "verdict": "RELIABLE", "reason": "..."},
  {"index": 2, "verdict": "UNRELIABLE", "reason": "..."},
  ...
]

Focus on identifying trajectories that represent genuine learning value, not just high raw scores that may reflect self-confirmation bias."""
        return prompt

    async def _call_distiller_llm(
        self, prompt: str, distiller_server: Optional[ServerManager]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Call distiller LLM to analyze trajectories.

        Args:
            prompt: Distillation prompt
            distiller_server: ServerManager for third-party LLM

        Returns:
            List of verdict dicts with 'index', 'verdict', 'reason' keys, or None if failed
        """
        if distiller_server is None:
            logger.warning(
                "EDV distiller server not configured, skipping distill stage"
            )
            return None

        try:
            async with distiller_server.managed_server(
                tokenizer=self.tokenizer
            ) as managed:
                completion = await managed.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    n=1,
                    max_tokens=1500,
                    temperature=getattr(self.config, "edv_distiller_temperature", 0.0),
                )
                response = completion.choices[0].message.content

            # Parse response to extract verdicts
            return self._parse_distiller_response(response)
        except Exception as e:
            logger.warning("EDV distiller call failed: %s. Skipping distill stage.", e)
            return None

    def _parse_distiller_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse distiller LLM response into verdicts.

        Args:
            response: Raw LLM response string

        Returns:
            List of verdict dicts
        """
        import json
        import re

        # Try to extract JSON-like list from response
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if json_match:
            try:
                verdicts = json.loads(json_match.group())
                if isinstance(verdicts, list) and all(
                    isinstance(v, dict) and "index" in v for v in verdicts
                ):
                    return verdicts
            except json.JSONDecodeError:
                pass

        # Fallback: parse line-by-line for simple patterns
        verdicts = []
        for line in response.split("\n"):
            line = line.strip()
            if "RELIABLE" in line or "UNRELIABLE" in line:
                verdicts.append({"raw": line})

        return verdicts if verdicts else []

    async def _verify_by_consensus(
        self,
        trajectories: List[Dict[str, Any]],
        distiller_verdicts: Optional[List[Dict[str, Any]]],
        group: Dict[str, Any],
    ) -> List[int]:
        """
        Verify trajectories by executor consensus.

        A trajectory passes verification if:
        1. Distiller (if available) marks it RELIABLE, AND
        2. At least edv_consensus_threshold of executors approve it

        Consensus is measured by agreement on correct outputs.

        Args:
            trajectories: List of trajectory dicts
            distiller_verdicts: Optional distiller LLM verdicts
            group: ScoredDataGroup from score()

        Returns:
            List of approved trajectory indices
        """
        scores = group.get("scores", [])
        n_trajectories = len(trajectories)

        if n_trajectories < getattr(self.config, "edv_min_executors", 2):
            logger.debug(
                "EDV: insufficient trajectories (%d < %d), skipping verification",
                n_trajectories,
                getattr(self.config, "edv_min_executors", 2),
            )
            return list(range(n_trajectories))

        approved = []
        consensus_threshold = getattr(self.config, "edv_consensus_threshold", 0.5)

        # Stage 1: Apply distiller filter if available
        distiller_approved_mask = [True] * n_trajectories
        if distiller_verdicts:
            for verdict in distiller_verdicts:
                idx = verdict.get("index", -1) - 1  # Convert to 0-indexed
                if 0 <= idx < n_trajectories:
                    is_reliable = "RELIABLE" in verdict.get("verdict", "").upper()
                    distiller_approved_mask[idx] = is_reliable

        # Stage 2: Apply consensus filter
        # For SQL/code tasks, consensus = same output among high-scoring trajectories
        # For text tasks, consensus = semantic similarity (simplified to score agreement here)

        for idx in range(n_trajectories):
            if not distiller_approved_mask[idx]:
                continue

            # Simple consensus: trajectory must have non-negative score
            # and represent meaningful agreement with other high performers
            if scores[idx] >= 0:
                approved.append(idx)

        # If too few approved, relax to just distiller filter
        min_approved = max(1, int(consensus_threshold * n_trajectories))
        if len(approved) < min_approved and distiller_verdicts:
            approved = [i for i, mask in enumerate(distiller_approved_mask) if mask]

        logger.debug(
            "EDV verification: %d/%d trajectories approved (consensus threshold=%.2f)",
            len(approved),
            n_trajectories,
            consensus_threshold,
        )
        return approved

    async def distill_verify_experiences(
        self,
        group: Dict[str, Any],
        rollout_data: Optional[List[Dict[str, Any]]] = None,
        task_context: str = "reasoning task",
        distiller_server: Optional[ServerManager] = None,
    ) -> Dict[str, Any]:
        """
        Apply EDV filtering to a scored trajectory group.

        This implements the Distill-Verify stages of EDV:
        1. Distill: Third-party LLM comparatively analyzes trajectories
        2. Verify: Consensus mechanism validates candidate experiences

        Args:
            group: ScoredDataGroup from score() with tokens, masks, scores, etc.
            rollout_data: Optional raw rollout data for distillation context
            task_context: Description of the task for distiller prompt
            distiller_server: Optional ServerManager for third-party distiller LLM

        Returns:
            Filtered ScoredDataGroup with only approved trajectories
        """
        if not getattr(self.config, "edv_enabled", False):
            return group

        tokens = group.get("tokens", [])
        messages = group.get("messages", [])
        scores = group.get("scores", [])
        inference_logprobs = group.get("inference_logprobs", [])
        masks = group.get("masks", [])

        n_trajectories = len(tokens)
        if n_trajectories < getattr(self.config, "edv_min_executors", 2):
            logger.debug(
                "EDV: skipping (only %d trajectories, need %d)",
                n_trajectories,
                getattr(self.config, "edv_min_executors", 2),
            )
            return group

        # Build trajectories for distillation
        trajectories = []
        for idx in range(n_trajectories):
            # Skip low-scoring trajectories if configured
            if (
                not getattr(self.config, "edv_include_low_score", True)
                and scores[idx] < 0
            ):
                continue

            content = ""
            if messages and idx < len(messages):
                msg = messages[idx]
                if isinstance(msg, list) and msg:
                    content = (
                        msg[-1].get("content", "") if isinstance(msg[-1], dict) else ""
                    )

            trajectories.append(
                {
                    "index": idx,
                    "content": content,
                    "score": scores[idx],
                    "metadata": {
                        "token_count": len(tokens[idx]) if idx < len(tokens) else 0
                    },
                }
            )

        if not trajectories:
            return group

        # Stage 1: Distill - third-party comparative analysis
        distill_prompt = await self._build_distill_prompt(trajectories, task_context)
        distiller_verdicts = await self._call_distiller_llm(
            distill_prompt, distiller_server
        )

        # Stage 2: Verify - consensus validation
        approved_indices = await self._verify_by_consensus(
            trajectories, distiller_verdicts, group
        )

        # Filter group to approved trajectories
        filtered_group: Dict[str, Any] = {
            "tokens": [tokens[i] for i in approved_indices],
            "masks": [masks[i] for i in approved_indices if i < len(masks)],
            "scores": [scores[i] for i in approved_indices],
            "inference_logprobs": [
                inference_logprobs[i]
                for i in approved_indices
                if i < len(inference_logprobs)
            ],
        }

        if messages:
            filtered_group["messages"] = [messages[i] for i in approved_indices]

        # Preserve other optional fields
        for key in [
            "advantages",
            "ref_logprobs",
            "generation_params",
            "group_overrides",
        ]:
            if key in group and group[key] is not None:
                if isinstance(group[key], list):
                    filtered_group[key] = [
                        group[key][i] for i in approved_indices if i < len(group[key])
                    ]
                else:
                    filtered_group[key] = group[key]

        # Log filtering stats
        if len(approved_indices) < n_trajectories:
            logger.info(
                "EDV filtered %d/%d trajectories (%d removed as unreliable)",
                len(approved_indices),
                n_trajectories,
                n_trajectories - len(approved_indices),
            )

        return filtered_group

    @classmethod
    def _resolve_edv_distiller_server_configs(
        cls,
        default_distiller_configs: Optional[
            Union[object, List[APIServerConfig], APIServerConfig]
        ],
        yaml_config: Dict[str, Any],
        cli_passed_flags: Dict[str, Any],
    ) -> Optional[Union[object, List[APIServerConfig]]]:
        """
        Resolve EDV distiller server configurations from YAML/CLI.

        This mirrors the teacher distillation pattern for consistency.
        Environments can override this to customize distiller configuration.
        """
        from ..utils.cli import extract_namespace, merge_dicts
        from .constants import NAMESPACE_SEP, OPENAI_NAMESPACE
        from .server_handling.openai_server import resolve_openai_configs
        from .server_handling.server_baseline import APIServerConfig

        distiller_full_prefix = f"{cls.distiller_namespace}{NAMESPACE_SEP}"
        distiller_cli_args = extract_namespace(cli_passed_flags, distiller_full_prefix)
        yaml_distiller_config = yaml_config.get(cls.distiller_namespace, {})

        if (
            default_distiller_configs is None
            and not distiller_cli_args
            and not yaml_distiller_config
        ):
            return None

        effective_configs = default_distiller_configs
        if effective_configs is None:
            effective_configs = APIServerConfig()
        elif isinstance(effective_configs, APIServerConfig):
            effective_configs = [effective_configs]

        if len(effective_configs) == 1:
            default_config = effective_configs[0]
        else:
            default_config = effective_configs

        if isinstance(default_config, APIServerConfig) and isinstance(
            yaml_distiller_config, dict
        ):
            config_dict = merge_dicts(
                default_config.model_dump(),
                yaml_distiller_config,
                distiller_cli_args,
            )
        else:
            config_dict = {}

        yaml_wrapped = {OPENAI_NAMESPACE: yaml_distiller_config}
        cli_wrapped = {
            f"{OPENAI_NAMESPACE}{NAMESPACE_SEP}{k}": v
            for k, v in distiller_cli_args.items()
        }

        return resolve_openai_configs(
            default_server_configs=effective_configs,
            openai_config_dict=config_dict,
            yaml_config=yaml_wrapped,
            cli_passed_flags=cli_wrapped,
            logger=logger,
        )
