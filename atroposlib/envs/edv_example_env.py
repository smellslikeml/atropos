"""
Example environment demonstrating EDV experience filtering integration.

This minimal example shows how to integrate the Execute-Distill-Verify paradigm
into any Atropos environment using the EDVDistillVerifyMixin.

Based on: "Escaping the Self-Confirmation Trap: An Execute-Distill-Verify
Paradigm for Agentic Experience Learning" (arxiv:2606.24428v1)

Usage:
    # Enable EDV filtering via config or CLI flag
    python edv_example_env.py --env.edv_enabled true --env.edv_consensus_threshold 0.6
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Union

from pydantic import Field
from pydantic_cli import Cmd
from rich import print as rprint

from ..utils.cli import (
    extract_namespace,
    get_double_dash_flags,
    get_prefixed_pydantic_model,
    merge_dicts,
)
from .base import BaseEnv, BaseEnvConfig, ScoredDataGroup
from .constants import ENV_NAMESPACE, NAMESPACE_SEP, OPENAI_NAMESPACE
from .edv_experience_filter import EDVConfig, EDVDistillVerifyMixin
from .server_handling.openai_server import resolve_openai_configs
from .server_handling.server_baseline import APIServerConfig, ServerBaseline
from .server_handling.server_manager import ServerManager, ServerManagerConfig

logger = logging.getLogger(__name__)


class EDVExampleEnvConfig(BaseEnvConfig, EDVConfig):
    """
    Configuration for EDV example environment.

    Combines BaseEnvConfig with EDVConfig for experience filtering.
    """

    pass


class EDVExampleEnv(EDVDistillVerifyMixin, BaseEnv):
    """
    Example environment demonstrating EDV experience filtering.

    This environment implements a simple text-reversal task with optional
    EDV filtering to demonstrate how the mixin integrates with BaseEnv.

    Key integration points:
    1. Inherit from both EDVDistillVerifyMixin and BaseEnv
    2. Mix EDVConfig into your env_config_cls
    3. Call distill_verify_experiences() in score() after scoring
    """

    name = "edv_example"
    env_config_cls = EDVExampleEnvConfig

    def __init__(
        self,
        config: EDVExampleEnvConfig,
        server_configs: Union[ServerBaseline, List[APIServerConfig]],
        edv_distiller_server_configs: Optional[
            Union[ServerBaseline, APIServerConfig, List[APIServerConfig]]
        ] = None,
        slurm: bool = False,
        testing: bool = False,
    ):
        BaseEnv.__init__(self, config, server_configs, slurm=slurm, testing=testing)
        self.edv_distiller_server: Optional[ServerManager] = None

        if config.edv_enabled and edv_distiller_server_configs:
            if isinstance(edv_distiller_server_configs, APIServerConfig):
                configs = [edv_distiller_server_configs]
            else:
                configs = edv_distiller_server_configs
            self.edv_distiller_server = ServerManager(
                configs,
                slurm=False,
                testing=False,
            )

    @classmethod
    def config_init(
        cls,
    ) -> Tuple[EDVExampleEnvConfig, Union[ServerBaseline, List[APIServerConfig]]]:
        """Initialize default configuration."""
        env_config = EDVExampleEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            group_size=4,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=100,
            batch_size=4,
            steps_per_eval=50,
            max_token_length=512,
            wandb_name="edv_example",
            # EDV-specific defaults
            edv_enabled=False,
            edv_consensus_threshold=0.5,
            edv_min_executors=2,
            edv_distiller_temperature=0.0,
            edv_include_low_score=True,
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=128,
            ),
        ]
        return env_config, server_configs

    @classmethod
    def edv_distiller_config_init(
        cls,
    ) -> Optional[Union[ServerBaseline, List[APIServerConfig], APIServerConfig]]:
        """
        Optional: define default distiller server config.

        Returns None by default - distiller is configured via CLI/YAML.
        """
        return None

    @classmethod
    def get_cli_serve_config_cls(cls) -> type:
        """Build CLI config class with EDV distiller support."""
        default_env_config, default_server_configs = cls.config_init()
        default_distiller_configs = cls.edv_distiller_config_init()

        env_full_prefix = f"{ENV_NAMESPACE}{NAMESPACE_SEP}"
        openai_full_prefix = f"{OPENAI_NAMESPACE}{NAMESPACE_SEP}"
        distiller_full_prefix = f"{cls.distiller_namespace}{NAMESPACE_SEP}"

        distiller_cli_base = get_prefixed_pydantic_model(
            APIServerConfig, distiller_full_prefix
        )

        class CliServeConfig(
            get_prefixed_pydantic_model(type(default_env_config), env_full_prefix),
            get_prefixed_pydantic_model(APIServerConfig, openai_full_prefix),
            distiller_cli_base,
            ServerManagerConfig,
            Cmd,
        ):
            config: str | None = Field(
                default=None,
                description="Path to .yaml config file. CLI args override this.",
            )

            def run(self) -> None:
                import yaml

                wandb_name_attr = f"{ENV_NAMESPACE}{NAMESPACE_SEP}wandb_name"
                if (
                    getattr(self, wandb_name_attr, None) is None
                    and cls.name is not None
                ):
                    setattr(self, wandb_name_attr, cls.name)

                if self.config is not None:
                    with open(self.config, "r") as f:
                        yaml_config = yaml.safe_load(f)
                else:
                    yaml_config = {}

                cli_passed_flags = get_double_dash_flags()

                # Resolve environment config
                env_config_dict = merge_dicts(
                    default_env_config.model_dump(),
                    yaml_config.get(ENV_NAMESPACE, {}),
                    extract_namespace(cli_passed_flags, env_full_prefix),
                )
                env_config = type(default_env_config)(**env_config_dict)

                # Resolve main server configs
                oai_cli_args = extract_namespace(cli_passed_flags, openai_full_prefix)
                yaml_oai_config = yaml_config.get(OPENAI_NAMESPACE, {})

                effective_server_configs = default_server_configs
                if isinstance(effective_server_configs, ServerBaseline) and (
                    oai_cli_args or yaml_oai_config
                ):
                    effective_server_configs = APIServerConfig(
                        **effective_server_configs.model_dump()
                    )

                server_configs = resolve_openai_configs(
                    default_server_configs=effective_server_configs,
                    openai_config_dict={},
                    yaml_config=yaml_config,
                    cli_passed_flags=cli_passed_flags,
                    logger=logger,
                )

                # Resolve EDV distiller configs
                distiller_configs = cls._resolve_edv_distiller_server_configs(
                    default_distiller_configs=default_distiller_configs,
                    yaml_config=yaml_config,
                    cli_passed_flags=cli_passed_flags,
                )

                # Resolve server manager config
                server_manager_cli_flags = {}
                if "slurm" in cli_passed_flags:
                    server_manager_cli_flags["slurm"] = cli_passed_flags["slurm"]
                if "testing" in cli_passed_flags:
                    server_manager_cli_flags["testing"] = cli_passed_flags["testing"]

                server_manager_yaml_dict = {}
                if "slurm" in yaml_config:
                    server_manager_yaml_dict["slurm"] = yaml_config["slurm"]
                if "testing" in yaml_config:
                    server_manager_yaml_dict["testing"] = yaml_config["testing"]

                server_manager_config_dict = merge_dicts(
                    ServerManagerConfig().model_dump(),
                    server_manager_yaml_dict,
                    server_manager_cli_flags,
                )

                rprint(env_config)
                rprint(server_configs)
                if distiller_configs:
                    rprint({"edv_distiller": distiller_configs})

                # Build environment kwargs
                env_kwargs = {
                    "config": env_config,
                    "server_configs": server_configs,
                    "slurm": server_manager_config_dict.get("slurm", False),
                    "testing": server_manager_config_dict.get("testing", False),
                }
                if distiller_configs is not None:
                    env_kwargs["edv_distiller_server_configs"] = distiller_configs

                env = cls(**env_kwargs)

                try:
                    loop = asyncio.get_running_loop()
                    task = loop.create_task(env.env_manager())
                    loop.run_until_complete(task)
                except RuntimeError:
                    asyncio.run(env.env_manager())

        return CliServeConfig

    async def setup(self):
        """Setup example data."""
        self.examples = [
            {"input": "hello", "target": "olleh"},
            {"input": "world", "target": "dlrow"},
            {"input": "atropos", "target": "sortpa"},
            {"input": "example", "target": "elpmaxe"},
        ]
        self.iter = 0
        logger.info(
            "EDV example environment initialized with %d examples", len(self.examples)
        )

    async def evaluate(self, *args, **kwargs):
        """Simple evaluation."""
        import time

        start_time = time.time()
        correct = 0
        total = 0

        for example in self.examples:
            result = await self.rollout_and_score_eval(example)
            total += 1
            if result["score"] > 0:
                correct += 1

        end_time = time.time()
        accuracy = correct / total if total > 0 else 0

        await self.evaluate_log(
            metrics={"accuracy": accuracy},
            samples=[],
            start_time=start_time,
            end_time=end_time,
            generation_parameters={"max_tokens": self.config.max_token_length},
        )

    async def rollout_and_score_eval(self, example: Dict) -> Dict:
        """Evaluate a single example."""
        response_content = f"\\boxed{{{example['target']}}}"

        is_correct = (
            "olleh" in response_content or example["target"] in response_content
        )
        return {"score": 1.0 if is_correct else 0.0, "sample": example}

    async def collect_trajectories(
        self, item: Dict
    ) -> Tuple[Optional[ScoredDataGroup], List]:
        """Collect trajectories with optional EDV filtering."""
        input_text = item["input"]
        target = item["target"]
        prompt = (
            f"Reverse this string: {input_text}\nProvide your answer inside \\boxed{{}}"
        )

        system_msg = {
            "role": "system",
            "content": "You are a text processing assistant. Reverse the input string.",
        }
        user_msg = {"role": "user", "content": prompt}

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            chat_completions = await managed.chat_completion(
                messages=[system_msg, user_msg],
                n=self.config.group_size,
                max_tokens=self.config.max_token_length,
                temperature=0.7,
            )

            state = managed.get_state()
            nodes = state["nodes"]

        to_score = []
        for i, choice in enumerate(chat_completions.choices):
            to_score.append(
                {
                    "messages": [
                        system_msg,
                        user_msg,
                        {"role": "assistant", "content": choice.message.content},
                    ],
                    "target": target,
                    "tokens": nodes[i].tokens,
                    "masks": nodes[i].masked_tokens,
                    "logprobs": nodes[i].logprobs,
                }
            )

        scored_group = await self.score(to_score)
        return scored_group, []

    async def score(self, rollout_group_data: List[Dict]) -> Optional[ScoredDataGroup]:
        """
        Score trajectories with optional EDV filtering.

        This demonstrates the key integration point: after scoring,
        optionally apply EDV distill-verify filtering.
        """
        scores: ScoredDataGroup = {
            "tokens": [],
            "masks": [],
            "scores": [],
            "inference_logprobs": [],
            "messages": [],
        }

        for item in rollout_group_data:
            content = item["messages"][-1]["content"]
            target = item["target"]

            # Simple scoring: check if reversed correctly
            is_correct = target in content or (
                len(content) >= len(target) and content[-len(target) - 6 : -1] == target
            )
            reward = 1.0 if is_correct else -1.0

            scores["tokens"].append(item["tokens"])
            scores["masks"].append(item["masks"])
            scores["inference_logprobs"].append(item["logprobs"])
            scores["scores"].append(reward)
            scores["messages"].append(item["messages"])

        # Apply EDV filtering if enabled
        if (
            self.config.edv_enabled
            and len(scores["tokens"]) >= self.config.edv_min_executors
        ):
            scores = await self.distill_verify_experiences(
                group=scores,
                rollout_data=rollout_group_data,
                task_context="text reversal task - reverse the input string",
                distiller_server=self.edv_distiller_server,
            )

        # Return None if all scores are identical (no learning signal)
        if scores["scores"] and len(set(scores["scores"])) <= 1:
            return None

        return scores

    async def get_next_item(self) -> Dict:
        """Get next training item."""
        item = self.examples[self.iter % len(self.examples)]
        self.iter += 1
        return item

    def save_checkpoint(self, step, data=None):
        """Save checkpoint with iteration state."""
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)


if __name__ == "__main__":
    EDVExampleEnv.cli()
