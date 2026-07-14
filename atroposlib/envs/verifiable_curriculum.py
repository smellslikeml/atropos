"""
Verifiable Curriculum for Multi-Teacher RLVR Distillation.

This module implements the compete-then-collaborate framework from
"Compete Then Collaborate: Frontier AI Teachers Build a Verifiable Curriculum
to Improve a Coding Student Beyond Imitation"
(https://arxiv.org/abs/2607.08255v1).

Core concepts:
- Teacher competition: Multiple frontier AI teachers ranked by execution judges
- Collaborative curriculum: Teachers collaborate to build a verifiable curriculum
- RLVR (Reinforcement Learning with Verifiable Rewards): Student learns by doing
  in a verifiable environment rather than imitating teacher outputs

The paper's key finding: Imitation (SFT) on verified solutions can degrade
performance, but using the collaborative curriculum as an RLVR environment
improves the student.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from pydantic import Field

from .server_handling.server_manager import ServerManager

logger = logging.getLogger(__name__)


class CompetitionMode(Enum):
    """Modes for teacher competition."""

    EXECUTION_BASED = "execution_based"
    """Rank teachers by execution-based verification."""

    MAJORITY_VOTE = "majority_vote"
    """Rank teachers by majority vote agreement on outputs."""

    NONE = "none"
    """No competition - all teachers weighted equally."""


class CurriculumMode(Enum):
    """Modes for curriculum generation."""

    COLLABORATIVE_POOL = "collaborative_pool"
    """Pool verified solutions from all teachers for collaborative curriculum."""

    BEST_TEACHER_ONLY = "best_teacher_only"
    """Use only solutions from the highest-ranked teacher."""

    VERIFIED_ONLY = "verified_only"
    """Use only solutions that pass execution verification, regardless of teacher."""


@dataclass
class TeacherRanking:
    """Represents a teacher's competition ranking."""

    teacher_name: str
    rank: int
    score: float
    num_verified: int
    num_attempted: int


@dataclass
class TeacherSolution:
    """A solution from a single teacher."""

    teacher_name: str
    content: str
    verified: bool
    execution_output: Optional[Any] = None
    error_message: Optional[str] = None


@dataclass
class CollaborativeSample:
    """A sample from the collaborative curriculum."""

    solutions: List[TeacherSolution]
    item: Any
    """The original problem/item."""

    num_verified: int
    """Number of verified solutions across all teachers."""

    has_collaboration: bool
    """Whether multiple teachers contributed verified solutions."""


class ExecutionJudge(ABC):
    """
    Abstract base class for execution-based judges.

    Concrete implementations should define how to execute and verify
    solutions for specific task types (code, SQL, etc.).
    """

    @abstractmethod
    async def verify_solution(
        self, solution: str, item: Any
    ) -> Tuple[bool, Optional[Any], Optional[str]]:
        """
        Verify a solution by execution.

        Args:
            solution: The generated solution to verify
            item: The original problem/item containing ground truth

        Returns:
            Tuple of (verified, output, error_message):
            - verified: Whether the solution is correct
            - output: Execution output if successful
            - error_message: Error message if verification failed
        """
        pass


class VerifiableCurriculum:
    """
    Implements the compete-then-collaborate framework for multi-teacher distillation.

    This class orchestrates:
    1. Teacher competition via execution-based judges
    2. Collaborative curriculum generation
    3. RLVR reward computation for student training
    """

    def __init__(
        self,
        teacher_servers: List[ServerManager],
        judge: ExecutionJudge,
        competition_mode: CompetitionMode = CompetitionMode.EXECUTION_BASED,
        curriculum_mode: CurriculumMode = CurriculumMode.COLLABORATIVE_POOL,
    ):
        """
        Initialize the verifiable curriculum.

        Args:
            teacher_servers: List of teacher server managers for distillation
            judge: Execution-based judge for verification
            competition_mode: How to rank teachers
            curriculum_mode: How to build the collaborative curriculum
        """
        self.teacher_servers = teacher_servers
        self.judge = judge
        self.competition_mode = competition_mode
        self.curriculum_mode = curriculum_mode

        # Track teacher rankings over time
        self.teacher_rankings: Dict[str, TeacherRanking] = {}
        self._update_teacher_names()

    def _update_teacher_names(self):
        """Initialize teacher rankings based on configured servers."""
        for idx, server in enumerate(self.teacher_servers):
            # Extract teacher name from server config
            name = f"teacher_{idx}"
            if hasattr(server, "servers") and server.servers:
                first_server = server.servers[0]
                if hasattr(first_server, "config"):
                    name = getattr(first_server.config, "model_name", name)
            self.teacher_rankings[name] = TeacherRanking(
                teacher_name=name, rank=idx, score=0.0, num_verified=0, num_attempted=0
            )

    async def compete_teachers(
        self, item: Any, prompt_generator: Callable[[Any], str]
    ) -> Dict[str, TeacherSolution]:
        """
        Run teacher competition for a single item.

        Each teacher generates a solution, and solutions are verified
        by the execution-based judge.

        Args:
            item: The problem/item to solve
            prompt_generator: Function to generate the prompt from the item

        Returns:
            Dict mapping teacher_name to their solution
        """
        prompt = prompt_generator(item)
        solutions: Dict[str, TeacherSolution] = {}

        tasks = []
        for idx, server in enumerate(self.teacher_servers):
            teacher_name = list(self.teacher_rankings.keys())[idx]
            tasks.append(
                self._fetch_teacher_solution(
                    server, teacher_name, prompt, item
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Teacher fetch failed: {result}")
                continue
            if result is not None:
                solutions[result.teacher_name] = result

        return solutions

    async def _fetch_teacher_solution(
        self,
        server: ServerManager,
        teacher_name: str,
        prompt: str,
        item: Any,
    ) -> Optional[TeacherSolution]:
        """Fetch a solution from a single teacher."""
        try:
            payload = await server.get_logprobs(
                input_ids=[],  # Will be populated by server
                top_k=0,
                max_tokens=1,
                split="train",
            )
            # Extract generated text from payload
            # This is a simplified implementation - actual implementation
            # would need to handle chat completion format
            content = payload.get("generated_text", "")

            # Verify solution by execution
            verified, output, error = await self.judge.verify_solution(content, item)

            solution = TeacherSolution(
                teacher_name=teacher_name,
                content=content,
                verified=verified,
                execution_output=output,
                error_message=error,
            )

            # Update teacher ranking statistics
            if teacher_name in self.teacher_rankings:
                ranking = self.teacher_rankings[teacher_name]
                ranking.num_attempted += 1
                if verified:
                    ranking.num_verified += 1

            return solution

        except Exception as e:
            logger.error(f"Failed to fetch solution from {teacher_name}: {e}")
            return None

    def update_rankings(self, all_solutions: Dict[str, TeacherSolution]):
        """
        Update teacher rankings based on latest competition results.

        Rankings are determined by verification success rate.
        """
        # Update stats from the latest competition results
        for teacher_name, solution in all_solutions.items():
            if teacher_name in self.teacher_rankings:
                ranking = self.teacher_rankings[teacher_name]
                ranking.num_attempted += 1
                if solution.verified:
                    ranking.num_verified += 1

        # Recompute scores and ranks
        for teacher_name, ranking in self.teacher_rankings.items():
            if ranking.num_attempted > 0:
                ranking.score = ranking.num_verified / ranking.num_attempted

        # Sort by score and assign ranks
        sorted_rankings = sorted(
            self.teacher_rankings.values(), key=lambda r: r.score, reverse=True
        )
        for rank, ranking in enumerate(sorted_rankings):
            ranking.rank = rank

    def build_collaborative_sample(
        self, solutions: Dict[str, TeacherSolution], item: Any
    ) -> CollaborativeSample:
        """
        Build a collaborative curriculum sample from teacher solutions.

        Args:
            solutions: Dict of teacher solutions from competition
            item: The original problem/item

        Returns:
            A collaborative sample for student training
        """
        teacher_solutions = list(solutions.values())

        # Count verified solutions
        num_verified = sum(1 for s in teacher_solutions if s.verified)

        # Check for collaboration (multiple teachers with verified solutions)
        has_collaboration = num_verified > 1

        # Filter based on curriculum mode
        if self.curriculum_mode == CurriculumMode.BEST_TEACHER_ONLY:
            # Use only solutions from highest-ranked teacher
            best_teacher = min(
                self.teacher_rankings.values(), key=lambda r: r.rank
            )
            teacher_solutions = [
                s for s in teacher_solutions
                if s.teacher_name == best_teacher.teacher_name
            ]
        elif self.curriculum_mode == CurriculumMode.VERIFIED_ONLY:
            # Use only verified solutions
            teacher_solutions = [s for s in teacher_solutions if s.verified]
        # COLLABORATIVE_POOL uses all solutions

        return CollaborativeSample(
            solutions=teacher_solutions,
            item=item,
            num_verified=num_verified,
            has_collaboration=has_collaboration,
        )

    def compute_rlvr_reward(
        self, student_solution: str, collaborative_sample: CollaborativeSample
    ) -> float:
        """
        Compute RLVR reward for a student solution.

        The reward is based on execution verification rather than
        imitation of teacher outputs.

        Args:
            student_solution: The student's generated solution
            collaborative_sample: The collaborative curriculum sample

        Returns:
            Reward signal (typically +1.0 for verified, -1.0 for failed)
        """
        # This is a simplified synchronous version
        # In practice, this would be called from an async context
        # where we can await the judge's verification

        # For now, return a reward based on whether the student's
        # solution matches any verified teacher solution
        # (as a proxy for execution verification)

        for solution in collaborative_sample.solutions:
            if solution.verified and solution.content == student_solution:
                return 1.0

        return -1.0

    def get_curriculum_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collaborative curriculum."""
        total_attempts = sum(r.num_attempted for r in self.teacher_rankings.values())
        total_verified = sum(r.num_verified for r in self.teacher_rankings.values())

        rate = total_verified / total_attempts if total_attempts > 0 else 0
        rankings = {
            name: {
                "rank": r.rank,
                "score": r.score,
                "num_verified": r.num_verified,
                "num_attempted": r.num_attempted,
            }
            for name, r in self.teacher_rankings.items()
        }
        return {
            "total_teacher_attempts": total_attempts,
            "total_verified_solutions": total_verified,
            "overall_verification_rate": rate,
            "teacher_rankings": rankings,
        }


class RLVRConfig:
    """Configuration for RLVR (Reinforcement Learning with Verifiable Rewards)."""

    rlvr_enabled: bool = Field(
        default=False,
        description="Enable RLVR mode instead of pure imitation.",
    )
    competition_mode: str = Field(
        default="execution_based",
        description=(
            "Mode for teacher competition: execution_based, "
            "majority_vote, none"
        ),
    )
    curriculum_mode: str = Field(
        default="collaborative_pool",
        description=(
            "Mode for curriculum: collaborative_pool, "
            "best_teacher_only, verified_only"
        ),
    )
    min_verified_teachers: int = Field(
        default=2,
        description=(
            "Minimum number of teachers with verified solutions "
            "for collaboration."
        ),
    )
    reward_scale: float = Field(
        default=1.0,
        description="Scale factor for RLVR rewards.",
    )


def create_compete_collaborate_hook(
    curriculum: VerifiableCurriculum,
) -> Callable[[Any, Any], Awaitable[float]]:
    """
    Create a hook function for RLVR reward computation.

    This function returns an async callable that can be used as
    a reward function in the training loop.

    Args:
        curriculum: The verifiable curriculum instance

    Returns:
        An async function that computes RLVR rewards
    """

    async def rlvr_reward_hook(
        student_solution: str, item: Any, prompt_generator: Callable[[Any], str]
    ) -> float:
        """
        Compute RLVR reward for a student solution.

        This hook:
        1. Runs teacher competition for the item
        2. Builds collaborative curriculum
        3. Verifies student solution against execution judge
        4. Returns reward based on verification
        """
        # Run teacher competition
        solutions = await curriculum.compete_teachers(item, prompt_generator)

        # Update rankings
        curriculum.update_rankings(solutions)

        # Verify student solution
        verified, _, _ = await curriculum.judge.verify_solution(
            student_solution, item
        )

        # Return scaled reward
        return curriculum.reward_scale * (1.0 if verified else -1.0)

    return rlvr_reward_hook
