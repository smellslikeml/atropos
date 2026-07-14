"""
Consensus-based reward function implementing the EDV (Execute-Distill-Verify) paradigm.

This module provides a reward function that implements collaborative verification
for agentic experience learning, adapted from the EDV framework:
https://arxiv.org/abs/2606.24428v1

The EDV paradigm addresses the self-confirmation trap by:
1. Execute: Multiple agents explore the same task space in parallel
2. Distill: Comparative analysis identifies consistent patterns across candidates
3. Verify: Consensus mechanism validates candidates before memory insertion

This adapted implementation uses heuristic-based consensus verification suitable
for SQL query generation and other structured output tasks.
"""

import logging
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter

from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)


def normalize_sql_query(query: str) -> str:
    """
    Normalize SQL query for comparison.

    - Convert to lowercase
    - Remove extra whitespace
    - Normalize keywords

    Args:
        query: SQL query string

    Returns:
        Normalized SQL query
    """
    if not query:
        return ""

    # Basic normalization
    normalized = query.lower().strip()
    # Replace multiple spaces with single space
    normalized = re.sub(r'\s+', ' ', normalized)
    # Remove trailing semicolon if present
    normalized = normalized.rstrip(';').strip()

    return normalized


def extract_sql_structure(query: str) -> Dict[str, Any]:
    """
    Extract structural components from a SQL query.

    Args:
        query: SQL query string

    Returns:
        Dictionary with structural components
    """
    structure = {
        'type': None,
        'columns': [],
        'tables': [],
        'has_join': False,
        'has_where': False,
        'has_aggregation': False,
        'has_order_by': False,
    }

    if not query:
        return structure

    query_lower = query.lower()

    # Extract query type
    if 'select' in query_lower:
        structure['type'] = 'SELECT'
    elif 'insert' in query_lower:
        structure['type'] = 'INSERT'
    elif 'update' in query_lower:
        structure['type'] = 'UPDATE'
    elif 'delete' in query_lower:
        structure['type'] = 'DELETE'

    # Check for structural elements
    structure['has_join'] = 'join' in query_lower
    structure['has_where'] = 'where' in query_lower
    structure['has_aggregation'] = any(
        agg in query_lower for agg in ['count', 'sum', 'avg', 'max', 'min', 'group by']
    )
    structure['has_order_by'] = 'order by' in query_lower

    return structure


def compute_structural_similarity(struct1: Dict[str, Any], struct2: Dict[str, Any]) -> float:
    """
    Compute similarity between two SQL structures.

    Args:
        struct1: First SQL structure
        struct2: Second SQL structure

    Returns:
        Similarity score between 0 and 1
    """
    if struct1.get('type') != struct2.get('type'):
        return 0.0

    similarity = 1.0

    # Penalize differences in structural elements
    for key in ['has_join', 'has_where', 'has_aggregation', 'has_order_by']:
        if struct1.get(key) != struct2.get(key):
            similarity -= 0.15

    return max(0.0, similarity)


def extract_boxed_sql(text: str) -> Optional[str]:
    """
    Extract SQL query from boxed format.

    Args:
        text: Text potentially containing boxed SQL

    Returns:
        Extracted SQL query or None
    """
    if not text:
        return None

    # Try to extract from \boxed{} format
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()

    # Try to extract from code blocks
    match = re.search(r'```(?:sql)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


@registry.register
class ConsensusReward(RewardFunction):
    """
    Reward function implementing EDV-style consensus verification.

    This reward function analyzes multiple candidate outputs and computes
    scores based on:
    1. Consensus agreement across candidates
    2. Structural consistency for structured outputs (e.g., SQL)
    3. Execution-based validation (when execution results are provided)

    The distillation stage identifies patterns across candidates, while
    the verification stage applies consensus-based filtering to reduce
    the self-confirmation trap.
    """

    def __init__(
        self,
        consensus_threshold: float = 0.5,
        structural_weight: float = 0.3,
        agreement_weight: float = 0.7,
        min_agreement_count: int = 2,
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the consensus reward function.

        Args:
            consensus_threshold: Minimum consensus score for a candidate to be considered valid
            structural_weight: Weight for structural similarity in consensus computation
            agreement_weight: Weight for cross-candidate agreement
            min_agreement_count: Minimum number of candidates that must agree for consensus
            weight: Overall weight for this reward function
            **kwargs: Additional configuration
        """
        super().__init__(weight=weight, **kwargs)
        self.consensus_threshold = consensus_threshold
        self.structural_weight = structural_weight
        self.agreement_weight = agreement_weight
        self.min_agreement_count = min_agreement_count

    def _extract_candidates(
        self, completions: List[Any]
    ) -> List[Tuple[str, Optional[str]]]:
        """
        Extract candidate queries from completions.

        Args:
            completions: List of completions

        Returns:
            List of (content, extracted_sql) tuples
        """
        candidates = []

        for completion in completions:
            content = self.get_content(completion)
            sql_query = extract_boxed_sql(content)
            candidates.append((content, sql_query))

        return candidates

    def _compute_pairwise_agreement(
        self, candidates: List[Tuple[str, Optional[str]]]
    ) -> List[List[float]]:
        """
        Compute pairwise agreement scores between candidates.

        Args:
            candidates: List of (content, extracted_sql) tuples

        Returns:
            Matrix of agreement scores
        """
        n = len(candidates)
        agreement_matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                _, sql_i = candidates[i]
                _, sql_j = candidates[j]

                if sql_i and sql_j:
                    # Normalize and compare
                    norm_i = normalize_sql_query(sql_i)
                    norm_j = normalize_sql_query(sql_j)

                    if norm_i == norm_j:
                        agreement_matrix[i][j] = 1.0
                        agreement_matrix[j][i] = 1.0
                    else:
                        # Compute structural similarity
                        struct_i = extract_sql_structure(norm_i)
                        struct_j = extract_sql_structure(norm_j)
                        similarity = compute_structural_similarity(struct_i, struct_j)
                        agreement_matrix[i][j] = similarity
                        agreement_matrix[j][i] = similarity

        return agreement_matrix

    def _compute_consensus_scores(
        self,
        candidates: List[Tuple[str, Optional[str]]],
        agreement_matrix: List[List[float]],
        execution_scores: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Compute consensus scores for each candidate.

        Args:
            candidates: List of (content, extracted_sql) tuples
            agreement_matrix: Pairwise agreement matrix
            execution_scores: Optional execution-based scores

        Returns:
            List of consensus scores
        """
        n = len(candidates)
        consensus_scores = []

        for i in range(n):
            # Average agreement with other candidates
            agreement_scores = [agreement_matrix[i][j] for j in range(n) if i != j]
            avg_agreement = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.0

            # Count strong agreements (above threshold)
            strong_agreements = sum(1 for score in agreement_scores if score >= self.consensus_threshold)

            # Compute weighted consensus score
            consensus_score = (
                self.agreement_weight * avg_agreement +
                self.structural_weight * (strong_agreements / max(1, n - 1))
            )

            # If we have execution scores, incorporate them
            if execution_scores and execution_scores[i] is not None:
                # Boost consensus score for candidates with good execution results
                if execution_scores[i] > 0:
                    consensus_score = max(consensus_score, 0.7)

            consensus_scores.append(consensus_score)

        return consensus_scores

    def _apply_consensus_filtering(
        self,
        consensus_scores: List[float],
        candidates: List[Tuple[str, Optional[str]]],
    ) -> List[bool]:
        """
        Apply consensus filtering to identify valid candidates.

        Args:
            consensus_scores: Consensus scores for each candidate
            candidates: List of (content, extracted_sql) tuples

        Returns:
            List of booleans indicating which candidates pass consensus
        """
        is_valid = []
        n = len(candidates)

        # Find the score threshold (adaptive based on distribution)
        sorted_scores = sorted(consensus_scores, reverse=True)
        if len(sorted_scores) >= self.min_agreement_count:
            # Use the score of the min_agreement_count-th best candidate as threshold
            threshold = max(
                self.consensus_threshold,
                sorted_scores[self.min_agreement_count - 1]
            )
        else:
            threshold = self.consensus_threshold

        for score in consensus_scores:
            is_valid.append(score >= threshold)

        return is_valid

    def compute(
        self,
        completions: List[Any],
        execution_scores: Optional[List[float]] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> List[float]:
        """
        Compute consensus-based rewards for completions.

        Args:
            completions: List of model completions to evaluate
            execution_scores: Optional execution-based scores for each completion
            ground_truth: Optional ground truth answer
            **kwargs: Additional context (schema, question, etc.)

        Returns:
            List of reward scores based on consensus analysis
        """
        if not completions:
            return []

        # Extract candidates
        candidates = self._extract_candidates(completions)

        # If only one candidate, return neutral score
        if len(candidates) == 1:
            if execution_scores and execution_scores[0] is not None:
                return [execution_scores[0]]
            return [0.5]

        # Compute pairwise agreement
        agreement_matrix = self._compute_pairwise_agreement(candidates)

        # Compute consensus scores
        consensus_scores = self._compute_consensus_scores(
            candidates, agreement_matrix, execution_scores
        )

        # Apply consensus filtering
        is_valid = self._apply_consensus_filtering(consensus_scores, candidates)

        # Generate final rewards
        rewards = []
        for i, (valid, consensus_score) in enumerate(zip(is_valid, consensus_scores)):
            if valid:
                # Valid candidates get their consensus score (possibly boosted by execution)
                reward = consensus_score
                if execution_scores and execution_scores[i] is not None:
                    # Incorporate execution score for valid candidates
                    reward = (reward + execution_scores[i]) / 2
            else:
                # Invalid candidates get low reward
                reward = -1.0

            rewards.append(reward)

        logger.info(
            f"Consensus: {sum(is_valid)}/{len(is_valid)} valid, "
            f"avg score: {sum(rewards)/len(rewards):.3f}"
        )

        return rewards


@registry.register
class EDVConsensusReward(RewardFunction):
    """
    Extended EDV consensus reward with distillation capabilities.

    This extends the basic consensus reward with explicit distillation stage
    that analyzes trajectories comparatively to identify consistent patterns
    before verification.
    """

    def __init__(
        self,
        distill_temperature: float = 0.7,
        verify_strictness: float = 0.6,
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the EDV consensus reward.

        Args:
            distill_temperature: Temperature for distillation (higher = more inclusive)
            verify_strictness: Strictness threshold for verification
            weight: Overall weight for this reward function
            **kwargs: Additional configuration
        """
        super().__init__(weight=weight, **kwargs)
        self.distill_temperature = distill_temperature
        self.verify_strictness = verify_strictness
        self.base_consensus = ConsensusReward(
            consensus_threshold=verify_strictness,
            structural_weight=0.3,
            agreement_weight=0.7,
        )

    def _distill_trajectories(
        self, candidates: List[Tuple[str, Optional[str]]]
    ) -> Dict[str, Any]:
        """
        Distill stage: Analyze trajectories comparatively.

        Args:
            candidates: List of (content, extracted_sql) tuples

        Returns:
            Distillation insights dictionary
        """
        insights = {
            'structures': [],
            'normalized_queries': [],
            'common_patterns': Counter(),
        }

        for content, sql_query in candidates:
            if sql_query:
                insights['structures'].append(extract_sql_structure(sql_query))
                insights['normalized_queries'].append(normalize_sql_query(sql_query))

        # Identify common patterns
        if insights['normalized_queries']:
            for query in insights['normalized_queries']:
                if 'select' in query:
                    insights['common_patterns']['select'] += 1
                if 'where' in query:
                    insights['common_patterns']['where'] += 1
                if 'join' in query:
                    insights['common_patterns']['join'] += 1

        return insights

    def compute(
        self,
        completions: List[Any],
        execution_scores: Optional[List[float]] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> List[float]:
        """
        Compute EDV consensus rewards with distillation.

        Args:
            completions: List of model completions to evaluate
            execution_scores: Optional execution-based scores
            ground_truth: Optional ground truth answer
            **kwargs: Additional context

        Returns:
            List of reward scores
        """
        if not completions:
            return []

        # Extract candidates
        candidates = [
            (self.get_content(c), extract_boxed_sql(self.get_content(c)))
            for c in completions
        ]

        # Distill stage: Analyze trajectories comparatively
        insights = self._distill_trajectories(candidates)

        logger.info(
            f"EDV Distill: {len(insights['normalized_queries'])} queries, "
            f"patterns: {dict(insights['common_patterns'])}"
        )

        # Verify stage: Use consensus mechanism
        rewards = self.base_consensus.compute(
            completions,
            execution_scores=execution_scores,
            ground_truth=ground_truth,
            **kwargs,
        )

        return rewards


# Legacy function for backward compatibility
def consensus_reward(
    completions: List[Any],
    execution_scores: Optional[List[float]] = None,
    **kwargs,
) -> List[float]:
    """
    Legacy function wrapper for ConsensusReward.

    Args:
        completions: List of model completions to evaluate
        execution_scores: Optional execution-based scores
        **kwargs: Additional parameters

    Returns:
        List of reward scores
    """
    reward_fn = ConsensusReward()
    return reward_fn.compute(completions, execution_scores=execution_scores, **kwargs)
