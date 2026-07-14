"""
Inverse Dynamics Reward Function for EnvRL auxiliary objectives.

This module implements the inverse dynamics auxiliary objective from EnvRL,
which encourages the agent to learn environment dynamics by predicting the
action that led to a state transition.

In the context of LLM agents:
- State: message history / observations
- Action: model response
- Next state: next observation

The reward function measures how well the agent can predict the action that
was taken to transition from one state to another, which helps internalize
environment dynamics and the effect of actions.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from .reward_function import RewardFunction
from .registry import registry

logger = logging.getLogger(__name__)


@registry.register
class InverseDynamicsReward(RewardFunction):
    """
    Reward function that encourages inverse dynamics learning for environment dynamics.

    This implements the inverse dynamics auxiliary objective from EnvRL. Given a
    state transition (state -> next_state), this reward function evaluates how
    well the agent can predict the action that was taken to cause this transition.

    The prediction quality is measured using:
    - Similarity between predicted and actual action
    - Optional consideration of action sequence consistency

    This auxiliary objective helps the agent understand the causal relationship
    between actions and state transitions, improving planning and generalization.
    """

    def __init__(
        self,
        weight: float = 0.1,
        use_embeddings: bool = False,
        sequence_consistency_weight: float = 0.2,
        **kwargs,
    ):
        """
        Initialize the inverse dynamics reward function.

        Args:
            weight: Importance factor for this auxiliary objective (default 0.1,
                   as this is typically used alongside a primary reward)
            use_embeddings: Whether to use embedding-based similarity (requires
                          sentence-transformers). Falls back to text overlap if False.
            sequence_consistency_weight: Weight for sequence consistency bonus
            **kwargs: Additional configuration
        """
        super().__init__(weight=weight, **kwargs)
        self.use_embeddings = use_embeddings
        self.sequence_consistency_weight = sequence_consistency_weight

        # Lazy load embedding model if needed
        self._embedding_model = None

    def _get_embedding_model(self):
        """Lazy load sentence transformer model."""
        if self._embedding_model is None and self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded sentence transformer for inverse dynamics")
            except ImportError:
                logger.warning(
                    "sentence-transformers not available, falling back to text overlap"
                )
                self.use_embeddings = False
        return self._embedding_model

    def _compute_text_overlap(self, text1: str, text2: str) -> float:
        """
        Compute text overlap similarity between two texts.

        Uses word-level Jaccard similarity as a simple proxy for semantic similarity.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _compute_embedding_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between embeddings of two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        model = self._get_embedding_model()
        if model is None:
            return self._compute_text_overlap(text1, text2)

        try:
            emb1 = model.encode(text1, convert_to_tensor=True)
            emb2 = model.encode(text2, convert_to_tensor=True)

            import torch
            similarity = torch.nn.functional.cosine_similarity(
                emb1.unsqueeze(0), emb2.unsqueeze(0)
            ).item()

            # Scale from [-1, 1] to [0, 1]
            return (similarity + 1) / 2
        except Exception as e:
            logger.warning(f"Error computing embedding similarity: {e}")
            return self._compute_text_overlap(text1, text2)

    def _extract_action_from_trajectory(
        self, trajectory: Dict[str, Any]
    ) -> Optional[str]:
        """
        Extract action from trajectory data.

        Handles different trajectory formats:
        - Dict with 'action' key
        - Dict with 'response' or 'completion' key

        Args:
            trajectory: Trajectory data

        Returns:
            Extracted action as string, or None if not found
        """
        if isinstance(trajectory, dict):
            for key in ["action", "response", "completion", "model_output"]:
                if key in trajectory and trajectory[key]:
                    return str(trajectory[key])

        return None

    def _extract_state_from_trajectory(
        self, trajectory: Dict[str, Any]
    ) -> Optional[str]:
        """
        Extract state representation from trajectory data.

        Args:
            trajectory: Trajectory data

        Returns:
            Extracted state as string, or None if not found
        """
        if isinstance(trajectory, dict):
            for key in ["state", "observation", "context"]:
                if key in trajectory and trajectory[key]:
                    return str(trajectory[key])

        return None

    def _compute_sequence_consistency(
        self,
        current_action: str,
        previous_actions: List[str],
    ) -> float:
        """
        Compute sequence consistency bonus.

        Rewards actions that are consistent with previous action patterns.
        This helps the agent learn coherent action sequences.

        Args:
            current_action: Current action to evaluate
            previous_actions: List of previous actions in the trajectory

        Returns:
            Consistency bonus between 0 and 1
        """
        if not previous_actions:
            return 0.0

        # Compute average similarity with previous actions
        similarities = []
        for prev_action in previous_actions[-3:]:  # Look at last 3 actions
            if self.use_embeddings:
                sim = self._compute_embedding_similarity(current_action, prev_action)
            else:
                sim = self._compute_text_overlap(current_action, prev_action)
            similarities.append(sim)

        # Return average similarity (with some penalty for being too similar,
        # as we want diversity within coherence)
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # Bonus for moderate similarity (0.2-0.6) - coherence without repetition
        if 0.2 <= avg_similarity <= 0.6:
            return avg_similarity
        elif avg_similarity < 0.2:
            # Too different, might be incoherent
            return avg_similarity * 0.5
        else:
            # Too similar, might be repetitive
            return (1.0 - avg_similarity) * 0.5

    def compute(
        self,
        completions: List[Any],
        trajectories: Optional[List[Dict[str, Any]]] = None,
        actions: Optional[List[str]] = None,
        action_history: Optional[List[List[str]]] = None,
        **kwargs,
    ) -> List[float]:
        """
        Compute inverse dynamics rewards.

        Evaluates how well the agent can predict the action that led to a
        state transition (from current state to next state).

        Args:
            completions: List of model completions (predicted actions)
            trajectories: Optional list of trajectory data containing state-action pairs
            actions: Optional explicit list of actual actions taken
            action_history: Optional list of action histories for sequence consistency
            **kwargs: Additional context (may include 'states', 'next_states', etc.)

        Returns:
            List of reward scores based on prediction quality
        """
        rewards = []

        # Extract states from various possible kwargs
        states = kwargs.get("states", kwargs.get("observations", None))
        next_states = kwargs.get("next_states", kwargs.get("next_observations", None))

        # Ensure we have lists of the right length
        if states is not None and not isinstance(states, list):
            states = [states] * len(completions)
        if next_states is not None and not isinstance(next_states, list):
            next_states = [next_states] * len(completions)
        if actions is not None and not isinstance(actions, list):
            actions = [actions] * len(completions)
        if action_history is not None and not isinstance(action_history, list):
            action_history = [action_history] * len(completions)

        # Compute rewards
        for i, completion in enumerate(completions):
            try:
                predicted_action = self.get_content(completion)

                # Get the actual action if available
                actual_action = None
                if actions is not None and i < len(actions):
                    actual_action = actions[i]
                elif trajectories is not None and i < len(trajectories):
                    actual_action = self._extract_action_from_trajectory(trajectories[i])

                # For inverse dynamics, we compare predicted vs actual action
                # The state transition context provides additional signal
                state_context = ""
                if states is not None and i < len(states):
                    state_context += str(states[i]) + " "
                if next_states is not None and i < len(next_states):
                    state_context += "-> " + str(next_states[i])

                # Base reward: similarity between predicted and actual action
                if actual_action is not None:
                    actual_action_str = str(actual_action)

                    if self.use_embeddings:
                        similarity = self._compute_embedding_similarity(
                            predicted_action, actual_action_str
                        )
                    else:
                        similarity = self._compute_text_overlap(
                            predicted_action, actual_action_str
                        )

                    base_reward = similarity
                else:
                    # Fallback: reward based on action quality given state context
                    # This provides a weak signal when ground truth action is unavailable
                    if state_context and len(predicted_action) > 0:
                        # Reward for non-empty action in context of state transition
                        base_reward = min(1.0, len(predicted_action) / 50.0)
                    else:
                        base_reward = 0.0

                # Add sequence consistency bonus if action history is available
                consistency_bonus = 0.0
                if action_history is not None and i < len(action_history):
                    previous_actions = action_history[i]
                    if previous_actions:
                        consistency_bonus = self._compute_sequence_consistency(
                            predicted_action, previous_actions
                        )

                # Combine base reward and consistency bonus
                reward = (
                    base_reward * (1 - self.sequence_consistency_weight) +
                    consistency_bonus * self.sequence_consistency_weight
                )

            except Exception as e:
                logger.warning(f"Error in inverse_dynamics_reward: {e}")
                logger.exception(e)
                reward = 0.0

            rewards.append(reward)

        return rewards


# Legacy function wrapper for backward compatibility
def inverse_dynamics_reward(
    completions: List[Any],
    trajectories: Optional[List[Dict[str, Any]]] = None,
    actions: Optional[List[str]] = None,
    action_history: Optional[List[List[str]]] = None,
    **kwargs,
) -> List[float]:
    """
    Legacy function wrapper for InverseDynamicsReward.

    Args:
        completions: List of model completions (predicted actions)
        trajectories: Optional list of trajectory data
        actions: Optional list of actual actions taken
        action_history: Optional list of action histories
        **kwargs: Additional context

    Returns:
        List of reward scores
    """
    reward_fn = InverseDynamicsReward()
    return reward_fn.compute(
        completions,
        trajectories=trajectories,
        actions=actions,
        action_history=action_history,
        **kwargs
    )
