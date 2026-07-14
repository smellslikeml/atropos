"""
State Prediction Reward Function for EnvRL auxiliary objectives.

This module implements the state prediction auxiliary objective from EnvRL,
which encourages the agent to learn environment dynamics by predicting the next
state given the current state and action.

In the context of LLM agents:
- State: message history / observations
- Action: model response
- Next state: next observation

The reward function measures how well the agent can predict the next state,
which helps internalize environment dynamics.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from .reward_function import RewardFunction
from .registry import registry

logger = logging.getLogger(__name__)


@registry.register
class StatePredictionReward(RewardFunction):
    """
    Reward function that encourages state prediction for learning environment dynamics.

    This implements the state prediction auxiliary objective from EnvRL. Given a
    trajectory of (state, action, next_state) transitions, this reward function
    evaluates how well the agent can predict the next state from the current
    state and action.

    The prediction quality is measured using:
    - Text similarity (overlap) between predicted and actual next state
    - Optional embedding-based similarity (requires sentence-transformers)

    This auxiliary objective helps the agent build an internal model of the
    environment dynamics, which improves generalization and sample efficiency.
    """

    def __init__(
        self,
        weight: float = 0.1,
        use_embeddings: bool = False,
        min_prediction_length: int = 10,
        **kwargs,
    ):
        """
        Initialize the state prediction reward function.

        Args:
            weight: Importance factor for this auxiliary objective (default 0.1,
                   as this is typically used alongside a primary reward)
            use_embeddings: Whether to use embedding-based similarity (requires
                          sentence-transformers). Falls back to text overlap if False.
            min_prediction_length: Minimum length of state to use for prediction
            **kwargs: Additional configuration
        """
        super().__init__(weight=weight, **kwargs)
        self.use_embeddings = use_embeddings
        self.min_prediction_length = min_prediction_length

        # Lazy load embedding model if needed
        self._embedding_model = None

    def _get_embedding_model(self):
        """Lazy load sentence transformer model."""
        if self._embedding_model is None and self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded sentence transformer for state prediction")
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

    def _extract_state_from_trajectory(
        self, trajectory: Dict[str, Any]
    ) -> Optional[str]:
        """
        Extract state representation from trajectory data.

        Handles different trajectory formats:
        - List of messages (chat format)
        - Dict with 'messages' key
        - Dict with 'state' or 'observation' key

        Args:
            trajectory: Trajectory data

        Returns:
            Extracted state as string, or None if not found
        """
        if isinstance(trajectory, dict):
            # Try different keys for state/observation
            for key in ["state", "observation", "context", "prompt"]:
                if key in trajectory and trajectory[key]:
                    return str(trajectory[key])

            # Try messages format
            if "messages" in trajectory and isinstance(trajectory["messages"], list):
                messages = trajectory["messages"]
                if len(messages) > 0:
                    # Concatenate all messages for context
                    return " ".join([
                        msg.get("content", "")
                        for msg in messages
                        if msg.get("content")
                    ])

        elif isinstance(trajectory, list) and len(trajectory) > 0:
            # List of messages
            return " ".join([
                (msg.get("content") if isinstance(msg, dict) else str(msg))
                for msg in trajectory
                if (msg.get("content") if isinstance(msg, dict) else msg)
            ])

        return None

    def compute(
        self,
        completions: List[Any],
        trajectories: Optional[List[Dict[str, Any]]] = None,
        next_states: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        """
        Compute state prediction rewards.

        Evaluates how well the agent can predict the next state from the current
        state and action.

        Args:
            completions: List of model completions (actions)
            trajectories: Optional list of trajectory data containing states
            next_states: Optional explicit list of next states for comparison
            **kwargs: Additional context (may include 'states', 'observations', etc.)

        Returns:
            List of reward scores based on prediction quality
        """
        rewards = []

        # Extract states from various possible kwargs
        states = kwargs.get("states", kwargs.get("observations", None))

        # If next_states not provided, try to extract from trajectories or other kwargs
        if next_states is None:
            next_states = kwargs.get("next_states", kwargs.get("next_observations", None))

        if trajectories is not None and next_states is None:
            # Try to extract next_states from trajectories
            next_states = []
            for traj in trajectories:
                if isinstance(traj, dict) and "next_state" in traj:
                    next_states.append(traj["next_state"])
                else:
                    next_states.append(None)

        # Ensure we have lists of the right length
        if states is not None and not isinstance(states, list):
            states = [states] * len(completions)
        if next_states is not None and not isinstance(next_states, list):
            next_states = [next_states] * len(completions)

        # Compute rewards
        for i, completion in enumerate(completions):
            try:
                action = self.get_content(completion)

                # Get current state if available
                current_state = None
                if states is not None and i < len(states):
                    current_state = states[i]
                elif trajectories is not None and i < len(trajectories):
                    current_state = self._extract_state_from_trajectory(trajectories[i])

                # For state prediction, we compare predicted vs actual next state
                # In this implementation, we use the action as a proxy for "predicted state"
                # and compare it to the actual next state
                predicted_state = action
                actual_next_state = None

                if next_states is not None and i < len(next_states):
                    actual_next_state = next_states[i]
                elif "next_state" in kwargs:
                    actual_next_state = kwargs["next_state"]

                # If we don't have explicit next states, use a heuristic:
                # The "prediction" is how well the action aligns with the completion
                # This is a simplified version when full trajectory data isn't available
                if actual_next_state is None:
                    # Fallback: reward based on action quality (length, coherence)
                    # This provides a weak signal when full trajectory data is unavailable
                    if len(predicted_state) < self.min_prediction_length:
                        reward = 0.0
                    else:
                        # Simple heuristic: reward for non-empty, reasonable length actions
                        reward = min(1.0, len(predicted_state) / 100.0)
                else:
                    actual_next_state_str = str(actual_next_state)

                    # Compute similarity between predicted and actual next state
                    if self.use_embeddings:
                        similarity = self._compute_embedding_similarity(
                            predicted_state, actual_next_state_str
                        )
                    else:
                        similarity = self._compute_text_overlap(
                            predicted_state, actual_next_state_str
                        )

                    reward = similarity

            except Exception as e:
                logger.warning(f"Error in state_prediction_reward: {e}")
                logger.exception(e)
                reward = 0.0

            rewards.append(reward)

        return rewards


# Legacy function wrapper for backward compatibility
def state_prediction_reward(
    completions: List[Any],
    trajectories: Optional[List[Dict[str, Any]]] = None,
    next_states: Optional[List[str]] = None,
    **kwargs,
) -> List[float]:
    """
    Legacy function wrapper for StatePredictionReward.

    Args:
        completions: List of model completions (actions)
        trajectories: Optional list of trajectory data
        next_states: Optional list of actual next states
        **kwargs: Additional context

    Returns:
        List of reward scores
    """
    reward_fn = StatePredictionReward()
    return reward_fn.compute(
        completions,
        trajectories=trajectories,
        next_states=next_states,
        **kwargs
    )
