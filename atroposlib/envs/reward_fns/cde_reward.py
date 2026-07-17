"""
Curiosity-Driven Exploration (CDE) exploration bonus for RLVR.

Adapted from: CDE: Curiosity-Driven Exploration for Efficient Reinforcement
Learning in Large Language Models (https://arxiv.org/abs/2509.09675v1)

CDE addresses poor exploration in Reinforcement Learning with Verifiable
Rewards (RLVR) — premature convergence and entropy collapse — by adding an
intrinsic curiosity bonus to the verifiable reward:

    r(x, y) = r_verifiable(x, y) + alpha * B_actor(x, y) + beta * B_critic(x, y)

This module implements the paper's two curiosity signals:

1. **Actor-wise bonus**: the perplexity of the generated response under the
   actor policy, log PPL(y) = -(1/T) * sum_t log pi(y_t | y_<t, x). Higher
   perplexity means the policy was less certain about its own output, so the
   bonus inherently penalizes overconfident (low-perplexity) errors and
   promotes diversity among correct responses. When per-token logprobs are
   supplied via the ``logprobs`` kwarg this signal is computed exactly. When
   they are absent (reward functions run server-side, often without
   logprobs), it is approximated by the mean surprisal of the response under
   an add-k-smoothed bigram model estimated from recent completions — a
   parameter-free proxy for sequence perplexity.

2. **Critic-wise bonus**: the standard deviation of value estimates across
   the heads of a multi-head critic, supplied per completion via the
   ``values`` kwarg as a list of K head estimates. The paper connects this
   epistemic-uncertainty signal to the count-based exploration bonus
   ~ 1/sqrt(N(s)). A single scalar value has zero head variance and yields
   no bonus — value magnitude is not a curiosity signal.

Substitution notes (Mode 2 adapted port): the actor's token probabilities
are replaced by the count-based bigram proxy only when logprobs are
unavailable; the multi-head critic itself lives in the trainer, so per-head
estimates are passed in by the caller rather than computed here. The
verifiable reward and group-relative normalization (GRPO/PPO) remain in
their existing Atropos components — this function contributes only the
exploration bonus, composed with verifiable rewards via CombinedReward.
"""

import logging
import math
import re
from collections import Counter
from typing import Any, List, Optional, Set

from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)


@registry.register
class CDEReward(RewardFunction):
    """
    Curiosity-Driven Exploration bonus for RLVR.

    Combines the paper's actor-wise perplexity bonus and critic-wise
    multi-head value-variance bonus:

        bonus = actor_weight * B_actor + critic_weight * B_critic

    Both components are normalized to [0, 1]; the weights act as the paper's
    alpha/beta coefficients on top of the verifiable reward.

    Components:
        - B_actor: normalized log-perplexity of the response (exact when
          token logprobs are provided, bigram-proxy otherwise)
        - B_critic: normalized standard deviation across multi-head value
          estimates (0 when fewer than two head estimates are supplied)
    """

    def __init__(
        self,
        actor_weight: float = 1.0,
        critic_weight: float = 0.5,
        max_log_perplexity: float = 10.0,
        max_value_std: float = 1.0,
        smoothing: float = 1.0,
        min_tokens: int = 10,
        history_size: int = 100,
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Initialize the CDE reward function.

        Args:
            actor_weight: Coefficient for the actor perplexity bonus
                (the paper's alpha, default: 1.0)
            critic_weight: Coefficient for the critic variance bonus
                (the paper's beta, default: 0.5)
            max_log_perplexity: Log-perplexity clipped and normalized against
                this scale (default: 10.0)
            max_value_std: Value-head standard deviation clipped and
                normalized against this scale (default: 1.0)
            smoothing: Add-k smoothing for the bigram perplexity proxy
                (default: 1.0)
            min_tokens: Minimum token count for a completion to be scored
                (default: 10)
            history_size: Maximum completions kept in the proxy's sliding
                corpus window (default: 100)
            weight: Overall weight for this reward function (default: 1.0)
            **kwargs: Additional configuration
        """
        super().__init__(weight=weight, **kwargs)
        self.actor_weight = actor_weight
        self.critic_weight = critic_weight
        self.max_log_perplexity = max_log_perplexity
        self.max_value_std = max_value_std
        self.smoothing = smoothing
        self.min_tokens = min_tokens
        self._history_size = history_size

        # Sliding-window corpus backing the bigram perplexity proxy.
        self._completion_history: List[str] = []
        self._unigram_counts: Counter = Counter()
        self._bigram_counts: Counter = Counter()
        self._vocab: Set[str] = set()
        self._total_unigrams: int = 0

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        """
        Compute CDE exploration bonuses for the given completions.

        Args:
            completions: List of completions to evaluate
            **kwargs: Additional context, may include:
                - logprobs: Per-completion token logprobs of the generated
                  response under the actor policy (exact actor bonus)
                - values: Per-completion list of multi-head critic value
                  estimates (critic bonus)

        Returns:
            List of exploration bonuses (higher = more curious/uncertain)
        """
        contents = [self.get_content(completion) for completion in completions]
        logprobs = kwargs.get("logprobs")
        values = kwargs.get("values")

        bonuses = []
        for idx, content in enumerate(contents):
            tokens = self._tokenize(content)
            if len(tokens) < self.min_tokens:
                bonuses.append(0.0)
                continue

            token_logprobs = (
                logprobs[idx] if logprobs is not None and idx < len(logprobs) else None
            )
            head_values = (
                values[idx] if values is not None and idx < len(values) else None
            )

            actor_bonus = self._actor_bonus(tokens, token_logprobs)
            critic_bonus = self._critic_bonus(head_values)

            bonuses.append(
                self.actor_weight * actor_bonus + self.critic_weight * critic_bonus
            )

        # Update the proxy corpus after scoring so bonuses reflect surprise
        # relative to completions generated *before* this batch.
        self._update_history(contents)

        if bonuses:
            logger.debug(
                f"CDE bonuses: min={min(bonuses):.3f}, max={max(bonuses):.3f}, "
                f"mean={sum(bonuses) / len(bonuses):.3f}"
            )

        return bonuses

    def _actor_bonus(self, tokens: List[str], logprobs: Optional[Any]) -> float:
        """
        Actor-wise curiosity bonus: normalized log-perplexity of the response.

        Uses exact token logprobs when available; falls back to the bigram
        surprisal proxy otherwise.

        Returns:
            Bonus in [0, 1]; higher = higher perplexity = more exploratory.
        """
        log_ppl = None
        if logprobs is not None:
            log_ppl = self._log_perplexity_from_logprobs(logprobs)
        if log_ppl is None:
            log_ppl = self._proxy_log_perplexity(tokens)
        clipped = min(max(log_ppl, 0.0), self.max_log_perplexity)
        return clipped / self.max_log_perplexity

    @staticmethod
    def _log_perplexity_from_logprobs(logprobs: Any) -> Optional[float]:
        """
        Exact actor signal: log PPL(y) = -(1/T) * sum_t log pi(y_t | y_<t, x).

        Accepts any per-token sequence of logprobabilities (list, numpy
        array, torch tensor). Returns None if not interpretable.
        """
        try:
            vals = [float(lp) for lp in logprobs]
        except (TypeError, ValueError):
            return None
        if not vals:
            return None
        return -sum(vals) / len(vals)

    def _proxy_log_perplexity(self, tokens: List[str]) -> float:
        """
        Parameter-free perplexity proxy: mean surprisal under an
        add-k-smoothed bigram model estimated from recent completions.

        Responses whose token transitions are rare in recent history score
        high surprisal (novel/diverse); responses repeating recent history
        score low. With an empty history the model is uniform over the
        completion's own vocabulary, so surprisal reduces to log |vocab|.
        """
        vocab_size = max(1, len(self._vocab | set(tokens)))
        k = self.smoothing
        total_surprisal = 0.0
        prev = None
        for tok in tokens:
            if prev is None:
                # Unigram backoff for the first token.
                count = self._unigram_counts.get(tok, 0)
                denom = self._total_unigrams + k * vocab_size
            else:
                count = self._bigram_counts.get((prev, tok), 0)
                denom = self._unigram_counts.get(prev, 0) + k * vocab_size
            total_surprisal += -math.log((count + k) / denom)
            prev = tok
        return total_surprisal / len(tokens)

    def _critic_bonus(self, value_estimates: Optional[Any]) -> float:
        """
        Critic-wise curiosity bonus: standard deviation of value estimates
        across the heads of a multi-head critic, normalized to [0, 1].

        The paper ties this epistemic-uncertainty signal to the count-based
        exploration bonus ~ 1/sqrt(N(s)). A single scalar estimate has zero
        head variance and yields no bonus.
        """
        if value_estimates is None:
            return 0.0
        try:
            vals = [float(v) for v in value_estimates]
        except (TypeError, ValueError):
            # A lone scalar carries no multi-head uncertainty signal.
            return 0.0
        if len(vals) < 2:
            return 0.0
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(variance)
        return min(std, self.max_value_std) / self.max_value_std

    def _tokenize(self, text: str) -> List[str]:
        """Simple word-level tokenization."""
        return re.findall(r"\b\w+\b", text.lower())

    def _update_history(self, contents: List[str]) -> None:
        """Add the current batch to the proxy corpus (sliding window)."""
        for content in contents:
            tokens = self._tokenize(content)
            if len(tokens) < self.min_tokens:
                continue
            self._completion_history.append(content)
            self._add_to_model(tokens)
            while len(self._completion_history) > self._history_size:
                evicted = self._completion_history.pop(0)
                self._remove_from_model(self._tokenize(evicted))

    def _add_to_model(self, tokens: List[str]) -> None:
        prev = None
        for tok in tokens:
            self._unigram_counts[tok] += 1
            self._total_unigrams += 1
            self._vocab.add(tok)
            if prev is not None:
                self._bigram_counts[(prev, tok)] += 1
            prev = tok

    def _remove_from_model(self, tokens: List[str]) -> None:
        prev = None
        for tok in tokens:
            self._unigram_counts[tok] -= 1
            self._total_unigrams -= 1
            if self._unigram_counts[tok] <= 0:
                del self._unigram_counts[tok]
                self._vocab.discard(tok)
            if prev is not None:
                key = (prev, tok)
                self._bigram_counts[key] -= 1
                if self._bigram_counts[key] <= 0:
                    del self._bigram_counts[key]
            prev = tok


# Legacy function for backward compatibility
def cde_reward(
    completions: List[Any],
    actor_weight: float = 1.0,
    critic_weight: float = 0.5,
    **kwargs,
) -> List[float]:
    """
    Legacy function wrapper for the CDE exploration bonus.

    Args:
        completions: List of completions to evaluate
        actor_weight: Coefficient for the actor perplexity bonus
        critic_weight: Coefficient for the critic variance bonus
        **kwargs: Additional parameters including logprobs, values

    Returns:
        List of exploration bonuses (higher = more curious/uncertain)
    """
    reward_fn = CDEReward(actor_weight=actor_weight, critic_weight=critic_weight)
    return reward_fn.compute(completions, **kwargs)
