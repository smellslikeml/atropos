"""
Deterministic reward floor for preventing verbatim-copy reward hacking.

This module implements the ``S_r`` "reward floor" component from:

    "Designing Reward Signals for Portable Query Generation: A Case Study in
    Industrial Semantic Job Search" (arXiv:2606.27291)

The paper generates *portable* job-search queries from candidate profiles with
an RLAIF pipeline. Under GRPO, the group-relative advantage normalization is
uniquely sensitive to spurious reward, so the policy learns to reward-hack the
LLM-as-judge by copying identifiers verbatim out of the input profile. The paper
mitigates this with a deterministic, rule-based floor that clamps the reward to
``-1.0`` (the largest single-lever effect in their empirical matrix, +0.147 on a
cross-family judge) whenever the generated query either:

    1. contains a 6-gram that appears verbatim in the input profile, or
    2. lifts a date-range fragment (e.g. "2019-2021", "Jan 2020 - Mar 2021")
       straight out of a profile entry.

Only these two rules exist in the paper; both trigger the same uniform ``-1.0``
hard cap.

Adaptation notes (Mode 2 — adapted port into atropos' generic reward_fns
framework):

  * The paper's ``(generated_query, member_profile)`` pair is generalized to the
    framework's ``(completion, profile/reference/prompt)`` inputs, since atropos
    has no job-search environment or LinkedIn dataset.
  * The paper's ``r_rubric`` (a Qwen3-8B LLM-judge over a 5-dimension portability
    rubric) is NOT reproduced here — only the deterministic ``S_r`` floor, which
    is the portable, parameter-free part of the reward.
  * INTENTIONAL DEVIATION — grader skip: the paper clamps to ``-1.0`` *and* skips
    the expensive judge call entirely. atropos' ``RewardFunction`` contract returns
    a scalar per completion with no side channel to short-circuit a downstream
    grader in a ``CombinedReward`` pipeline. We therefore reproduce the reward
    signal (the ``-1.0`` floor still dominates GRPO's group-relative advantage) but
    do not realize the judge-compute saving the paper reports.
"""

import logging
import re
from typing import Any, List, Optional

from .registry import registry
from .reward_function import RewardFunction

logger = logging.getLogger(__name__)

# A date-range endpoint is an optional month prefix followed by a 4-digit year,
# e.g. "2019" or "Jan 2020" / "March 2021".
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_YEAR = r"(?:19|20)\d{2}"
_ENDPOINT = rf"(?:{_MONTH}\s+)?{_YEAR}"
# Separator between two endpoints: a hyphen/en-dash/em-dash or the word "to".
_SEP = r"(?:\s*[-–—]\s*|\s+to\s+)"
# A full range: start endpoint, separator, then an end endpoint or "present"/"current".
_DATE_RANGE_RE = re.compile(
    rf"{_ENDPOINT}{_SEP}(?:{_ENDPOINT}|present|current)",
    re.IGNORECASE,
)


class DegeneracyDetector:
    """
    The paper's two deterministic ``S_r`` triggers.

    Both operate on the (generated query, source profile) pair and return a bool:
    a hit on either means the completion reward-hacks by lifting content verbatim.
    """

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase word-token split shared by both detectors."""
        return re.findall(r"\b\w+\b", text.lower())

    @staticmethod
    def six_gram_verbatim_overlap(completion: str, profile: str, n: int = 6) -> bool:
        """
        Return True if any n-gram (default 6-gram) of the completion appears
        verbatim in the profile.

        This is the paper's exact-substring rule, not an approximate similarity
        ratio: for every n-token window of the completion, check whether that
        exact window occurs as a contiguous n-token run in the profile.
        """
        if not completion or not profile:
            return False

        comp_tokens = DegeneracyDetector._tokenize(completion)
        prof_tokens = DegeneracyDetector._tokenize(profile)
        if len(comp_tokens) < n or len(prof_tokens) < n:
            return False

        profile_ngrams = {
            tuple(prof_tokens[i : i + n]) for i in range(len(prof_tokens) - n + 1)
        }
        for i in range(len(comp_tokens) - n + 1):
            if tuple(comp_tokens[i : i + n]) in profile_ngrams:
                return True
        return False

    @staticmethod
    def _normalize_range(fragment: str) -> str:
        """Canonicalize a date-range fragment for cross-text comparison."""
        fragment = fragment.lower()
        fragment = re.sub(r"[–—]", "-", fragment)  # en/em dash -> hyphen
        fragment = re.sub(r"\s*-\s*", "-", fragment)  # tighten spacing around dash
        fragment = re.sub(r"\s+", " ", fragment).strip()
        return fragment

    @staticmethod
    def date_range_lifted(completion: str, profile: str) -> bool:
        """
        Return True if a date-range fragment in the completion also appears in
        the profile (i.e. the range was lifted from a profile entry).

        Recognizes year ranges ("2019-2021"), month-year ranges
        ("Jan 2020 - Mar 2021"), and open ranges ("2019 - present").
        """
        if not completion or not profile:
            return False

        norm_profile = DegeneracyDetector._normalize_range(profile)
        for match in _DATE_RANGE_RE.finditer(completion):
            fragment = DegeneracyDetector._normalize_range(match.group(0))
            if fragment and fragment in norm_profile:
                return True
        return False


@registry.register
class RewardFloor(RewardFunction):
    """
    Deterministic reward floor (the paper's ``S_r``) to prevent verbatim-copy
    reward hacking under GRPO.

    Implements exactly the two rules from "Designing Reward Signals for Portable
    Query Generation" (arXiv:2606.27291): a 6-gram verbatim overlap against the
    input profile, and a lifted date-range fragment. A completion that trips
    either rule receives ``floor_value`` (``-1.0`` by default, the paper's hard
    cap); otherwise it receives ``pass_value`` (``0.0``, neutral under additive
    composition).

    Usage:
        # Standalone: -1.0 on any degenerate completion, 0.0 otherwise.
        floor = RewardFloor()
        floor.compute(completions, profile=member_profile)

        # Composed with a judge reward. Because atropos' CombinedReward sums
        # sub-rewards, the -1.0 floor is added to (rather than a true clamp on)
        # the judge score; it still dominates the group-relative advantage that
        # the paper identifies as the reward-hacking vector.
        combined = CombinedReward([
            {"type": "accuracy", "weight": 1.0},
            {"type": "reward_floor", "weight": 1.0},
        ])
    """

    def __init__(
        self,
        rules: Optional[List[str]] = None,
        floor_value: float = -1.0,
        pass_value: float = 0.0,
        ngram_size: int = 6,
        weight: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            rules: Subset of {"six_gram_verbatim", "date_range"} to enforce.
                   Defaults to both (the paper's full S_r).
            floor_value: Reward for a completion that trips any rule. Defaults to
                         the paper's -1.0 hard cap.
            pass_value: Reward for a completion that trips no rule. Defaults to
                        0.0 (neutral when summed with other rewards).
            ngram_size: n for the verbatim-overlap detector (paper uses 6).
            weight: Weight for this reward function.
            **kwargs: Additional configuration (ignored).
        """
        super().__init__(weight=weight, **kwargs)

        valid_rules = {"six_gram_verbatim", "date_range"}
        self.rules = rules if rules is not None else ["six_gram_verbatim", "date_range"]
        for rule in self.rules:
            if rule not in valid_rules:
                logger.warning(f"Unknown rule '{rule}', ignoring")

        self.floor_value = floor_value
        self.pass_value = pass_value
        self.ngram_size = ngram_size

    def _source_text(self, kwargs: dict, index: int, n: int) -> str:
        """
        Assemble the profile text a completion is checked against.

        Pulls from ``profile`` / ``reference`` / ``solution`` / ``prompt`` kwargs,
        supporting either a single shared value or a per-completion list.
        """
        parts = []
        for key in ("profile", "reference", "solution", "prompt"):
            value = kwargs.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                if index < len(value) and value[index]:
                    parts.append(str(value[index]))
            else:
                parts.append(str(value))
        return "\n".join(parts)

    def compute(self, completions: List[Any], **kwargs) -> List[float]:
        """
        Return ``floor_value`` for completions that trip a rule, ``pass_value``
        otherwise. Context (the source profile) is read from the ``profile`` /
        ``reference`` / ``solution`` / ``prompt`` kwargs.
        """
        rewards = []
        for i, completion in enumerate(completions):
            try:
                content = self.get_content(completion)
                profile = self._source_text(kwargs, i, self.ngram_size)

                triggered = None
                if "six_gram_verbatim" in self.rules:
                    if DegeneracyDetector.six_gram_verbatim_overlap(
                        content, profile, n=self.ngram_size
                    ):
                        triggered = f"{self.ngram_size}-gram_verbatim_overlap"

                if triggered is None and "date_range" in self.rules:
                    if DegeneracyDetector.date_range_lifted(content, profile):
                        triggered = "date_range_lifted"

                if triggered is not None:
                    logger.info(f"Reward floor triggered: {triggered}")
                    rewards.append(self.floor_value)
                else:
                    rewards.append(self.pass_value)

            except Exception as e:
                logger.error(f"Error in reward floor computation: {e}")
                logger.exception(e)
                # On error, do not penalize: return the neutral pass value.
                rewards.append(self.pass_value)

        return rewards


# Legacy function wrapper, matching the convention of the other reward modules.
def reward_floor(
    completions: List[Any],
    rules: Optional[List[str]] = None,
    floor_value: float = -1.0,
    **kwargs,
) -> List[float]:
    """Functional wrapper around :class:`RewardFloor`."""
    reward_fn = RewardFloor(rules=rules, floor_value=floor_value)
    return reward_fn.compute(completions, **kwargs)
