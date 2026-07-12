## PR Type
<!-- Please check ONE of the following options -->
- [ ] RL Environment PR - Complete Environment Snapshot & Zero-Training sections
- [x] Non-Environment PR - Complete Description, Related Issues & Type of Change sections

---

## 📝 General Information
### Description
**What this PR delivers:** a new `metacognitive` reward function under
`atroposlib/envs/reward_fns/` that scores completions by *metacognitive
faithfulness* — how well a model's self-reported confidence matches whether it
was actually correct. It adapts the RLMF paradigm from *Reinforcement Learning
with Metacognitive Feedback Elicits Faithful Uncertainty Expression in LLMs*
(arXiv:2606.32032) into Atropos' existing reward-registry path.

- `atroposlib/envs/reward_fns/metacognitive_reward.py` — `MetacognitiveReward`,
  which uses a Brier proper score (`faithfulness = 1 - (p - outcome) ** 2`) over
  a parser-extracted self-reported confidence (`<confidence>0.8</confidence>` or
  `Confidence: 80%`). It auto-registers alongside `accuracy_reward` /
  `format_reward` and is invoked from `DatasetEnv.score` via
  `registry.create(...)`.
- `atroposlib/envs/reward_fns/metacognitive_accuracy.py` —
  `MetacognitiveAccuracyReward`, the accuracy-conditioned companion signal
  (registered as `metacognitiveaccuracy`).
- `atroposlib/envs/reward_fns/__init__.py` — eager import so both rewards
  register on package load.

The signal plugs into the existing GRPO / PipelineRL reward path through
`CombinedReward` (combine with `accuracy_reward` so task correctness still
dominates), rather than introducing a custom optimizer. The paper's separate
LLM judge and standalone faithful-calibration benchmark are intentionally out
of scope for this PR.

### Related Issues
<!-- Link any relevant issues here. -->
Implements the reward signal from arXiv:2606.32032 (RLMF). N/A — no tracking
issue.

### Type of Change
- [x] New feature (non-breaking change which adds functionality)

> **Note (autonomous discovery):** this PR was drafted by an autonomous
> paper-implementation pass. The change is confined to `atroposlib` library
> code (a reward function and its colocated tests); it is not a runnable RL
> environment, hence the *Non-Environment PR* selection above.

---

## ✅ Developer & Reviewer Checklist
<!-- Common checklist for all PR types - adapt as needed for your PR type -->
- [x] Code follows project style (black, isort, flake8 pass with pre-commit)
- [x] I have performed a self-review of my own code
- [x] My changes generate no new warnings
- [x] New and existing unit tests pass locally with my changes
- [x] Docstrings added for all new public classes / functions

### Test plan
New colocated pytest modules mirror the source files under
`atroposlib/tests/`:

- `atroposlib/tests/test_metacognitive_reward.py` — covers the Brier
  faithfulness scoring and confidence parsing (`<confidence>` tag and
  `Confidence: N%` line) for `MetacognitiveReward`.
- `atroposlib/tests/test_metacognitive_accuracy.py` — covers the
  accuracy-conditioned `MetacognitiveAccuracyReward` behaviour.

```
# before (main): metacognitive modules absent
$ pytest -v atroposlib/tests/
... (no metacognitive tests collected)

# after (this PR):
$ pytest -v atroposlib/tests/test_metacognitive_reward.py atroposlib/tests/test_metacognitive_accuracy.py
11 passed, 0 failed, 0 warnings
```
