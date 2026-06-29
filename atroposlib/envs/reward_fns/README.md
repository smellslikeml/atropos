# Reward Functions

`atroposlib.envs.reward_fns` is the reward-function subsystem used by dataset
and online/gymnasium environments. A reward function scores a list of model
completions and returns one `float` per completion. Reward functions are plain
Python classes that live alongside this README; the registry wires them up to a
string name so environments can request them by name from config.

The public surface is re-exported from the package `__init__`:

```python
from atroposlib.envs.reward_fns import RewardFunction, registry, CombinedReward, RankCalibratedReward
```

## `RewardFunction` base class

`reward_function.RewardFunction` is the abstract base every reward function
subclasses. It exposes:

- `compute(completions, **kwargs) -> List[float]` — the method each subclass
  implements; returns one reward per completion.
- `__call__(completions, **kwargs)` — wrapper that applies `self.weight` to the
  computed rewards (and logs to WandB when a logger is attached). Callers
  normally invoke the instance rather than `compute` directly so the weight is
  applied.
- `get_content(completion)` — static helper that extracts assistant text from
  the common completion formats (plain string, `{"role": "assistant", ...}`
  dict, nested `message`, or a list of messages).
- `weight` (default `1.0`) — importance factor applied by `__call__` when this
  reward is combined with others.

## Registry

`registry.Registry` (instance `registry` in `registry.py`) holds the map from
name → reward-function class. Reward functions register themselves with the
`@registry.register` decorator:

```python
from .registry import registry
from .reward_function import RewardFunction

@registry.register
class MyReward(RewardFunction):
    def compute(self, completions, **kwargs):
        return [...]
```

The registered name is derived from the class name: the class name is
lower-cased and, if it ends in `reward`, that suffix is stripped. So
`CombinedReward` registers as `"combined"`, `RankCalibratedReward` as
`"rankcalibrated"`, and `AccuracyReward` as `"accuracy"`. Pass
`@registry.register(name="custom_name")` to override.

Create an instance by name or config dict:

```python
reward_fn = registry.create("rankcalibrated")            # by name
reward_fn = registry.create({"type": "combined",          # by config dict
                             "rewards": ["rankcalibrated", "format"],
                             "normalization": "sum"})
scores = reward_fn(completions, scores=[...])
```

`registry.list_registered()` returns the names registered so far.

## Built-in reward functions

| Registry name | Class | Module | Purpose |
| --- | --- | --- | --- |
| `combined` | `CombinedReward` | `combined_reward.py` | Meta reward that sums several reward functions (with optional normalization). |
| `rankcalibrated` | `RankCalibratedReward` | `rank_calibrated_reward.py` | Group-relative shaping for continuous execution scores (no ground-truth solution needed). |

Additional reward functions (e.g. accuracy, format, cosine-scaled,
reasoning-steps, repetition-penalty, and the r1 format-reasoning reward) live as
sibling modules in this directory and are loaded on demand by name. Their exact
registry names follow the lower-cased-and-`reward`-stripped rule above.

### `combined` — `CombinedReward`

Meta reward function that combines multiple reward functions. Pass a list of
sub-rewards (each a registry name or a config dict); their outputs are summed,
with optional `normalization` of `"none"` (default), `"sum"` (divide by total
weight), or `"minmax"` (scale to `[0, 1]`).

```python
reward_fn = registry.create({
    "type": "combined",
    "rewards": ["rankcalibrated", "format"],
    "normalization": "sum",
    "weight": 1.0,
})
```

### `rankcalibrated` — `RankCalibratedReward`

Calibrated, group-relative reward for continuous execution scores, for
score-based tasks that have no ground-truth solution (e.g. execution-feedback
scores from a Heuristic Contest). Adapted from RiVER ("Reinforcement Learning
without Ground-Truth Solutions can Improve LLMs").

It counters two group-relative failure modes at the reward layer:

- **scale dominance** — raw score magnitudes vary wildly across instances, so a
  few large-magnitude instances dominate policy updates. Scores are min-max
  normalized *within each group* so every instance contributes on a comparable
  scale.
- **frequency dominance** — frequently sampled mediocre solutions can crowd out
  rare stronger ones. A rank-based `emphasis` gives the top-ranked solver the
  full reward while keeping bounded, non-negative feedback for the rest.

The shaping is **instance-wise**: only completions evaluated against the same
instance are compared, so call `compute` once per prompt/group. The reward for
each completion in the group is

```
c_i = (s_i - s_min) / (s_max - s_min)          # in [0, 1], within the group
r_i = min_reward + (c_i ** emphasis) * (max_reward - min_reward)
```

where `s_i` is the raw execution score of completion `i`. When every completion
in the group has the same score, the rewards are the bounded, equal
`min_reward` (the group is indistinguishable, so the downstream group-relative
advantage centers it to ~0).

**Options** (`__init__`):

- `emphasis` (`float`, default `2.0`) — exponent `>= 1` applied to the
  calibrated score; larger values sharpen the gap so the top-ranked solver
  receives the full reward. `1.0` disables emphasis (pure linear calibration).
- `min_reward` (`float`, default `0.0`) — reward assigned to the weakest
  completion in the group.
- `max_reward` (`float`, default `1.0`) — reward assigned to the top-ranked
  solver.
- `higher_is_better` (`bool`, default `True`) — whether larger raw scores are
  better (`False` for cost-style scores where lower is better).
- `weight` (`float`, default `1.0`) — importance factor applied by
  `RewardFunction.__call__`.

**The `scores` kwarg** (passed to `compute`):

- A list of raw scores aligned one-to-one with `completions`.
- A scalar, broadcast to every completion.
- Omitted (`None`), in which case a numeric score is parsed from each
  completion's content as a convenience for environments that print a score in
  the model output.

```python
reward_fn = registry.create("rankcalibrated", emphasis=2.0, min_reward=-1.0, max_reward=1.0)
rewards = reward_fn.compute(completions, scores=execution_scores)
```

A legacy `rank_calibrated_reward(completions, scores=None, **kwargs)` function
wrapper is also exported for function-style use.

## Adding a reward function

Reward functions are colocated in this directory (one module per function) and
self-register via `@registry.register`. To add one:

1. Create `<name>_reward.py` here implementing a `RewardFunction` subclass with
   a `compute` method.
2. Decorate the class with `@registry.register` (or pass an explicit `name`).
3. Re-export the class from `__init__.py`.
4. Add a row to the table above and, for user-facing functions, a dedicated
   section like the `rankcalibrated` one.
