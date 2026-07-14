# Reward Functions (`reward_fns`)

This package provides a small framework for creating, composing, and applying
reward functions that score model outputs. Reward functions can be used from
both dataset environments and online/gymnasium environments.

## Core components

- **`RewardFunction`** (`reward_function.py`): abstract base class. Subclass it
  and implement `compute(self, completions, **kwargs) -> List[float]`.
- **`registry`** (`registry.py`): registers reward functions and instantiates
  them by name or config dict. Decorate a class with `@registry.register` to
  make it available, then create it with `registry.create(...)`.
- **`CombinedReward`** (`combined_reward.py`): meta reward function that sums a
  list of sub-rewards (with optional normalization).

## Usage

```python
from atroposlib.envs.reward_fns import registry

# Create by registry key ...
reward_fn = registry.create("accuracy", weight=1.5)

# ... or by config dict.
reward_fn = registry.create({"type": "accuracy", "weight": 1.5})

scores = reward_fn(completions, **kwargs)
```

Each module in this directory contributes one reward function; see the module
docstrings for the specifics of each. New reward functions should follow the
same pattern: a `RewardFunction` subclass registered with `@registry.register`,
plus an optional legacy functional wrapper.

## `reward_floor` — deterministic anti-reward-hacking floor

`RewardFloor` (`reward_floor.py`, registry key `reward_floor`) is a
deterministic, rule-based reward floor that clamps a completion's reward when it
reward-hacks by lifting content verbatim out of its input context. It reproduces
the `S_r` "reward floor" component from *Designing Reward Signals for Portable
Query Generation: A Case Study in Industrial Semantic Job Search*
(arXiv:2606.27291).

It applies two deterministic rules against the source context (read from the
`profile` / `reference` / `solution` / `prompt` kwargs):

1. **6-gram verbatim overlap** — any 6-token window of the completion that also
   appears verbatim in the source context.
2. **Lifted date range** — a date-range fragment (e.g. `2019-2021`,
   `Jan 2020 - Mar 2021`, `2019 - present`) that also appears in the source.

A completion that trips either rule receives `floor_value` (`-1.0` by default);
otherwise it receives `pass_value` (`0.0`, neutral under additive composition).

```python
from atroposlib.envs.reward_fns import registry

# Standalone: -1.0 on any degenerate completion, 0.0 otherwise.
floor = registry.create("reward_floor")
floor.compute(completions, profile=member_profile)

# Composed with a judge reward via CombinedReward.
combined = registry.create({
    "type": "combined",
    "rewards": [
        {"type": "accuracy", "weight": 1.0},
        {"type": "reward_floor", "weight": 1.0},
    ],
})
```
