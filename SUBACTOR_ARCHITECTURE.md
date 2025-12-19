# Subactor Architecture in Hieros

## Overview

This document explains the hierarchical subactor architecture used in the Hieros (HIERarchical imagination On Structured State Space Sequence Models) implementation, including how many subactors are created, where they are created, their purpose, and how this approach compares to the baseline models (DreamerV3 and S5) that Hieros is built upon.

## How Many Subactors Are Created?

The number of subactors in Hieros is **dynamic** and can grow during training. The key parameters controlling this are:

1. **Initial Number**: **1 subactor** (Subactor-0) is created at initialization
2. **Maximum Number**: Configurable via `max_hierarchy` parameter (default: 3)
3. **Growth Mechanism**: New subactors are added incrementally during training based on the `add_hierarchy_every` parameter

### Configuration Parameters

From `hieros/configs.yaml`:
```yaml
max_hierarchy: 3              # Maximum number of hierarchical layers (subactors)
add_hierarchy_every: 1        # Frequency of adding new hierarchy layers
```

### Dynamic Growth Example

With default settings (`max_hierarchy=3`, `add_hierarchy_every=1`):
- **Start**: 1 subactor (Subactor-0)
- **After first checkpoint**: 2 subactors (Subactor-0, Subactor-1)
- **After second checkpoint**: 3 subactors (Subactor-0, Subactor-1, Subactor-2)
- **Maximum**: Growth stops at 3 subactors

## Where Subactors Are Created

### 1. Initial Creation (File: `hieros/hieros.py`, lines 60-73)

The first subactor (Subactor-0) is created in the `Hieros.__init__()` method:

```python
class Hieros(nn.Module):
    def __init__(self, obs_space, act_space, config, prefilled_replay):
        # ... initialization code ...
        
        # Create the initial subactor (lowest level in hierarchy)
        self._subactors = nn.ModuleList(
            [
                SubActor(
                    "Subactor-0",           # Name
                    obs_space,              # Observation space
                    act_space,              # Action space
                    self._subgoal_shape,    # Subgoal shape [8, 8]
                    new_config,             # Configuration
                    prefilled_replay,       # Replay buffer
                    compute_subgoal=config.max_hierarchy > 1,
                )
            ]
        )
```

### 2. Dynamic Creation (File: `hieros/hieros.py`, lines 262-271 and 371-473)

Additional subactors are created dynamically during training in the `policy()` method:

```python
def policy(self, obs, state=None, mode="train"):
    # Check if we should add a new hierarchy layer
    if (
        self._should_add_hierarchy(self._environment_time_steps)
        and (self._environment_time_steps > 0 or self._config.add_hierarchy_every == 0)
        and len(self._subactors) < self._config.max_hierarchy
    ):
        print("Adding a new hierarchy layer")
        self._create_subactor()
```

The actual creation happens in `_create_subactor()` method (lines 371-473):

```python
def _create_subactor(self):
    new_config = copy.deepcopy(self._config)
    
    # Configure the new subactor based on architecture type
    if self._config.subactor_encoding_architecture == "mlp":
        new_config.encoder["mlp_keys"] = ".*"
        new_config.decoder["mlp_keys"] = ".*"
        new_config.encoder["cnn_keys"] = "$^"
        new_config.decoder["cnn_keys"] = "$^"
    elif self._config.subactor_encoding_architecture == "cnn":
        new_config.encoder["mlp_keys"] = "$^"
        new_config.decoder["mlp_keys"] = "$^"
        new_config.encoder["cnn_keys"] = ".*"
        new_config.decoder["cnn_keys"] = ".*"
    
    # Append new subactor to the hierarchy
    self._subactors.append(
        SubActor(
            f"Subactor-{len(self._subactors)}",              # Name (e.g., "Subactor-1")
            self._subactors[-1].encoded_obs_space(),        # Encoded obs from previous layer
            {"action": embodied.Space(np.float32, self._subgoal_shape)},  # Action space is subgoal
            self._subgoal_shape,                            # Subgoal shape
            new_config,                                     # Modified configuration
            make_replay(
                self._config,
                self._config.traindir / f"replay-{len(self._subactors)}",
            ),                                              # Separate replay buffer
            buffer_obs=self._config.subactor_encode_intermediate,
            buffer_obs_keys=list(self._subactors[-1].encoded_obs_space().keys()),
            use_world_model=use_world_model,
            other_world_model=self._subactors[0]._wm if not use_world_model else None,
        )
    )
```

### Key Features of Dynamic Creation

1. **Incremental Addition**: Subactors are added one at a time during training
2. **Hierarchical Dependency**: Each new subactor operates on the encoded observation space of the previous subactor
3. **Separate Replay Buffers**: Each subactor has its own replay buffer (`replay-0`, `replay-1`, etc.)
4. **World Model Sharing**: Depending on configuration (`hierarchical_world_models`), higher-level subactors may share the world model with Subactor-0 or have their own

## Purpose of Creating Subactors

The hierarchical subactor architecture serves several critical purposes in the Hieros framework:

### 1. Temporal Abstraction

Each subactor operates at a different temporal scale:

- **Subactor-0** (Lowest Level): Operates at the base action frequency, directly interacting with the environment
- **Subactor-1** (Mid Level): Operates at `subactor_update_every` × frequency (default: 4× slower)
- **Subactor-2** (Highest Level): Operates at `subactor_update_every²` × frequency (default: 16× slower)

From `hieros/configs.yaml`:
```yaml
subactor_update_every: 4      # Each higher level operates 4× slower
subactor_train_every: 4       # Each higher level trains 4× less frequently
```

### 2. Goal-Conditioned Hierarchical Control

Higher-level subactors generate **subgoals** for lower-level subactors:

- **Subactor-2** (if exists) → generates subgoals for Subactor-1
- **Subactor-1** → generates subgoals for Subactor-0
- **Subactor-0** → generates actions for the environment

This creates a hierarchical decision-making structure where:
- High-level subactors handle long-term planning
- Low-level subactors handle short-term execution

### 3. Observation Encoding and Abstraction

Each subactor encodes its observations before passing them to the next level:

```python
def encode_obs(self):
    """Encodes observation and adds latent information for next hierarchy layer"""
    # Combines raw observations with world model state (stoch and deter)
    # Creates increasingly abstract representations up the hierarchy
```

This allows:
- **Progressive abstraction**: Higher levels work with more abstract representations
- **Information compression**: Reducing dimensionality as we go up the hierarchy
- **Latent state integration**: Including model dynamics in the encoded observations

### 4. Reward Decomposition

Each subactor optimizes a combination of rewards:

```python
rewards = {
    "extrinsic": subactor._wm.heads["reward"](subactor._wm.dynamics.get_feat(s)),
    "subgoal": subactor._subgoal_reward(s, subgoal),
    "novelty": subactor._novelty_reward(s),
}
```

- **Extrinsic reward**: From the environment (can be disabled for higher levels)
- **Subgoal reward**: Reward for achieving subgoals set by higher-level subactors
- **Novelty reward**: Intrinsic exploration bonus

Configuration from `hieros/configs.yaml`:
```yaml
only_subgoal_reward: false            # If true, higher levels ignore extrinsic reward
extrinsic_reward_weight: 1.0
subgoal_reward_weight: 0.3
novelty_reward_weight: 0.1
novelty_only_higher_level: false      # If true, only higher levels get novelty reward
```

### 5. Separate Learning and Replay

Each subactor maintains:
- **Separate replay buffer**: Independent experience storage
- **Independent training schedule**: Can train at different frequencies
- **Own world model** (optional): Can model environment dynamics at different abstractions

### 6. Improved Credit Assignment

The hierarchical structure improves credit assignment by:
- Breaking down long-horizon tasks into shorter subgoal-achievement problems
- Each level focuses on its own temporal scale
- Reducing the effective planning horizon at each level

### 7. Scalability and Modularity

The architecture supports:
- **Dynamic growth**: Adding complexity as training progresses
- **Configurable hierarchy depth**: Can use anywhere from 1 to `max_hierarchy` subactors
- **Optional features**: Can enable/disable hierarchical world models, observation encoding, etc.

## Comparison with DreamerV3 and S5

### DreamerV3 Architecture

Hieros is built on top of DreamerV3 (as indicated in the README and code references). Let's examine the key differences:

#### DreamerV3 (Baseline - File: `hieros/dreamer.py`)

```python
class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, replay=None):
        # Single-level architecture
        self._wm = models_dreamer.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models_dreamer.ImagBehavior(
            config, self._wm, config.behavior_stop_grad
        )
        # Single replay buffer
        self._replay = replay
        self._dataset = self.dataset(self._replay.dataset)
```

**Key Characteristics:**
- **Single actor**: No hierarchical structure
- **Flat temporal scale**: All decisions at the same frequency
- **Single replay buffer**: All experience in one buffer
- **Single world model**: One model for all planning
- **Direct action generation**: Actor directly outputs environment actions

#### Hieros Architecture (File: `hieros/hieros.py`)

```python
class Hieros(nn.Module):
    def __init__(self, obs_space, act_space, config, prefilled_replay):
        # Hierarchical multi-actor architecture
        self._subactors = nn.ModuleList([
            SubActor("Subactor-0", obs_space, act_space, ...)
        ])
        # Can grow to max_hierarchy subactors dynamically
        # Multiple replay buffers (one per subactor)
        # Multiple world models (optional)
        # Hierarchical action/subgoal generation
```

**Key Characteristics:**
- **Multiple subactors**: Up to `max_hierarchy` (default: 3)
- **Hierarchical temporal scales**: Each level operates at different frequencies
- **Multiple replay buffers**: Separate experience storage per level
- **Multiple world models** (optional): Can have separate world models per level
- **Subgoal-based control**: Higher levels generate subgoals for lower levels

### Differences Summary

| Feature | DreamerV3 | S5 | Hieros |
|---------|-----------|-----|---------|
| **Architecture** | Single actor | Single actor | Multiple hierarchical subactors (1-3+) |
| **Temporal Scale** | Single frequency | Single frequency | Multi-scale (1×, 4×, 16×, ...) |
| **Planning Horizon** | Fixed | Fixed | Hierarchical (different per level) |
| **Action Generation** | Direct actions | Direct actions | Hierarchical subgoals → actions |
| **Replay Buffers** | 1 buffer | 1 buffer | N buffers (one per subactor) |
| **World Models** | 1 model | 1 model | 1-N models (configurable) |
| **Observation Encoding** | Single encoder | Single encoder | Progressive encoding up hierarchy |
| **Reward Structure** | Extrinsic only | Extrinsic only | Extrinsic + Subgoal + Novelty |
| **State Space Model** | RSSM (Recurrent) | S5 (Structured) | S5 + Hierarchical extension |
| **Credit Assignment** | Standard | Standard | Hierarchical decomposition |
| **Long-term Planning** | Imagination rollouts | Imagination rollouts | Multi-level hierarchical planning |

### S5 (Structured State Space Model)

S5 is not a reinforcement learning architecture itself, but rather a **sequence modeling approach** used as the dynamics model. The comparison is:

#### In DreamerV3
- Uses **RSSM** (Recurrent State Space Model) for world model dynamics
- Recurrent neural network-based state transitions

#### In Hieros
- Uses **S5** (Structured State Space Model) for world model dynamics (configurable)
- Can also use RSSM for compatibility
- S5 provides better long-range dependency modeling
- Implementation in `resettable_s5/` directory

**Key Point**: S5 is the **dynamics model component**, not the full RL architecture. Hieros extends both DreamerV3's RL framework and (optionally) uses S5 for dynamics modeling, while adding the hierarchical subactor structure on top.

### Why Hieros Uses Multiple Subactors (But DreamerV3 and S5 Don't)

The hierarchical subactor approach is **unique to Hieros** and not found in DreamerV3 or standard S5 implementations:

1. **DreamerV3's Limitation**: Single temporal scale struggles with long-horizon tasks requiring both high-level strategy and low-level control

2. **Hieros's Solution**: Hierarchical subactors decompose the problem:
   - High levels: Strategic, long-term planning with abstract subgoals
   - Low levels: Tactical, short-term execution with concrete actions

3. **Inspiration**: The hierarchical approach draws from:
   - **Hierarchical RL literature**: Options framework, feudal networks, HAM
   - **Temporal abstraction theory**: Operating at multiple time scales
   - **Goal-conditioned RL**: Subgoal generation and achievement

4. **Novel Contribution**: Combining hierarchical temporal abstraction with:
   - World model-based planning (from DreamerV3)
   - Structured state space models (from S5)
   - Progressive observation encoding
   - Multi-level replay and learning

## Implementation Details

### Subactor Class (File: `hieros/hieros.py`, lines 602-1000)

Each `SubActor` contains:

```python
class SubActor(nn.Module):
    def __init__(self, name, obs_space, act_space, subgoal_shape, config, replay, ...):
        self._wm = models.WorldModel(obs_space, act_space, ...)  # World model
        self._task_behavior = models.ImagBehavior(...)            # Policy/actor-critic
        self.subgoal_autoencoder = models.SubgoalAutoencoder(...) # Subgoal compression
        self._replay = replay                                     # Replay buffer
        self._dataset = make_dataset(self._replay.dataset, config)
```

### Update Frequency Scaling

The temporal hierarchy is enforced through update frequency scaling:

```python
# From hieros.py, lines 454-465
self._should_update_subactor.append(
    tools.Every(
        self._should_update_subactor[-1]._every * self._config.subactor_update_every,
    )
)
self._should_train_subactor.append(
    tools.Every(
        self._should_train_subactor[-1]._every * self._config.subactor_train_every,
    )
)
```

This creates exponential scaling:
- Subactor-0: Every 1 step
- Subactor-1: Every 4 steps (if `subactor_update_every=4`)
- Subactor-2: Every 16 steps

### Training Flow (File: `hieros/hieros.py`, lines 130-160)

```python
def train(self, data=None, state=None):
    for subactor, should_train in zip(self._subactors, self._should_train_subactor):
        if should_train(self._training_steps):
            subactor.train()
            if len(subactor._replay) >= self._config.batch_size:
                if (self._config.fix_dataset and subactor._name != "Subactor-0"):
                    # Higher levels can optionally train on Subactor-0's start states
                    data = self._subactors[0]._last_start
                else:
                    data = next(subactor._dataset)
                
                subactor_metrics = subactor._train(data)
                metrics.update({f"{subactor._name}/{key}": value 
                                for key, value in subactor_metrics.items()})
```

## Configuration Options

Key configuration parameters for controlling subactor behavior:

```yaml
# Hierarchy structure
max_hierarchy: 3                      # Maximum number of subactors
add_hierarchy_every: 1                # When to add new layers (environment steps)

# Temporal scaling
subactor_update_every: 4              # Update frequency multiplier
subactor_train_every: 4               # Training frequency multiplier

# Architecture
subactor_encoding_architecture: "mlp" # "mlp" or "cnn" for higher levels
subactor_encode_intermediate: True    # Whether to encode observations between levels

# World models
hierarchical_world_models: True       # Each subactor has own world model
higher_level_wm: true                 # Higher levels use world models

# Rewards
only_subgoal_reward: false            # Higher levels only optimize for subgoals
extrinsic_reward_weight: 1.0
subgoal_reward_weight: 0.3
novelty_reward_weight: 0.1
novelty_only_higher_level: false

# Subgoal representation
subgoal_shape: [8, 8]                 # Dimensionality of subgoals
decompress_subgoal_for_input: False   # Whether to decompress subgoals

# Optional: Size reduction per level
hierarchy_decrease_sizes:
  enabled: False
  sizes: ["dyn_hidden", "dyn_deter", "units"]
  factor: 2.0
  min: 32
```

## Usage Examples

### 1. Single-Level (Equivalent to DreamerV3)

```bash
python hieros/train.py --configs atari100k --task=atari_pong --max_hierarchy=1
```

### 2. Two-Level Hierarchy

```bash
python hieros/train.py --configs atari100k --task=atari_pong --max_hierarchy=2
```

### 3. Three-Level Hierarchy (Default)

```bash
python hieros/train.py --configs atari100k --task=atari_pong --max_hierarchy=3
```

### 4. Custom Temporal Scaling

```bash
python hieros/train.py --configs atari100k --task=atari_pong \
    --max_hierarchy=3 \
    --subactor_update_every=8 \
    --subactor_train_every=8
```

## Summary

### How Many Subactors?
- **Minimum**: 1 subactor (Subactor-0) - equivalent to baseline DreamerV3
- **Default**: Up to 3 subactors (configurable via `max_hierarchy`)
- **Dynamic**: Grows from 1 to `max_hierarchy` during training

### Where Are They Created?
- **Initial**: Subactor-0 created in `Hieros.__init__()` (lines 60-73)
- **Dynamic**: Additional subactors created in `_create_subactor()` method (lines 371-473)
- **Trigger**: New subactors added in `policy()` based on training progress (lines 262-271)

### Purpose of Subactors
1. **Temporal abstraction**: Multi-scale decision making
2. **Hierarchical control**: Goal-conditioned subgoal generation
3. **Progressive abstraction**: Encoded observation representations
4. **Reward decomposition**: Multiple reward components per level
5. **Improved credit assignment**: Decomposed learning problem
6. **Separate learning**: Independent replay and training schedules
7. **Modularity**: Configurable hierarchy depth and features

### Comparison with DreamerV3 and S5
- **DreamerV3**: Single-level actor, no hierarchical structure
- **S5**: Sequence model (dynamics component), not an RL architecture
- **Hieros**: Multi-level hierarchical extension of DreamerV3, optionally using S5 for dynamics
- **Unique to Hieros**: The hierarchical subactor architecture is a novel contribution not present in the baseline models

## References

1. **Repository Structure**: See `/hieros/hieros.py` for main implementation
2. **Configuration**: See `/hieros/configs.yaml` for all parameters
3. **Baseline DreamerV3**: See `/hieros/dreamer.py` for comparison
4. **S5 Implementation**: See `/resettable_s5/` for structured state space model
5. **Paper**: See `/documentation/hieros_icml.pdf` and `/documentation/hieros_thesis.pdf` for theoretical background

## Notes

- The term "subactor" is used consistently throughout the codebase to refer to each hierarchical level
- Each subactor is a complete agent with its own world model, policy, and replay buffer
- The architecture supports flexible configurations, allowing ablation studies on various components
- The hierarchical structure is optional - setting `max_hierarchy=1` reduces to standard DreamerV3
