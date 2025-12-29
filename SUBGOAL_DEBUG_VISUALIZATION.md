# Subgoal Debug Visualization

## Overview

This feature provides a debugging visualization that shows the **fixed subgoal representation** (without stochastic state) and the action sequences taken at each hierarchy level.

## Difference from `subgoal_visualization`

- **`subgoal_visualization`**: Shows subgoals with stochastic state added, making it difficult to see what is being commanded to the lower layer.
- **`subgoal_debug_visualization`**: Shows the **fixed deterministic subgoal representation** (the actual action/command sent to the lower layer) without stochastic state added.

## Usage

Enable the debug visualization by setting the configuration flag in your training command:

```bash
python hieros/train.py --configs atari100k --task=atari_alien --subgoal_debug_visualization=True
```

Or by modifying `hieros/configs.yaml`:

```yaml
defaults:
  subgoal_debug_visualization: True  # Enable debug visualization
  subgoal_visualization: True        # Can be used together with regular visualization
```

## What it shows

The debug visualization creates videos that display:

1. **Fixed subgoal representations**: The deterministic part of the subgoal (what is actually commanded to lower layers)
2. **Action sequences**: The sequence of actions taken at each hierarchy level
3. **Original observations**: The actual environment observations for reference

## Output

When enabled, the visualization will be logged to WandB under the key `subgoal_debug_visualization` and can be viewed alongside other training metrics.

## Implementation Details

- Actions are cached during policy execution in `_action_cache`
- The `_visualize_subgoals_debug()` method processes the cached actions
- Only the deterministic component of subgoals is visualized (stochastic state is zeroed out)
- The feature is designed to have minimal performance impact when disabled (default: False)

## Notes

- This feature requires `image` observations to be present in the observation space
- The debug visualization is only generated during training/exploration modes
- Videos are generated periodically based on the `log_every` configuration
