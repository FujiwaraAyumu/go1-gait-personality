"""Designing quadruped gait "personalities" with reward weights.

The package is a thin layer on top of MuJoCo Playground (Go1JoystickFlatTerrain):

- styles:     reward-weight presets ("personalities")
- training:   PPO training / fine-tuning with checkpoints
- evaluation: fixed command schedule rollouts and gait metrics
- video:      rendering and side-by-side comparison videos
- plots:      trade-off figures
"""

from gait_personality.styles import ENV_NAME, STYLES, Style, describe, make_env_config

__all__ = ["ENV_NAME", "STYLES", "Style", "describe", "make_env_config"]
