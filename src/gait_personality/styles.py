"""Gait "personalities" defined only by reward weights.

Every style starts from the default reward configuration of
``Go1JoystickFlatTerrain`` in MuJoCo Playground and overrides a few entries.
Observations, actions and the network stay identical, so every style can be
fine-tuned from the same base (``standard``) checkpoint.

Edit ``STYLES`` to try other personalities. Keep multipliers moderate
(about 2-5x of the default): the per-step reward is clipped at 0 by the
environment, so very large penalties make the reward signal vanish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

from ml_collections import config_dict
from mujoco_playground import registry

ENV_NAME = "Go1JoystickFlatTerrain"


@dataclass(frozen=True)
class Style:
    """A gait personality = a set of reward-weight overrides."""

    name: str  # folder-friendly key
    label: str  # English label (videos / plots)
    label_ja: str  # Japanese label (docs / report / video)
    description: str
    subtitle_ja: str = ""  # one-line Japanese explanation for the video
    # Overrides for env_cfg.reward_config.scales[...]
    scales: Dict[str, float] = field(default_factory=dict)
    # Overrides for other env_cfg.reward_config entries (e.g. max_foot_height)
    reward_params: Dict[str, float] = field(default_factory=dict)


STYLES: Dict[str, Style] = {
    "standard": Style(
        name="standard",
        label="Standard",
        label_ja="標準",
        description="Default reward weights of MuJoCo Playground (baseline).",
        subtitle_ja="既定の報酬（比較の基準）",
    ),
    "serving": Style(
        name="serving",
        label="Serving",
        label_ja="配膳モード",
        description="Keep the torso level and steady, as if carrying a tray.",
        subtitle_ja="胴体の傾き・揺れへの罰を強化（物を載せて運ぶ想定）",
        scales={
            "orientation": -15.0,  # default -5.0  : torso tilt
            "ang_vel_xy": -0.25,  # default -0.05 : roll / pitch rate
            "lin_vel_z": -2.0,  # default -0.5  : vertical bobbing
        },
    ),
    "eco": Style(
        name="eco",
        label="Eco",
        label_ja="省エネモード",
        description="Spend less energy: penalize power, torque and jerky actions; lift feet lower.",
        subtitle_ja="エネルギー・急な動きへの罰を強化し、足を低く運ぶ",
        scales={
            "energy": -0.003,  # default -0.001  : |joint velocity x torque|
            "torques": -0.0006,  # default -0.0002 : torque magnitude
            "action_rate": -0.03,  # default -0.01   : action changes
        },
        reward_params={"max_foot_height": 0.07},  # default 0.10 m
    ),
    "energetic": Style(
        name="energetic",
        label="Energetic",
        label_ja="元気モード",
        description="High, bouncy steps with long swing phases.",
        subtitle_ja="足を高く上げ、滞空時間を長くする",
        scales={
            "feet_air_time": 0.5,  # default 0.1     : reward long swing phases
            "energy": -0.0005,  # default -0.001  : allow more power
        },
        reward_params={"max_foot_height": 0.15},  # default 0.10 m
    ),
}


def _resolve(style: Union[str, Style]) -> Style:
    if isinstance(style, Style):
        return style
    if style not in STYLES:
        raise KeyError(f"Unknown style '{style}'. Available: {list(STYLES)}")
    return STYLES[style]


def make_env_config(
    style: Union[str, Style] = "standard", impl: Optional[str] = None
) -> config_dict.ConfigDict:
    """Default env config with the style's reward overrides applied.

    Args:
      style: style key in ``STYLES`` or a ``Style`` instance.
      impl: physics backend. ``None`` keeps the Playground default ("warp");
        use "jax" if MuJoCo Warp does not work on your GPU.
    """
    s = _resolve(style)
    cfg = registry.get_default_config(ENV_NAME)
    for key, value in s.scales.items():
        if key not in cfg.reward_config.scales:
            raise KeyError(f"'{key}' is not a reward term of {ENV_NAME}")
        cfg.reward_config.scales[key] = float(value)
    for key, value in s.reward_params.items():
        if key not in cfg.reward_config:
            raise KeyError(f"'{key}' is not a reward_config entry of {ENV_NAME}")
        cfg.reward_config[key] = float(value)
    if impl is not None:
        cfg.impl = impl
    return cfg


def describe(style: Union[str, Style]) -> str:
    """Human-readable list of what the style changes compared with the default."""
    s = _resolve(style)
    base = registry.get_default_config(ENV_NAME).reward_config
    lines = [f"{s.label} ({s.label_ja}): {s.description}"]
    if not s.scales and not s.reward_params:
        lines.append("  (no changes - default weights)")
    for key, value in s.scales.items():
        lines.append(f"  scales.{key}: {base.scales[key]} -> {value}")
    for key, value in s.reward_params.items():
        lines.append(f"  {key}: {base[key]} -> {value}")
    return "\n".join(lines)
