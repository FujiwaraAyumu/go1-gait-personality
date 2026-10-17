"""Evaluation rollouts with a fixed command schedule, and gait metrics.

All styles are driven by exactly the same joystick commands, so differences in
the metrics come only from the learned "personality".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jp
import numpy as np
import pandas as pd
from mujoco_playground import registry
from scipy import stats

from gait_personality.styles import ENV_NAME, Style, make_env_config

# (duration [s], vx [m/s], vy [m/s], yaw rate [rad/s], label_en, label_ja)
COMMAND_SCHEDULE: Tuple[Tuple[float, float, float, float, str, str], ...] = (
    (2.0, 0.0, 0.0, 0.0, "stand still", "その場で静止"),
    (5.0, 0.6, 0.0, 0.0, "walk forward 0.6 m/s", "前進 0.6 m/s"),
    (5.0, 1.2, 0.0, 0.0, "walk forward 1.2 m/s", "前進 1.2 m/s"),
    (4.0, 0.6, 0.0, 0.8, "walk + turn left", "前進しながら左旋回"),
    (4.0, 0.0, 0.4, 0.0, "side-step left", "左へ横歩き"),
)

GRAVITY = 9.81


def command_array(
    dt: float,
    schedule: Sequence[Tuple[float, float, float, float, str, str]] = COMMAND_SCHEDULE,
    japanese: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """Expand the schedule into per-step commands (T, 3) and per-step labels."""
    cmds, labels = [], []
    for duration, vx, vy, wz, label_en, label_ja in schedule:
        n = int(round(duration / dt))
        cmds.extend([[vx, vy, wz]] * n)
        labels.extend([label_ja if japanese else label_en] * n)
    return np.asarray(cmds, dtype=np.float32), labels


def make_eval_env(style: Union[str, Style], impl: Optional[str] = None):
    """Environment for evaluation: same rewards as training, no external pushes."""
    cfg = make_env_config(style, impl=impl)
    cfg.pert_config.enable = False
    return registry.load(ENV_NAME, config=cfg)


def total_mass(env) -> float:
    return float(env.mj_model.body_mass.sum())


def rollout(env, inference_fn, commands: np.ndarray, seeds: Sequence[int]) -> Dict[str, np.ndarray]:
    """Run deterministic-policy rollouts for several seeds in parallel.

    The joystick command is overwritten at every step, so the environment's own
    random command resampling never takes effect.

    Returns:
      dict of numpy arrays with leading axes (num_seeds, T, ...).
    """
    commands = jp.asarray(commands)

    def one_episode(seed):
        rng = jax.random.PRNGKey(seed)
        rng, reset_key = jax.random.split(rng)
        state = env.reset(reset_key)

        def body(carry, cmd):
            state, rng = carry
            state.info["command"] = cmd
            rng, act_key = jax.random.split(rng)
            action, _ = inference_fn(state.obs, act_key)
            state = env.step(state, action)
            data = state.data
            log = {
                "command": cmd,
                "local_linvel": env.get_local_linvel(data),
                "global_linvel": env.get_global_linvel(data),
                "global_angvel": env.get_global_angvel(data),
                "gyro": env.get_gyro(data),
                "upvector": env.get_upvector(data),
                "qpos": data.qpos,
                "qvel": data.qvel,
                "actuator_force": data.actuator_force,
                "action": action,
                "swing_peak": state.info["swing_peak"],
                "contact": state.info["last_contact"],
                "done": state.done,
            }
            return (state, rng), log

        _, logs = jax.lax.scan(body, (state, rng), commands)
        return logs

    logs = jax.jit(jax.vmap(one_episode))(jp.asarray(seeds, dtype=jp.int32))
    return {k: np.asarray(v) for k, v in jax.device_get(logs).items()}


def joint_power(logs: Dict[str, np.ndarray]) -> np.ndarray:
    """Mechanical power sum_j |tau_j * qdot_j| per step, shape (S, T) [W]."""
    return np.sum(np.abs(logs["qvel"][..., 6:] * logs["actuator_force"]), axis=-1)


def _first_done(done: np.ndarray) -> int:
    idx = np.flatnonzero(done > 0.5)
    return int(idx[0]) + 1 if idx.size else len(done)


def episode_metrics(
    logs: Dict[str, np.ndarray], seed_index: int, mass: float, dt: float
) -> Dict[str, float]:
    """Gait metrics for one episode (one seed)."""
    s = seed_index
    t_end = _first_done(logs["done"][s])
    sl = slice(0, t_end)

    cmd = logs["command"][s, sl]
    v_local = logs["local_linvel"][s, sl]
    v_global = logs["global_linvel"][s, sl]
    angvel = logs["global_angvel"][s, sl]
    gyro = logs["gyro"][s, sl]
    up_z = np.clip(logs["upvector"][s, sl, 2], -1.0, 1.0)
    torso_z = logs["qpos"][s, sl, 2]
    action = logs["action"][s, sl]
    swing = logs["swing_peak"][s, sl]
    power = joint_power(logs)[s, sl]

    moving = np.linalg.norm(cmd[:, :2], axis=-1) > 1e-3
    speed = np.linalg.norm(v_global[:, :2], axis=-1)

    vel_err = np.sqrt(np.mean(np.sum((cmd[:, :2] - v_local[:, :2]) ** 2, axis=-1)))
    yaw_err = np.sqrt(np.mean((cmd[:, 2] - gyro[:, 2]) ** 2))

    distance = np.sum(speed[moving]) * dt
    energy = np.sum(power[moving]) * dt
    cot = energy / (mass * GRAVITY * distance) if distance > 1e-3 else np.nan

    rp_rate = np.sqrt(np.mean(np.sum(angvel[moving, :2] ** 2, axis=-1)))
    tilt_deg = np.degrees(np.arccos(up_z[moving]))
    tilt_rms = np.sqrt(np.mean(tilt_deg**2))
    height_std_cm = np.std(torso_z[moving]) * 100.0
    jerk = np.mean(np.sum(np.diff(action, axis=0) ** 2, axis=-1)) if len(action) > 1 else np.nan
    # swing_peak holds the highest foot height of the current swing and resets
    # to 0 at touchdown, so the value just before each reset is one step's peak.
    touchdown = (swing[:-1] > 0) & (swing[1:] == 0) & moving[1:, None]
    peaks = swing[:-1][touchdown]
    foot_lift_cm = float(np.mean(peaks) * 100.0) if peaks.size else np.nan

    return {
        "vel_err": float(vel_err),  # m/s, RMS of xy velocity tracking error
        "yaw_err": float(yaw_err),  # rad/s, RMS of yaw-rate tracking error
        "cot": float(cot),  # cost of transport (dimensionless)
        "mean_power_w": float(np.mean(power[moving])),  # W while moving
        "rp_rate_rms": float(rp_rate),  # rad/s, torso roll/pitch rate (shaking)
        "tilt_rms_deg": float(tilt_rms),  # deg, torso tilt from vertical
        "height_std_cm": float(height_std_cm),  # cm, torso bobbing
        "action_jerk": float(jerk),  # mean squared action change per step
        "foot_lift_cm": float(foot_lift_cm),  # cm, mean peak foot height per swing
        "fell": float(t_end < len(logs["done"][s])),
        "steps": int(t_end),
    }


METRIC_INFO = {
    # key: (label for plots, lower_is_better)
    "vel_err": ("Velocity tracking error [m/s]", True),
    "yaw_err": ("Yaw-rate tracking error [rad/s]", True),
    "cot": ("Cost of transport [-]", True),
    "mean_power_w": ("Mean joint power [W]", True),
    "rp_rate_rms": ("Torso roll/pitch rate RMS [rad/s]", True),
    "tilt_rms_deg": ("Torso tilt RMS [deg]", True),
    "height_std_cm": ("Torso height std [cm]", True),
    "action_jerk": ("Action change per step [-]", True),
    "foot_lift_cm": ("Foot height at swing peak [cm]", False),
}


SHORT_JA = {
    "vel_err": "速度の追従誤差",
    "yaw_err": "旋回の追従誤差",
    "cot": "移動コスト",
    "mean_power_w": "消費パワー",
    "rp_rate_rms": "胴体の揺れ",
    "tilt_rms_deg": "胴体の傾き",
    "height_std_cm": "胴体の上下動",
    "action_jerk": "動きの急変",
    "foot_lift_cm": "足の高さ",
}
KEY_METRICS = ("cot", "vel_err", "rp_rate_rms", "height_std_cm", "foot_lift_cm")


def metrics_table(style_name: str, logs: Dict[str, np.ndarray], mass: float, dt: float) -> pd.DataFrame:
    rows = []
    for s in range(logs["done"].shape[0]):
        row = episode_metrics(logs, s, mass, dt)
        row["style"] = style_name
        row["seed_index"] = s
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Mean and t-based confidence half-width per style and metric."""
    out = []
    for style, g in df.groupby("style", sort=False):
        row = {"style": style, "n": len(g), "fall_rate": g["fell"].mean()}
        for m in METRIC_INFO:
            x = g[m].dropna().to_numpy()
            n = len(x)
            mean = float(np.mean(x)) if n else np.nan
            if n > 1:
                half = float(stats.t.ppf(0.5 + confidence / 2, n - 1) * np.std(x, ddof=1) / np.sqrt(n))
            else:
                half = np.nan
            row[m] = mean
            row[f"{m}_ci"] = half
        out.append(row)
    return pd.DataFrame(out)


def relative_changes(
    summary: pd.DataFrame, baseline: str = "standard", metrics: Sequence[str] = KEY_METRICS
) -> pd.DataFrame:
    """Percent change of each style's mean metric versus the baseline style."""
    base = summary.set_index("style").loc[baseline]
    rows = []
    for _, r in summary.iterrows():
        if r["style"] == baseline:
            continue
        row = {"style": r["style"]}
        for m in metrics:
            b = base[m]
            row[m] = float((r[m] - b) / abs(b) * 100.0) if b and np.isfinite(b) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def highlights_ja(
    summary: pd.DataFrame, baseline: str = "standard", metrics: Sequence[str] = KEY_METRICS, top: int = 3
) -> Dict[str, str]:
    """One Japanese line per style with its largest changes versus the baseline."""
    rel = relative_changes(summary, baseline, metrics)
    out = {}
    for _, r in rel.iterrows():
        items = sorted(
            ((m, r[m]) for m in metrics if np.isfinite(r[m])), key=lambda kv: -abs(kv[1])
        )[:top]
        out[r["style"]] = " / ".join(f"{SHORT_JA[m]} {v:+.0f}%" for m, v in items)
    return out


def summary_markdown(summary: pd.DataFrame, labels: Dict[str, str], metrics: Sequence[str] = KEY_METRICS) -> str:
    """Markdown table: mean ± 95% CI half-width per style."""
    head = "| 性格 | " + " | ".join(METRIC_INFO[m][0] for m in metrics) + " | 転倒率 |"
    sep = "|---" * (len(metrics) + 2) + "|"
    lines = [head, sep]
    for _, r in summary.iterrows():
        cells = []
        for m in metrics:
            ci = r.get(f"{m}_ci", np.nan)
            cells.append(f"{r[m]:.3g} ± {ci:.2g}" if np.isfinite(ci) else f"{r[m]:.3g}")
        lines.append(f"| {labels.get(r['style'], r['style'])} | " + " | ".join(cells) + f" | {r['fall_rate']:.0%} |")
    return "\n".join(lines)
