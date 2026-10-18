"""PPO training and fine-tuning with Brax, following the official Playground notebook.

Checkpoints are written to ``<ckpt_dir>/<cumulative_step:012d>/`` after every
training epoch, and a ``DONE`` marker is written when a run finishes. If a
Colab session dies, calling ``train_or_resume`` again continues from the latest
checkpoint instead of starting over.
"""

from __future__ import annotations

import functools
import json
import time
from typing import Any, Callable, Dict, List, Optional, Union

from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
from flax.training import orbax_utils
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params
import numpy as np
from orbax import checkpoint as ocp

from gait_personality.styles import ENV_NAME, Style, make_env_config

_NET_CONFIG = "ppo_network_config.json"


def ppo_config(
    num_timesteps: Optional[int] = None,
    num_envs: Optional[int] = None,
    num_evals: Optional[int] = None,
):
    """Tuned PPO hyper-parameters from Playground, with optional overrides."""
    params = locomotion_params.brax_ppo_config(ENV_NAME)
    if num_timesteps is not None:
        params.num_timesteps = int(num_timesteps)
    if num_envs is not None:
        params.num_envs = int(num_envs)
    if num_evals is not None:
        params.num_evals = int(num_evals)
    return params


def format_steps(n: float) -> str:
    """Human-readable step count: '14.7M' (or '800' for tiny test runs)."""
    return f"{n / 1e6:.1f}M" if abs(n) >= 1e5 else f"{int(n):,}"


def effective_timesteps(ppo_kwargs: Dict[str, Any]) -> int:
    """Number of environment steps Brax PPO will actually run.

    Brax splits training into ``num_evals - 1`` evaluation intervals, each made of
    ``max(num_resets_per_eval, 1)`` epochs of whole training steps, so the requested
    ``num_timesteps`` is rounded *up* to a multiple of that unit (about 14.7M
    steps with the Playground settings for Go1). Defaults mirror ``ppo.train``.
    """
    per_step = (
        int(ppo_kwargs.get("batch_size", 32))
        * int(ppo_kwargs.get("unroll_length", 10))
        * int(ppo_kwargs.get("num_minibatches", 16))
        * int(ppo_kwargs.get("action_repeat", 1))
    )
    intervals = max(int(ppo_kwargs.get("num_evals", 1)) - 1, 1)
    resets = max(int(ppo_kwargs.get("num_resets_per_eval", 0)), 1)
    epochs = int(np.ceil(int(ppo_kwargs["num_timesteps"]) / (intervals * per_step * resets)))
    return intervals * resets * epochs * per_step


def latest_checkpoint(ckpt_dir: Union[str, epath.Path]) -> Optional[epath.Path]:
    """Return the newest ``<step>`` sub-directory, or None if there is none."""
    ckpt_dir = epath.Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None
    steps = [
        p for p in ckpt_dir.iterdir()
        if p.is_dir() and p.name.isdigit() and (p / _NET_CONFIG).exists()
    ]
    if not steps:
        return None
    return max(steps, key=lambda p: int(p.name))


def is_done(ckpt_dir: Union[str, epath.Path]) -> bool:
    return (epath.Path(ckpt_dir) / "DONE").exists()


def load_history(ckpt_dir: Union[str, epath.Path]) -> List[Dict[str, float]]:
    path = epath.Path(ckpt_dir) / "history.json"
    return json.loads(path.read_text()) if path.exists() else []


def _save_params(path: epath.Path, params: Any, obs_size: Dict[str, Any], action_size: int) -> None:
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(path, params, force=True, save_args=orbax_utils.save_args_from_target(params))
    (path / _NET_CONFIG).write_text(
        json.dumps({
            "observation_size": {k: {"shape": list(v)} for k, v in obs_size.items()},
            "action_size": int(action_size),
        })
    )


def train_style(
    style: Union[str, Style],
    ckpt_dir: Union[str, epath.Path],
    num_timesteps: int,
    restore_from: Optional[Union[str, epath.Path]] = None,
    step_offset: int = 0,
    num_envs: Optional[int] = None,
    num_evals: Optional[int] = None,
    impl: Optional[str] = None,
    seed: int = 0,
    progress: Optional[Callable[[Dict[str, float]], None]] = None,
    ppo_overrides: Optional[Dict[str, Any]] = None,
):
    """Train (or fine-tune, if ``restore_from`` is given) one style.

    Args:
      step_offset: added to the step numbers of saved checkpoints / history
        (used when resuming, so numbering stays cumulative).
      ppo_overrides: extra keyword arguments for Brax ``ppo.train``
        (e.g. ``{"num_eval_envs": 64}``); mainly for quick CPU tests.

    Returns:
      (make_inference_fn, params, history)
    """
    env_cfg = make_env_config(style, impl=impl)
    env = registry.load(ENV_NAME, config=env_cfg)
    eval_env = registry.load(ENV_NAME, config=env_cfg)
    obs_size = {k: tuple(v) for k, v in env.observation_size.items()}

    params = ppo_config(num_timesteps, num_envs, num_evals)
    kwargs = dict(params)
    kwargs.update(ppo_overrides or {})
    net_kwargs = kwargs.pop("network_factory", None)
    network_factory = (
        functools.partial(ppo_networks.make_ppo_networks, **net_kwargs)
        if net_kwargs is not None
        else ppo_networks.make_ppo_networks
    )

    ckpt_dir = epath.Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "env_config.json").write_text(json.dumps(env_cfg.to_dict(), indent=2, default=str))

    actual_timesteps = effective_timesteps(kwargs)
    if actual_timesteps != int(num_timesteps):
        name = style if isinstance(style, str) else style.name
        print(f"[{name}] 学習量: 指定 {format_steps(num_timesteps)} → 実際は {format_steps(actual_timesteps)} ステップ"
              "（Brax が評価の区切りに合わせて切り上げるため）", flush=True)

    history = load_history(ckpt_dir)
    t0 = time.time()

    def _progress(step: int, metrics: Dict[str, Any]) -> None:
        row = {
            "step": int(step) + step_offset,
            "reward": float(metrics.get("eval/episode_reward", float("nan"))),
            "reward_std": float(metrics.get("eval/episode_reward_std", float("nan"))),
            "elapsed_s": time.time() - t0,
            "target_step": actual_timesteps + step_offset,
        }
        history.append(row)
        (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))
        if progress is not None:
            progress(row)

    def _policy_params_fn(step: int, make_policy, p) -> None:
        del make_policy
        if step == 0:  # initial (or restored) parameters - nothing new to save
            return
        _save_params(ckpt_dir / f"{step + step_offset:012d}", p, obs_size, env.action_size)

    make_inference_fn, trained_params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=network_factory,
        randomization_fn=registry.get_domain_randomizer(ENV_NAME),
        progress_fn=_progress,
        policy_params_fn=_policy_params_fn,
        restore_checkpoint_path=str(restore_from) if restore_from else None,
        seed=seed,
        **kwargs,
    )
    return make_inference_fn, trained_params, history


def _is_oom(err: Exception) -> bool:
    msg = str(err)
    return "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower()


def train_or_resume(
    style: Union[str, Style],
    ckpt_dir: Union[str, epath.Path],
    total_timesteps: int,
    init_from: Optional[Union[str, epath.Path]] = None,
    num_envs: Optional[int] = None,
    min_num_envs: int = 1024,
    **kwargs,
) -> List[Dict[str, float]]:
    """Train until ``total_timesteps`` (cumulative), resuming if checkpoints exist.

    - Already finished (``DONE`` marker) -> does nothing.
    - Some checkpoints exist -> continues from the latest one.
    - Otherwise starts from ``init_from`` (e.g. the base policy) or from scratch.
    - On GPU out-of-memory, retries with half as many parallel environments.

    Returns the training history.
    """
    ckpt_dir = epath.Path(ckpt_dir)
    if is_done(ckpt_dir):
        print(f"[skip] {ckpt_dir.name}: already trained")
        return load_history(ckpt_dir)

    latest = latest_checkpoint(ckpt_dir)
    if latest is not None:
        offset, restore = int(latest.name), latest
        print(f"[resume] {ckpt_dir.name}: continuing from step {offset:,}")
    else:
        offset, restore = 0, init_from
    remaining = int(total_timesteps) - offset

    envs = num_envs or int(ppo_config().num_envs)
    while remaining > 0:
        try:
            train_style(
                style, ckpt_dir, remaining, restore_from=restore,
                step_offset=offset, num_envs=envs, **kwargs,
            )
            break
        except Exception as err:  # noqa: BLE001 - re-raised unless it is an OOM
            if _is_oom(err) and envs // 2 >= min_num_envs:
                envs //= 2
                print(f"[retry] GPU memory is not enough -> num_envs = {envs}")
                continue
            raise
    (ckpt_dir / "DONE").write_text(json.dumps({"total_timesteps": int(total_timesteps)}))
    return load_history(ckpt_dir)


def load_policy(ckpt_step_dir: Union[str, epath.Path], deterministic: bool = True):
    """Load an inference function ``(obs, rng) -> (action, extras)`` from a checkpoint.

    The network is rebuilt from the same PPO settings used for training and only
    the parameters are read from disk.
    """
    ckpt_step_dir = epath.Path(ckpt_step_dir)
    params = ppo_checkpoint.load(ckpt_step_dir)
    cfg = json.loads((ckpt_step_dir / _NET_CONFIG).read_text())
    obs_size = {k: tuple(v["shape"]) for k, v in cfg["observation_size"].items()}
    net_kwargs = dict(ppo_config().network_factory)
    network = ppo_networks.make_ppo_networks(
        obs_size,
        cfg["action_size"],
        preprocess_observations_fn=running_statistics.normalize,
        **net_kwargs,
    )
    return ppo_networks.make_inference_fn(network)(params, deterministic=deterministic)
