"""CPU smoke test: runs the same pipeline as the notebook with tiny settings.

It checks that the code paths work (base training, resumable checkpoints,
fine-tuning, evaluation rollouts, metrics, figures, and the assembled video).
The resulting policies are meaningless; this only proves the pipeline runs.

    MUJOCO_GL=egl python tests/smoke_test_cpu.py
    SMOKE_OUT=/tmp/gait_smoke python tests/smoke_test_cpu.py   # reuse checkpoints
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from gait_personality import STYLES, describe, training  # noqa: E402
from gait_personality.pipeline import (  # noqa: E402
    RunConfig, evaluate, finetune_styles, make_figures, make_videos, train_base,
)

TINY_PPO = {
    "batch_size": 8,
    "num_minibatches": 2,
    "unroll_length": 5,
    "num_updates_per_batch": 1,
    "episode_length": 20,
    "num_eval_envs": 4,
}


def main() -> None:
    t0 = time.time()
    run_dir = os.environ.get("SMOKE_OUT") or tempfile.mkdtemp(prefix="gait_smoke_")
    cfg = RunConfig(
        run_dir=run_dir, styles=("serving",), base_timesteps=160, finetune_timesteps=160,
        num_envs=8, num_evals=2, impl="jax", eval_seeds=2, ppo_overrides=TINY_PPO,
    )
    print("run dir:", run_dir)
    for name in STYLES:
        print(describe(name))

    train_base(cfg)
    assert training.is_done(cfg.path("checkpoints", "standard"))
    # second call must be a no-op (resumable / skip logic)
    train_base(cfg)
    finetune_styles(cfg)
    assert training.latest_checkpoint(cfg.path("checkpoints", "serving")) is not None

    df, summary, logs = evaluate(cfg, commands_slice=slice(90, 150))
    print(summary.round(3).to_string())
    figures = make_figures(cfg, summary)
    videos = make_videos(cfg, summary, logs, figures, size=(360, 640), clip_seconds=(0.0, 0.6))

    expected = [cfg.path("results", n) for n in ("metrics.csv", "summary.csv", "summary.md")]
    expected += list(figures.values()) + list(videos.values())
    for p in expected:
        assert p.exists() and os.path.getsize(str(p)) > 0, p
    print(cfg.path("results", "summary.md").read_text())
    print(f"SMOKE TEST OK in {time.time() - t0:.0f}s -> {run_dir}")


if __name__ == "__main__":
    main()
