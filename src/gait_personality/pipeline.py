"""End-to-end steps used by the Colab notebook (and the CPU smoke test).

Directory layout under ``run_dir`` (put it on Google Drive in Colab):

    checkpoints/<style>/<step>/   PPO parameters, DONE marker, history.json
    eval/<style>_logs.npz         raw evaluation rollouts
    results/                      metrics.csv, summary.csv, summary.md, figures
    videos/                       final_video.mp4, comparison_grid.mp4
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from etils import epath

from gait_personality import evaluation, plots, training, video
from gait_personality.styles import STYLES

BASE_STYLE = "standard"


@dataclass
class RunConfig:
    run_dir: str
    styles: Sequence[str] = ("serving", "eco", "energetic")  # fine-tuned from the base
    base_timesteps: int = 100_000_000
    finetune_timesteps: int = 40_000_000
    num_envs: Optional[int] = None  # None = Playground default (8192)
    num_evals: Optional[int] = None  # None = Playground default
    impl: Optional[str] = None  # None = Playground default ("warp"); "jax" as fallback
    eval_seeds: int = 8
    seed: int = 0
    ppo_overrides: Dict[str, Any] = field(default_factory=dict)  # for tests

    @property
    def all_styles(self) -> List[str]:
        return [BASE_STYLE] + [s for s in self.styles if s != BASE_STYLE]

    def path(self, *parts: str) -> epath.Path:
        p = epath.Path(self.run_dir).joinpath(*parts)
        return p

    def save(self) -> None:
        self.path().mkdir(parents=True, exist_ok=True)
        self.path("run_config.json").write_text(json.dumps(asdict(self), indent=2, default=str))


def _progress_printer(style: str):
    def show(row: Dict[str, float]) -> None:
        done, target = row["step"], max(row["target_step"], 1)
        eta = ""
        if 0 < done < target and row["elapsed_s"] > 0:
            # elapsed covers only this (possibly resumed) run; good enough for an estimate
            rate = done / row["elapsed_s"] if row["elapsed_s"] else 0
            if rate > 0:
                eta = f" | 残り約 {max(target - done, 0) / rate / 60:.0f} 分"
        print(f"[{style}] step {done/1e6:6.1f}M / {target/1e6:.0f}M | reward {row['reward']:7.2f}"
              f" | 経過 {row['elapsed_s']/60:5.1f} 分{eta}", flush=True)
    return show


# ------------------------------------------------------------------------ training
def train_base(cfg: RunConfig) -> List[Dict[str, float]]:
    """Train the standard gait from scratch (resumable)."""
    cfg.save()
    return training.train_or_resume(
        BASE_STYLE, cfg.path("checkpoints", BASE_STYLE), cfg.base_timesteps,
        num_envs=cfg.num_envs, num_evals=cfg.num_evals, impl=cfg.impl, seed=cfg.seed,
        progress=_progress_printer(BASE_STYLE), ppo_overrides=cfg.ppo_overrides,
    )


def finetune_styles(cfg: RunConfig) -> Dict[str, List[Dict[str, float]]]:
    """Fine-tune every personality from the latest base checkpoint (resumable)."""
    base = training.latest_checkpoint(cfg.path("checkpoints", BASE_STYLE))
    if base is None:
        raise RuntimeError("Base policy not found. Run train_base first.")
    histories = {}
    for style in cfg.styles:
        if style == BASE_STYLE:
            continue
        histories[style] = training.train_or_resume(
            style, cfg.path("checkpoints", style), cfg.finetune_timesteps, init_from=base,
            num_envs=cfg.num_envs, num_evals=cfg.num_evals, impl=cfg.impl, seed=cfg.seed + 1,
            progress=_progress_printer(style), ppo_overrides=cfg.ppo_overrides,
        )
    return histories


def histories(cfg: RunConfig) -> Dict[str, List[Dict[str, float]]]:
    return {s: training.load_history(cfg.path("checkpoints", s)) for s in cfg.all_styles}


# ---------------------------------------------------------------------- evaluation
def evaluate(cfg: RunConfig, commands_slice: Optional[slice] = None):
    """Roll out every trained style with the same commands; save metrics."""
    rows, logs_by_style = [], {}
    cfg.path("eval").mkdir(parents=True, exist_ok=True)
    cfg.path("results").mkdir(parents=True, exist_ok=True)
    seeds = list(range(1000, 1000 + cfg.eval_seeds))
    for style in cfg.all_styles:
        ckpt = training.latest_checkpoint(cfg.path("checkpoints", style))
        if ckpt is None:
            print(f"[skip] {style}: no checkpoint yet")
            continue
        env = evaluation.make_eval_env(style, impl=cfg.impl)
        cmds, _ = evaluation.command_array(env.dt)
        if commands_slice is not None:
            cmds = cmds[commands_slice]
        logs = evaluation.rollout(env, training.load_policy(ckpt), cmds, seeds)
        np.savez_compressed(cfg.path("eval", f"{style}_logs.npz"), **logs)
        logs_by_style[style] = logs
        rows.append(evaluation.metrics_table(style, logs, evaluation.total_mass(env), env.dt))
        print(f"[eval] {style}: {len(seeds)} episodes from {ckpt.name}")
    df = pd.concat(rows, ignore_index=True)
    summary = evaluation.summarize(df)
    df.to_csv(cfg.path("results", "metrics.csv"), index=False)
    summary.to_csv(cfg.path("results", "summary.csv"), index=False)
    labels = {s: f"{STYLES[s].label_ja}（{STYLES[s].label}）" for s in summary["style"]}
    md = [
        "# 評価結果（自動生成）", "",
        f"各性格につき {cfg.eval_seeds} エピソード。値は平均 ± 95%信頼区間の半幅。", "",
        evaluation.summary_markdown(summary, labels), "",
    ]
    if BASE_STYLE in set(summary["style"]):
        md += ["## 標準との比較", ""]
        for style, line in evaluation.highlights_ja(summary).items():
            md.append(f"- {labels[style]}: {line}")
    cfg.path("results", "summary.md").write_text("\n".join(md) + "\n")
    return df, summary, logs_by_style


def load_eval_logs(cfg: RunConfig) -> Dict[str, Dict[str, np.ndarray]]:
    out = {}
    for style in cfg.all_styles:
        p = cfg.path("eval", f"{style}_logs.npz")
        if p.exists():
            with p.open("rb") as f:
                out[style] = dict(np.load(f))
    return out


# ------------------------------------------------------------------------- figures
def make_figures(cfg: RunConfig, summary: pd.DataFrame) -> Dict[str, epath.Path]:
    out = {}
    cfg.path("results").mkdir(parents=True, exist_ok=True)
    figs = {
        "tradeoff": plots.tradeoff_plot(summary),
        "metrics": plots.metric_bars(summary),
        "training": plots.training_curves(histories(cfg)),
    }
    for name, fig in figs.items():
        path = cfg.path("results", f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        out[name] = path
    return out


# -------------------------------------------------------------------------- videos
def make_videos(
    cfg: RunConfig,
    summary: pd.DataFrame,
    logs_by_style: Dict[str, Dict[str, np.ndarray]],
    figures: Dict[str, epath.Path],
    size=(720, 1280),
    every: int = 2,
    clip_seconds: Sequence[float] = (2.0, 14.0),
) -> Dict[str, epath.Path]:
    """Assemble the submission video (~1.5 min) and a short comparison video."""
    jp_text = video.has_cjk_font()
    h, w = size
    styles = [s for s in cfg.all_styles if s in logs_by_style]
    env = evaluation.make_eval_env(BASE_STYLE, impl=cfg.impl)  # same robot model for all styles
    dt = env.dt
    fps = 1.0 / (dt * every)
    _, cmd_labels = evaluation.command_array(dt, japanese=jp_text)
    powers = {s: evaluation.joint_power(logs_by_style[s]) for s in styles}

    def name(s: str) -> str:
        return f"{STYLES[s].label_ja}  {STYLES[s].label}" if jp_text else STYLES[s].label

    out_dir = cfg.path("videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path, grid_path = out_dir / "final_video.mp4", out_dir / "comparison_grid.mp4"

    if jp_text:
        title = ["報酬で「歩き方の性格」をデザインする", "四足ロボット Unitree Go1 × 強化学習（PPO, MuJoCo Playground）", "等速再生"]
        concept = [
            "同じロボット・同じ学習アルゴリズム",
            "変えたのは「何を良しとするか」= 報酬の重みだけ",
            "標準の歩き方を学習 → 報酬を変えて追加学習（ファインチューニング）",
            " / ".join(STYLES[s].label_ja for s in styles),
        ]
        closing_head = "まとめ（標準との比較、評価 {} 回の平均）".format(cfg.eval_seeds)
    else:
        title = ["Designing gait personalities with rewards", "Unitree Go1 x reinforcement learning (PPO, MuJoCo Playground)", "real time (1x)"]
        concept = [
            "Same robot, same learning algorithm",
            "Only the reward weights are changed",
            "Train a standard gait, then fine-tune each personality",
            " / ".join(STYLES[s].label for s in styles),
        ]
        closing_head = f"Summary vs. standard (mean of {cfg.eval_seeds} episodes)"

    closing = [closing_head]
    if BASE_STYLE in set(summary["style"]):
        for s, line in evaluation.highlights_ja(summary).items():
            closing.append(f"{STYLES[s].label_ja}: {line}" if jp_text else f"{STYLES[s].label}: {line}")

    full = video.Renderer(env.mj_model, h, w)
    half = video.Renderer(env.mj_model, h // 2, w // 2)
    start, stop = (int(round(t / dt)) for t in clip_seconds)
    with video.MultiWriter([final_path], h, w, fps) as final:
        final.hold(video.text_card(title, h, w), 4.0)
        final.hold(video.text_card(concept, h, w, sizes=[44, 34, 30, 30]), 7.0)
        for s in styles:
            video.write_style_clip(
                final, full, logs_by_style[s], powers[s], name(s),
                STYLES[s].subtitle_ja if jp_text else STYLES[s].description,
                cmd_labels, jp_text, start, stop, every,
            )
        with video.MultiWriter([grid_path], h, w, fps) as grid_only:
            video.write_grid(
                video.Tee(final, grid_only), half, [logs_by_style[s] for s in styles], [powers[s] for s in styles],
                [name(s) for s in styles], cmd_labels, jp_text, every, out_size=(h, w),
            )
        if "tradeoff" in figures:
            final.hold(video.image_card(str(figures["tradeoff"]), h, w,
                                        "トレードオフ（左下ほど効率的で正確）" if jp_text else "Trade-offs"), 6.0)
        if "metrics" in figures:
            final.hold(video.image_card(str(figures["metrics"]), h, w,
                                        "指標ごとの比較（平均と95%信頼区間）" if jp_text else "Metrics (mean, 95% CI)"), 6.0)
        final.hold(video.text_card(closing, h, w, sizes=[40] + [30] * (len(closing) - 1)), 7.0)
    full.close()
    half.close()
    return {"final": final_path, "grid": grid_path}


def load_results(cfg: RunConfig):
    """Reload evaluation results saved by ``evaluate`` / ``make_figures`` (e.g. after a restart)."""
    summary = pd.read_csv(cfg.path("results", "summary.csv"))
    logs = load_eval_logs(cfg)
    figures = {
        name: cfg.path("results", f"{name}.png")
        for name in ("tradeoff", "metrics", "training")
        if cfg.path("results", f"{name}.png").exists()
    }
    return summary, logs, figures
