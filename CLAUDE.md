# CLAUDE.md

このリポジトリで作業する Claude 向けのメモです。
公開リポジトリに含まれるため、**個人情報（氏名・所属・メールアドレス・PC のユーザー名を含む絶対パス）は書かない**こと。

## プロジェクト

- 目的：Physical AI講座 応用編1 最終課題「AI×ロボティクスによる創造的タスクの実現」
- テーマ：報酬で「歩き方の性格」をデザインする四足ロボット
  - Unitree Go1 × MuJoCo Playground（`Go1JoystickFlatTerrain`）× PPO（Brax）
  - 標準の歩き方を学習 → 報酬の重みだけを変えて性格ごとにファインチューニング → 同じ指令で評価・比較
- 提出物：動画（1〜3分、YouTube、限定公開以上）／コード（このリポジトリ、public）／レポート PDF
- 締切：2026-11-02 10:00（日本時間）

## 進め方の方針

- **作業は最大でも丸1日で終わる設計を最優先**。範囲が広がる提案は「余裕があれば」として分ける
- 人手のデモ収集・アノテーションは使わない（強化学習と報酬設計だけで作る）
- 学習・評価・動画作成は Google Colab（GPU）。ローカル（Windows）では編集と Git 操作だけ
- ユーザーとのやりとりとドキュメントは日本語。コードの識別子とコメントは英語
- 説明は手順を具体的に書く。ロボット制御・環境構築の前提知識は仮定しない
- 機械学習・統計の部分（指標の定義、信頼区間、トレードオフの読み取り）は、手を抜かずに正確に書く

## 構成

```text
notebooks/go1_gait_personality.ipynb  Colab の入口。各セルは pipeline.py の関数を呼ぶだけ
src/gait_personality/
  styles.py      性格 = 報酬の重みの上書き（STYLES）
  training.py    学習・ファインチューニング、再開可能なチェックポイント、OOM 時の自動縮小
  evaluation.py  指令スケジュール（COMMAND_SCHEDULE）、ロールアウト、指標、要約
  plots.py       図（色は検証済みパレット。標準=グレー、配膳=青、省エネ=橙、元気=青緑）
  video.py       描画・字幕（Pillow）・MP4 へ逐次書き出し
  pipeline.py    RunConfig と、train_base / finetune_styles / evaluate / make_figures / make_videos
tests/smoke_test_cpu.py  GPU なしで全工程を極小設定で通す
results/       Colab の結果のうち小さいもの（md・csv・png）だけ
docs/report_outline.md   レポートの骨子
```

Colab 上の保存先（Google ドライブ）：`<DRIVE_DIR>/runs/<profile>/{checkpoints,eval,results,videos}`

## 技術的な注意点

- **JAX は 0.9.x に固定**。brax 0.14.2 の PPO が `jax.device_put_replicated` を使っており、JAX 0.10 以降では学習が動かない。依存を変えるときは `requirements-colab.txt` と `requirements-cpu.txt` を揃え、CPU スモークテストを通してから更新する
- 物理計算は MuJoCo Warp が既定（`impl=None`）。不具合時は `impl="jax"`
- 環境の報酬は1ステップごとに **0 未満が切り捨てられる**（`jp.clip(sum * dt, 0, ...)`）。罰の重みを大きくしすぎると学習信号が消えるので、既定値の2〜5倍程度に抑える
- 性格の違いは報酬の重みだけ（観測・行動・ネットワークは共通）なので、標準の方策から `restore_checkpoint_path` でファインチューニングできる
- Brax の PPO は学習量を「(num_evals − 1) × num_resets_per_eval × 学習ステップ」単位に**切り上げる**。Go1 の設定では約1,470万ステップ単位（trial の 10M・5M はどちらも 14.7M、pro は 206M + 73.7M）。`training.effective_timesteps` で同じ計算をして、進捗表示の目標値に使っている
- 学習中の評価報酬（`eval/episode_reward`）は確率的な方策で測った値で、**性格ごとに報酬の定義が違う**ため性格間で比べられない。比較は評価セル（`evaluate`）の共通指標で行う
- 速度の実測（Colab Pro の A100）：約400万ステップ/分。pro 全体で約2時間、約5コンピューティングユニット/時
- 図は返す前に `plt.close` する（ノートブックで二重に表示されるのを防ぐ。`display(fig)` と `savefig` はそのまま使える）
- チェックポイントは `checkpoints/<style>/<累積ステップ:012d>/`（Brax の `ppo_network_config.json` 相当の最小設定つき）。`DONE` があれば学習済みとしてスキップ、無ければ最新から再開する
- Brax 標準の `load_policy` は初期化関数が未設定の設定ファイルで失敗するため、`training.load_policy` でネットワークを組み直して読み込む
- 評価は `jax.vmap` × `jax.lax.scan` で複数 seed を並列に実行。指令は毎ステップ上書きする（環境側のランダムな指令変更を無効にするため）
- 動画は 1280×720・25fps。日本語字幕には `fonts-noto-cjk` が必要（ノートブックの「1.」で導入）

## よく使うコマンド

```bash
# CPU での動作確認（Linux / WSL。10分前後）
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt
MUJOCO_GL=egl python tests/smoke_test_cpu.py
# 学習済みのチェックポイントを使い回す場合
SMOKE_OUT=/tmp/gait_smoke MUJOCO_GL=egl python tests/smoke_test_cpu.py
```

## ルール

- 個人情報をコード・ドキュメント・図・動画・コミットに入れない。コミットのメールアドレスは GitHub の noreply アドレスを使う
- 講義資料（このリポジトリの外にある `../講義資料`）は参照のみ。リポジトリに含めない、内容を転載しない
- チェックポイント・評価ログ（npz）・動画（mp4）はコミットしない（`.gitignore` 済み）
- サンプルや他人の作品を丸写ししない。参考にしたものは README の参考文献に追加する
- README・レポートに書く数値は `results/summary.md` の値をそのまま使い、誇張しない。うまくいかなかった点も考察として残す

## 進捗

- [x] テーマ決定、コード一式・ノートブック作成、CPU スモークテスト通過
- [x] GitHub に public で push
- [x] Colab で `trial` を最後まで実行（A100 で約35分・約3ユニット。学習量が少ないので全性格が転倒し、指標の差もほぼない。パイプラインの動作確認のみ）
- [ ] 本番の PROFILE（`t4` または `pro`）で学習・評価・動画作成
- [ ] `results/` を反映し、README の「結果」を更新
- [ ] YouTube にアップロード（タイトル：`PAI最終課題_omnicampusアカウント名`）
- [ ] レポート PDF（`説明資料_PAI最終課題_omnicampusアカウント名.pdf`）を作成
- [ ] omnicampus で提出（PDF・GitHub URL・YouTube URL）
