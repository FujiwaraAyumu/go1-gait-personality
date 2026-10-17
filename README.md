# 報酬で「歩き方の性格」をデザインする四足ロボット

四足ロボット **Unitree Go1** に強化学習（PPO）で歩き方を学習させ、**報酬の重みだけ**を変えて
「標準」「配膳モード」「省エネモード」「元気モード」の歩き方を作り分けるプロジェクトです。
シミュレーターには [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) を使い、
学習から評価・動画作成までを Google Colab の GPU で行います。

> Physical AI講座 応用編1 最終課題「AI×ロボティクスによる創造的タスクの実現」の提出物です。

- 動画：（YouTube の URL をここに貼る）
- レポート：提出用 PDF の骨子は [docs/report_outline.md](docs/report_outline.md)

## アイデア

同じロボットでも、用途によって望ましい歩き方は違います。物を運ぶなら胴体を揺らさない歩き方、
長く動かすならエネルギーを節約する歩き方、段差が多い場所なら足を高く上げる歩き方です。

強化学習では、ロボットは報酬（何を良しとするか）を最大化するように動き方を学びます。
そこで、**観測・行動・ネットワーク・学習アルゴリズムはすべて同じ**にして、報酬の重みだけを変えたときに
歩き方の「性格」がどう変わり、何を得て何を失うのか（トレードオフ）を数値で比べます。

| 性格 | 報酬の変更点（既定値 → 変更後） | 想定する用途 |
|---|---|---|
| 標準 | 変更なし（MuJoCo Playground の既定値） | 比較の基準 |
| 配膳モード | 胴体の傾き −5 → −15、ロール・ピッチ角速度 −0.05 → −0.25、上下速度 −0.5 → −2.0 | 背中に物を載せて運ぶ |
| 省エネモード | エネルギー −0.001 → −0.003、トルク −0.0002 → −0.0006、行動の変化 −0.01 → −0.03、足の目標高さ 10 → 7 cm | バッテリーを長持ちさせる |
| 元気モード | 滞空時間 0.1 → 0.5、エネルギー −0.001 → −0.0005、足の目標高さ 10 → 15 cm | 段差や凸凹道 |

重みは [`src/gait_personality/styles.py`](src/gait_personality/styles.py) で定義しています。

## 方法

1. **標準の歩き方を学習**：`Go1JoystickFlatTerrain`（速度指令に追従して歩くタスク）を、Playground の調整済み PPO 設定で学習
2. **性格ごとに追加学習**：標準の方策を出発点に、報酬の重みだけを変えてファインチューニング
3. **同じ指令で評価**：全性格に同じ指令を与え、乱数を変えて複数回実行
   - 静止 2秒 → 前進 0.6 m/s 5秒 → 前進 1.2 m/s 5秒 → 前進しながら左旋回 4秒 → 左へ横歩き 4秒
4. **指標で比較**（平均と95%信頼区間）

| 指標 | 定義 |
|---|---|
| 移動コスト（Cost of transport） | Σ\|関節トルク × 関節角速度\|·Δt ÷（全質量 × 重力加速度 × 移動距離）。小さいほど省エネ |
| 速度の追従誤差 | 指令速度と実際の水平速度の差の RMS [m/s] |
| 胴体の揺れ | 胴体のロール・ピッチ方向の角速度の RMS [rad/s] |
| 胴体の上下動 | 胴体の高さの標準偏差 [cm] |
| 足の高さ | 1歩ごとの足先の最高点の平均 [cm] |

## 結果

（Colab で実行後、`results/summary.md` の表と `results/tradeoff.png` をここに貼る）

## 実行方法（Google Colab）

1. このリポジトリを GitHub に public で置く
2. Colab で「ファイル」→「ノートブックを開く」→「GitHub」タブから
   `notebooks/go1_gait_personality.ipynb` を開く
3. 「ランタイムのタイプを変更」で GPU を選ぶ
4. 「0. 設定」の `REPO_URL` を自分のリポジトリの URL にして、上から順に実行する

| PROFILE | 学習量（標準 + 各性格） | 目安 |
|---|---|---|
| trial | 1,000万 + 500万ステップ | 動作確認用（20〜30分） |
| t4 | 1億 + 4,000万ステップ | 無料版 T4（1〜2時間） |
| pro | 2億 + 6,000万ステップ | L4 / A100（40〜90分） |

時間はインストール・コンパイル・評価・動画作成を含めた目安です。学習中は残り時間が表示されます。
途中で切断されても、同じセルを実行し直せば続きから再開します。

## リポジトリ構成

```text
.
├── notebooks/go1_gait_personality.ipynb  # Colab で実行するノートブック（入口）
├── src/gait_personality/
│   ├── styles.py       # 性格 = 報酬の重みの定義
│   ├── training.py     # PPO 学習・ファインチューニング（再開可能なチェックポイント）
│   ├── evaluation.py   # 同じ指令でのロールアウトと指標計算
│   ├── plots.py        # 図
│   ├── video.py        # 描画・字幕・動画の書き出し
│   └── pipeline.py     # ノートブックから呼ぶ一連の処理
├── tests/smoke_test_cpu.py  # GPU なしで全工程を極小設定で通す動作確認
├── results/            # Colab の結果のうち小さいファイル（表・図）
├── docs/report_outline.md   # レポートの骨子
├── requirements-colab.txt   # Colab 用（バージョン固定）
└── requirements-cpu.txt     # CPU での動作確認用
```

## 動作確認（GPU なし）

学習量を極小にして、学習 → チェックポイント → 追加学習 → 評価 → 図 → 動画の全工程が通るかを確認します。
Linux（Windows では WSL）で実行してください。

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt
MUJOCO_GL=egl python tests/smoke_test_cpu.py
```

## バージョンについて

`brax 0.14.2` の PPO は `jax.device_put_replicated` を使っていますが、この関数は JAX 0.10 で削除されました。
そのため JAX は 0.9.2 に固定しています（`requirements-colab.txt`）。

## 参考文献・クレジット

- MuJoCo Playground（Apache-2.0）: https://github.com/google-deepmind/mujoco_playground
- Brax: https://github.com/google/brax
- MuJoCo: https://github.com/google-deepmind/mujoco
- G. Margolis, P. Agrawal. *Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior*. CoRL 2022. https://arxiv.org/abs/2212.03238

開発には AI コーディングアシスタント（Claude）を利用しました。
