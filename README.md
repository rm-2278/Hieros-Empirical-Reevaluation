# 階層的世界モデルHierosの実証的再評価 / Empirical Re-evaluation of Hieros

> 🌐 **言語選択 / Language Selection**
> 
> このREADMEには日本語版と英語版の両方が含まれています。下のセクションをクリックして展開してください。
> 
> This README contains both Japanese and English versions. Click on the sections below to expand.

---

<details>
<summary><h2>🇯🇵 日本語版 (Japanese Version)</h2></summary>

# 階層的世界モデルHierosの実証的再評価

## 論文

📄 **[論文PDF](docs/paper.pdf)**

このリポジトリは、論文「階層的世界モデルHierosの実証的再評価：内部表現解析と階層構造の影響分析」で使用した実験コードとデータを含んでいます。

## 概要

階層的強化学習と世界モデルを結びつけた手法は、長期タスクの学習において期待されていますが、その実用性や内部メカニズムについては十分な検証がされていません。本研究では、階層的世界モデルの代表例である**Hieros**に着目し、性能評価と内部状態の可視化を通じてその実態を検証しました。

**主な発見：**
- **ハイパーパラメータへの高い感度**: Visual Pinpad環境での実験により、Hierosはハイパーパラメータ設定に対して高い感度を示し、報酬設計や更新頻度の変更に対する頑健性に限界があることがわかりました
- **単純な行動パターンの学習**: Atari環境での方策可視化では、高スコアを示しているにも関わらず単純な行動パターンのみが学習されており、階層性を活かしたサブゴールの学習が実現されていないことを確認しました
- **階層数増加による学習不安定化**: 階層数の比較実験により、階層数の増加が学習の安定性を低下させることが確認されました

これらの結果は、現在の階層的世界モデルにおいて理論的期待と実際の性能の間にギャップがあることを示しており、より頑健な階層的学習手法の必要性を示唆しています。

## 実験環境

本研究では以下の環境で評価を行いました：

- **Visual Pinpad**: エージェントが特定の順番でタイルを踏むタスク
- **Pinpad-easy**: 末尾一致度に基づく報酬設計を導入した改良版
- **Atari 100k**: Freeway, Breakout, Krull, Battle Zoneなど

## 実験内容

`experiments/`ディレクトリには論文で使用した全ての実験設定とスクリプトが含まれています：

- **サブゴール更新頻度の変更**: `subactor_update_every`パラメータの影響分析
- **方策エントロピーの変更**: 異なるエントロピー設定での探索範囲の変化
- **報酬割り当て係数**: external reward, subgoal reward, intrinsic rewardの比率変更
- **報酬設計の変更**: flat, progressive, sparse, decayingなど7種類の報酬設計
- **階層数の影響**: `max_hierarchy`パラメータの影響分析

## 再現性

実験結果を再現するには：

1. 以下の[インストール](#installation-ja)セクションに従って依存関係をインストール
2. `experiments/configs/`の実験設定を使用
3. `experiments/scripts/`のスクリプトで実験を実行
4. `notebooks/`のノートブックで結果を可視化

## ベースモデル

本実装はHIEROS (HIERarchical imagination On Structured State Space Sequence Models) のPyTorch実装に基づいています。

参考リポジトリ：
- [Hieros](https://github.com/Snagnar/Hieros)
- [Director](https://github.com/danijar/director)
- [DreamerV3](https://github.com/danijar/dreamerv3)
- [DreamerV3 in PyTorch](https://github.com/NM512/dreamerv3-torch)
- [S5 in PyTorch](https://github.com/i404788/s5-pytorch)

<a id="installation-ja"></a>
## インストール

pip依存関係をインストール：
```
pip install -r requirements.txt
```

必要なツールをインストール：
```
sudo apt update && sudo apt install -y wget unrar
```

Atari ROMをインストール：
```
bash embodied/scripts/install-atari.sh
```

## 使用方法

Atariゲームでモデルを訓練するには：
```
python hieros/train.py --configs atari100k --task=atari_alien
```

`--task`フラグでタスクを指定できます。利用可能なタスク：
```
atari_alien, atari_amidar, atari_assault, atari_asterix, atari_bank_heist, atari_battle_zone, atari_boxing, atari_breakout, atari_chopper_command, atari_crazy_climber, atari_demon_attack, atari_freeway, atari_frostbite, atari_gopher, atari_hero, atari_jamesbond, atari_kangaroo, atari_krull, atari_kung_fu_master, atari_ms_pacman, atari_pong, atari_private_eye, atari_qbert, atari_road_runner, atari_seaquest
```

その他のベンチマークもサポートしています。`hieros/config.yml`を参照して異なる設定を見つけてください。例えば、`dmc_vision`タスクで訓練するには：
```
python hieros/train.py --configs dmc_vision --task=dmc_cheetah_run
```

`hieros/config.yml`で利用可能な全てのフラグはコマンドライン引数として設定可能です。例えば、異なる階層数で`atari_alien`タスクを訓練するには：
```
python hieros/train.py --configs atari100k --task=atari_alien --max_hierarchy=2
```

オリジナルの[DreamerV3](https://github.com/NM512/dreamerv3-torch)モデルの実装も含まれており、`--model_name=dreamer`でアクセス可能です。

メトリクスはデフォルトでtensorboardに記録されます。訓練の進捗を可視化するには：
```
tensorboard --logdir=logs
```
これらの訓練統計を使用して、論文のプロットを再現することもできます。

## リポジトリ構成

```
root/
├─ docs/                    -- ドキュメントファイル
│   ├─ *.md                 -- Markdownドキュメント
│   └─ *.pdf                -- PDFレポートと論文
│
├─ experiments/
│   ├─ configs/             -- YAML/JSON実験設定
│   ├─ results/             -- 実験出力（ログ、メトリクス）
│   └─ scripts/             -- 実験起動スクリプト
│
├─ hieros/                  -- HIEROSモデルの実装と訓練コード
│
├─ embodied/                -- 基本ツール（ロギング、リプレイバッファ、環境）
│                              主にDreamerV3からコピー
│
├─ resettable_s5/           -- S5WM用のリセット可能なS5モデル実装
│                              pytorch S5実装に基づく
│
├─ tests/                   -- テストコード（ユニット/スモークテスト）
│
├─ data/
│   ├─ raw/                 -- 生データ（git管理外）
│   └─ processed/           -- 前処理済みデータ
│
├─ notebooks/               -- 分析と可視化スクリプト
│
├─ docker/                  -- Dockerファイルとコンテナ設定
│
├─ .github/                 -- GitHubワークフローとテンプレート
│
├─ README.md                -- このファイル
├─ LICENSE
├─ requirements.txt         -- Python依存関係
└─ .gitignore               -- プッシュしないファイル/フォルダ
```

## デバッグ：サブゴール可視化

`subgoal_debug_visualization: True`使用時にテンソル次元の不一致エラーが発生した場合、包括的なデバッグツールを提供しています：

**クイックスタート：**
設定でデバッグモードを有効にする：
```yaml
debug: True
subgoal_debug_visualization: True
```

これにより、問題の診断に役立つ詳細なテンソル形状情報がログに記録されます。

**ドキュメント：**
- 📖 [完全なデバッグガイド](docs/DEBUG_SUBGOAL_VISUALIZATION.md) - 問題と解決策の詳細な説明
- 📋 [クイックリファレンス](docs/DEBUG_README.md) - デバッグ機能の概要
- 📝 [実装サマリー](docs/IMPLEMENTATION_SUMMARY.md) - 実装の技術的詳細
- 💻 [使用例](docs/examples_debug_usage.py) - 実践的なコード例

**テスト：**
```bash
# 構造検証テストを実行（依存関係なし）
python tests/test_debug_structure.py

# 機能テストを実行（torchが必要）
python tests/test_subgoal_debug.py

# 使用例を実行（torchが必要）
python docs/examples_debug_usage.py
```

詳細については、[docs/DEBUG_README.md](docs/DEBUG_README.md)を参照してください。

</details>

---

<details>
<summary><h2>🇬🇧 English Version</h2></summary>

# Empirical Re-evaluation of the Hierarchical World Model Hieros

## Paper

📄 **[Paper PDF](docs/paper.pdf)**

This repository contains the experiments and data used in our paper "Empirical Re-evaluation of the Hierarchical World Model Hieros: Internal Representation Analysis and the Impact of Hierarchical Structure".

## Overview

Hierarchical reinforcement learning combined with world models is a promising approach for learning long-horizon tasks, but its practical effectiveness and internal mechanisms have not been sufficiently validated. In this study, we focus on **Hieros**, a representative hierarchical world model, and examine its performance and internal state visualization.

**Key Findings:**
- **High Sensitivity to Hyperparameters**: Experiments in the Visual Pinpad environment revealed that Hieros is highly sensitive to hyperparameter settings and has limited robustness to changes in reward design and update frequency
- **Learning of Simple Action Patterns**: Policy visualization in Atari environments confirmed that despite achieving high scores, only simple action patterns are learned, and subgoal learning utilizing hierarchy is not realized
- **Decreased Learning Stability with More Hierarchy Levels**: Comparative experiments on the number of hierarchy levels confirmed that increasing the number of levels decreases learning stability

These results indicate a gap between theoretical expectations and actual performance in current hierarchical world models, suggesting the need for more robust hierarchical learning methods.

## Experimental Environments

We evaluated on the following environments:

- **Visual Pinpad**: A task where agents step on tiles in a specific order
- **Pinpad-easy**: An improved version with suffix-matching reward design
- **Atari 100k**: Including Freeway, Breakout, Krull, Battle Zone, etc.

## Experiments

The `experiments/` directory contains all experiment configurations and scripts used in the paper:

- **Subgoal Update Frequency**: Analysis of the impact of `subactor_update_every` parameter
- **Policy Entropy**: Changes in exploration range with different entropy settings
- **Reward Allocation Coefficients**: Changes in ratio of external reward, subgoal reward, intrinsic reward
- **Reward Design**: 7 types of reward designs including flat, progressive, sparse, decaying
- **Hierarchy Level Impact**: Analysis of the impact of `max_hierarchy` parameter

## Reproducibility

To reproduce our results:

1. Install dependencies following the [Installation](#installation-en) section
2. Use experiment configurations in `experiments/configs/`
3. Run experiments using scripts in `experiments/scripts/`
4. Visualize results using notebooks in `notebooks/`

## Base Model

This implementation is based on the PyTorch implementation of HIEROS (HIERarchical imagination On Structured State Space Sequence Models).

Reference repositories:
- [Hieros](https://github.com/Snagnar/Hieros)
- [Director](https://github.com/danijar/director)
- [DreamerV3](https://github.com/danijar/dreamerv3)
- [DreamerV3 in PyTorch](https://github.com/NM512/dreamerv3-torch)
- [S5 in PyTorch](https://github.com/i404788/s5-pytorch)

<a id="installation-en"></a>
## Installation

Install pip dependencies:
```
pip install -r requirements.txt
```

Install required tools:
```
sudo apt update && sudo apt install -y wget unrar
```

Install atari roms:
```
bash embodied/scripts/install-atari.sh
```

## Usage

To train a model on a atari game, run:
```
python hieros/train.py --configs atari100k --task=atari_alien
```

You can specify the task to train on with the `--task` flag. The available tasks are:
```
atari_alien, atari_amidar, atari_assault, atari_asterix, atari_bank_heist, atari_battle_zone, atari_boxing, atari_breakout, atari_chopper_command, atari_crazy_climber, atari_demon_attack, atari_freeway, atari_frostbite, atari_gopher, atari_hero, atari_jamesbond, atari_kangaroo, atari_krull, atari_kung_fu_master, atari_ms_pacman, atari_pong, atari_private_eye, atari_qbert, atari_road_runner, atari_seaquest
```

We also support a wide range of other benchmarks. For this, please reference the `hieros/config.yml` to find different configurations. For example, to train on the `dmc_vision` task, run:
```
python hieros/train.py --configs dmc_vision --task=dmc_cheetah_run
```

All flags available in `hieros/config.yml` are configurable as command line arguments. For example, to train on the `atari_alien` task with a different number of layers, run:
```
python hieros/train.py --configs atari100k --task=atari_alien --max_hierarchy=2
```

We also include an implementation of the original [DreamerV3](https://github.com/NM512/dreamerv3-torch) model, which is accessible with `--model_name=dreamer`.

The metrics are logged to tensorboard by default. To visualize the training progress, run:
```
tensorboard --logdir=logs
```

With these training statistics, you can also reproduce the plots in the paper.

## Repository Structure

```
root/
├─ docs/                    -- Documentation files
│   ├─ *.md                 -- Markdown documentation
│   └─ *.pdf                -- PDF reports and papers
│
├─ experiments/
│   ├─ configs/             -- YAML/JSON experiment configurations
│   ├─ results/             -- Experiment outputs (logs, metrics)
│   └─ scripts/             -- Experiment launch scripts
│
├─ hieros/                  -- Implementation and training code of the HIEROS model
│
├─ embodied/                -- Basic tools (logging, replay buffers, environments)
│                              Largely copied from DreamerV3
│
├─ resettable_s5/           -- Resettable S5 model implementation for S5WM
│                              Based on pytorch S5 implementation
│
├─ tests/                   -- Test code (unit / smoke tests)
│
├─ data/
│   ├─ raw/                 -- Raw data (not git managed)
│   └─ processed/           -- Preprocessed data
│
├─ notebooks/               -- Analysis and visualization scripts
│
├─ docker/                  -- Docker files and container setup
│
├─ .github/                 -- GitHub workflows and templates
│
├─ README.md                -- This file
├─ LICENSE
├─ requirements.txt         -- Python dependencies
└─ .gitignore               -- Files/folders not to push
```

## Debugging: Subgoal Visualization

If you encounter tensor dimension mismatch errors when using `subgoal_debug_visualization: True`, we provide comprehensive debugging tools:

**Quick Start:**
Enable debug mode in your config:
```yaml
debug: True
subgoal_debug_visualization: True
```

This will log detailed tensor shape information to help diagnose issues.

**Documentation:**
- 📖 [Complete Debugging Guide](docs/DEBUG_SUBGOAL_VISUALIZATION.md) - Detailed explanation of the issue and solutions
- 📋 [Quick Reference](docs/DEBUG_README.md) - Fast overview of debugging features
- 📝 [Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md) - Technical details of the implementation
- 💻 [Usage Examples](docs/examples_debug_usage.py) - Practical code examples

**Tests:**
```bash
# Run structure validation tests (no dependencies)
python tests/test_debug_structure.py

# Run functional tests (requires torch)
python tests/test_subgoal_debug.py

# Run usage examples (requires torch)
python docs/examples_debug_usage.py
```

For more information, see [docs/DEBUG_README.md](docs/DEBUG_README.md).

</details>
