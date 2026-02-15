# 階層的世界モデルHierosの実証的再評価
# Empirical Re-evaluation of the Hierarchical World Model Hieros

## 論文 / Paper

📄 **[論文PDF / Paper PDF](docs/paper.pdf)**

このリポジトリは、論文「階層的世界モデルHierosの実証的再評価：内部表現解析と階層構造の影響分析」で使用した実験コードとデータを含んでいます。

This repository contains the experiments and data used in our paper "Empirical Re-evaluation of the Hierarchical World Model Hieros: Internal Representation Analysis and the Impact of Hierarchical Structure".

## 概要 / Overview

### 日本語

階層的強化学習と世界モデルを結びつけた手法は、長期タスクの学習において期待されていますが、その実用性や内部メカニズムについては十分な検証がされていません。本研究では、階層的世界モデルの代表例である**Hieros**に着目し、性能評価と内部状態の可視化を通じてその実態を検証しました。

**主な発見：**
- **ハイパーパラメータへの高い感度**: Visual Pinpad環境での実験により、Hierosはハイパーパラメータ設定に対して高い感度を示し、報酬設計や更新頻度の変更に対する頑健性に限界があることがわかりました
- **単純な行動パターンの学習**: Atari環境での方策可視化では、高スコアを示しているにも関わらず単純な行動パターンのみが学習されており、階層性を活かしたサブゴールの学習が実現されていないことを確認しました
- **階層数増加による学習不安定化**: 階層数の比較実験により、階層数の増加が学習の安定性を低下させることが確認されました

これらの結果は、現在の階層的世界モデルにおいて理論的期待と実際の性能の間にギャップがあることを示しており、より頑健な階層的学習手法の必要性を示唆しています。

### English

Hierarchical reinforcement learning combined with world models is a promising approach for learning long-horizon tasks, but its practical effectiveness and internal mechanisms have not been sufficiently validated. In this study, we focus on **Hieros**, a representative hierarchical world model, and examine its performance and internal state visualization.

**Key Findings:**
- **High Sensitivity to Hyperparameters**: Experiments in the Visual Pinpad environment revealed that Hieros is highly sensitive to hyperparameter settings and has limited robustness to changes in reward design and update frequency
- **Learning of Simple Action Patterns**: Policy visualization in Atari environments confirmed that despite achieving high scores, only simple action patterns are learned, and subgoal learning utilizing hierarchy is not realized
- **Decreased Learning Stability with More Hierarchy Levels**: Comparative experiments on the number of hierarchy levels confirmed that increasing the number of levels decreases learning stability

These results indicate a gap between theoretical expectations and actual performance in current hierarchical world models, suggesting the need for more robust hierarchical learning methods.

## 実験環境 / Experimental Environments

本研究では以下の環境で評価を行いました / We evaluated on the following environments:

- **Visual Pinpad**: エージェントが特定の順番でタイルを踏むタスク / A task where agents step on tiles in a specific order
- **Pinpad-easy**: 末尾一致度に基づく報酬設計を導入した改良版 / An improved version with suffix-matching reward design
- **Atari 100k**: Freeway, Breakout, Krull, Battle Zoneなど / Including Freeway, Breakout, Krull, Battle Zone, etc.

## 実験内容 / Experiments

`experiments/`ディレクトリには論文で使用した全ての実験設定とスクリプトが含まれています。

The `experiments/` directory contains all experiment configurations and scripts used in the paper:

- **サブゴール更新頻度の変更 / Subgoal Update Frequency**: `subactor_update_every`パラメータの影響分析
- **方策エントロピーの変更 / Policy Entropy**: 異なるエントロピー設定での探索範囲の変化
- **報酬割り当て係数 / Reward Allocation Coefficients**: external reward, subgoal reward, intrinsic rewardの比率変更
- **報酬設計の変更 / Reward Design**: flat, progressive, sparse, decayingなど7種類の報酬設計
- **階層数の影響 / Hierarchy Level Impact**: `max_hierarchy`パラメータの影響分析

## 再現性 / Reproducibility

実験結果を再現するには / To reproduce our results:

1. 以下の[インストール](#installation)セクションに従って依存関係をインストール
2. `experiments/configs/`の実験設定を使用
3. `experiments/scripts/`のスクリプトで実験を実行
4. `notebooks/`のノートブックで結果を可視化

Install dependencies following the [Installation](#installation) section, use experiment configurations in `experiments/configs/`, run experiments using scripts in `experiments/scripts/`, and visualize results using notebooks in `notebooks/`.

## ベースモデル / Base Model

本実装はHIEROS (HIERarchical imagination On Structured State Space Sequence Models) のPyTorch実装に基づいています。

This implementation is based on the PyTorch implementation of HIEROS (HIERarchical imagination On Structured State Space Sequence Models).

参考リポジトリ / Reference repositories:
- [Hieros](https://github.com/Snagnar/Hieros)
- [Director](https://github.com/danijar/director)
- [DreamerV3](https://github.com/danijar/dreamerv3)
- [DreamerV3 in PyTorch](https://github.com/NM512/dreamerv3-torch)
- [S5 in PyTorch](https://github.com/i404788/s5-pytorch)

<a id="installation"></a>
# インストール / Installation

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

# 使用方法 / Usage

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

# リポジトリ構成 / Repository Structure

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

# デバッグ：サブゴール可視化 / Debugging Subgoal Visualization

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
