# HIDA
Implementation of HIDA: Hierarchical Imagination with Dynamic Adaptation.

## Paper

This repository contains the experiments and data used in our paper. You can find the paper in the [docs folder](docs/paper.pdf).

## Members

shiropa-uk, t-yamada02, ziwoo3244

## Overview

We propose:
- A hierarchical model with dynamic temporal abstraction.
- A model that automatically balances exploration parameters among hierarchy.
- Addressing the non-stationary problem of higher layers using prioritised experience replay or hindsight relabeling inspired by [Gu et al. 18].

It is based on the implemenation of the HIERarchical imagionation On Structured State Space Sequence Models (HIEROS) paper in pytorch. Hieros repository is based on the [DreamerV3](https://github.com/danijar/dreamerv3), [DreamerV3 in pytorch](https://github.com/NM512/dreamerv3-torch) and [S5 in pytorch](https://github.com/i404788/s5-pytorch) repositories.

# Installation

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

# Usage

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

# Repository Structure

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

# Debugging Subgoal Visualization

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
