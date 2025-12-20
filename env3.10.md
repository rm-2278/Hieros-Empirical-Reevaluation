# Hieros on Python 3.10

1. Environment Setup
```
conda create -n hieros python=3.10 -y
conda activate hieros
conda install -c conda-forge wget unrar cmake zlib -y
```
(since s5_model.py requires match syntax from python>=3.10) \
(atari-py requires python<3.10 so requires cmake to be installed) \
(conda-forge channel, which is a community-driven channel, is needed to install unrar. Should use conda-forge for all packages to avoid conflicts.) \

In repository root folder, run: 
```
pip install "pip<24.0"
pip install -r requirements.txt
bash embodied/scripts/install-atari.sh
```

1.1. Setup for pinpad
```
pip uninstall mujoco
pip install mujoco==2.3.7
```

1.5. How to use w&b
login to your wandb account:
```
wandb login
```
Change the "wandb_name", "wandb_prefix" in hieros/config.yml to your desired names. \
