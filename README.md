# Stylized Offline Reinforcement Learning: Extracting Diverse High-Quality Behaviors from Heterogeneous Datasets

Code used for the paper "[Stylized Offline Reinforcement Learning: Extracting Diverse High-Quality Behaviors from Heterogeneous Datasets](https://openreview.net/forum?id=rnHNDihrIT)". The code is adapted from https://github.com/joonaspu/video-game-behavioural-cloning.

Atari-HEAD dataset can be downlaoded at https://zenodo.org/record/3451402. The `ATARI_HEAD_DIR` should point at a directory that has subdirectories for each game (i.e. `montezuma_revenge`, `ms_pacman` and `space_invaders`). The `.tar.bz2` archives inside each game's directory should also be extracted.

First train the dqn model for the usage of advantage function.

```
python3 train_dqn.py ATARI_HEAD_DIR game models_atari_head/dqn_models/space_invaders --epochs 10 --workers 16 --framestack 2 --l2 0.00001 --save-freq 1 --merge --atari-head --env_id SpaceInvaders-v0
```

Then train the SORL algorithm.

```
python3 train_style.py ATARI_HEAD_DIR game models_atari_head/space_invaders --epochs 60 --workers 16 --framestack 2 --l2 0.00001 --save-freq 1 --merge --atari-head --env_id SpaceInvaders-v0 --load_ppo ./models_atari_head/dqn_models/space_invaders_10.pt
```