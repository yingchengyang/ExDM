# ExDM

[![arXiv](https://img.shields.io/badge/arXiv-2502.07279-b31b1b.svg)](https://arxiv.org/abs/2502.07279)

- This is the official implementation for [Exploratory Diffusion Model for Unsupervised Reinforcement Learning](https://arxiv.org/abs/2502.07279).

- The training code is based on [URLB](https://github.com/rll-research/url_benchmark).

## Setup
```sh
conda create -n exdm python=3.8
conda activate exdm
pip install -r requirements.txt
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## Pretrain in Maze
```sh
cd URL
chmod +x pretrain_maze.sh

# you can choose task and domain from maze_square_a/maze_square_b/maze_square_c/maze_square_d/maze_square_tree/maze_square_bottleneck/maze_square_large
export MUJOCO_EGL_DEVICE_ID=0
python pretrain.py configs/agent=exdm_maze task=maze_square_a device=cuda:0 domain=maze_square_a num_train_frames=100010 seed=0 save_snapshot=true

# calculate the state coverage for each maze
python result_maze.py
```

## Citation

If you find this work helpful, please cite our paper.

```
@article{ying2025exploratory,
  title={Exploratory Diffusion Model for Unsupervised Reinforcement Learning},
  author={Ying, Chengyang and Chen, Huayu and Zhou, Xinning and Hao, Zhongkai and Su, Hang and Zhu, Jun},
  journal={arXiv preprint arXiv:2502.07279},
  year={2025}
}
```