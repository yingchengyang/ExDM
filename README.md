# ExDM

[![arXiv](https://img.shields.io/badge/arXiv-2502.07279-b31b1b.svg)](https://arxiv.org/abs/2502.07279) [![Project Page](https://img.shields.io/badge/project-page-blue)](https://yingchengyang.github.io/exdm)

- This is the Official implementation for [Exploratory Diffusion Model for Unsupervised Reinforcement Learning](https://openreview.net/forum?id=k0Kb1ynFbt) (ICLR 2026 Oral)

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

# you can choose task and domain from maze_square_a/maze_square_b/maze_square_c/maze_square_d/maze_square_tree/maze_square_bottleneck/maze_square_large
export MUJOCO_EGL_DEVICE_ID=0
python pretrain.py configs/agent=exdm_maze task=maze_square_a device=cuda:0 domain=maze_square_a num_train_frames=100010 seed=0 save_snapshot=true

# calculate the state coverage for each maze
python result_maze.py
```

## Pretrain in URLB and Finetune the Gaussian Policy with DDPG
```sh
cd URL

# you can set DOMAIN as walker, quadruped, jaco, or hopper

if [ "$DOMAIN" == "walker" ]
then
    ALL_TASKS=("walker_stand" "walker_walk" "walker_run" "walker_flip")
    TASK="walker_stand"
elif [ "$DOMAIN" == "quadruped" ]
then
    ALL_TASKS=("quadruped_stand" "quadruped_walk" "quadruped_run" "quadruped_jump")
    TASK="quadruped_stand"
elif [ "$DOMAIN" == "jaco" ]
then
    ALL_TASKS=("jaco_reach_top_left" "jaco_reach_top_right" "jaco_reach_bottom_left" "jaco_reach_bottom_right")
    TASK="jaco_reach_top_left"
elif [ "$DOMAIN" == "cheetah" ]
then
    ALL_TASKS=("cheetah_run" "cheetah_run_backward" "cheetah_flip" "cheetah_flip_backward")
    TASK="cheetah_run"
elif [ "$DOMAIN" == "hopper" ]
then
    ALL_TASKS=("hopper_hop" "hopper_hop_backward" "hopper_flip" "hopper_flip_backward")
    TASK="hopper_hop"
else
    ALL_TASKS=()
    echo "No matching tasks"
fi

export MUJOCO_EGL_DEVICE_ID=0
python pretrain.py configs/agent=exdm_urlb task=${TASK} seed=0 device=cuda:0 domain=${DOMAIN}
for string in "${ALL_TASKS[@]}"
do
    export MUJOCO_EGL_DEVICE_ID=0
    python finetune.py configs/agent=exdm_urlb task=${string} domain=${DOMAIN} seed=0 device=cuda:0 snapshot_ts=2000000 num_train_frames=100010 
done
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