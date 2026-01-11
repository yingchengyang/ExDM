
import os
import numpy as np
from maze_env.maze_env import MazeEnv
import matplotlib.pyplot as plt
import torch
import utils
from collections import OrderedDict
from scipy.spatial import ConvexHull
import itertools
import math
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import polygonize
from typing import Any, Literal
import pandas as pd
import argparse

import os
import numpy as np
from maze_env.maze_env import MazeEnv
import matplotlib.pyplot as plt
import torch
import utils
from collections import OrderedDict

# colors for plot
COLORS = ([
    '#d73027',
    # (0.33725490196078434, 0.7058823529411765, 0.9137254901960784),
    (0.792156862745098, 0.5686274509803921, 0.3803921568627451),
    (0.984313725490196, 0.6862745098039216, 0.8941176470588236),
    # (0.5803921568627451, 0.5803921568627451, 0.5803921568627451),
    '#74add1', 
    (0.9254901960784314, 0.8823529411764706, 0.2),
    (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
    (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
    (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
    (0.8352941176470589, 0.3686274509803922, 0.0),
    (0.8, 0.47058823529411764, 0.7372549019607844),
    # deepmind style
    # '#F0E442',
    # '#D55E00',
    # '#009E73',
    # '#CC79A7',
    '#f46d43',


    # deepmind style
    '#009E73', # 绿
    '#CC79A7', # 粉
    '#D55E00',
    '#0072B2', # 蓝
    '#F0E442',
    'red',

    # built-in color
    # 'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
    # 'brown', 'orange', 'teal',  'lightblue', 'lime', 'lavender', 'turquoise',
    # 'darkgreen', 'tan', 'salmon', 'gold',  'darkred', 'darkblue',
    # personal color
    '#313695',  # DARK BLUE
    '#74add1',  # LIGHT BLUE
    '#4daf4a',  # GREEN
    '#f46d43',  # ORANGE
    '#d73027',  # RED
    '#984ea3',  # PURPLE
    '#f781bf',  # PINK
    '#ffc832',  # YELLOW
    '#000000',  # BLACK
    'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
])

from matplotlib import colors



def collect_state_coverage(directory_path, env_params, max_trajectories, x_interval, y_interval):
    env = MazeEnv(**env_params)
    all_walls = env.maze._walls
    # print(all_walls)
    x_values = []
    y_values = []
    
    # 算wall围起来的面积    
    # 提取所有墙的线段
    # all_walls 里面的数据格式是 (x1, x2), (y1, y2) 非常nt
    lines = [LineString([(x1, y1), (x2, y2)]) for (x1, x2), (y1, y2) in all_walls]

    # 使用 polygonize 将线段组合成多边形
    polygons = list(polygonize(lines))
    area = sum(polygon.area for polygon in polygons)
    # 算wall围起来的面积

    for (x1, x2), (y1, y2) in all_walls:
        x_values.extend([x1, x2])  # 添加 x 坐标
        y_values.extend([y1, y2])  # 添加 y 坐标
    
    # 计算 x 和 y 的最大值和最小值
    # 这里我们需要加上/减去1e-4，因为边界点可能会被算到一个新的小矩形内，需要把这种情况排除
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    # print(x_min, x_max, y_min, y_max)
    
    full_area = (x_max - x_min) * (y_max - y_min)
    # print(area, full_area)

    # x_interval = 0.05
    # y_interval = 0.05
    x_grid_density = int((x_max - x_min) / x_interval)
    y_grid_density = int((y_max - y_min) / y_interval)
    state_coverage = np.zeros((x_grid_density, y_grid_density))
    
    # grid_density = 100
    # state_coverage = np.zeros((grid_density, grid_density))
    # x_interval = (x_max - x_min) / grid_density
    # y_interval = (y_max - y_min) / grid_density
    state_coverage_sum = []
    
    loaded_data = {}
    index = 0
    path_names = sorted(os.listdir(directory_path))
    for filename in path_names:
        index += 1
        if index > max_trajectories:
            break
        
        if filename.endswith(".npz"):
            file_path = os.path.join(directory_path, filename)
            try:
                data = np.load(file_path)
                loaded_data[filename] = data
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
            # print(data['observation'])
            episode = [data['observation'][:, 0], data['observation'][:, 1]]
            # print(data['observation'].shape) # 51, 2
            for point_idx in range(data['observation'].shape[0]):
                # point_xy = Point(data['observation'][point_idx][0], data['observation'][point_idx][1])
                # is_inside = any(polygon.contains(point_xy) for polygon in polygons)
                # if not is_inside:
                #     # 环境有点小bug，有时候点会跳出这个迷宫，导致state converage > 1.0, 我们不统计不在多边形内的点
                #     continue

                x_index = (data['observation'][point_idx][0] - x_min) / (x_max - x_min)
                y_index = (data['observation'][point_idx][1] - y_min) / (y_max - y_min)
                # assert x_index >= 0 and x_index <= 1
                # assert y_index >= 0 and y_index <= 1
                x_index = min(max(x_index, 0.0), 1.0 - 1e-4)
                y_index = min(max(y_index, 0.0), 1.0 - 1e-4)
                x_index = int(x_index * x_grid_density)
                y_index = int(y_index * y_grid_density)
                
                # 我们统计的比较保守，只有当点距离方块中心的距离小于方块长度的一半时，我们才统计这个方块
                # if abs(data['observation'][point_idx][0] - (x_min + (x_max - x_min) * (x_index+0.5) / x_grid_density)) > x_interval / 2.0:
                #     continue
                # if abs(data['observation'][point_idx][1] - (y_min + (y_max - y_min) * (y_index+0.5) / y_grid_density)) > y_interval / 2.0:
                #     continue

                state_coverage[x_index][y_index] = 1.0
            state_coverage_sum.append(state_coverage.sum() / (x_grid_density*y_grid_density) / (area / full_area))
        
        # if index % 1000 == 0:
        #     print(index, ':', state_coverage.sum())
    return area, full_area, state_coverage_sum



def list_subdirectories(folder_path):
    subdirectories = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    return subdirectories


base_dir = './exp_local'
domains = [
    'maze_square_a', 
    'maze_square_b', 
    'maze_square_c', 
    'maze_square_d', 
    'maze_square_tree', 
    'maze_square_bottleneck', 
    'maze_square_large'
    ]

all_env_params = {
    'maze_square_a': {
        "n": 50,
        "maze_type": "square_a",
        "done_on_success": False
    },
    'maze_square_b': {
        "n": 50,
        "maze_type": "square_b",
        "done_on_success": False
    },
    'maze_square_c': {
        "n": 50,
        "maze_type": "square_c",
        "done_on_success": False
    },
    'maze_square_d': {
        "n": 50,
        "maze_type": "square_d",
        "done_on_success": False
    },
    'maze_square_tree': {
        "n": 50,
        "maze_type": "square_tree",
        "done_on_success": False
    },
    'maze_square_bottleneck': {
        "n": 50,
        "maze_type": "square_bottleneck",
        "done_on_success": False
    },
    'maze_square_large': {
        "n": 50,
        "maze_type": "square_large",
        "done_on_success": False
    },
}


interval = {
    'maze_square_a': (0.05, 0.05),
    'maze_square_b': (0.05, 0.05),
    'maze_square_c': (0.05, 0.05),
    'maze_square_d': (0.05, 0.05),
    'maze_square_tree': (0.05, 0.05),
    'maze_square_bottleneck': (0.1, 0.1),
    'maze_square_large': (0.1, 0.1),
}

max_trajectories = 2000

domains = [
        'maze_square_a', 
        'maze_square_b', 
        'maze_square_c', 
        'maze_square_d', 
        'maze_square_tree', 
        'maze_square_bottleneck', 
        'maze_square_large'
        ]

algs = [
    'exdm'
]

# 算state coverage
for domain in domains:
    plt.figure()
    idx = 0
    env_params = all_env_params[domain]
    print('*'*10, 'current domain:', domain, '*'*10)
    for alg in algs:
    # for alg in ALGS:
        best_train_results = []
        last_train_results = []
        
        folder_path = os.path.join(base_dir, domain, 'pretrain', alg)
        subdirectories = list_subdirectories(folder_path)
        subdirectories = [a + '/buffer' for a in subdirectories]
    
        all_state_coverage_sum = []
        for path in subdirectories:
            save_path = path[:path.rfind('/')] + '/state_coverage.npy'

            # area, full_area, state_coverage_sum = collect_state_coverage(path, env_params, max_trajectories=max_trajectories, x_interval=interval[domain][0], y_interval=interval[domain][1])  # 调用绘制函数
            # np.save(save_path, np.array(state_coverage_sum))
            
            if os.path.exists(save_path):
                state_coverage_sum = np.load(save_path)
                state_coverage_sum = state_coverage_sum.tolist()
            else:
                area, full_area, state_coverage_sum = collect_state_coverage(path, env_params, max_trajectories=max_trajectories, x_interval=interval[domain][0], y_interval=interval[domain][1])  # 调用绘制函数
                np.save(save_path, np.array(state_coverage_sum))
            # print('algo:', alg, path, area, full_area, state_coverage_sum[-1])
            all_state_coverage_sum.append(state_coverage_sum)
    
        all_state_coverage_sum = np.array(all_state_coverage_sum)
        all_state_coverage_mean = np.mean(all_state_coverage_sum, axis=0)
        all_state_coverage_std = np.std(all_state_coverage_sum, axis=0)
    
        # print('current task:', domain, 'alg', alg, 'results:', all_state_coverage_sum[:, -1])
        print('alg', alg, 'mean and std:', all_state_coverage_mean[-1], all_state_coverage_std[-1])
        print(all_state_coverage_sum[:, -1])

        plt.plot(np.arange(all_state_coverage_mean.shape[0]), all_state_coverage_mean, label=alg, color=COLORS[idx], linewidth=2)
        plt.fill_between(np.arange(all_state_coverage_mean.shape[0]), all_state_coverage_mean - all_state_coverage_std, all_state_coverage_mean + all_state_coverage_std, color=COLORS[idx], alpha=0.2)
        idx += 1

    plt.legend()
    plt.savefig('./figs/{}.png'.format(domain))