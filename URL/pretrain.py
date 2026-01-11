import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRETRAIN_TASKS


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment, cfg.agent.pretrain_name, cfg.domain, cfg.obs_type,
                str(cfg.seed)
            ])
            wandb.init(project="urlb", group=cfg.agent.pretrain_name, name=exp_name)

        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create envs
        # task = PRETRAIN_TASKS[self.cfg.domain]
        # self.train_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
        #                           cfg.action_repeat, cfg.seed)
        # self.eval_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
        #                          cfg.action_repeat, cfg.seed)

        # create envs
        if cfg.task != 'none':
            # single task
            tasks = [cfg.task]
        else:
            # pre-define multi-task
            tasks = PRETRAIN_TASKS[self.cfg.domain]
        # task = cfg.task if cfg.task != 'none' else PRETRAIN_TASKS[self.cfg.domain] # -> which is the URLB default
        
        self.train_envs = [dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                    cfg.action_repeat, cfg.seed, )
                           for task in tasks]
        self.tasks_name = tasks
        self.train_envs_number = len(self.train_envs)
        self.current_train_id = 0
        self.eval_env = [dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                  cfg.action_repeat, cfg.seed, )
                         for task in tasks]

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_envs[0].observation_spec(),
                                self.train_envs[0].action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_envs[0].observation_spec(),
                      self.train_envs[0].action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'),
                      specs.Array((1,), np.float32, 'done'),)

        # create data storage
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')

        # create replay buffer
        save_fake = cfg.agent.get('save_fake', False)
        print('save fake:', save_fake)
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                cfg.save_snapshot, cfg.nstep, cfg.discount,
                                                save_fake=save_fake)
        self._replay_iter = None

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        pass

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_envs[self.current_train_id].reset()
        time_step['done'] = time_step["is_last"] or time_step["is_terminal"]
        print('current task is', self.tasks_name[self.current_train_id])
        meta = self.agent.init_meta()
        self.replay_storage.add(time_step, meta)
        self.train_video_recorder.init(time_step['observation'])
        metrics = None
        while train_until_step(self.global_step):
            # if time_step.last():
            if time_step['is_last']:
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                self.current_train_id = (self.current_train_id + 1) % self.train_envs_number
                time_step = self.train_envs[self.current_train_id].reset()
                print('current task is', self.tasks_name[self.current_train_id])
                time_step['done'] = time_step["is_last"] or time_step["is_terminal"]

                meta = self.agent.init_meta()
                self.replay_storage.add(time_step, meta)
                self.train_video_recorder.init(time_step['observation'])
                # try to save snapshot
                episode_step = 0
                episode_reward = 0
                if self.global_frame in self.cfg.snapshots:
                    self.save_snapshot()

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            meta = self.agent.update_meta(meta, self.global_step, time_step)
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step['observation'],
                                        meta,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_envs[self.current_train_id].step(action)
            time_step['done'] = time_step["is_last"] or time_step["is_terminal"]
            episode_reward += time_step['reward']
            self.replay_storage.add(time_step, meta)
            self.train_video_recorder.record(time_step['observation'])
            episode_step += 1
            self._global_step += 1
        
        # if self.cfg.update_type == 0 and 'diffusion' in self.cfg.agent.pretrain_name:
        #     for score_epoch in range(2000):
        #         metrics = self.agent.update(self.replay_iter, self.global_step, update_type=2)
        #         print(score_epoch, metrics)
        #     self._global_step = 10000000000 # TODO: save name
        #     self.save_snapshot()

    def save_snapshot(self):
        # we save in two paths
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)
            
        # snapshot_dir = self.work_dir
        # snapshot_dir.mkdir(exist_ok=True, parents=True)
        # snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        # keys_to_save = ['agent', '_global_step', '_global_episode']
        # payload = {k: self.__dict__[k] for k in keys_to_save}
        # with snapshot.open('wb') as f:
        #     torch.save(payload, f)


@hydra.main(config_path='.', config_name='pretrain')
def main(cfg):
    from pretrain import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
