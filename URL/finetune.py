import warnings
import random
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from collections import OrderedDict
from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
from dmc_benchmark import PRETRAIN_TASKS, FINETUNE_TASKS

torch.backends.cudnn.benchmark = True


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
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)

        # create envs
        if cfg.task != 'none':
            # single task
            tasks = [cfg.task]
        else:
            # pre-define multi-task
            tasks = FINETUNE_TASKS[self.cfg.finetune_domain]

        self.train_envs = [dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                    cfg.action_repeat, cfg.seed)
                           for task in tasks]
        self.train_tasks_name = tasks
        self.train_envs_number = len(self.train_envs)
        self.current_train_id = 0

        # create envs
        if cfg.task != 'none':
            # single task
            eval_tasks = [cfg.task]
        else:
            # pre-define multi-task
            eval_tasks = FINETUNE_TASKS[self.cfg.finetune_domain + '_eval']
        self.eval_envs = [dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                   cfg.action_repeat, cfg.seed)
                          for task in eval_tasks]
        self.eval_tasks_name = eval_tasks
        self.eval_envs_number = len(self.eval_envs)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_envs[0].observation_spec(),
                                self.train_envs[0].action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # initialize from pretrained
        if cfg.snapshot_path != 'none':
            pretrained_agent = self.load_snapshot(cfg.snapshot_path)['agent']
            self.agent.init_from(pretrained_agent)
        elif cfg.snapshot_ts > 0:
            pretrained_agent = self.load_snapshot()['agent']
            self.agent.init_from(pretrained_agent)
        else:
            print("!!!!!! we donot load model before finetuning")

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
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount)
        self._replay_iter = None

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        
        self.meta = self.agent.init_meta()
        
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
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(int(self.cfg.num_eval_episodes / 2))
        
        # cesd fix the meta
        if 'cesd' in self.cfg.agent.name:
            print('in cesd, we fix meta')
            meta = self.meta
        else:
            meta = self.agent.init_meta()

        for eval_id in range(self.eval_envs_number):
            current_episode = 0
            print('we evaluate task', self.eval_tasks_name[eval_id])
            while eval_until_episode(current_episode):
                time_step = self.eval_envs[eval_id].reset()
                time_step['done'] = time_step["is_last"] or time_step["is_terminal"]
                self.video_recorder.init(self.eval_envs[eval_id], enabled=(current_episode == 0))
                while not bool(time_step['is_last']):
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(time_step['observation'],
                                                meta,
                                                self.global_step,
                                                eval_mode=True)
                    time_step = self.eval_envs[eval_id].step(action)
                    time_step['done'] = time_step["is_last"] or time_step["is_terminal"]
                    self.video_recorder.record(self.eval_envs[eval_id])
                    total_reward += time_step['reward']
                    step += 1

                self.video_recorder.save(f'{self.eval_tasks_name[eval_id]}_{self.global_frame}.mp4')
                episode += 1
                current_episode += 1

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

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
        print('current task is', self.train_tasks_name[self.current_train_id])
        
        if 'cesd' in self.cfg.agent.name:
            meta = self.meta
        else:
            meta = self.agent.init_meta()
        
        self.replay_storage.add(time_step, meta)
        self.train_video_recorder.init(time_step['observation'])
        metrics = None
        while train_until_step(self.global_step):
            if bool(time_step['is_last']):
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
                        if 'diffusion' in self.cfg.agent.name or 'edp' in self.cfg.agent.name:
                            if 'q0loss' in metrics.keys() and 'qtloss' in metrics.keys():
                                print('q0loss:', metrics['q0loss'], 'qtloss:', metrics['qtloss'])

                # reset env
                self.current_train_id = (self.current_train_id + 1) % self.train_envs_number
                time_step = self.train_envs[self.current_train_id].reset()
                time_step['done'] = time_step["is_last"] or time_step["is_terminal"]
                print('current task is', self.train_tasks_name[self.current_train_id])
                
                if 'cesd' in self.cfg.agent.name:
                    meta = self.meta
                else:
                    meta = self.agent.init_meta()
                
                self.replay_storage.add(time_step, meta)
                self.train_video_recorder.init(time_step['observation'])
                
                # try to save snapshot
                if self.global_frame in self.cfg.snapshots:
                    self.save_snapshot()

                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            if 'cesd' in self.cfg.agent.name:
                meta = self.meta
            else:
                meta = self.agent.update_meta(meta, self.global_step, time_step)

            if hasattr(self.agent, "regress_meta"):
                repeat = self.cfg.action_repeat
                every = self.agent.update_task_every_step // repeat
                init_step = self.agent.num_init_steps
                if self.global_step > (init_step // repeat) and self.global_step % every == 0:
                    meta = self.agent.regress_meta(self.replay_iter, self.global_step)

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step['observation'],
                                        meta,
                                        self.global_step,
                                        eval_mode=False)
                
            # try to update the agent
            if not seed_until_step(self.global_step):
                if 'diffusion' in self.cfg.agent.name or 'edp' in self.cfg.agent.name:
                    if self.global_step % self.cfg.every_score_step == 0:
                        # print(self.global_step)
                        metrics = self.agent.update(self.replay_iter, self.global_step, iter_num=self.cfg.every_score_update)
                        self.logger.log_metrics(metrics, self.global_frame, ty='train')
                else:
                    metrics = self.agent.update(self.replay_iter, step=self.global_step)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_envs[self.current_train_id].step(action)
            time_step['done'] = time_step["is_last"] or time_step["is_terminal"]
            episode_reward += time_step['reward']
            self.replay_storage.add(time_step, meta)
            self.train_video_recorder.record(time_step['observation'])
            episode_step += 1
            self._global_step += 1

    def load_snapshot(self, snapshot_path=None):
        if snapshot_path is not None:
            snapshot = Path(snapshot_path)
            with snapshot.open('rb') as f:
                # payload = torch.load(f)
                payload = torch.load(f, map_location=self.device)
            return payload
            
        current_path = os.getcwd()
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        # domain, _ = self.cfg.task.split('_', 1)
        snapshot_dir = snapshot_base_dir / self.cfg.obs_type / self.cfg.domain / f'{self.cfg.agent.pretrain_name}'

        def try_load(seed):
            print('current path', current_path)
            snapshot = snapshot_dir / str(seed) / f'snapshot_{self.cfg.snapshot_ts}.pt'
            print('load path:', snapshot)
            if not snapshot.exists():
                print('no such path')
                return None
            with snapshot.open('rb') as f:
                # payload = torch.load(f)
                payload = torch.load(f, map_location=self.device)
            return payload

        # try to load current seed
        payload = try_load(self.cfg.seed)
        if payload is not None:
            print('we have load the models')
            return payload
        # otherwise try random seed
        print('we have not load the model successfully')
        while True:
            seed = np.random.randint(1, 11)
            payload = try_load(seed)
            if payload is not None:
                return payload
        return None

    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

@hydra.main(config_path='.', config_name='finetune')
def main(cfg):
    from finetune import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
