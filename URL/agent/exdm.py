import copy

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import random

import utils
from agent.ddpg import DDPGAgent
from diffusion_SDE.schedule import marginal_prob_std
from diffusion_SDE.model import ScoreNet, State_ScoreNet

# pretrain方式: 用diffusion估计log p(s)作为intrinsic reward
# finetune方式: 目前支持
#  DDPG更新 gaussian policy
#  ExDM更新 diffusion policy
#  GFT更新 diffusion policy

#  baseline
#  IDQL更新 diffusion policy √
#  DQL更新 diffusion policy √
#  QSM更新 diffusion policy ×
#  DIPO更新 diffusion policy √
class ExDM_Agent(DDPGAgent):
    def __init__(self, update_encoder, critic_type, guidance_scale, 
                 tau=0.9, finetune_type='DDPG', diffusion_scale=1., alpha=3.0, 
                 score_train_iter=1, diffusion_step=15, 
                 score_time_step=10, domain="walker", **kwargs):
        super().__init__(**kwargs)
        self.update_encoder = update_encoder
        
        # our score model
        self.diffusion_scale = diffusion_scale
        self.score_reward_rms = utils.RMS(device=self.device)
        self.guidance_scale = guidance_scale
        self.alpha = alpha
        
        if domain == "walker" or domain == "quadruped" or domain == "jaco" or domain == "hopper":
            self.score_time_step = score_time_step[domain]
            self.score_train_iter = score_train_iter[domain]
        else:
            self.score_time_step = score_time_step
            self.score_train_iter = score_train_iter
            
        print('!!!each iter we update the score', score_train_iter)
        self.device = self.device
        self.diffusion_steps = diffusion_step
        print("We sample from dpm-solver with", diffusion_step, "steps")
        
        self.critic_type = critic_type
        marginal_prob_std_fn = functools.partial(marginal_prob_std, device=self.device)
        self.marginal_prob_std_fn = marginal_prob_std_fn
        self.score_model = ScoreNet(input_dim=self.obs_dim + self.action_dim,
                                    output_dim=self.action_dim,
                                    marginal_prob_std=marginal_prob_std_fn,
                                    critic_type=critic_type,
                                    device=self.device,
                                    actor_blocks=3, 
                                    q_layer=2, 
                                    alpha=3.0).to(self.device)
        self.score_model.q[0].set_guidance(guidance_scale)
        if critic_type == 'IQL':
            self.score_model.q[0].set_tau(tau)
        self.score_model.q[0].set_alpha(alpha)
        self.score_model_opt = torch.optim.Adam(self.score_model.parameters(), lr=1e-4)
        
        self.state_score_model = State_ScoreNet(
                                    output_dim=self.obs_dim,
                                    marginal_prob_std=marginal_prob_std_fn,
                                    device=self.device,
                                    actor_blocks=3).to(self.device)
        self.state_score_model_opt = torch.optim.Adam(self.state_score_model.parameters(), lr=1e-4)
        self.state_score_reward_rms = utils.RMS(device=self.device)

        self.finetune_type = finetune_type
        if self.finetune_type == 'DDPG':
            print('!!!we use finetune style 0: we directly finetune the gaussian policy')
        elif self.finetune_type == 'ExDM':
            print('!!!we use finetune style 1: we use the cep score to finetune the score model during the finetune stage')
        elif self.finetune_type == 'IDQL':
            print('!!!we use finetune the diffusion policy with IDQL')
        elif self.finetune_type == 'DQL':
            print('!!!we use finetune the diffusion policy with DQL')
        elif self.finetune_type == 'QSM':
            print('!!!we use finetune the diffusion policy with QSM')
        elif self.finetune_type == 'DIPO':
            print('!!!we use finetune the diffusion policy with DIPO')
        else:
            raise NotImplementedError


        if self.finetune_type == 'DDPG':
            pass
        elif self.finetune_type == 'ExDM':
            self.finetuned_score_model = ScoreNet(input_dim=self.obs_dim + self.action_dim,
                                                  output_dim=self.action_dim,
                                                  marginal_prob_std=marginal_prob_std_fn,
                                                  critic_type=critic_type,
                                                  device=self.device,
                                                  actor_blocks=3, 
                                                  q_layer=2, 
                                                  alpha=3.0).to(self.device)
            self.finetuned_score_model.q[0].set_guidance(guidance_scale)
            if critic_type == 'IQL':
                self.finetuned_score_model.q[0].set_tau(tau)
            self.finetuned_score_model.q[0].set_alpha(alpha)
            self.finetuned_score_model_opt = torch.optim.Adam(self.finetuned_score_model.parameters(), lr=1e-4)
        elif self.finetune_type == 'IDQL' or self.finetune_type == 'DQL' or self.finetune_type == 'QSM' or self.finetune_type == 'DIPO':
            pass
        else:
            raise NotImplementedError

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        if self.init_critic:
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)
        print('we load the score model')
        utils.hard_update_params(other.score_model, self.score_model)
        if self.finetune_type == 'DDPG' or self.finetune_type == 'IDQL' or self.finetune_type == 'DQL' or self.finetune_type == 'QSM' or self.finetune_type == 'DIPO':
            pass
        elif self.finetune_type == 'ExDM':
            utils.hard_update_params(other.score_model, self.finetuned_score_model)
        else:
            raise NotImplementedError
        
    def load_all_models(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        if self.init_critic:
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)
        print('we load the score model')
        utils.hard_update_params(other.score_model, self.score_model)
        utils.hard_update_params(other.finetuned_score_model, self.finetuned_score_model)

    def act(self, obs, meta, step, eval_mode, sample_type='cond'):
        if self.reward_free or self.finetune_type == 'DDPG':
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            h = self.encoder(obs)
            inputs = [h]
            for value in meta.values():
                value = torch.as_tensor(value, device=self.device).unsqueeze(0)
                inputs.append(value)
            inpt = torch.cat(inputs, dim=-1)
            #assert obs.shape[-1] == self.obs_shape[-1]
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(inpt, stddev)
            if eval_mode:
                action = dist.mean
            else:
                action = dist.sample(clip=None)
                if step < self.num_expl_steps:
                    action.uniform_(-1.0, 1.0)
            return action.cpu().numpy()[0]
        else:
            # diffusion policy sample action
            if self.finetune_type == 'ExDM':
                action = self.finetuned_score_model.select_actions(obs.reshape(1, -1), diffusion_steps=self.diffusion_steps, sample_type=sample_type)
            elif self.finetune_type == 'IDQL':
                # IDQL采样方式类似SfBC，采多个action，然后取Q函数最高的那个
                action = self.score_model.select_actions_sfbc(obs.reshape(1, -1), diffusion_steps=self.diffusion_steps)
            elif self.finetune_type == 'DQL' or self.finetune_type == 'QSM' or self.finetune_type == 'DIPO':
                action = self.score_model.select_actions(obs.reshape(1, -1), diffusion_steps=self.diffusion_steps, sample_type='uncond')
            else:
                raise NotImplementedError
            
            if not eval_mode:
                action = action[0]
            return action

    def update_score(self, obs, action):
        metrics = dict()
        # Update diffusion behavior
        self.score_model.train()

        all_s = obs
        all_a = action
        random_t = torch.rand(all_a.shape[0], device=all_a.device) * (1. - 1e-3) + 1e-3  
        z = torch.randn_like(all_a)
        alpha_t, std = self.marginal_prob_std_fn(random_t)
        perturbed_x = all_a * alpha_t[:, None] + z * std[:, None]
        episilon = self.score_model(perturbed_x, random_t, all_s)
        loss = torch.mean(torch.sum((episilon * std[:, None] + z)**2, dim=(1,)))

        self.score_model_opt.zero_grad()
        loss.backward()  
        self.score_model_opt.step()
        metrics['score_loss'] = loss.item()
        return metrics

    def update_DQL_score(self, obs, action):
        metrics = dict()
        # Update diffusion with DQL
        # DQL loss = BC loss - eta * \nabla_\theta Q(s, \pi(s))
        self.score_model.train()

        all_s = obs
        all_a = action
        random_t = torch.rand(all_a.shape[0], device=all_a.device) * (1. - 1e-3) + 1e-3  
        z = torch.randn_like(all_a)
        alpha_t, std = self.marginal_prob_std_fn(random_t)
        perturbed_x = all_a * alpha_t[:, None] + z * std[:, None]
        episilon = self.score_model(perturbed_x, random_t, all_s)
        bc_loss = torch.mean(torch.sum((episilon * std[:, None] + z)**2, dim=(1,)))

        new_action_with_grad = self.score_model.select_actions_with_grad(obs, diffusion_steps=self.diffusion_steps)

        # q_values = self.score_model.q[0].q0(new_action_with_grad, obs)
        # q_loss = - q_values.mean()

        q1_values, q2_values = self.score_model.q[0].q0.both(new_action_with_grad, obs)
        if np.random.uniform() > 0.5:
            q_loss = - q1_values.mean() / q2_values.abs().mean().detach()
        else:
            q_loss = - q2_values.mean() / q1_values.abs().mean().detach()

        eta = 1.0
        loss = bc_loss + eta * q_loss

        self.score_model_opt.zero_grad()
        loss.backward()  
        self.score_model_opt.step()
        metrics['score_loss'] = bc_loss.item()
        metrics['q_loss'] = q_loss.item()
        return metrics

    def update_QSM_score(self, obs, action):
        metrics = dict()
        # Update diffusion with QSM
        # QSM loss = \| s_theta (a_t) - \nabla_a_t Q \|_2
        self.score_model.train()

        all_s = obs
        all_a = action
        random_t = torch.rand(all_a.shape[0], device=all_a.device) * (1. - 1e-3) + 1e-3  
        z = torch.randn_like(all_a)
        alpha_t, std = self.marginal_prob_std_fn(random_t)
        perturbed_x = all_a * alpha_t[:, None] + z * std[:, None]
        episilon = self.score_model(perturbed_x, random_t, all_s)
        
        detach_a = perturbed_x.detach().requires_grad_(True)
        qs = self.score_model.q[0].q0.both(detach_a , obs)
        guidance_q1 = torch.autograd.grad(qs[0].sum(), detach_a)[0]
        guidance_q2 = torch.autograd.grad(qs[1].sum(), detach_a)[0]
        guidance = torch.stack((guidance_q1, guidance_q2), 0).mean(0).detach()

        q_grad_coeff = 10.0
        loss = F.mse_loss(episilon * std[:, None], q_grad_coeff * guidance)

        self.score_model_opt.zero_grad()
        loss.backward()  
        self.score_model_opt.step()
        metrics['qsm_loss'] = loss.item()
        return metrics
    
    def update_state_score(self, obs):
        metrics = dict()
        # Update diffusion behavior
        self.state_score_model.train()

        all_s = obs
        random_t = torch.rand(all_s.shape[0], device=all_s.device) * (1. - 1e-3) + 1e-3  
        z = torch.randn_like(all_s)
        alpha_t, std = self.marginal_prob_std_fn(random_t)
        perturbed_x = all_s * alpha_t[:, None] + z * std[:, None]
        episilon = self.state_score_model(perturbed_x, random_t)
        loss = torch.mean(torch.sum((episilon * std[:, None] + z)**2, dim=(1,)))

        self.state_score_model_opt.zero_grad()
        loss.backward()  
        self.state_score_model_opt.step()
        metrics['state_score_loss'] = loss.item()
        return metrics
    
    def update_cep_score(self, obs, action):
        # used in finetune_type=1
        # finetune_score(o, a) = score(o, a) + \nabla f
        metrics = dict()
        all_s = obs
        all_a = action
        
        random_t = torch.rand(all_a.shape[0], device=all_a.device) * (1. - 1e-3) + 1e-3   # unifrom sample from U(eps, 1)
        z = torch.randn_like(all_a)
        alpha_t, std = self.marginal_prob_std_fn(random_t)
        perturbed_x = all_a * alpha_t[:, None] + z * std[:, None]
        score = self.finetuned_score_model(perturbed_x, random_t, all_s)
                
        with torch.no_grad():
            target_score = self.score_model(perturbed_x, random_t, all_s) + self.score_model.q[0].calculate_guidance(perturbed_x, random_t, all_s)
        loss_ = torch.sum((score - target_score) ** 2 * std[:, None], dim=(1,))
        loss = torch.mean(loss_)
        self.finetuned_score_model_opt.zero_grad()
        loss.backward()
        self.finetuned_score_model_opt.step()
        metrics['cep_score_loss'] = loss.item()
        
        return metrics

    def compute_intr_reward(self, obs, action, step):
        # score reward
        all_s = obs
        all_a = action
        
        # intr_rews = []
        # eps=1e-3
        # # time_step=10
        # time_step = self.score_time_step
        # for tt in np.arange(eps, 1+eps, 1./time_step):
        #     # random_t = torch.full((a.shape[0],), float(tt), device=a.device)
        #     random_t = torch.rand(all_a.shape[0], device=all_a.device) * (1. - eps) + eps # unifrom sample from U(eps, 1)
        #     z = torch.randn_like(all_a) # sample from N(0, 1)
        #     alpha_t, std = self.marginal_prob_std_fn(random_t)
        #     perturbed_x = all_a * alpha_t[:, None] + z * std[:, None]
        #     score = self.score_model(perturbed_x, random_t, all_s)
        #     intr_rew = torch.sum((score * std[:, None] + z)**2, dim=(1,))
        #     intr_rews.append(intr_rew)
        # score_loss_ = torch.stack(intr_rews).mean(dim=0)
        # score_loss_ = score_loss_.reshape(-1, 1)
        # _, score_reward_var = self.score_reward_rms(score_loss_)
        # score_reward = score_loss_ * self.diffusion_scale / (torch.sqrt(score_reward_var) + 1e-8)
        
        state_intr_rews = []
        eps=1e-3
        # time_step=10
        time_step = self.score_time_step
        for tt in np.arange(eps, 1+eps, 1./time_step):
            random_t = torch.rand(all_s.shape[0], device=all_s.device) * (1. - eps) + eps # unifrom sample from U(eps, 1)
            z = torch.randn_like(all_s) # sample from N(0, 1)
            alpha_t, std = self.marginal_prob_std_fn(random_t)
            perturbed_x = all_s * alpha_t[:, None] + z * std[:, None]
            score = self.state_score_model(perturbed_x, random_t)
            state_intr_rew = torch.sum((score * std[:, None] + z)**2, dim=(1,))
            state_intr_rews.append(state_intr_rew)
        state_score_loss_ = torch.stack(state_intr_rews).mean(dim=0)
        state_score_loss_ = state_score_loss_.reshape(-1, 1)
        _, state_score_reward_var = self.state_score_reward_rms(state_score_loss_)
        state_score_reward = state_score_loss_ * self.diffusion_scale / (torch.sqrt(state_score_reward_var) + 1e-8)
        
        reward = state_score_reward
        return reward

    def update(self, replay_iter, step, iter_num=1):
        metrics = dict()
        if self.reward_free:
            for _ in range(self.score_train_iter):
                batch = next(replay_iter)
                obs, action, extr_reward, discount, next_obs, done = utils.to_torch(batch, self.device)
                # augment and encode
                obs = self.aug_and_encode(obs)
                with torch.no_grad():
                    next_obs = self.aug_and_encode(next_obs)
                
                # update score model
                metrics.update(self.update_score(obs, action))
                metrics.update(self.update_state_score(obs))
                
                with torch.no_grad():
                    intr_reward = self.compute_intr_reward(obs, action, step)

                if self.use_tb or self.use_wandb:
                    metrics['intr_reward'] = intr_reward.mean().item()
                reward = intr_reward

                if not self.update_encoder:
                    obs = obs.detach()
                    next_obs = next_obs.detach()

                # update critic
                metrics.update(
                    self.update_critic(obs.detach(), action, reward, discount,
                                    next_obs.detach(), step))

                # update actor
                metrics.update(self.update_actor(obs.detach(), step))

                # update critic target
                utils.soft_update_params(self.critic, self.critic_target,
                                        self.critic_target_tau)
            
            if step % 500 == 0:
                print('step:', step, 'score loss', metrics['score_loss'], 'state score loss', metrics['state_score_loss'])
        elif self.finetune_type == 'DDPG':
            if step % self.update_every_steps != 0:
                return metrics

            batch = next(replay_iter)

            obs, action, extr_reward, discount, next_obs, done = utils.to_torch(
                batch, self.device)

            with torch.no_grad():
                obs = self.aug_and_encode(obs)
                next_obs = self.aug_and_encode(next_obs)

            reward = extr_reward

            if self.use_tb or self.use_wandb:
                metrics['batch_reward'] = reward.mean().item()

            # update critic
            metrics.update(
                self.update_critic(obs, action, reward, discount, next_obs, step))

            # update actor
            metrics.update(self.update_actor(obs, step))

            # update critic target
            utils.soft_update_params(self.critic, self.critic_target,
                                    self.critic_target_tau)
        else:
            # we first train q0, then train qt
            # for training q0, QGPO need the fake a', IQL does not need it
            for update_iter in range(iter_num):
                batch = next(replay_iter)
                obs, action, extr_reward, discount, next_obs, done = utils.to_torch(batch, self.device)

                # augment and encode
                obs = self.aug_and_encode(obs)
                with torch.no_grad():
                    next_obs = self.aug_and_encode(next_obs)
                reward = extr_reward
                
                if self.finetune_type == 'ExDM':
                    cep_action = self.score_model.select_actions(obs.cpu(), diffusion_steps=self.diffusion_steps, sample_type='cond')
                    cep_action = np.vstack(cep_action)
                    cep_action = torch.tensor(cep_action, device=self.device) # batch_size, action_size
                    metrics.update(self.update_cep_score(obs, cep_action))
                    if (update_iter + 1) % 1000 == 0:
                        print('step:', update_iter + 1, 'score loss:', metrics['cep_score_loss'])
                elif self.finetune_type == 'IDQL':
                    # IDQL训练policy就是对着采样的action进行蒸馏
                    # 由于IDQL采样的时候就是用的sfbc采样，所以似乎我们直接对着action去拟合diffusion model就行
                    metrics.update(self.update_score(obs, action))

                    # sfbc_action = self.score_model.select_actions_sfbc(obs.cpu(), diffusion_steps=self.diffusion_steps)
                    # sfbc_action = np.vstack(sfbc_action)
                    # sfbc_action = torch.tensor(sfbc_action, device=self.device) # batch_size, action_size
                    # metrics.update(self.update_score(obs, sfbc_action))
                    if (update_iter + 1) % 1000 == 0:
                        print('step:', update_iter + 1, 'score loss:', metrics['score_loss'])
                elif self.finetune_type == 'DQL':
                    metrics.update(self.update_DQL_score(obs, action))
                    if (update_iter + 1) % 1000 == 0:
                        print('step:', update_iter + 1, 'score loss:', metrics['score_loss'], 'q loss:', metrics['q_loss'])
                elif self.finetune_type == 'QSM':
                    metrics.update(self.update_QSM_score(obs, action))
                    if (update_iter + 1) % 1000 == 0:
                        print('step:', update_iter + 1, 'QSM loss:', metrics['qsm_loss'])
                elif self.finetune_type == 'DIPO':
                    # action = action + eta \nabla_a Q(s, a)
                    dipo_action = action.detach().clone()
                    dipo_action_optim = torch.optim.Adam([dipo_action], lr=0.03, eps=1e-5)
                    for _ in range(20):
                        dipo_action.requires_grad_(True)
                        q_values = self.score_model.q[0].q0(dipo_action, obs)
                        dipo_q_loss = -q_values
                        # guidance_q = torch.autograd.grad(-q_values.mean(), dipo_action)[0]

                        dipo_action_optim.zero_grad()
                        dipo_q_loss.backward(torch.ones_like(dipo_q_loss))
                        torch.nn.utils.clip_grad_norm_([dipo_action], max_norm=0.1 * dipo_action.shape[-1], norm_type=2,)
                        dipo_action_optim.step()

                        dipo_action.requires_grad_(False)
                        dipo_action.clamp_(-1.0, 1.0)
                    dipo_action = dipo_action.detach()
                    
                    metrics.update(self.update_score(obs, dipo_action))
                    if (update_iter + 1) % 1000 == 0:
                        print('step:', update_iter + 1, 'score loss:', metrics['score_loss'])

                data = {'s': obs,
                        'a': action,
                        'r': reward,
                        's_': next_obs,
                        'd': done,}
                
                if self.critic_type == 'QGPO':
                    # QGPO里面q0的训练需要fake a'
                    # IQL不需要
                    data['fake_a_'] = torch.Tensor(self.score_model.sample(next_obs.cpu(), sample_per_state=32, diffusion_steps=self.diffusion_steps, sample_type='uncond').astype(np.float32)).to(self.device)
                if self.finetune_type == 'ExDM':
                    # CEP里面qt的训练需要fake a
                    # 其他方法不需要
                    data['fake_a'] = torch.Tensor(self.finetuned_score_model.sample(obs.cpu(), sample_per_state=32, diffusion_steps=self.diffusion_steps, sample_type='uncond').astype(np.float32)).to(self.device)
                
                if self.finetune_type == 'ExDM':
                    # 像CEP一样更新q0和qt
                    loss1 = self.score_model.q[0].update_q0(data)
                    loss2 = self.score_model.q[0].update_qt(data)
                elif self.finetune_type == 'IDQL' or self.finetune_type == 'DQL' or self.finetune_type == 'QSM' or self.finetune_type == 'DIPO':
                    # 只更新q0，不更新qt
                    loss1 = self.score_model.q[0].update_q0(data)
                    loss2 = 0.0

                if (update_iter + 1) % 1000 == 0:
                    print('step:', update_iter+1, 'q0loss:', loss1, 'qtloss:', loss2)

            metrics['q0loss'] = loss1
            metrics['qtloss'] = loss2

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        return metrics