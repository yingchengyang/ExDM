import torch
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
from diffusion_SDE import dpm_solver_pytorch
from diffusion_SDE import schedule
from scipy.special import softmax


def update_target(new, target, tau):
    # Update the frozen target models
    for param, target_param in zip(new.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[..., None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    

def mlp(dims, activation=nn.ReLU, output_activation=None):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'
    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net

    
class TwinQ(nn.Module):
    def __init__(self, action_dim, state_dim, layers=2):
        super().__init__()
        dims = [state_dim + action_dim] + [256] * layers + [1]
        # dims = [state_dim + action_dim, 256, 256, 1] # TODO
        self.q1 = mlp(dims)
        self.q2 = mlp(dims)

    def both(self, action, condition=None):
        as_ = torch.cat([action, condition], -1) if condition is not None else action
        return self.q1(as_), self.q2(as_)

    def forward(self, action, condition=None):
        return torch.min(*self.both(action, condition))


class ValueFunction(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        dims = [state_dim, 256, 256, 1]
        self.v = mlp(dims)

    def forward(self, state):
        return self.v(state)


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class GuidanceQt(nn.Module):
    def __init__(self, action_dim, state_dim):
        super().__init__()
        dims = [action_dim + 32 + state_dim, 256, 256, 256, 256, 1]
        self.qt = mlp(dims, activation=SiLU)
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=32), nn.Linear(32, 32))

    def forward(self, action, t, condition=None):
        embed = self.embed(t)
        ats = torch.cat([action, embed, condition], -1) if condition is not None else torch.cat([action, embed], -1)
        return self.qt(ats)


class Critic_Guide(nn.Module):
    def __init__(self, adim, sdim) -> None:
        super().__init__()
        # is sdim is 0  means unconditional guidance
        self.conditional_sampling = False if sdim == 0 else True
        self.q0 = None
        self.qt = None

    def forward(self, a, condition=None):
        return self.q0(a, condition)
    
    def set_guidance(self, guidance):
        print("!!!we set guidance as", guidance)
        self.guidance_scale = guidance
        
    def set_tau(self, tau):
        print("!!!we set tau as", tau)
        self.tau = tau
        
    def set_alpha(self, alpha):
        print("!!!we set alpha as", alpha)
        self.alpha = alpha

    def calculate_guidance(self, a, t, condition=None):
        with torch.enable_grad():
            a.requires_grad_(True)
            Q_t = self.qt(a, t, condition)
            guidance = self.guidance_scale * torch.autograd.grad(torch.sum(Q_t), a)[0]
        return guidance.detach()

    def calculateQ(self, a, condition=None):
        return self(a, condition)

    def update_q0(self, data):
        raise NotImplementedError

    def update_qt(self, data):
        # input  many s <bz, S>  action <bz, M, A>,
        s = data['s']
        a = data['a']
        fake_a = data['fake_a']
        energy = self.q0_target(fake_a, torch.stack([s] * fake_a.shape[1], axis=1)).detach().squeeze()

        self.all_mean = torch.mean(energy, dim=-1).detach().cpu().squeeze().numpy()
        self.all_std = torch.std(energy, dim=-1).detach().cpu().squeeze().numpy()

        if self.method == "mse":
            random_t = torch.rand(a.shape[0], device=s.device) * (1. - 1e-3) + 1e-3
            z = torch.randn_like(a)
            alpha_t, std = schedule.marginal_prob_std(random_t)
            perturbed_a = a * alpha_t[..., None] + z * std[..., None]

            # calculate sample based baselines
            # sample_based_baseline = torch.max(energy, dim=-1, keepdim=True)[0]  #<bz , 1>
            sample_based_baseline = 0.0
            self.debug_used = (self.q0_target(a,
                                              s).detach() * self.alpha - sample_based_baseline * self.alpha).detach().cpu().squeeze().numpy()
            loss = torch.mean((self.qt(perturbed_a, random_t, s) - self.q0_target(a,
                                                                                  s).detach() * self.alpha + sample_based_baseline * self.alpha) ** 2)
        elif self.method == "emse":
            random_t = torch.rand(a.shape[0], device=s.device) * (1. - 1e-3) + 1e-3
            z = torch.randn_like(a)
            alpha_t, std = schedule.marginal_prob_std(random_t)
            perturbed_a = a * alpha_t[..., None] + z * std[..., None]

            # calculate sample based baselines
            # sample_based_baseline = (torch.logsumexp(energy*self.alpha, dim=-1, keepdim=True)- np.log(energy.shape[1])) /self.alpha   #<bz , 1>
            sample_based_baseline = torch.max(energy, dim=-1, keepdim=True)[0]  # <bz , 1>
            self.debug_used = (self.q0_target(a,
                                              s).detach() * self.alpha - sample_based_baseline * self.alpha).detach().cpu().squeeze().numpy()

            def unlinear_func(value, alpha, clip=False):
                if clip:
                    return torch.exp(torch.clamp(value * alpha, -100, 4.5))
                else:
                    return torch.exp(value * alpha)

            loss = torch.mean((unlinear_func(self.qt(perturbed_a, random_t, s), 1.0, clip=True) - unlinear_func(
                self.q0_target(a, s).detach() - sample_based_baseline, self.alpha, clip=True)) ** 2)
        elif self.method == "CEP":
            # CEP guidance method, as proposed in the paper
            logsoftmax = nn.LogSoftmax(dim=1)
            softmax = nn.Softmax(dim=1)

            x0_data_energy = energy * self.alpha
            # random_t = torch.rand((fake_a.shape[0], fake_a.shape[1]), device=s.device) * (1. - 1e-3) + 1e-3
            random_t = torch.rand((fake_a.shape[0],), device=s.device) * (1. - 1e-3) + 1e-3
            random_t = torch.stack([random_t] * fake_a.shape[1], dim=1)
            z = torch.randn_like(fake_a)
            alpha_t, std = schedule.marginal_prob_std(random_t, device=self.device)
            perturbed_fake_a = fake_a * alpha_t[..., None] + z * std[..., None]
            xt_model_energy = self.qt(perturbed_fake_a, random_t, torch.stack([s] * fake_a.shape[1], axis=1)).squeeze()
            p_label = softmax(x0_data_energy)
            self.debug_used = torch.flatten(p_label).detach().cpu().numpy()
            loss = -torch.mean(torch.sum(p_label * logsoftmax(xt_model_energy), axis=-1))  # <bz,M>
        else:
            raise NotImplementedError

        self.qt_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.qt_optimizer.step()

        return loss.detach().cpu().numpy()


class QGPO_Critic(Critic_Guide):
    def __init__(self, adim, sdim, device, q_layer, alpha, method='CEP') -> None:
        super().__init__(adim, sdim)
        # is sdim is 0  means unconditional guidance
        assert sdim > 0
        # only apply to conditional sampling here
        self.q0 = TwinQ(adim, sdim).to(device)
        self.q0_target = copy.deepcopy(self.q0).requires_grad_(False).to(device)
        self.qt = GuidanceQt(adim, sdim).to(device)
        self.q_optimizer = torch.optim.Adam(self.q0.parameters(), lr=3e-4)
        self.qt_optimizer = torch.optim.Adam(self.qt.parameters(), lr=3e-4)
        self.discount = 0.99

        self.alpha = alpha
        self.guidance_scale = 1.0
        print('guidance', self.guidance_scale, 'alpha:', self.alpha)
        
        self.method = method
        self.device = device

    def update_q0(self, data):
        s = data["s"]
        a = data["a"]
        r = data["r"]
        s_ = data["s_"]
        d = data["d"]

        # fake_a = data['fake_a']
        fake_a_ = data['fake_a_']
        with torch.no_grad():
            softmax = nn.Softmax(dim=1)
            next_energy = self.q0_target(fake_a_, torch.stack([s_] * fake_a_.shape[1],
                                         axis=1)).detach().squeeze(dim=2)  # <bz, 16>
            next_v = torch.sum(softmax(self.alpha * next_energy) * next_energy, dim=-1, keepdim=True)

        # Update Q function
        targets = r + (1. - d.float()) * self.discount * next_v.detach()
        qs = self.q0.both(a, s)
        # import pdb; pdb.set_trace()
        # r: 1024, 1; d: 1024, 1
        # targets: [1024, 1024]; qs[0]: [1024, 1] 
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target
        update_target(self.q0, self.q0_target, 0.005)
        
        return q_loss.cpu().item()


class IQL_Critic(Critic_Guide):
    def __init__(self, adim, sdim, device, q_layer, alpha, method='CEP') -> None:
        super().__init__(adim, sdim)
        # is sdim is 0  means unconditional guidance
        assert sdim > 0
        # only apply to conditional sampling here
        self.q0 = TwinQ(adim, sdim, layers=q_layer).to(device)
        self.q0_target = copy.deepcopy(self.q0).to(device)

        self.vf = ValueFunction(sdim).to(device)
        self.q_optimizer = torch.optim.Adam(self.q0.parameters(), lr=3e-4)
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=3e-4)    

        self.qt = GuidanceQt(adim, sdim).to(device)
        self.qt_optimizer = torch.optim.Adam(self.qt.parameters(), lr=3e-4)
        
        self.discount = 0.99

        self.alpha = alpha
        self.guidance_scale = 1.0
        
        # self.tau = 0.9 if "maze" in args.env else 0.7
        self.tau = 0.7
        print('tau', self.tau, 'guidance', self.guidance_scale, 'alpha', self.alpha)
        
        self.method = method
        self.device = device
   
    def update_q0(self, data):
        s = data["s"]
        a = data["a"]
        r = data["r"]
        s_ = data["s_"]
        d = data["d"]
        with torch.no_grad():
            target_q = self.q0_target(a, s).detach()
            next_v = self.vf(s_).detach()

        # Update value function
        v = self.vf(s)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()
        
        # Update Q function
        targets = r + (1. - d.float()) * self.discount * next_v.detach()
        qs = self.q0.both(a, s)
        self.v = v.mean()
        q_loss = sum(torch.nn.functional.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        self.v_loss = v_loss
        self.q_loss = q_loss
        self.q = target_q.mean()
        self.v = next_v.mean()
        # Update target
        update_target(self.q0, self.q0_target, 0.005)        
        
        return q_loss.cpu().item()

def sigmoid(x):
    return x * torch.sigmoid(x)


class Residual_Block(nn.Module):
    def __init__(self, input_dim, output_dim, t_dim=128, last=False):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SiLU(),
            nn.Linear(t_dim, output_dim),
        )
        self.dense1 = nn.Sequential(nn.Linear(input_dim, output_dim), SiLU())
        self.dense2 = nn.Sequential(nn.Linear(output_dim, output_dim), SiLU())
        self.modify_x = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x, t):
        h1 = self.dense1(x) + self.time_mlp(t)
        h2 = self.dense2(h1)
        return h2 + self.modify_x(x)

  
class MLPResNetBlock(nn.Module):
    """MLPResNet block."""
    def __init__(self, features, act, dropout_rate=None, use_layer_norm=False):
        super(MLPResNetBlock, self).__init__()
        self.features = features
        self.act = act
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(features)

        self.fc1 = nn.Linear(features, features * 4)
        self.fc2 = nn.Linear(features * 4, features)
        self.residual = nn.Linear(features, features)

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None and dropout_rate > 0.0 else None

    def forward(self, x, training=False):
        residual = x
        if self.dropout is not None:
            x = self.dropout(x)

        if self.use_layer_norm:
            x = self.layer_norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        if residual.shape != x.shape:
            residual = self.residual(residual)

        return residual + x


class MLPResNet(nn.Module):
    def __init__(self, num_blocks, input_dim, out_dim, dropout_rate=None, use_layer_norm=False, hidden_dim=256, activations=F.relu):
        super(MLPResNet, self).__init__()
        self.num_blocks = num_blocks
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.activations = activations

        self.fc = nn.Linear(input_dim+128, self.hidden_dim)

        self.blocks = nn.ModuleList([MLPResNetBlock(self.hidden_dim, self.activations, self.dropout_rate, self.use_layer_norm)
                                     for _ in range(self.num_blocks)])

        self.out_fc = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x, training=False):
        x = self.fc(x)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.activations(x)
        x = self.out_fc(x)

        return x


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class ScoreBase(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=32, critic_type='IQL', device='cuda', q_layer=2, alpha=3.0):
        super().__init__()
        self.output_dim = output_dim
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
        self.device = device
        self.noise_schedule = dpm_solver_pytorch.NoiseScheduleVP(schedule='linear')
        self.dpm_solver = dpm_solver_pytorch.DPM_Solver(self.forward_dmp_wrapper_fn, self.noise_schedule,
                                                        predict_x0=True)
        self.uncond_dpm_solver = dpm_solver_pytorch.DPM_Solver(self.uncond_forward_dmp_wrapper_fn, self.noise_schedule,
                                                               predict_x0=True)
        # self.dpm_solver = dpm_solver_pytorch.DPM_Solver(self.forward_dmp_wrapper_fn, self.noise_schedule)
        self.marginal_prob_std = marginal_prob_std
        self.q = []
        self.critic_type = critic_type
        if self.critic_type == 'QGPO':
            self.q.append(QGPO_Critic(adim=output_dim, sdim=input_dim - output_dim,
                                    device=device, q_layer=q_layer, alpha=alpha, method='CEP'))
        elif self.critic_type == 'IQL':
            self.q.append(IQL_Critic(adim=output_dim, sdim=input_dim - output_dim,
                                    device=device, q_layer=q_layer, alpha=alpha, method='CEP'))
        else:
            raise NotImplementedError()

    def forward_dmp_wrapper_fn(self, x, t):
        # q guided sampling
        score = self(x, t)
        result = - (score + self.q[0].calculate_guidance(x, t, self.condition)) * self.marginal_prob_std(t)[1][..., None]
        return result
    
    def uncond_forward_dmp_wrapper_fn(self, x, t):
        # unconditional sampling
        score = self(x, t)
        result = - score * self.marginal_prob_std(t)[1][..., None]
        return result

    def dpm_wrapper_sample(self, dim, batch_size, sample_type, sample_per_state=16, **kwargs):
        with torch.no_grad():
            init_x = torch.randn(batch_size, dim, device=self.device)
            if sample_type=='cond':
                return self.dpm_solver.sample(init_x, **kwargs).cpu().numpy()
            elif sample_type=='uncond':
                return self.uncond_dpm_solver.sample(init_x, **kwargs).cpu().numpy()
            elif sample_type=='dql_train':
                # dql训练的时候的采样方法，特点是无guidance采样，并且采样带梯度
                return self.uncond_dpm_solver.sample_with_grad(init_x, **kwargs)
            else:
                raise NotImplementedError

    def calculateQ(self, s, a, t=None):
        if s is None:
            if self.condition.shape[0] == a.shape[0]:
                s = self.condition
            elif self.condition.shape[0] == 1:
                s = torch.cat([self.condition] * a.shape[0])
            else:
                assert False
        return self.q[0](a, s)

    def forward(self, x, t, condition=None):
        raise NotImplementedError

    def select_actions(self, states, diffusion_steps=15, sample_type='cond'):
        self.eval()
        multiple_input = True
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            if states.dim == 1:
                states = states.unsqueeze(0)
                multiple_input = False
            num_states = states.shape[0]
            self.condition = states
            results = self.dpm_wrapper_sample(self.output_dim, batch_size=states.shape[0], sample_type=sample_type, steps=diffusion_steps, order=2)
            actions = results.reshape(num_states, self.output_dim).copy()  # <bz, A>
            self.condition = None
        out_actions = [actions[i] for i in range(actions.shape[0])] if multiple_input else actions[0]
        self.train()
        return out_actions

    def select_actions_sfbc(self, states, diffusion_steps=15, sample_per_state=4, average_num=1):
        # select actions from sfbc, i.e., a1, a2, ..., an \sim \mu, then sample \propto e^{Q(s,ai)}
        self.eval()
        num_states = states.shape[0]
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            states = torch.repeat_interleave(states, sample_per_state, dim=0)
            self.condition = states
            results = self.dpm_wrapper_sample(self.output_dim, batch_size=states.shape[0], sample_type='uncond', steps=diffusion_steps, order=2)
            # num_states: 1024, sample_per_state: 32, self.output_dim: 6
            actions = results[:, :].reshape(num_states, sample_per_state, self.output_dim).copy() # bs, sample_per_state, action_dim
            # import pdb; pdb.set_trace()
            states = states.reshape(num_states, sample_per_state, -1) # bs, sample_per_state, state_dim
            actions_tensor = torch.tensor(actions, dtype=states.dtype, device=states.device)
            q_values = self.q[0].q0_target(actions_tensor, states).to("cpu").detach().numpy() # bs, sample_per_state, 1
            out_actions = []
            alpha = 100.0
            for i in range(actions.shape[0]):
                returns = q_values[i, :, 0]
                index = np.argmax(returns)
                out_actions.append(actions[i][index])
                
                # soft sample
                # returns = returns * alpha
                # allowed_max = np.sort(returns)[-average_num] + 40
                # returns[returns > allowed_max] = allowed_max
                # unnormalised_p = np.exp(returns - np.max(returns))
                # p = unnormalised_p / np.sum(unnormalised_p)
                # # replace: can we repeat sample the same item
                # index = np.random.choice(actions[i].shape[0], p=p, size=average_num, replace=False) # average_num actions
                # out_actions.append(np.mean(actions[i][index], axis=0)) # action_dim
            self.condition = None
        self.train()
        return out_actions

    def select_actions_with_grad(self, states, diffusion_steps=15):
        # 这个函数的输入要求是tensor，目前只在dql中使用，作用是采样动作并且保留梯度
        multiple_input = True
        if states.dim == 1:
            states = states.unsqueeze(0)
            multiple_input = False
        num_states = states.shape[0]
        self.condition = states
        results = self.dpm_wrapper_sample(self.output_dim, batch_size=states.shape[0], sample_type='dql_train', steps=diffusion_steps, order=2)
        actions = results.reshape(num_states, self.output_dim)  # <bz, A>
        self.condition = None
        out_actions = [actions[i] for i in range(actions.shape[0])] if multiple_input else actions[0]
        return actions

    def sample(self, states, sample_per_state=16, diffusion_steps=15, sample_type='cond'):
        self.eval()
        num_states = states.shape[0]
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            states = torch.repeat_interleave(states, sample_per_state, dim=0)
            self.condition = states
            results = self.dpm_wrapper_sample(self.output_dim, batch_size=states.shape[0], sample_type=sample_type, steps=diffusion_steps, order=2)
            # num_states: 1024, sample_per_state: 32, self.output_dim: 6
            actions = results[:, :].reshape(num_states, sample_per_state, self.output_dim).copy()
            self.condition = None
        self.train()
        return actions


class ScoreNet(ScoreBase):
    def __init__(self, input_dim, output_dim, marginal_prob_std, embed_dim=64, critic_type='IQL', device='cuda', actor_blocks=3, q_layer=2, alpha=3.0):
        super().__init__(input_dim, output_dim, marginal_prob_std, embed_dim, critic_type, device, q_layer, alpha)
        self.output_dim = output_dim
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim))
        self.device= device
        self.marginal_prob_std = marginal_prob_std
        self.main = MLPResNet(actor_blocks, input_dim, output_dim, dropout_rate=0.1, use_layer_norm=True, hidden_dim=256, activations=Mish())
        self.cond_model = mlp([64, 128, 128], output_activation=None, activation=Mish)

        # The swish activation function
        # self.act = lambda x: x * torch.sigmoid(x)
        
    def forward(self, x, t, condition=None):
        if condition is None:
            if self.condition.shape[0] == x.shape[0]:
                condition = self.condition
            elif self.condition.shape[0] == 1:
                condition = torch.cat([self.condition] * x.shape[0])
            else:
                assert False
                        
        embed = self.cond_model(self.embed(t))
        all = torch.cat([x, condition, embed], dim=-1)
        h = self.main(all)
        return h


class State_ScoreNet(nn.Module):
    def __init__(self, output_dim, marginal_prob_std, embed_dim=64, device='cuda', actor_blocks=3):
        super().__init__()
        self.output_dim = output_dim
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
        self.device = device
        self.noise_schedule = dpm_solver_pytorch.NoiseScheduleVP(schedule='linear')
        self.dpm_solver = dpm_solver_pytorch.DPM_Solver(self.forward_dmp_wrapper_fn, self.noise_schedule, predict_x0=True)
        self.marginal_prob_std = marginal_prob_std
        
        self.output_dim = output_dim
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim))
        self.device= device
        self.marginal_prob_std = marginal_prob_std
        self.main = MLPResNet(actor_blocks, output_dim, output_dim, dropout_rate=0.1, use_layer_norm=True, hidden_dim=256, activations=Mish())
        self.cond_model = mlp([64, 128, 128], output_activation=None, activation=Mish)
    
    def forward_dmp_wrapper_fn(self, x, t):
        # unconditional sampling
        score = self(x, t)
        result = - score * self.marginal_prob_std(t)[1][..., None]
        return result

    def dpm_wrapper_sample(self, dim, batch_size, sample_type, sample_per_state=16, **kwargs):
        with torch.no_grad():
            init_x = torch.randn(batch_size, dim, device=self.device)
            return self.dpm_solver.sample(init_x, **kwargs).cpu().numpy()

    def forward(self, x, t):
        embed = self.cond_model(self.embed(t))
        all = torch.cat([x, embed], dim=-1)
        h = self.main(all)
        return h

