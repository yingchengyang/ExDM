import torch
import numpy as np
def loss_fn(model, x, marginal_prob_std, eps=1e-3, each_loss=False):
    """The loss function for training score-based generative models.

    Args:
    model: A PyTorch model instance that represents a
        time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
    eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps # unifrom sample from U(eps, 1)
    z = torch.randn_like(x) # sample from N(0, 1)
    alpha_t, std = marginal_prob_std(random_t)
    perturbed_x = x * alpha_t[:, None] + z * std[:, None]
    score = model(perturbed_x, random_t)
    loss_ = torch.sum((score * std[:, None] + z)**2, dim=(1,))
    loss = torch.mean(loss_)
    if each_loss:
        return loss_, loss
    return loss

def intr_reward(model, x, marginal_prob_std, eps=1e-3, time_step=10):
    """The loss function for training score-based generative models.

    Args:
    model: A PyTorch model instance that represents a
        time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
    eps: A tolerance value for numerical stability.
    """
    intr_rews = []
    for tt in np.arange(eps, 1+eps, 1./time_step):
        # random_t = torch.full_like(x, float(tt))
        random_t = torch.full((x.shape[0],), float(tt), device=x.device)
        # random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps # unifrom sample from U(eps, 1)
        z = torch.randn_like(x) # sample from N(0, 1)
        alpha_t, std = marginal_prob_std(random_t)
        perturbed_x = x * alpha_t[:, None] + z * std[:, None]
        score = model(perturbed_x, random_t)
        intr_rew = torch.sum((score * std[:, None] + z)**2, dim=(1,))
        intr_rews.append(intr_rew)
    intr_rew = torch.stack(intr_rews).mean(dim=0)
    return intr_rew

def cep_loss_fn(model, target_model, x, marginal_prob_std, eps=1e-3, each_loss=False):
    """The loss function for training score-based generative models.

    Args:
    model: A PyTorch model instance that represents a
        time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
    eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps # unifrom sample from U(eps, 1)
    z = torch.randn_like(x) # sample from N(0, 1)
    alpha_t, std = marginal_prob_std(random_t)
    perturbed_x = x * alpha_t[:, None] + z * std[:, None]
    score = model(perturbed_x, random_t)
    
    with torch.no_grad():
        target_score = target_model(perturbed_x, random_t) + target_model.q[0].calculate_guidance(perturbed_x, random_t, target_model.condition)
    loss_ = torch.sum((score - target_score) ** 2 * std[:, None], dim=(1,))
    # loss_ = torch.sum((score * std[:, None] + z)**2, dim=(1,))
    loss = torch.mean(loss_)
    if each_loss:
        return loss_, loss
    return loss