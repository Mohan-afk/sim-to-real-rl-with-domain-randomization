"""
Sim-to-Real RL with Domain Randomization

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - set_pendulum_mass
def set_pendulum_mass(env, mass):
    """Set a Pendulum environment's mass physics parameter in place.

    Args:
        env: Gymnasium Pendulum-v1 environment.
        mass: Positive float mass to assign.

    Returns:
        The same env with unwrapped mass updated.
    """
    # TODO: update the unwrapped env mass and return env
    env.unwrapped.m = mass

    return env

# Step 2 - set_pendulum_length
def set_pendulum_length(env, length):
    """Set a Pendulum env's rod length physics parameter and return env."""
    # TODO: Set the unwrapped env rod length in-place and return env
    env.unwrapped.l = length  
    return env

# Step 3 - set_pendulum_gravity
def set_pendulum_gravity(env, gravity):
    """Set a Pendulum environment's gravity physics parameter to a given value."""
    # TODO: Update the env's unwrapped gravity in-place and return env
    env.unwrapped.g = gravity
    return env

# Step 4 - sample_physics_config
def sample_physics_config(mass_range, length_range, gravity_range, rng):
    """Sample a physics config (mass, length, gravity) uniformly from ranges.

    Args:
        mass_range: (min, max) float tuple for pendulum mass.
        length_range: (min, max) float tuple for rod length.
        gravity_range: (min, max) float tuple for gravity.
        rng: numpy.random.Generator used for all sampling.

    Returns:
        Dict with keys 'mass', 'length', 'gravity' (floats).
    """
    # TODO: Sample a physics configuration from given min/max ranges\
    mass = float(rng.uniform(mass_range[0],mass_range[1]))
    length = float(rng.uniform(length_range[0],length_range[1]))
    gravity = float(rng.uniform(gravity_range[0],gravity_range[1]))

    return {
        'mass':mass,
        'length':length,
        'gravity':gravity
    }

# Step 5 - build_parallel_pendulum_envs
def build_parallel_pendulum_envs(n_envs, mass_range, length_range, gravity_range, seed):
    """Build parallel Pendulum-v1 envs each with its own sampled physics.

    Args:
        n_envs: Number of environments to create.
        mass_range: (min, max) float tuple for pendulum mass.
        length_range: (min, max) float tuple for rod length.
        gravity_range: (min, max) float tuple for gravity.
        seed: Integer seed for the physics-sampling RNG.

    Returns:
        envs: List of n_envs Gymnasium Pendulum-v1 environments.
        configs: List of physics dicts with keys 'mass', 'length', 'gravity'.
    """
    # TODO: Build n_envs Pendulum envs each with its own sampled physics...
    rng = np.random.default_rng(seed)
    envs = []
    configs = []

    for _ in range(n_envs):
        config = sample_physics_config(mass_range,length_range,gravity_range,rng)
        env = gym.make('Pendulum-v1')
        set_pendulum_mass(env,config['mass'])
        set_pendulum_length(env, config['length'])
        set_pendulum_gravity(env, config['gravity'])

        envs.append(env)
        configs.append(config)

    return envs, configs

# Step 6 - shape_upright_hold_reward
def shape_upright_hold_reward(obs, base_reward, action, angle_thresh=0.2, angvel_thresh=0.5, hold_bonus=1.0):
    """Shape reward with a bonus for holding the pendulum upright and still.

    Args:
        obs: np.ndarray of shape (3,) or (n, 3) as [cos(theta), sin(theta), theta_dot].
        base_reward: float or np.ndarray of shape (n,) from the environment.
        action: float or np.ndarray (accepted for wrapper compatibility).
        angle_thresh: max absolute angle from upright to count as upright.
        angvel_thresh: max absolute angular velocity to count as still.
        hold_bonus: extra reward added when upright and still.

    Returns:
        Shaped reward with the same shape as base_reward.
    """
    # TODO: Add hold_bonus when upright and still; else return base_reward

    obs_arr = np.asarray(obs, dtype=float)

    sin_theta = obs_arr[...,1]
    cos_theta = obs_arr[...,0]
    theta_dot = obs_arr[...,2]

    theta = np.arctan2(sin_theta,cos_theta)
    threshold = (np.abs(theta)<angle_thresh) & (np.abs(theta_dot)<angvel_thresh)
    shaped_reward = base_reward + hold_bonus*threshold.astype(float)

    if np.isscalar(base_reward) or isinstance(base_reward,(float,int)):
        return float(shaped_reward)
    
    return shaped_reward

# Step 7 - build_actor_network
import torch 
import torch.nn as nn 

class GaussianActor(nn.Module):
    def __init__(self,obs_dim,action_dim,hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    def forward(self,obs):
        mean = self.net(obs)
        std = torch.exp(self.log_std)
        return mean, std
    
def build_actor_network(obs_dim, action_dim, hidden_dim=64):
    """Build a Gaussian actor: forward(obs) -> (mean, std).

    Store the learnable log-std as an attribute named `log_std`
    (an nn.Parameter of shape (action_dim,), initialized to zeros) --
    later steps read it via `actor.log_std`.
    """
    # TODO: Construct a Gaussian actor network mapping obs to (mean, std)
    # with two Tanh hidden layers and std = exp(log_std).
    return GaussianActor(obs_dim, action_dim, hidden_dim)

# Step 8 - build_critic_network
import torch
import torch.nn as nn
def build_critic_network(obs_dim, hidden_dim=64):
    # TODO: Build a critic network that maps an observation to a single state-value estimate.
    return nn.Sequential(
        nn.Linear(obs_dim,hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim,hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 1)
    )

# Step 9 - sample_action_log_prob_entropy
import torch
import torch.nn as nn 
from torch.distributions import Normal
def sample_action_log_prob_entropy(actor, obs, deterministic=False):
    """Sample actions from the Gaussian policy; return (actions, log_probs, entropy).

    Use the actor's mean output and its learnable `log_std` parameter
    (std = exp(actor.log_std)). Sum log-probs and entropy over action dims.
    """
    # TODO: Build a Normal(mean, std), take the mean when deterministic=True
    # (otherwise sample), and return actions, summed log-probs, and entropy.
    actor_out = actor(obs)
    mean = actor_out[0] if isinstance(actor_out, tuple) else actor_out

    std = torch.exp(actor.log_std)
    dist = Normal(mean, std)
    
    if deterministic:
        actions = mean
    else:
        actions = dist.sample()
    
    log_probs = dist.log_prob(actions).sum(dim = -1)
    entropy = dist.entropy().sum(dim=-1)
    return actions, log_probs, entropy

# Step 10 - collect_rollout
import torch
import numpy as np

def collect_rollout(envs, actor, critic, n_steps, device='cpu'):
    """Collect a fixed-length rollout from parallel envs into a trajectory dict."""
    n_envs = len(envs)
    
    # 1. Reset each environment to get starting observations
    curr_obs_list = [env.reset()[0] for env in envs]
    curr_obs = np.array(curr_obs_list, dtype=np.float32)
    obs_dim = curr_obs.shape[1]
    
    # Get action_dim directly without drawing a random sample (preserves torch random state!)
    if hasattr(actor, 'log_std'):
        action_dim = actor.log_std.shape[-1]
    else:
        action_dim = envs[0].action_space.shape[0]
    
    # Preallocate buffers
    obs_buf = torch.zeros((n_steps, n_envs, obs_dim), dtype=torch.float32, device=device)
    actions_buf = torch.zeros((n_steps, n_envs, action_dim), dtype=torch.float32, device=device)
    rewards_buf = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
    dones_buf = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
    values_buf = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
    log_probs_buf = torch.zeros((n_steps, n_envs), dtype=torch.float32, device=device)
    
    last_dones = np.zeros(n_envs, dtype=np.float32)
    
    for t in range(n_steps):
        obs_tensor = torch.as_tensor(curr_obs, dtype=torch.float32, device=device)
        obs_buf[t] = obs_tensor
        
        with torch.no_grad():
            actions, log_probs, _ = sample_action_log_prob_entropy(actor, obs_tensor)
            values = critic(obs_tensor).squeeze(-1)
            
        actions_buf[t] = actions
        log_probs_buf[t] = log_probs
        values_buf[t] = values
        
        actions_np = actions.cpu().numpy()
        next_obs_list = []
        step_rewards = []
        step_dones = []
        
        for i, env in enumerate(envs):
            act = actions_np[i]
            next_o, r, terminated, truncated, _ = env.step(act)
            done = bool(terminated or truncated)
            
            r_shaped = float(shape_upright_hold_reward(next_o, r, act))
            step_rewards.append(r_shaped)
            step_dones.append(float(done))
            
            if done:
                next_o, _ = env.reset()
            next_obs_list.append(next_o)
            
        rewards_buf[t] = torch.as_tensor(step_rewards, dtype=torch.float32, device=device)
        dones_buf[t] = torch.as_tensor(step_dones, dtype=torch.float32, device=device)
        
        curr_obs = np.array(next_obs_list, dtype=np.float32)
        last_dones = np.array(step_dones, dtype=np.float32)
        
    return {
        'obs': obs_buf,
        'actions': actions_buf,
        'rewards': rewards_buf,
        'dones': dones_buf,
        'values': values_buf,
        'log_probs': log_probs_buf,
        'last_obs': torch.as_tensor(curr_obs, dtype=torch.float32, device=device),
        'last_dones': torch.as_tensor(last_dones, dtype=torch.float32, device=device)
    }

# Step 11 - rollout_observations
def rollout_observations(rollout):
    """Extract the recorded observations tensor from a collected rollout.

    Args:
        rollout: dict produced by collect_rollout.

    Returns:
        torch.Tensor of shape (n_steps, n_envs, obs_dim) stored under key 'obs'.
    """
    # TODO: Return the observations tensor from the rollout dict
    return rollout['obs']

# Step 12 - rollout_actions
def rollout_actions(rollout):
    """Extract the recorded actions tensor from a collected rollout.

    Args:
        rollout: dict produced by collect_rollout.

    Returns:
        torch.Tensor of shape (n_steps, n_envs, action_dim).
    """
    # TODO: Return the actions tensor stored under the 'actions' key.
    return rollout['actions']

# Step 13 - rollout_rewards
def rollout_rewards(rollout):
    """Extract the recorded rewards tensor from a collected rollout.

    Args:
        rollout: dict returned by collect_rollout.

    Returns:
        torch.Tensor of shape (n_steps, n_envs) under key 'rewards'.
    """
    # TODO: Extract the recorded rewards tensor from a collected rollout.
    return rollout['rewards']

# Step 14 - rollout_dones
def rollout_dones(rollout):
    """Extract the recorded episode-termination flags from a collected rollout.

    Args:
        rollout: dict produced by collect_rollout.

    Returns:
        torch.Tensor of shape (n_steps, n_envs) under key 'dones'.
    """
    # TODO: return the tensor of episode-termination flags from the rollout
    return rollout['dones']

# Step 15 - rollout_values
def rollout_values(rollout):
    """Extract the recorded critic value estimates from a collected rollout.

    Args:
        rollout: dict produced by collect_rollout, containing a 'values' tensor.

    Returns:
        torch.Tensor of shape (n_steps, n_envs) with critic value estimates.
    """
    # TODO: return the tensor of critic value estimates from the rollout
    return rollout['values']

# Step 16 - rollout_log_probs
def rollout_log_probs(rollout):
    """Extract the recorded action log-probabilities from a collected rollout.

    Args:
        rollout: dict returned by collect_rollout.

    Returns:
        torch.Tensor of shape (n_steps, n_envs) under key 'log_probs'.
    """
    # TODO: Return the tensor of recorded action log-probabilities
    return rollout['log_probs']

# Step 17 - compute_gae
def compute_gae(rewards, values, dones, last_values, last_dones, gamma=0.99, lam=0.95):
    """Compute GAE advantages and value targets from a rollout.

    Args:
        rewards: Tensor (T, N) of per-step rewards.
        values: Tensor (T, N) of critic values V(s_t).
        dones: Tensor (T, N) of episode-termination flags.
        last_values: Tensor (N,) bootstrap values after the final step.
        last_dones: Tensor (N,) done flags after the final step.
        gamma: Discount factor (default 0.99).
        lam: GAE lambda (default 0.95).

    Returns:
        advantages: Tensor (T, N).
        returns: Tensor (T, N), equal to advantages + values.
    """
    # TODO: Compute advantage estimates and value targets with GAE
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)

    last_gae_lam = torch.zeros(N, dtype=rewards.dtype, device=rewards.device)

    next_values = last_values
    next_non_terminal = 1.0 - last_dones.to(dtype=rewards.dtype)

    for t in reversed(range(T)):
        delta = rewards[t] + gamma*next_values *next_non_terminal - values[t]

        last_gae_lam = delta + gamma * lam * next_non_terminal*last_gae_lam
        advantages[t] = last_gae_lam

        next_values = values[t]
        next_non_terminal = 1.0- dones[t].to(dtype=rewards.dtype)
    
    returns = advantages + values
    return advantages, returns

# Step 18 - normalize_advantages
def normalize_advantages(advantages, eps=1e-8):
    # TODO: Normalize advantages to zero mean and unit standard deviation...
    mean = advantages.mean()
    std = advantages.std()
    return (advantages - mean)/(std + eps)

# Step 19 - clipped_surrogate_objective
def clipped_surrogate_objective(new_log_probs, old_log_probs, advantages, clip_eps=0.2):
    """Compute the PPO clipped surrogate policy objective from log-probs and advantages."""
    # TODO: Compute the PPO clipped surrogate policy objective from log-probs and advantages.
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio*advantages
    surr2 = torch.clamp(ratio, 1.0-clip_eps,1.0+clip_eps)*advantages
    return -torch.mean(torch.min(surr1,surr2))

# Step 20 - value_loss_and_entropy_bonus
def value_loss_and_entropy_bonus(values_pred, value_targets, entropy, value_coef=0.5, entropy_coef=0.01):
    """Compute value-function loss and entropy bonus for PPO.

    Args:
        values_pred: Predicted state values, shape (batch,).
        value_targets: Target returns, shape (batch,).
        entropy: Per-sample entropy (batch,) or a scalar mean.
        value_coef: Scale on the MSE value loss (default 0.5).
        entropy_coef: Scale on mean entropy (default 0.01).

    Returns:
        (value_loss, entropy_bonus) as scalar torch.Tensors.
    """
    # TODO: Compute the value-function loss and the entropy bonus terms used by PPO.
    mse = torch.mean((values_pred - value_targets)**2)
    value_loss = value_coef*mse

    mean_entropy = entropy.mean() if entropy.dim()>0 else entropy
    entropy_bonus = entropy_coef * mean_entropy
    return value_loss, entropy_bonus

# Step 21 - ppo_loss
def ppo_loss(policy_loss, value_loss, entropy_bonus):
    """Combine clipped surrogate, value loss, and entropy bonus into one PPO loss.

    Args:
        policy_loss: Scalar tensor from the clipped surrogate objective.
        value_loss: Scalar tensor value-function loss.
        entropy_bonus: Scalar tensor entropy bonus term.

    Returns:
        Scalar torch.Tensor total_loss = policy_loss + value_loss - entropy_bonus.
    """
    # TODO: Combine the three terms into one PPO loss
    return policy_loss + value_loss - entropy_bonus

# Step 22 - ppo_update_epoch
import torch
import torch.nn as nn
from torch.distributions import Normal

def ppo_update_epoch(
    actor,
    critic,
    optimizer,
    rollout,
    advantages,
    returns,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    minibatch_size=64,
):
    """Run one PPO update epoch over shuffled minibatches with gradient clipping."""
    # 1. Extract and flatten rollout observations and actions
    obs = rollout['observations'] if 'observations' in rollout else rollout['obs']
    batch_size = obs.shape[0] if obs.dim() == 2 else obs.shape[0] * obs.shape[1]
    
    obs = obs.reshape(batch_size, -1)
    actions = rollout['actions'].reshape(batch_size, -1)
    old_log_probs = rollout['log_probs'].reshape(batch_size).detach()
    
    advantages = advantages.reshape(batch_size).detach()
    returns = returns.reshape(batch_size).detach()
    
    # 2. Normalize advantages once over the full rollout batch
    norm_advantages = normalize_advantages(advantages).reshape(batch_size)
    
    # 3. Generate random permutation of indices
    indices = torch.randperm(batch_size)
    
    epoch_policy_loss = 0.0
    epoch_value_loss = 0.0
    epoch_entropy = 0.0
    epoch_total_loss = 0.0
    num_batches = 0
    
    # 4. Iterate through minibatches
    for start in range(0, batch_size, minibatch_size):
        end = min(start + minibatch_size, batch_size)
        mb_idx = indices[start:end]
        
        mb_obs = obs[mb_idx]
        mb_actions = actions[mb_idx]
        mb_old_lp = old_log_probs[mb_idx]
        mb_adv = norm_advantages[mb_idx]
        mb_ret = returns[mb_idx]
        
        # Forward pass policy
        actor_out = actor(mb_obs)
        if isinstance(actor_out, tuple):
            mean, log_std = actor_out
            std = torch.exp(log_std)
        else:
            mean = actor_out
            std = torch.exp(actor.log_std)
            
        dist = Normal(mean, std)
        new_log_probs = dist.log_prob(mb_actions).sum(dim=-1).view(-1)
        entropy = dist.entropy().sum(dim=-1).view(-1)
        
        # Forward pass critic
        values_pred = critic(mb_obs).view(-1)
        
        # Compute objectives using scaffold helper functions
        policy_loss = clipped_surrogate_objective(
            new_log_probs, mb_old_lp, mb_adv, clip_eps=clip_eps
        )
        value_loss, entropy_bonus = value_loss_and_entropy_bonus(
            values_pred, mb_ret, entropy, value_coef=value_coef, entropy_coef=entropy_coef
        )
        total_loss = ppo_loss(policy_loss, value_loss, entropy_bonus)
        
        # Optimization step
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(actor.parameters()) + list(critic.parameters()),
            max_grad_norm
        )
        optimizer.step()
        
        # Metrics accumulation
        epoch_policy_loss += float(policy_loss.item())
        epoch_value_loss += float(value_loss.item())
        epoch_entropy += float(entropy.mean().item())
        epoch_total_loss += float(total_loss.item())
        num_batches += 1
        
    return {
        'policy_loss': epoch_policy_loss / num_batches,
        'value_loss': epoch_value_loss / num_batches,
        'entropy': epoch_entropy / num_batches,
        'total_loss': epoch_total_loss / num_batches,
    }

# Step 23 - train_ppo
import torch
import numpy as np

def train_ppo(
    actor,
    critic,
    optimizer,
    envs,
    n_iters=10,
    n_steps=128,
    n_epochs=4,
    minibatch_size=64,
    clip_eps=0.2,
    gamma=0.99,
    lam=0.95,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    mass_range=None,
    length_range=None,
    gravity_range=None,
    seed=None,
    device='cpu',
):
    """Run multi-iteration PPO training with optional domain randomization on parallel pendulum envs."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    returns_history = []
    has_dr = (mass_range is not None) or (length_range is not None) or (gravity_range is not None)

    for it in range(n_iters):
        # 1. Resample each environment's physics if DR ranges are supplied
        if has_dr:
            for env in envs:
                cfg = sample_physics_config(
                    mass_range,
                    length_range,
                    gravity_range,
                    np.random,
                )
                set_pendulum_mass(env, cfg['mass'])
                set_pendulum_length(env, cfg['length'])
                set_pendulum_gravity(env, cfg['gravity'])

        # 2. Collect fixed-length rollout
        rollout = collect_rollout(envs, actor, critic, n_steps=n_steps, device=device)

        # 3. Record mean rollout reward across all steps and environments
        rewards = rollout['rewards']
        mean_reward = float(rewards.mean().item())
        returns_history.append(mean_reward)

        # 4. Bootstrap final state values
        with torch.no_grad():
            last_obs = rollout['last_obs']
            last_values = critic(last_obs).squeeze(-1)

        # 5. Compute GAE advantages and returns
        advantages, returns = compute_gae(
            rewards=rollout['rewards'],
            values=rollout['values'],
            dones=rollout['dones'],
            last_values=last_values,
            last_dones=rollout['last_dones'],
            gamma=gamma,
            lam=lam,
        )

        # 6. Apply PPO updates
        for epoch in range(n_epochs):
            ppo_update_epoch(
                actor=actor,
                critic=critic,
                optimizer=optimizer,
                rollout=rollout,
                advantages=advantages,
                returns=returns,
                clip_eps=clip_eps,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                max_grad_norm=max_grad_norm,
                minibatch_size=minibatch_size,
            )

    return {'returns_history': returns_history}

# Step 24 - resample_envs_physics
def resample_envs_physics(envs, mass_range, length_range, gravity_range, rng):
    """Resample physics for every env and return the applied configs.

    Args:
        envs: List of Gymnasium Pendulum-v1 environments.
        mass_range: (min, max) float tuple for pendulum mass.
        length_range: (min, max) float tuple for rod length.
        gravity_range: (min, max) float tuple for gravity.
        rng: numpy.random.Generator used for all sampling.

    Returns:
        List of dicts with keys 'mass', 'length', 'gravity', one per env,
        in the same order as `envs`.
    """
    # TODO: Assign each env a freshly sampled physics configuration...
    applied_config = []

    for env in envs:
        cfg = sample_physics_config(mass_range,length_range,gravity_range,rng)
        set_pendulum_mass(env, cfg['mass'])
        set_pendulum_length(env, cfg['length'])
        set_pendulum_gravity(env, cfg['gravity'])
        applied_config.append(cfg)

    return applied_config

# Step 25 - evaluate_fixed_physics
import gymnasium as gym
import torch

def evaluate_fixed_physics(actor, mass, length, gravity, n_episodes=5, seed=0, max_steps=200):
    """Measure mean episodic return on one fixed Pendulum physics config.

    Args:
        actor: Trained actor network (Gaussian policy).
        mass: Pendulum mass to evaluate under.
        length: Rod length to evaluate under.
        gravity: Gravity to evaluate under.
        n_episodes: Number of evaluation episodes.
        seed: Base seed; episode i uses seed + i.
        max_steps: Max steps per episode before stopping.

    Returns:
        Mean episodic return as a Python float.
    """
    # TODO: Measure mean episodic return on a fixed-physics Pendulum env
    env = gym.make('Pendulum-v1')
    set_pendulum_mass(env, mass)
    set_pendulum_length(env, length)
    set_pendulum_gravity(env, gravity)

    total_reward = []

    try: 
        for i in range(n_episodes):
            obs, _ = env.reset(seed=seed+i)
            episode_reward = 0.0

            for _ in range(max_steps):
                obs_tensor = torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    action,_,_ = sample_action_log_prob_entropy(actor, obs_tensor, deterministic=True)
                action_np = action.squeeze(0).cpu().numpy()

                obs, reward, terminated, truncation , _ = env.step(action_np)
                episode_reward += float(reward)
                if terminated or truncation:
                    break
            total_reward.append(episode_reward)
    finally:
        env.close()
    
    return float(sum(total_reward))/len(total_reward)

