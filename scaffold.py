"""
Sim-to-Real RL with Domain Randomization scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Sim-to-Real RL with Domain Randomization: PPO on randomized Pendulum envs."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    mass_range = (0.8, 1.2)
    length_range = (0.8, 1.2)
    gravity_range = (8.0, 12.0)
    n_envs, n_steps, hidden = 4, 64, 32
    device = "cpu"

    # --- physics sampling & parallel envs ---
    rng = np.random.default_rng(0)
    cfg = sample_physics_config(mass_range, length_range, gravity_range, rng)
    print("sample_physics_config:", tuple(round(float(cfg[k]), 3) for k in ("mass", "length", "gravity")))

    envs, env_cfgs = build_parallel_pendulum_envs(
        n_envs, mass_range, length_range, gravity_range, seed=0
    )
    print("n_parallel_envs:", len(envs))

    obs_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.shape[0]

    # --- actor-critic ---
    actor = build_actor_network(obs_dim, action_dim, hidden_dim=hidden)
    critic = build_critic_network(obs_dim, hidden_dim=hidden)
    opt = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=3e-4
    )

    obs0 = torch.zeros(1, obs_dim)
    action, log_p, ent = sample_action_log_prob_entropy(actor, obs0, deterministic=True)
    print("action_shape:", tuple(action.shape), "log_prob_dim:", log_p.dim())

    # --- one randomized rollout + GAE ---
    resample_envs_physics(envs, mass_range, length_range, gravity_range, rng)
    rollout = collect_rollout(envs, actor, critic, n_steps, device=device)

    obs_t = rollout_observations(rollout)
    act_t = rollout_actions(rollout)
    rew_t = rollout_rewards(rollout)
    done_t = rollout_dones(rollout)
    val_t = rollout_values(rollout)
    lp_t = rollout_log_probs(rollout)
    print("rollout_obs:", tuple(obs_t.shape), "rewards_mean:", round(float(rew_t.mean()), 4))

    with torch.no_grad():
        last_obs = obs_t[-1]
        last_values = critic(last_obs).squeeze(-1)
        last_dones = done_t[-1]
    advantages, returns = compute_gae(
        rew_t, val_t, done_t, last_values, last_dones, gamma=0.99, lam=0.95
    )
    advantages = normalize_advantages(advantages)
    print("adv_mean:", round(float(advantages.mean()), 4), "returns_std:", round(float(returns.std()), 4))

    # --- single PPO epoch ---
    update_rollout = {"observations": obs_t, "actions": act_t, "log_probs": lp_t}
    loss_val = ppo_update_epoch(
        actor, critic, opt, update_rollout, advantages, returns,
        clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
        max_grad_norm=0.5, minibatch_size=32,
    )
    print("ppo_epoch_loss:", round(float(loss_val["total_loss"]), 4))

    # --- short DR training ---
    actor_tr = build_actor_network(obs_dim, action_dim, hidden_dim=hidden)
    critic_tr = build_critic_network(obs_dim, hidden_dim=hidden)
    opt_tr = torch.optim.Adam(
        list(actor_tr.parameters()) + list(critic_tr.parameters()), lr=3e-4
    )
    train_ppo(
        actor_tr, critic_tr, opt_tr, envs,
        n_iters=3, n_steps=n_steps, n_epochs=2, minibatch_size=32,
        gamma=0.99, lam=0.95, clip_eps=0.2,
        value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
        mass_range=mass_range, length_range=length_range,
        gravity_range=gravity_range, seed=0,
    )

    # --- fixed-physics baseline (brief) ---
    envs_fixed, _ = build_parallel_pendulum_envs(
        n_envs, (1.0, 1.0), (1.0, 1.0), (10.0, 10.0), seed=1
    )
    actor_fx = build_actor_network(obs_dim, action_dim, hidden_dim=hidden)
    critic_fx = build_critic_network(obs_dim, hidden_dim=hidden)
    opt_fx = torch.optim.Adam(
        list(actor_fx.parameters()) + list(critic_fx.parameters()), lr=3e-4
    )
    train_ppo(
        actor_fx, critic_fx, opt_fx, envs_fixed,
        n_iters=3, n_steps=n_steps, n_epochs=2, minibatch_size=32,
        mass_range=None, length_range=None, gravity_range=None, seed=1,
    )

    # --- evaluation & generalization ---
    ret_fixed = evaluate_fixed_physics(
        actor_tr, mass=1.0, length=1.0, gravity=10.0, n_episodes=2, seed=0
    )
    print("eval_nominal_return:", round(float(ret_fixed), 3))

    train_ranges = {"mass": mass_range, "length": length_range, "gravity": gravity_range}
    heldout_ranges = {"mass": (1.4, 1.6), "length": (1.4, 1.6), "gravity": (13.0, 15.0)}
    gap = measure_generalization_gap(
        actor_tr, train_ranges, heldout_ranges, n_configs=2, n_episodes=1, seed=0
    )
    print("generalization_gap:", {k: round(float(gap[k]), 3)
          for k in ("in_dist_return", "heldout_return", "gap")})

    sweep = sweep_physics_parameter(
        actor_tr, "mass", [0.7, 1.0, 1.5],
        base_mass=1.0, base_length=1.0, base_gravity=10.0, n_episodes=1, seed=0,
    )
    print("mass_sweep:", [(r["param_value"], round(float(r["mean_return"]), 3)) for r in sweep])

    heldout_configs = [
        {"mass": 1.5, "length": 1.5, "gravity": 14.0},
        {"mass": 0.6, "length": 0.6, "gravity": 7.0},
    ]
    cmp = compare_dr_vs_fixed_policy(
        actor_tr, actor_fx, heldout_configs, n_episodes=1, seed=0
    )
    print("dr_vs_fixed:", {k: round(float(cmp[k]), 3)
          for k in ("dr_mean", "fixed_mean", "dr_advantage")})

    for e in list(envs) + list(envs_fixed):
        e.close()


if __name__ == "__main__":
    main()
