# Sim-to-Real RL with Domain Randomization

Build an Isaac-style PPO pipeline on parallel Pendulum environments whose mass, length, and gravity are resampled every rollout. Train a robust actor-critic policy with GAE and clipped surrogates, then quantify the generalization gap and failure boundaries against fixed-physics baselines.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** set_pendulum_mass
- [x] **2.** set_pendulum_length
- [x] **3.** set_pendulum_gravity
- [x] **4.** sample_physics_config
- [x] **5.** build_parallel_pendulum_envs
- [x] **6.** shape_upright_hold_reward
- [x] **7.** build_actor_network
- [x] **8.** build_critic_network
- [x] **9.** sample_action_log_prob_entropy
- [x] **10.** collect_rollout
- [x] **11.** rollout_observations
- [x] **12.** rollout_actions
- [x] **13.** rollout_rewards
- [x] **14.** rollout_dones
- [x] **15.** rollout_values
- [x] **16.** rollout_log_probs
- [x] **17.** compute_gae
- [x] **18.** normalize_advantages
- [x] **19.** clipped_surrogate_objective
- [x] **20.** value_loss_and_entropy_bonus
- [x] **21.** ppo_loss
- [x] **22.** ppo_update_epoch
- [x] **23.** train_ppo
- [x] **24.** resample_envs_physics
- [x] **25.** evaluate_fixed_physics
- [x] **26.** measure_generalization_gap
- [x] **27.** sweep_physics_parameter
- [x] **28.** compare_dr_vs_fixed_policy

---

Built on Deep-ML.
