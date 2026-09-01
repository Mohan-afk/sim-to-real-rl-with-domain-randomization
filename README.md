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
- [ ] **4.** sample_physics_config
- [ ] **5.** build_parallel_pendulum_envs
- [ ] **6.** shape_upright_hold_reward
- [ ] **7.** build_actor_network
- [ ] **8.** build_critic_network
- [ ] **9.** sample_action_log_prob_entropy
- [ ] **10.** collect_rollout
- [ ] **11.** rollout_observations
- [ ] **12.** rollout_actions
- [ ] **13.** rollout_rewards
- [ ] **14.** rollout_dones
- [ ] **15.** rollout_values
- [ ] **16.** rollout_log_probs
- [ ] **17.** compute_gae
- [ ] **18.** normalize_advantages
- [ ] **19.** clipped_surrogate_objective
- [ ] **20.** value_loss_and_entropy_bonus
- [ ] **21.** ppo_loss
- [ ] **22.** ppo_update_epoch
- [ ] **23.** train_ppo
- [ ] **24.** resample_envs_physics
- [ ] **25.** evaluate_fixed_physics
- [ ] **26.** measure_generalization_gap
- [ ] **27.** sweep_physics_parameter
- [ ] **28.** compare_dr_vs_fixed_policy

---

Built on Deep-ML.
