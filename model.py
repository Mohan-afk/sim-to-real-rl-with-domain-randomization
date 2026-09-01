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

# Step 3 - set_pendulum_gravity (not yet solved)
# TODO: implement

# Step 4 - sample_physics_config (not yet solved)
# TODO: implement

# Step 5 - build_parallel_pendulum_envs (not yet solved)
# TODO: implement

# Step 6 - shape_upright_hold_reward (not yet solved)
# TODO: implement

# Step 7 - build_actor_network (not yet solved)
# TODO: implement

# Step 8 - build_critic_network (not yet solved)
# TODO: implement

# Step 9 - sample_action_log_prob_entropy (not yet solved)
# TODO: implement

# Step 10 - collect_rollout (not yet solved)
# TODO: implement

# Step 11 - rollout_observations (not yet solved)
# TODO: implement

# Step 12 - rollout_actions (not yet solved)
# TODO: implement

# Step 13 - rollout_rewards (not yet solved)
# TODO: implement

# Step 14 - rollout_dones (not yet solved)
# TODO: implement

# Step 15 - rollout_values (not yet solved)
# TODO: implement

# Step 16 - rollout_log_probs (not yet solved)
# TODO: implement

# Step 17 - compute_gae (not yet solved)
# TODO: implement

# Step 18 - normalize_advantages (not yet solved)
# TODO: implement

# Step 19 - clipped_surrogate_objective (not yet solved)
# TODO: implement

# Step 20 - value_loss_and_entropy_bonus (not yet solved)
# TODO: implement

# Step 21 - ppo_loss (not yet solved)
# TODO: implement

# Step 22 - ppo_update_epoch (not yet solved)
# TODO: implement

# Step 23 - train_ppo (not yet solved)
# TODO: implement

# Step 24 - resample_envs_physics (not yet solved)
# TODO: implement

# Step 25 - evaluate_fixed_physics (not yet solved)
# TODO: implement

# Step 26 - measure_generalization_gap (not yet solved)
# TODO: implement

# Step 27 - sweep_physics_parameter (not yet solved)
# TODO: implement

# Step 28 - compare_dr_vs_fixed_policy (not yet solved)
# TODO: implement

