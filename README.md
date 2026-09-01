# sim-to-real-rl-with-domain-randomization
Build an Isaac-style PPO pipeline on parallel Pendulum environments whose mass, length, and gravity are resampled every rollout. Train a robust actor-critic policy with GAE and clipped surrogates, then quantify the generalization gap and failure boundaries against fixed-physics baselines.
