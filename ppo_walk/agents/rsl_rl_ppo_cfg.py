"""rsl-rl PPO runner configuration for the G1 29-DOF walk task.

Ported from unitree_rl_lab (Apache License 2.0),
https://github.com/unitreerobotics/unitree_rl_lab

Every hyperparameter is carried over unchanged. The single addition is
``obs_groups``, which rsl-rl >= 4.0 requires: it is the explicit form of the
mapping the old ``RslRlVecEnvWrapper`` performed implicitly (the ``policy``
observation group feeds the actor, the privileged ``critic`` group feeds the
critic). Without it rsl-rl falls back to ``critic: ["policy"]`` and the
privileged critic observations would be silently dropped.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1WalkPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1WalkEstimatorPPORunnerCfg(G1WalkPPORunnerCfg):
    """Identical training recipe, used for the estimator-compatible task variant."""
