"""Bridge the rsl-rl PPO walk teacher into JOSE's estimator pipeline.

``estimator/pipeline.py`` drives a skrl agent through a skrl vector-env wrapper.
The walk teacher is an rsl-rl policy on a manager-based environment, so this
module supplies two thin shims with the same surface:

* :class:`RslRlTeacherAgent` - a callable deterministic policy that also answers
  ``enable_training_mode`` the way the pipeline expects.
* :class:`JoseRslRlEnvAdapter` - presents ``reset``/``step`` with skrl's return
  signature while keeping the full observation TensorDict internally, so the
  privileged critic group survives and the policy can be re-evaluated on an
  observation whose policy group was overwritten by an estimate.

Neither shim changes the estimator's method, schedule, or ablation axes.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class RslRlTeacherAgent:
    """Deterministic rsl-rl policy with the small skrl-compatible surface."""

    def __init__(self, runner, device: str):
        self.runner = runner
        self.device = device
        self.policy = runner.get_inference_policy(device=device)

    def enable_training_mode(self, mode: bool, apply_to_models: bool = True) -> None:
        """Mirror skrl's agent API; the pipeline only ever disables training."""
        if mode:
            self.runner.alg.train_mode()
        else:
            self.runner.alg.eval_mode()

    def __call__(self, observations: TensorDict) -> torch.Tensor:
        return self.policy(observations)


class JoseRslRlEnvAdapter:
    """skrl-shaped view over :class:`RslRlVecEnvWrapper`.

    The pipeline passes the *policy* observation tensor around and hands it back
    to the teacher, sometimes after overwriting estimator-owned columns. This
    adapter therefore keeps the most recent full observation TensorDict and
    rebuilds it around whatever policy tensor it is given.
    """

    def __init__(self, env: RslRlVecEnvWrapper):
        self.env = env
        self._obs: TensorDict | None = None

    # -- properties the pipeline and adapters reach through ----------------

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    def state(self):
        """skrl's optional privileged-state hook; unused by the rsl-rl policy."""
        return None

    # -- observation plumbing ---------------------------------------------

    def _split(self, obs: TensorDict) -> torch.Tensor:
        self._obs = obs
        return obs["policy"]

    def as_policy_input(self, policy_obs: torch.Tensor) -> TensorDict:
        """Rebuild the full observation around a (possibly edited) policy tensor."""
        if self._obs is None:
            raise RuntimeError("Environment must be reset before the policy is queried")
        obs = self._obs.clone(recurse=False)
        obs["policy"] = policy_obs
        return obs

    # -- skrl-shaped stepping ---------------------------------------------

    def reset(self):
        obs, extras = self.env.reset()
        return self._split(obs), extras

    def step(self, actions: torch.Tensor):
        obs, rewards, _, extras = self.env.step(actions)
        policy_obs = self._split(obs)
        # rsl-rl collapses the two flags into a single `dones`; read the manager
        # directly so genuine falls stay distinguishable from episode timeouts.
        manager = self.unwrapped.termination_manager
        terminated = manager.terminated.view(-1, 1)
        truncated = manager.time_outs.view(-1, 1)
        return policy_obs, rewards.view(-1, 1), terminated, truncated, extras

    def close(self):
        self.env.close()


def build_teacher(
    task: str, agent_cfg, env, checkpoint: str | None, device: str
) -> tuple[JoseRslRlEnvAdapter, RslRlTeacherAgent]:
    """Wrap ``env`` for rsl-rl, restore ``checkpoint``, and return both shims.

    Args:
        task: Gym task id, used only for error messages.
        agent_cfg: An ``RslRlOnPolicyRunnerCfg`` already passed through
            ``handle_deprecated_rsl_rl_cfg``.
        env: The ``gym.make`` result for a ``G1WalkEstimatorEnv`` task.
        checkpoint: Path to the ``model_*.pt`` file to load, or ``None`` to leave
            the policy untrained (used by action-log replay).
        device: Torch device string.
    """
    from rsl_rl.runners import OnPolicyRunner

    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"Task {task!r} uses unsupported runner class {agent_cfg.class_name!r}")

    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    if checkpoint is not None:
        runner.load(checkpoint)
    return JoseRslRlEnvAdapter(wrapped), RslRlTeacherAgent(runner, device)
