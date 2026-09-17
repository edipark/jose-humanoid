"""Build the teacher agent and environment for whichever RL stack a task uses.

JOSE's AMP and Direct-PPO teachers are SKRL agents driven through
``SkrlVecEnvWrapper``. The manager-based PPO walk teacher is an rsl-rl policy.
The estimator, DAgger and ablation code is written against one interface, so this
module hides the difference behind a single constructor and leaves everything
downstream, including the estimator methodology, untouched.
"""

from __future__ import annotations

from pathlib import Path


#: Adapters whose teacher is trained with rsl-rl rather than SKRL.
RSL_RL_ADAPTERS = frozenset({"ppo_walk"})

#: Hydra entry point holding the runner config for each stack.
SKRL_AGENT_ENTRY_POINT = "skrl_amp_cfg_entry_point"
RSL_RL_AGENT_ENTRY_POINT = "rsl_rl_cfg_entry_point"


def uses_rsl_rl(adapter: str) -> bool:
    """Whether ``adapter`` names an rsl-rl teacher."""
    return adapter.lower() in RSL_RL_ADAPTERS


def resolve_agent_entry_point(adapter: str, requested: str | None) -> str:
    """Pick the config entry point, defaulting to the one the adapter needs.

    Keeps ``--adapter ppo_walk`` usable without also restating
    ``--agent_cfg_entry_point rsl_rl_cfg_entry_point`` on every command.
    """
    if uses_rsl_rl(adapter):
        if requested in (None, SKRL_AGENT_ENTRY_POINT, "skrl_cfg_entry_point"):
            return RSL_RL_AGENT_ENTRY_POINT
        return requested
    return requested or SKRL_AGENT_ENTRY_POINT


def build_env_and_teacher(
    task: str,
    adapter: str,
    env_cfg,
    agent_cfg,
    checkpoint: str | Path | None,
    device: str,
    seed: int | None = None,
    render_mode: str | None = None,
    wrap_raw=None,
):
    """Create the environment and load the frozen teacher policy.

    Args:
        checkpoint: Teacher weights to restore. ``None`` builds the environment and
            an untrained agent, which is what action-log replay needs.
        wrap_raw: Optional callable applied to the raw ``gym`` environment before
            the RL-library wrapper, e.g. to attach ``gym.wrappers.RecordVideo``.

    Returns:
        ``(env, teacher_agent)`` where ``env`` exposes SKRL's
        ``reset``/``step`` signature and ``teacher_agent`` accepts the
        ``enable_training_mode`` call the estimator pipeline makes.
    """
    import gymnasium as gym

    checkpoint = str(Path(checkpoint).resolve()) if checkpoint is not None else None

    def _make():
        raw_env = gym.make(task, cfg=env_cfg, render_mode=render_mode)
        return wrap_raw(raw_env) if wrap_raw is not None else raw_env

    if uses_rsl_rl(adapter):
        import importlib.metadata as metadata

        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

        from jose.ppo_walk.rsl_rl_teacher import build_teacher

        if seed is not None:
            agent_cfg.seed = seed
        agent_cfg.device = device
        # The ported runner config uses the pre-4.0 `policy` form; translate it.
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        return build_teacher(task, agent_cfg, _make(), checkpoint, device)

    from skrl.utils.runner.torch import Runner

    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    from jose.skrl_compat import prepare_runner_config

    prepare_runner_config(agent_cfg)
    if seed is not None:
        agent_cfg["seed"] = seed
    env = SkrlVecEnvWrapper(_make(), ml_framework="torch")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, agent_cfg)
    if checkpoint is not None:
        runner.agent.load(checkpoint)
    return env, runner.agent


def teacher_policy_module(teacher_agent):
    """The teacher's policy network, for parameter counting. ``None`` if absent."""
    policy = getattr(teacher_agent, "policy", None)
    if policy is None and hasattr(teacher_agent, "models"):
        policy = teacher_agent.models.get("policy")
    if policy is None:
        # rsl-rl keeps the actor on the algorithm object.
        alg = getattr(getattr(teacher_agent, "runner", None), "alg", None)
        policy = getattr(alg, "actor", None) or getattr(alg, "policy", None)
    return policy
