"""SKRL 2.x version/config/checkpoint compatibility helpers."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from types import MethodType
from typing import Mapping, MutableMapping

from packaging.version import Version


MIN_SKRL = Version("2.0.0")
MAX_SKRL = Version("3.0.0")

OBSOLETE_KEYS = {
    "amp_state_preprocessor": "amp_observation_preprocessor",
    "amp_state_preprocessor_kwargs": "amp_observation_preprocessor_kwargs",
    "task_reward_weight": "task_reward_scale",
    "style_reward_weight": "style_reward_scale",
    "discriminator_reward_scale": "style_reward_scale (fold the old discriminator scale into this value)",
    "lambda": "gae_lambda",
    "clip_predicted_values": "remove it; value_clip controls value clipping in SKRL 2.x",
    "rewards_shaper_scale": "remove it or provide the SKRL 2.x rewards_shaper callable",
}


def installed_skrl_version() -> Version:
    try:
        return Version(version("skrl"))
    except PackageNotFoundError as exc:
        raise RuntimeError("skrl is not installed; install 'skrl>=2.0,<3.0'") from exc


def require_skrl_2() -> str:
    current = installed_skrl_version()
    if not MIN_SKRL <= current < MAX_SKRL:
        raise RuntimeError(
            f"Unsupported skrl version {current}. JOSE G1 supports skrl>=2.0,<3.0; "
            "SKRL 1.x checkpoints are not supported by the SOLO-aligned pipeline."
        )
    return str(current)


def validate_skrl_config(config: Mapping) -> None:
    """Fail before simulator startup when a legacy agent key is present."""
    agent = config.get("agent", config)
    found = {key: replacement for key, replacement in OBSOLETE_KEYS.items() if key in agent}
    if found:
        hints = "; ".join(f"{key!r} -> {replacement}" for key, replacement in found.items())
        raise ValueError(f"Obsolete SKRL 1.x configuration key(s): {hints}")
    if agent.get("class") == "AMP":
        for required in ("observation_preprocessor", "amp_observation_preprocessor", "task_reward_scale", "style_reward_scale"):
            if required not in agent:
                raise ValueError(f"SKRL 2.x AMP config is missing {required!r}")


def prepare_runner_config(config: MutableMapping) -> MutableMapping:
    require_skrl_2()
    validate_skrl_config(config)
    return config


def disable_velocity_termination_for_evaluation(env_cfg) -> float | None:
    """Disable only the optional low-forward-velocity termination.

    Evaluation should still report genuine falls through the height termination,
    while a policy that pauses or moves backwards must be allowed to continue.
    Return the configured threshold so callers can report what was overridden.
    """
    if not hasattr(env_cfg, "vel_window_min_vx"):
        return None
    threshold = float(env_cfg.vel_window_min_vx)
    env_cfg.vel_window_min_vx = 0.0
    return threshold


def force_skrl_isaaclab_reset(env) -> None:
    """Re-enable physical reset on SKRL's reset-once Isaac Lab wrapper."""
    current = env
    for _ in range(32):
        if hasattr(current, "_reset_once"):
            current._reset_once = True
        next_env = getattr(current, "_env", None)
        if next_env is None or next_env is current:
            break
        current = next_env


def deterministic_action(agent, observations, states=None):
    """Use only the public SKRL 2.x inference API."""
    outputs = agent.act(observations, states, timestep=0, timesteps=0)
    return outputs[-1].get("mean_actions", outputs[0])


def scaled_reward(raw_task, raw_style, task_scale: float = 0.0, style_scale: float = 2.0):
    """Return scaled components and their effective sum for logging/tests."""
    task = raw_task * task_scale
    style = raw_style * style_scale
    return task, style, task + style


def amp_reward_components(agent, amp_observations, raw_task):
    """Compute the SKRL 2.x AMP reward decomposition from a loaded agent.

    The agent remains responsible for checkpoint loading. This helper only uses
    the live AMP model/preprocessor so no serialized module key is assumed.
    """
    if not hasattr(agent, "discriminator") or not hasattr(agent, "_amp_observation_preprocessor"):
        return None
    import torch

    with torch.no_grad():
        logits, _ = agent.discriminator.act(
            {"observations": agent._amp_observation_preprocessor(amp_observations)}, role="discriminator"
        )
        raw_style = -torch.log(
            torch.maximum(1.0 - torch.sigmoid(logits), torch.tensor(1.0e-4, device=logits.device))
        ).view(raw_task.shape)
    task_scale = float(agent.cfg.task_reward_scale)
    style_scale = float(agent.cfg.style_reward_scale)
    scaled_task, scaled_style, total = scaled_reward(raw_task, raw_style, task_scale, style_scale)
    return {
        "raw_task": raw_task,
        "raw_style": raw_style,
        "scaled_task": scaled_task,
        "scaled_style": scaled_style,
        "effective_reward": total,
        "task_reward_scale": task_scale,
        "style_reward_scale": style_scale,
    }


def install_amp_reward_tracking(agent) -> bool:
    """Track AMP's effective reward instead of the raw environment reward.

    SKRL 2.x computes the discriminator reward only in ``AMP.update``. Its base
    ``Agent.record_transition`` therefore accumulates the environment reward,
    which omits the AMP discriminator component, for TensorBoard and best-checkpoint
    selection. This instance-level adapter keeps the original task reward in the
    rollout memory, but feeds the effective task + style reward to SKRL's
    statistics accumulator.

    Returns ``True`` when the adapter was installed and ``False`` for non-AMP
    agents. Calling the function more than once is safe.
    """
    if getattr(agent, "_jose_amp_reward_tracking", False):
        return True
    if not hasattr(agent, "discriminator") or not hasattr(agent, "_amp_observation_preprocessor"):
        return False

    try:
        from skrl.agents.torch import Agent as TorchAgent
    except ImportError:
        return False

    original_record_transition = agent.record_transition

    def record_transition_with_amp_reward(self, **transition):
        raw_reward = transition["rewards"]
        components = amp_reward_components(self, transition["infos"]["amp_obs"], raw_reward)
        effective_reward = raw_reward if components is None else components["effective_reward"]

        # AMP.record_transition must see the raw task reward because AMP.update
        # applies task/style scales itself. Suppress only the base-class reward
        # tracker during that call, then update it once with the effective reward.
        write_interval = self.write_interval
        self.write_interval = 0
        try:
            original_record_transition(**transition)
        finally:
            self.write_interval = write_interval

        if write_interval > 0:
            tracked_transition = dict(transition)
            tracked_transition["rewards"] = effective_reward
            TorchAgent.record_transition(self, **tracked_transition)

            if components is not None:
                self.track_data("Reward / AMP task reward (mean)", components["scaled_task"].mean().item())
                self.track_data("Reward / AMP style reward (mean)", components["scaled_style"].mean().item())
                self.track_data("Reward / AMP effective reward (mean)", effective_reward.mean().item())

    agent.record_transition = MethodType(record_transition_with_amp_reward, agent)
    agent._jose_amp_reward_tracking = True
    return True
