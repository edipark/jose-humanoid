"""Collection, supervised training, DAgger, evaluation, and checkpoint I/O."""

from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import numpy as np
from torch import nn

from jose.distillation.command_eval import reset_ids

from ..schema import JOINT_PRESETS, SCHEMA_VERSION
from ..skrl_compat import force_skrl_isaaclab_reset, require_skrl_2
from .adapters import PolicyAdapter
from .metrics import MetricAccumulator, step_metrics
from .models import NormalizedEstimator, build_estimator


@contextmanager
def frozen_rng():
    """Run a block without advancing the random stream.

    skrl's Gaussian policy samples from its distribution on every ``act()`` call
    and discards the sample when the caller wants the mean, so evaluating the
    policy one extra time purely to *measure* something would still shift every
    later draw -- changing the very trajectory being measured, and every DAgger
    round after it. Metric-only work goes inside this so adding a metric can
    never alter a result.
    """
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@dataclass
class RolloutDataset:
    histories: torch.Tensor
    targets: torch.Tensor
    frames: torch.Tensor
    teacher_actions: torch.Tensor

    def append(self, other: "RolloutDataset", max_size: int | None = None) -> "RolloutDataset":
        self_size = len(self.histories)
        other_size = len(other.histories)
        total_size = self_size + other_size
        if max_size is None or total_size <= max_size:
            return RolloutDataset(
                *(torch.cat((getattr(self, field), getattr(other, field))) for field in self.__dataclass_fields__)
            )

        # Do not concatenate the complete datasets before truncating. For a
        # 500k-sample w100 history, the old batch, new batch, concatenation and
        # indexed result would otherwise coexist and exceed 64 GB of RAM.
        device = self.histories.device
        selected = torch.randperm(total_size, device=device)[:max_size]
        values = []
        copy_chunk = 4096
        for field in self.__dataclass_fields__:
            first = getattr(self, field)
            second = getattr(other, field)
            if first.shape[1:] != second.shape[1:]:
                raise ValueError(f"Cannot append incompatible dataset field {field}")
            output = torch.empty((max_size, *first.shape[1:]), dtype=first.dtype, device=first.device)
            for start in range(0, max_size, copy_chunk):
                stop = min(start + copy_chunk, max_size)
                source = selected[start:stop]
                from_first = source < self_size
                if from_first.any():
                    destination = torch.arange(start, stop, device=device)[from_first]
                    output.index_copy_(0, destination, first.index_select(0, source[from_first]))
                if (~from_first).any():
                    destination = torch.arange(start, stop, device=device)[~from_first]
                    output.index_copy_(0, destination, second.index_select(0, source[~from_first] - self_size))
            values.append(output)
        return RolloutDataset(*values)

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.pid{os.getpid()}.tmp")
        try:
            torch.save({"dataset": self.__dict__, "metadata": metadata or {}}, temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: str | Path) -> "RolloutDataset":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls(**payload["dataset"])

    @classmethod
    def load_with_metadata(cls, path: str | Path) -> tuple["RolloutDataset", dict]:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls(**payload["dataset"]), payload.get("metadata", {})

    def project_joint_history(
        self,
        joint_ids: tuple[int, ...] | list[int],
        source_window: int,
        target_window: int,
        full_joint_count: int,
    ) -> "RolloutDataset":
        """Derive a shorter/subset estimator dataset without recollection.

        The source is expected to contain all joint positions followed by all
        joint velocities. Taking the newest history suffix is exactly
        equivalent to collecting with a shorter history buffer.
        """
        if target_window > source_window:
            raise ValueError(f"Cannot derive window {target_window} from cached window {source_window}")
        joint_ids = tuple(joint_ids)
        all_joints = joint_ids == tuple(range(full_joint_count))
        if target_window == source_window and all_joints:
            return self
        ids = list(joint_ids) + [full_joint_count + index for index in joint_ids]
        histories = self.histories[:, -target_window:]
        if not all_joints:
            histories = histories[:, :, ids]
            frames = self.frames[:, ids]
        else:
            # Clone a shorter suffix so deleting the source releases the much
            # larger backing storage of the longest-window cache.
            histories = histories.clone()
            frames = self.frames
        return RolloutDataset(
            histories=histories,
            targets=self.targets,
            frames=frames,
            teacher_actions=self.teacher_actions,
        )


class HistoryBuffer:
    def __init__(self, num_envs: int, window: int, input_dim: int, device: torch.device | str):
        self.values = torch.zeros((num_envs, window, input_dim), device=device)

    def push(self, frame: torch.Tensor) -> torch.Tensor:
        self.values = torch.roll(self.values, -1, dims=1)
        self.values[:, -1] = frame
        return self.values

    def reset(self, done: torch.Tensor) -> None:
        self.values[done] = 0.0


@torch.no_grad()
def collect_rollout(
    env,
    adapter: PolicyAdapter,
    teacher_agent,
    steps: int,
    window: int = 50,
    estimator: NormalizedEstimator | None = None,
    estimator_ratio: float = 0.0,
    action_noise: float = 0.0,
    max_samples: int | None = None,
) -> tuple[RolloutDataset, dict]:
    force_skrl_isaaclab_reset(env)
    observations, _ = env.reset()
    history = HistoryBuffer(observations.shape[0], window, adapter.input_dim, observations.device)
    histories, targets, frames, teacher_actions = [], [], [], []
    deaths = timeouts = 0
    returns = torch.zeros(observations.shape[0], device=observations.device)
    lengths = torch.zeros(observations.shape[0], device=observations.device)
    completed_returns: list[float] = []
    completed_lengths: list[float] = []
    metrics = MetricAccumulator()
    previous_action = torch.zeros((observations.shape[0], adapter.schema.action_dim), device=observations.device)

    if estimator is not None:
        estimator.eval()
    teacher_agent.enable_training_mode(False, apply_to_models=True)
    samples_per_step = observations.shape[0]
    if max_samples is not None:
        samples_per_step = max(1, min(observations.shape[0], max_samples // max(steps, 1)))
    for _ in range(steps):
        frame = adapter.estimator_input()
        target = adapter.estimator_target()
        sequence = history.push(frame)
        teacher_action = adapter.action(teacher_agent, observations)
        action = teacher_action
        estimator_action = None
        if estimator is not None and estimator_ratio > 0.0:
            estimate = estimator.predict(frame if estimator.__class__.__name__.startswith("MLP") else sequence)
            use_estimator = torch.rand(observations.shape[0], device=observations.device) < estimator_ratio
            estimated_obs = adapter.inject_estimate(observations, estimate)
            estimator_action = adapter.action(teacher_agent, estimated_obs)
            action = torch.where(use_estimator[:, None], estimator_action, teacher_action)
        if action_noise:
            action = action + action_noise * torch.randn_like(action)
        if samples_per_step < observations.shape[0]:
            sample_ids = torch.randperm(observations.shape[0], device=observations.device)[:samples_per_step]
        else:
            sample_ids = slice(None)
        histories.append(sequence[sample_ids].cpu().clone())
        targets.append(target[sample_ids].cpu().clone())
        frames.append(frame[sample_ids].cpu().clone())
        teacher_actions.append(teacher_action[sample_ids].cpu().clone())
        observations, rewards, terminated, truncated, _ = env.step(action)
        returns += rewards.flatten()
        lengths += 1
        metrics.add(
            step_metrics(
                adapter.core_env, adapter, teacher_agent,
                action=action, previous_action=previous_action, rewards=rewards,
                policy_action=estimator_action, teacher_action=teacher_action,
            )
        )
        previous_action = action
        done = (terminated | truncated).flatten()
        deaths += int(terminated.sum())
        timeouts += int((truncated & ~terminated).sum())
        if done.any():
            completed_returns.extend(returns[done].cpu().tolist())
            completed_lengths.extend(lengths[done].cpu().tolist())
            returns[done] = 0.0
            lengths[done] = 0.0
            history.reset(done)
            previous_action[done] = 0.0
            # Re-draw the IMU fault for the environments that just reset. The
            # model in distillation/imu.py holds a per-episode gyro bias and a
            # latency ring; carrying them across a reset would hand the estimator
            # one fixed offset for the whole run instead of a fresh one each
            # episode, which is a far easier problem than deployment.
            # `getattr` because only the randomized JOSE+IMU arm carries one.
            imu_corruptor = getattr(adapter, "imu_corruptor", None)
            if imu_corruptor is not None:
                imu_corruptor.reset(reset_ids(done))

    dataset = RolloutDataset(*(torch.cat(items) for items in (histories, targets, frames, teacher_actions)))
    stats = {
        "samples": len(dataset.targets),
        "deaths": deaths,
        "timeouts": timeouts,
        "death_rate": 100.0 * deaths / (deaths + timeouts) if deaths + timeouts else 0.0,
        "timeout_rate": 100.0 * timeouts / (deaths + timeouts) if deaths + timeouts else 0.0,
        "return_mean": sum(completed_returns) / len(completed_returns) if completed_returns else 0.0,
        "episode_length_mean": sum(completed_lengths) / len(completed_lengths) if completed_lengths else 0.0,
        "episode_length_std": float(np.std(completed_lengths)) if completed_lengths else 0.0,
        "success_rate": 100.0 * timeouts / (deaths + timeouts) if deaths + timeouts else 0.0,
        "velocity_source": "sim_joint_velocity",
        **metrics.mean(),
    }
    return dataset, stats


@torch.no_grad()
def evaluate_estimator_closed_loop(
    env,
    adapter: PolicyAdapter,
    teacher_agent,
    estimator: NormalizedEstimator,
    estimator_type: str,
    window: int,
    episodes: int = 200,
    max_episode_steps: int = 1000,
    seed: int | None = None,
) -> dict:
    """Evaluate estimator-injected teacher observations over completed episodes."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    force_skrl_isaaclab_reset(env)
    observations, _ = env.reset()
    history = HistoryBuffer(observations.shape[0], window, adapter.input_dim, observations.device)
    returns = torch.zeros(observations.shape[0], device=observations.device)
    lengths = torch.zeros_like(returns)
    completed_returns: list[float] = []
    completed_lengths: list[float] = []
    deaths = timeouts = 0
    squared_error = torch.zeros(adapter.schema.estimator_target_dim, dtype=torch.float64)
    sample_count = 0
    estimator.eval()
    teacher_agent.enable_training_mode(False, apply_to_models=True)
    # Same fidelity metrics every other method reports, so the estimator is not
    # a row of blanks in the comparison table. Every added call sits inside
    # frozen_rng(), which is what keeps the rollout -- and therefore every later
    # DAgger round -- bit-identical to runs made before these metrics existed.
    metrics = MetricAccumulator()
    previous_action = torch.zeros((observations.shape[0], adapter.schema.action_dim), device=observations.device)
    max_steps = max_episode_steps * max(1, (episodes + observations.shape[0] - 1) // observations.shape[0] + 1)
    for _ in range(max_steps):
        frame = adapter.estimator_input()
        target = adapter.estimator_target()
        sequence = history.push(frame)
        model_input = frame if estimator_type.upper() == "MLP" else sequence
        estimate = estimator.predict(model_input)
        squared_error += (estimate - target).double().square().sum(dim=0).cpu()
        sample_count += target.shape[0]
        action = adapter.action(teacher_agent, adapter.inject_estimate(observations, estimate))
        driving_observations = observations
        observations, rewards, terminated, truncated, _ = env.step(action)
        returns += rewards.flatten()
        lengths += 1
        with frozen_rng():
            # The same teacher, driven by the true state instead of the
            # estimate: the gap between the two is exactly this method's
            # imitation error, and is the quantity collect_rollout already
            # reports for the students.
            teacher_action = adapter.action(teacher_agent, driving_observations)
            metrics.add(
                step_metrics(
                    adapter.core_env, adapter, teacher_agent,
                    action=action, previous_action=previous_action, rewards=rewards,
                    policy_action=action, teacher_action=teacher_action,
                )
            )
        previous_action = action
        done = (terminated | truncated).flatten()
        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            remaining = episodes - len(completed_lengths)
            selected = done_ids[:remaining]
            completed_returns.extend(returns[selected].cpu().tolist())
            completed_lengths.extend(lengths[selected].cpu().tolist())
            deaths += int(terminated[selected].sum())
            timeouts += int((truncated[selected] & ~terminated[selected]).sum())
            returns[done_ids] = 0.0
            lengths[done_ids] = 0.0
            history.reset(done)
            previous_action[done] = 0.0
            imu_corruptor = getattr(adapter, "imu_corruptor", None)
            if imu_corruptor is not None:
                imu_corruptor.reset(reset_ids(done))
            if len(completed_lengths) >= episodes:
                break
    if not completed_lengths:
        raise RuntimeError("Closed-loop evaluation completed no episodes")
    target_rmse = (squared_error / max(sample_count, 1)).sqrt().float()
    return {
        "episodes": len(completed_lengths),
        "episode_length_mean": sum(completed_lengths) / len(completed_lengths),
        "episode_length_std": float(np.std(completed_lengths)),
        "return_mean": sum(completed_returns) / len(completed_returns),
        "deaths": deaths,
        "timeouts": timeouts,
        "death_rate": 100.0 * deaths / len(completed_lengths),
        "timeout_rate": 100.0 * timeouts / len(completed_lengths),
        "success_rate": 100.0 * timeouts / len(completed_lengths),
        "rmse": float(target_rmse.square().mean().sqrt()),
        "target_rmse": target_rmse.tolist(),
        **metrics.mean(),
    }


def uses_locomotion_eval(adapter: PolicyAdapter) -> bool:
    """Whether this adapter's task is judged by command tracking.

    Dispatching on the adapter rather than the task id keeps the decision with
    the object that already knows the observation contract, so a new locomotion
    task inherits the right evaluation by choosing the right adapter.
    """
    return adapter.name() == "ppo_walk"


@torch.no_grad()
def evaluate_locomotion_grid(
    env,
    adapter: PolicyAdapter,
    teacher_agent,
    estimator: NormalizedEstimator | None,
    estimator_type: str,
    window: int,
    settle_s: float = 1.0,
    measure_s: float = 4.0,
    seed: int | None = None,
    with_sanity_checks: bool = False,
) -> dict:
    """Command-tracking evaluation for the locomotion task.

    The episode-based evaluation above is kept because it produces
    `episode_length_mean` and the survival columns, but on this task those pin to
    their ceilings for any competent teacher and separate nothing. This drives
    the same policy over a fixed command grid instead and reports how well it
    still follows each command -- with `estimator` set, through the estimate; with
    it None, from the true privileged state, which is the teacher baseline the
    report subtracts against.

    Estimator RMSE is accumulated in the same rollout, so it is measured under
    exactly the states the tracking numbers were measured under.
    """
    from .locomotion import CommandEvaluator, resolve_feet_cfgs, run_sanity_checks, summarize

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    unwrapped = adapter.core_env
    feet_sensor_cfg, feet_asset_cfg = resolve_feet_cfgs(unwrapped.scene)
    step_dt = unwrapped.step_dt
    evaluator = CommandEvaluator(
        env,
        feet_sensor_cfg,
        feet_asset_cfg,
        settle_steps=int(round(settle_s / step_dt)),
        measure_steps=int(round(measure_s / step_dt)),
    )

    history: HistoryBuffer | None = None
    squared_error = torch.zeros(adapter.schema.estimator_target_dim, dtype=torch.float64)
    sample_count = 0

    if estimator is not None:
        estimator.eval()
    teacher_agent.enable_training_mode(False, apply_to_models=True)

    def act(observations):
        nonlocal history, squared_error, sample_count
        if estimator is None:
            return adapter.action(teacher_agent, observations)
        if history is None:
            history = HistoryBuffer(observations.shape[0], window, adapter.input_dim, observations.device)
        frame = adapter.estimator_input()
        target = adapter.estimator_target()
        sequence = history.push(frame)
        model_input = frame if estimator_type.upper() == "MLP" else sequence
        estimate = estimator.predict(model_input)
        squared_error += (estimate - target).double().square().sum(dim=0).cpu()
        sample_count += target.shape[0]
        return adapter.action(teacher_agent, adapter.inject_estimate(observations, estimate))

    def on_step(dones):
        if dones is not None and dones.any():
            if history is not None:
                history.reset(dones)
            imu_corruptor = getattr(adapter, "imu_corruptor", None)
            if imu_corruptor is not None:
                imu_corruptor.reset(reset_ids(dones))

    results = evaluator.run_all(act, on_step=on_step)
    metrics = summarize(results)
    if with_sanity_checks:
        # Run here rather than in the caller: the checks need the full per-command
        # records, which `summarize` deliberately trims down for the report.
        metrics["sanity_checks"] = [
            {"name": name, "passed": passed, "detail": detail}
            for name, passed, detail in run_sanity_checks(results)
        ]
    if estimator is not None and sample_count:
        target_rmse = (squared_error / sample_count).sqrt().float()
        metrics["grid_rmse"] = float(target_rmse.square().mean().sqrt())
        metrics["grid_target_rmse"] = target_rmse.tolist()
    return metrics


@torch.no_grad()
def _record_body_trajectory(env, adapter, policy, *, seed: int, horizon: int, on_reset=None):
    """Roll `policy` for `horizon` steps from a seeded reset, logging body positions."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    force_skrl_isaaclab_reset(env)
    observations, _ = env.reset()
    if on_reset is not None:
        on_reset()
    bodies, roots = [], []
    for _ in range(horizon):
        bodies.append(adapter.body_positions().clone())
        roots.append(adapter.root_position().clone())
        observations, _, _, _, _ = env.step(policy(observations))
    return torch.stack(bodies), torch.stack(roots)


def evaluate_paired_motion_fidelity(
    env,
    adapter: PolicyAdapter,
    teacher_agent,
    policy,
    *,
    seed: int,
    horizon: int = 100,
    on_reset=None,
) -> dict:
    """Teacher-relative MPJPE: how far the student's motion drifts from the teacher's.

    Both policies are rolled from the same seeded reset, so step t of each
    rollout starts from the same initial condition, and the per-body position
    gap at step t is the motion difference the policies themselves caused.
    Reported in millimetres, following PHC / ExBody2 / OmniH2O.

    Two rollouts cannot share an env, so this runs them back to back. That
    consumes RNG, which is why callers must invoke it *after* training rather
    than between DAgger rounds -- the RNG state is saved and restored here so a
    caller that ignores that still cannot perturb its own training stream.

    Trajectories diverge chaotically once the policies differ at all, so
    `horizon` is deliberately bounded and reported alongside the numbers: an
    MPJPE without its horizon is meaningless.

    `on_reset` is invoked after each env reset, for policies carrying state of
    their own (a history buffer, a sensor corruptor) that must be cleared so the
    second rollout really does start where the first one did.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    torch_state = torch.get_rng_state()
    numpy_state = np.random.get_state()
    try:
        teacher_bodies, teacher_roots = _record_body_trajectory(
            env, adapter, lambda obs: adapter.action(teacher_agent, obs),
            seed=seed, horizon=horizon, on_reset=on_reset,
        )
        student_bodies, student_roots = _record_body_trajectory(
            env, adapter, policy, seed=seed, horizon=horizon, on_reset=on_reset,
        )
    finally:
        torch.set_rng_state(torch_state)
        np.random.set_state(numpy_state)

    millimetres = 1000.0
    global_error = (student_bodies - teacher_bodies).norm(dim=-1) * millimetres
    local_error = (
        (student_bodies - student_roots.unsqueeze(2)) - (teacher_bodies - teacher_roots.unsqueeze(2))
    ).norm(dim=-1) * millimetres
    root_error = (student_roots - teacher_roots).norm(dim=-1) * millimetres
    return {
        "mpjpe_g": float(global_error.mean()),
        "mpjpe_l": float(local_error.mean()),
        "root_position_error": float(root_error.mean()),
        "mpjpe_horizon": horizon,
        "mpjpe_per_body": {
            name: float(global_error[:, :, index].mean())
            for index, name in enumerate(adapter.body_names())
        },
        # Kept so the divergence curve can be plotted later without re-running.
        "mpjpe_by_step": [
            {
                "step": step,
                "mpjpe_g": float(global_error[step].mean()),
                "mpjpe_l": float(local_error[step].mean()),
                "root_position_error": float(root_error[step].mean()),
            }
            for step in range(horizon)
        ],
    }


def _fit_model(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    epoch_callback: Callable[[dict], None] | None = None,
    seed: int = 0,
) -> dict:
    model.to(device)
    size = len(targets)
    validation_size = max(1, min(size // 10, 10000))
    # A local generator, not the global RNG. The split and the per-epoch shuffle
    # must depend only on `seed`, never on how much randomness was drawn before
    # this call -- otherwise a dataset cache *hit* (no rollout collection, so no
    # `torch.randperm` in `collect_rollout`) and a cache *miss* consume different
    # amounts of the global stream and fit different weights from identical data.
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(size, generator=generator)
    validation_ids, training_ids = order[:validation_size], order[validation_size:]
    if len(training_ids) == 0:
        raise ValueError("At least two samples are required for training")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    best_loss, best_state = float("inf"), None
    history = []
    gradient_steps = 0
    for epoch in range(epochs):
        model.train()
        permutation = training_ids[torch.randperm(len(training_ids), generator=generator)]
        train_total = 0.0
        for start in range(0, len(permutation), batch_size):
            ids = permutation[start : start + batch_size]
            prediction = model(inputs[ids].to(device))
            loss = nn.functional.mse_loss(prediction, targets[ids].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss) * len(ids)
            gradient_steps += 1
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                nn.functional.mse_loss(model(inputs[validation_ids].to(device)), targets[validation_ids].to(device))
            )
        row = {"epoch": epoch + 1, "train_mse": train_total / len(training_ids), "validation_mse": validation_loss}
        history.append(row)
        if epoch_callback is not None:
            epoch_callback(row)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {"best_validation_mse": best_loss, "epochs": history, "gradient_steps": gradient_steps}


def train_estimator(
    estimator: NormalizedEstimator,
    dataset: RolloutDataset,
    estimator_type: str,
    epochs: int = 50,
    batch_size: int = 1024,
    learning_rate: float = 1.0e-3,
    device: str = "cuda:0",
    epoch_callback: Callable[[dict], None] | None = None,
    seed: int = 0,
) -> dict:
    inputs = dataset.frames if estimator_type.upper() == "MLP" else dataset.histories
    estimator.to(inputs.device)
    estimator.set_normalization(inputs, dataset.targets)
    normalized_targets = estimator.normalized_targets(dataset.targets)
    return _fit_model(
        estimator, inputs, normalized_targets, epochs, batch_size, learning_rate, device,
        epoch_callback, seed,
    )


def evaluate_predictions(
    estimator: NormalizedEstimator,
    dataset: RolloutDataset,
    estimator_type: str,
    device: str,
    target_names: tuple[str, ...] | list[str] | None = None,
) -> dict:
    inputs = dataset.frames if estimator_type.upper() == "MLP" else dataset.histories
    estimator.eval().to(device)
    start = time.perf_counter()
    with torch.no_grad():
        prediction = estimator.predict(inputs.to(device)).cpu()
    elapsed = time.perf_counter() - start
    error = prediction - dataset.targets
    mse = error.square().mean(dim=0)
    variance = dataset.targets.var(dim=0).clamp_min(1.0e-8)
    metrics = {
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "r2": float((1.0 - mse / variance).mean()),
        "target_mae": error.abs().mean(dim=0).tolist(),
        "target_rmse": mse.sqrt().tolist(),
        "inference_ms_per_sample": elapsed * 1000.0 / len(inputs),
        "parameters": sum(parameter.numel() for parameter in estimator.parameters()),
        "trace_target": dataset.targets[:200].tolist(),
        "trace_prediction": prediction[:200].tolist(),
    }
    if target_names is not None:
        metrics["target_names"] = list(target_names)
    return metrics


def save_jose_checkpoint(
    path: str | Path,
    model: nn.Module,
    adapter: PolicyAdapter,
    task: str,
    window: int,
    metrics: dict,
    kind: str = "estimator",
) -> None:
    skrl_version = require_skrl_2()
    payload = {
        "jose_schema_version": SCHEMA_VERSION,
        "kind": kind,
        "skrl_version": skrl_version,
        "task": task,
        "adapter": adapter.name(),
        "observation_schema": adapter.schema.to_dict(),
        "joint_preset": adapter.joint_preset,
        "joint_names": JOINT_PRESETS[adapter.joint_preset],
        # What the estimator reads, so a loader can rebuild the adapter that
        # matches these weights instead of inferring it from the input width.
        # False for every checkpoint written before the JOSE+IMU arm existed;
        # `getattr` is what makes those still loadable.
        "imu_input": getattr(adapter, "use_imu", False),
        "velocity_source": "sim_joint_velocity",
        "window": window,
        "model_config": model.config() if hasattr(model, "config") else {"type": model.__class__.__name__},
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "module_manifest": sorted(model.state_dict()),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    json_payload = {
        key: value for key, value in payload.items() if key not in ("model_state_dict", "module_manifest")
    }
    path.with_suffix(".json").write_text(
        json.dumps(json_payload, indent=2),
        encoding="utf-8",
    )


def load_estimator(path: str | Path, device: str = "cpu") -> tuple[NormalizedEstimator, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("jose_schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported JOSE checkpoint schema")
    if payload.get("kind") != "estimator":
        raise ValueError("Checkpoint is not a JOSE state estimator")
    if payload.get("velocity_source") != "sim_joint_velocity":
        raise ValueError("Checkpoint was not trained with simulator joint velocities")
    config = payload["model_config"]
    estimator = build_estimator(
        config["type"], config["input_dim"], config["output_dim"],
        config.get("hidden_size", 256), config.get("num_layers", 2), tuple(config.get("channels", (64, 128, 128))),
        config.get("window", payload.get("window", 50)),
    ).to(device)
    estimator.load_state_dict(payload["model_state_dict"])
    estimator.eval()
    return estimator, payload
