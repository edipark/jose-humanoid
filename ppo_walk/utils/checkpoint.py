"""Partial checkpoint loading for the PPO walk tasks.

The reworked walk task (:mod:`jose.ppo_walk.walk_env_cfg`) is effectively a new
MDP: the reward set changed, so the old critic's value function and the old Adam
moments are stale, and resuming a full checkpoint would drag the standing policy's
value estimates into the new run. This module implements the "warm-start the actor
only" option instead: actor weights are loaded, the critic and the optimizer stay
freshly initialised.
"""

from __future__ import annotations

import torch


def load_actor_only(runner, path: str, device: str = "cpu") -> dict:
    """Load *only* the actor weights from ``path`` into ``runner``.

    The critic and the optimizer are left at their fresh initialisation, so the
    policy keeps whatever standing/balancing prior the old checkpoint had while
    the value function is re-learned under the new reward set.

    The walk task prepends ``base_lin_vel`` to the actor observation group, which
    widens the actor's input layer (480 -> 495 for the 5-step history). Because
    observation terms are concatenated in declaration order and ``base_lin_vel``
    is declared *first*, every old input keeps its relative offset: the old weight
    matrix is copied into the right-hand columns of the new one and the new
    left-hand columns are zeroed. A zero column means the warm-started actor
    initially ignores the new observation, which is exactly the old policy's
    behaviour, and PPO grows the weights from there.

    Args:
        runner: an ``rsl_rl.runners.OnPolicyRunner``.
        path: path to a ``model_*.pt`` checkpoint.
        device: device to map the checkpoint onto while loading.

    Returns:
        A report dict with the keys ``loaded``, ``zero_padded`` and ``skipped``.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "actor_state_dict" not in ckpt:
        raise KeyError(
            f"'{path}' has no 'actor_state_dict' (found {sorted(ckpt.keys())}). "
            "Expected an rsl-rl >= 4.0 checkpoint."
        )
    src = ckpt["actor_state_dict"]
    # rsl-rl >= 4.0 keeps the actor on the algorithm (``runner.alg.actor``);
    # older builds wrapped it in ``actor_critic``.
    actor = getattr(runner.alg, "actor", None)
    if actor is None:
        actor = runner.alg.actor_critic.actor
    dst = actor.state_dict()

    report = {"loaded": [], "zero_padded": [], "skipped": []}
    new_state = {}
    for key, dst_tensor in dst.items():
        if key not in src:
            report["skipped"].append(f"{key} (absent in checkpoint)")
            new_state[key] = dst_tensor
            continue
        src_tensor = src[key].to(dst_tensor.device, dst_tensor.dtype)
        if src_tensor.shape == dst_tensor.shape:
            new_state[key] = src_tensor
            report["loaded"].append(key)
            continue
        # only a widened input layer is recoverable: same rank, same output size,
        # more input columns, and the extra columns sit at the front.
        recoverable = (
            src_tensor.ndim == dst_tensor.ndim == 2
            and src_tensor.shape[0] == dst_tensor.shape[0]
            and src_tensor.shape[1] < dst_tensor.shape[1]
        )
        if not recoverable:
            report["skipped"].append(f"{key} ({tuple(src_tensor.shape)} -> {tuple(dst_tensor.shape)})")
            new_state[key] = dst_tensor
            continue
        pad = dst_tensor.shape[1] - src_tensor.shape[1]
        merged = torch.zeros_like(dst_tensor)
        merged[:, pad:] = src_tensor
        new_state[key] = merged
        report["zero_padded"].append(f"{key}: {pad} new input columns zeroed")

    actor.load_state_dict(new_state, strict=True)

    print(f"[INFO] actor-only warm start from: {path} (checkpoint iter {ckpt.get('iter', '?')})")
    print(f"[INFO]   loaded      : {len(report['loaded'])} tensors")
    for line in report["zero_padded"]:
        print(f"[INFO]   zero-padded : {line}")
    for line in report["skipped"]:
        print(f"[WARN]   skipped     : {line}")
    print("[INFO]   critic and optimizer are freshly initialised (not loaded).")
    return report
