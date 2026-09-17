"""Unitree-compatible IMU contract, validation, and training corruption.

Projected gravity is derived from attitude, never from normalized raw
accelerometer data.  The public convention is a scalar-first ``wxyz``
quaternion representing the IMU/body orientation in the world frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import torch


class IMUFault(str, Enum):
    NONE = "none"
    NAN = "nan"
    INVALID_QUATERNION = "invalid_quaternion"
    STALE_TIMESTAMP = "stale_timestamp"
    INVALID_GRAVITY = "invalid_projected_gravity"


@dataclass(frozen=True)
class IMUObservation:
    angular_velocity: torch.Tensor
    projected_gravity: torch.Tensor
    timestamp_s: float | torch.Tensor
    fault: IMUFault = IMUFault.NONE

    @property
    def valid(self) -> bool:
        return self.fault is IMUFault.NONE


def _normalize_quaternion(quaternion_wxyz: torch.Tensor, epsilon: float = 1.0e-6) -> torch.Tensor:
    if quaternion_wxyz.shape[-1] != 4:
        raise ValueError("Quaternion must have four components in wxyz order")
    norm = quaternion_wxyz.norm(dim=-1, keepdim=True)
    if torch.any(~torch.isfinite(quaternion_wxyz)):
        raise ValueError("Quaternion contains NaN or infinity")
    if torch.any(norm < epsilon):
        raise ValueError("Quaternion norm is too small")
    return quaternion_wxyz / norm


def quaternion_rotate_inverse(quaternion_wxyz: torch.Tensor, vector_world: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors into the quaternion's body frame."""
    q = _normalize_quaternion(quaternion_wxyz)
    scalar = q[..., :1]
    xyz = q[..., 1:]
    # R(q)^T v == R(q*) v. This closed form supports arbitrary batch shapes.
    return (
        (2.0 * scalar.square() - 1.0) * vector_world
        - 2.0 * scalar * torch.cross(xyz, vector_world, dim=-1)
        + 2.0 * xyz * (xyz * vector_world).sum(dim=-1, keepdim=True)
    )


def projected_gravity_from_quaternion(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Compute ``R_WB.T @ [0, 0, -1]`` for scalar-first quaternions."""
    gravity_world = torch.zeros_like(quaternion_wxyz[..., :3])
    gravity_world[..., 2] = -1.0
    gravity = quaternion_rotate_inverse(quaternion_wxyz, gravity_world)
    return gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _rotate(quaternion_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q = _normalize_quaternion(quaternion_wxyz)
    xyz = q[..., 1:]
    uv = torch.cross(xyz, vector, dim=-1)
    uuv = torch.cross(xyz, uv, dim=-1)
    return vector + 2.0 * (q[..., :1] * uv + uuv)


@dataclass(frozen=True)
class IMUObservationSpec:
    """Shared simulator/LowState observation ABI.

    ``imu_to_pelvis_wxyz`` represents the fixed rotation from IMU-frame
    vectors to the AMP pelvis frame. Unitree LowState gyro is rad/s and its
    quaternion is expected in wxyz order.
    """

    quaternion_order: str = "wxyz"
    quaternion_frame: str = "world_from_imu"
    output_frame: str = "pelvis"
    angular_velocity_unit: str = "rad/s"
    imu_to_pelvis_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    quaternion_norm_tolerance: float = 0.05
    gravity_norm_tolerance: float = 1.0e-3
    stale_after_s: float = 0.10

    def __post_init__(self) -> None:
        if self.quaternion_order != "wxyz":
            raise ValueError("JOSE deploy ABI requires wxyz quaternion order")
        if self.quaternion_frame != "world_from_imu":
            raise ValueError("JOSE deploy ABI requires world_from_imu attitude")
        if self.angular_velocity_unit != "rad/s":
            raise ValueError("JOSE deploy ABI requires gyro values in rad/s")
        if len(self.imu_to_pelvis_wxyz) != 4:
            raise ValueError("IMU-to-pelvis extrinsic must be a wxyz quaternion")
        norm = sum(value * value for value in self.imu_to_pelvis_wxyz) ** 0.5
        if abs(norm - 1.0) > self.quaternion_norm_tolerance:
            raise ValueError("IMU-to-pelvis extrinsic is not unit length")

    def to_dict(self) -> dict:
        return asdict(self)

    def observe(
        self,
        quaternion_wxyz: torch.Tensor,
        gyro_imu: torch.Tensor,
        timestamp_s: float | torch.Tensor,
        now_s: float | torch.Tensor | None = None,
    ) -> IMUObservation:
        if quaternion_wxyz.shape[-1] != 4 or gyro_imu.shape[-1] != 3:
            raise ValueError("Expected quaternion [...,4] and gyro [...,3]")
        if torch.any(~torch.isfinite(quaternion_wxyz)) or torch.any(~torch.isfinite(gyro_imu)):
            empty = torch.zeros_like(gyro_imu)
            return IMUObservation(empty, empty, timestamp_s, IMUFault.NAN)
        norm = quaternion_wxyz.norm(dim=-1)
        if torch.any((norm - 1.0).abs() > self.quaternion_norm_tolerance) or torch.any(norm < 1.0e-6):
            empty = torch.zeros_like(gyro_imu)
            return IMUObservation(empty, empty, timestamp_s, IMUFault.INVALID_QUATERNION)
        if now_s is not None:
            age = torch.as_tensor(now_s) - torch.as_tensor(timestamp_s)
            if bool(torch.any(age > self.stale_after_s)) or bool(torch.any(age < -1.0e-6)):
                empty = torch.zeros_like(gyro_imu)
                return IMUObservation(empty, empty, timestamp_s, IMUFault.STALE_TIMESTAMP)

        q_wi = _normalize_quaternion(quaternion_wxyz)
        q_pi = torch.as_tensor(self.imu_to_pelvis_wxyz, dtype=q_wi.dtype, device=q_wi.device)
        q_pi = q_pi.expand_as(q_wi)
        gravity_imu = projected_gravity_from_quaternion(q_wi)
        gravity_pelvis = _rotate(q_pi, gravity_imu)
        gyro_pelvis = _rotate(q_pi, gyro_imu)
        gravity_norm = gravity_pelvis.norm(dim=-1, keepdim=True)
        if torch.any((gravity_norm - 1.0).abs() > self.gravity_norm_tolerance):
            empty = torch.zeros_like(gyro_imu)
            return IMUObservation(empty, empty, timestamp_s, IMUFault.INVALID_GRAVITY)
        return IMUObservation(gyro_pelvis, gravity_pelvis / gravity_norm, timestamp_s)

    def from_low_state(
        self,
        low_state: Mapping[str, Any] | Any,
        now_s: float | None = None,
        timestamp_s: float | None = None,
    ) -> IMUObservation:
        """Adapt a dict/object mirroring Unitree ``LowState.imu_state``."""
        imu = low_state.get("imu_state", low_state) if isinstance(low_state, Mapping) else getattr(low_state, "imu_state", low_state)
        if callable(imu):
            imu = imu()

        def value(*names: str):
            for name in names:
                if isinstance(imu, Mapping) and name in imu:
                    return imu[name]
                if hasattr(imu, name):
                    candidate = getattr(imu, name)
                    return candidate() if callable(candidate) else candidate
            raise KeyError(f"LowState IMU is missing {names}")

        quaternion = torch.as_tensor(value("quaternion", "quat"), dtype=torch.float32)
        gyro = torch.as_tensor(value("gyroscope", "gyro"), dtype=torch.float32)
        if timestamp_s is None:
            try:
                timestamp_s = float(value("timestamp_s", "timestamp", "time"))
            except KeyError:
                # Unitree's generated IMUState does not guarantee an embedded
                # timestamp. The deployment receiver must then provide its
                # monotonic receive timestamp explicitly.
                raise ValueError("LowState adapter requires a monotonic timestamp") from None
        timestamp = float(timestamp_s)
        return self.observe(quaternion, gyro, timestamp, now_s)


@dataclass(frozen=True)
class SensorCorruptionCfg:
    gyro_noise_std: float = 0.015
    gyro_bias_std: float = 0.01
    gravity_tilt_std_rad: float = 0.015
    max_latency_steps: int = 2
    enabled: bool = True

    def __post_init__(self) -> None:
        if min(self.gyro_noise_std, self.gyro_bias_std, self.gravity_tilt_std_rad) < 0.0:
            raise ValueError("Sensor noise scales must be non-negative")
        if self.max_latency_steps < 0:
            raise ValueError("Sensor latency must be non-negative")


class SensorCorruptor:
    """Stateful, per-environment deploy-observation corruption."""

    def __init__(self, num_envs: int, device: torch.device | str, cfg: SensorCorruptionCfg):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = cfg
        self.bias = torch.zeros(num_envs, 3, device=device)
        self.latency = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._frames = torch.zeros(num_envs, cfg.max_latency_steps + 1, 6, device=device)
        self._initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self.bias[ids] = torch.randn(len(ids), 3, device=self.device) * self.cfg.gyro_bias_std
        self.latency[ids] = torch.randint(self.cfg.max_latency_steps + 1, (len(ids),), device=self.device)
        self._frames[ids] = 0.0
        self._initialized[ids] = False

    def __call__(self, angular_velocity: torch.Tensor, projected_gravity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.enabled:
            return angular_velocity, projected_gravity
        gyro = angular_velocity + self.bias + torch.randn_like(angular_velocity) * self.cfg.gyro_noise_std
        tilt = torch.randn_like(projected_gravity) * self.cfg.gravity_tilt_std_rad
        gravity = projected_gravity + torch.cross(tilt, projected_gravity, dim=-1)
        gravity = gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        frame = torch.cat((gyro, gravity), dim=-1)
        fresh = ~self._initialized
        if fresh.any():
            self._frames[fresh] = frame[fresh, None, :]
        existing = self._initialized
        if existing.any():
            self._frames[existing, 1:] = self._frames[existing, :-1].clone()
            self._frames[existing, 0] = frame[existing]
        self._initialized[:] = True
        selected = self._frames[torch.arange(self.num_envs, device=self.device), self.latency]
        return selected[:, :3], selected[:, 3:]
