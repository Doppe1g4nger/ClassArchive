"""Platform state for an :class:`~rfdes.system.RFSystem`.

A platform has a name and a 6-degree-of-freedom kinematic state: position and a
quaternion orientation (the six DOF), plus the corresponding velocities so motion
(and effects like Doppler) can be modeled. State is mutable because platforms
move during a simulation; :meth:`PlatformState.snapshot` takes an independent deep
copy for attaching to a transmitted signal.
"""

from __future__ import annotations

import copy as _copy
from dataclasses import dataclass, field

import numpy as np


def _vec3(value) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"expected a length-3 vector, got {arr.size} elements")
    return arr


def _quat(value) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 4:
        raise ValueError(f"expected a length-4 quaternion (qw,qx,qy,qz), got {arr.size}")
    return arr


@dataclass
class PlatformState:
    """6DOF kinematic state and identity of an RF platform.

    Attributes:
        name: Human-readable platform identifier.
        position: ``[x, y, z]`` (metres, world frame).
        velocity: ``[vx, vy, vz]`` (m/s).
        orientation: Attitude quaternion ``[qw, qx, qy, qz]`` (identity default).
        angular_velocity: Body angular rate ``[wx, wy, wz]`` (rad/s).
        metadata: Free-form additional state.
    """

    name: str = ""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Coerce list/tuple inputs to numpy arrays of the right shape.
        self.position = _vec3(self.position)
        self.velocity = _vec3(self.velocity)
        self.orientation = _quat(self.orientation)
        self.angular_velocity = _vec3(self.angular_velocity)

    def snapshot(self) -> "PlatformState":
        """Return a deep, independent copy (arrays duplicated)."""
        return _copy.deepcopy(self)
