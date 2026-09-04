"""Sensing subpackage.

`sensors/` may import from `systems/` and must not import from `trainables/`,
`rollout.py`, or anything above them in the dependency order.
"""
from __future__ import annotations

from typing import Callable, Dict, Type

from .base import DelayBuffer, Sensor
from .full_state import FullState, FullStateVelocity, NoisyPosition
from .landmarks import LandmarkCamera
from .range_sensor import RangeSensor
from .lens import LENSES, DoubleSphere, Pinhole, make_lens

SENSORS: Dict[str, Type[Sensor]] = {}


def register_sensor(name: str) -> Callable:
    def deco(cls):
        if name in SENSORS:
            raise KeyError(f"sensor {name!r} already registered")
        SENSORS[name] = cls
        return cls

    return deco


def make_sensor(name: str, system, **kw) -> Sensor:
    if name not in SENSORS:
        raise KeyError(f"unknown sensor {name!r}; registered: {sorted(SENSORS)}")
    return SENSORS[name](system, **kw)


register_sensor("full_state")(FullState)
register_sensor("full_state_velocity")(FullStateVelocity)
register_sensor("noisy_position")(NoisyPosition)
register_sensor("landmark_camera")(LandmarkCamera)
register_sensor("range")(RangeSensor)

__all__ = ["Sensor", "DelayBuffer", "FullState", "FullStateVelocity",
           "NoisyPosition", "LandmarkCamera", "RangeSensor", "Pinhole", "DoubleSphere",
           "make_lens", "LENSES",
           "SENSORS", "register_sensor", "make_sensor"]
