"""Sensing subpackage.

`sensors/` may import from `systems/` and must not import from `trainables/`,
`rollout.py`, or anything above them in the dependency order.
"""
from __future__ import annotations

from typing import Callable, Dict, Type

from .base import DelayBuffer, Sensor
from .full_state import FullState, FullStateVelocity, NoisyPosition, Tilt
from .landmarks import LandmarkCamera
from .map_view import MapPrior
from .depth_camera import DepthCamera
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
register_sensor("tilt")(Tilt)
register_sensor("full_state_velocity")(FullStateVelocity)
register_sensor("noisy_position")(NoisyPosition)
register_sensor("landmark_camera")(LandmarkCamera)
register_sensor("range")(RangeSensor)


class RangeDown(RangeSensor):
    """A short proximity fan under the vehicle: four beams, straight down and
    slightly splayed, two metres of range.  Body-fixed, like the front fan."""

    def __init__(self, system, n_beams: int = 4, max_range: float = 2.0, spread: float = 0.6,
                 sigma: float = 0.02, latency_steps: int = 1, **kw):
        super().__init__(system, n_beams=n_beams, max_range=max_range, spread=spread, sigma=sigma,
                         latency_steps=latency_steps, elevations=(-1.5707963267948966,), name="range_down", **kw)


register_sensor("range_down")(RangeDown)
register_sensor("depth_camera")(DepthCamera)
register_sensor("map_prior")(MapPrior)

__all__ = ["Sensor", "DelayBuffer", "FullState", "FullStateVelocity",
           "NoisyPosition", "LandmarkCamera", "RangeSensor", "Pinhole", "DoubleSphere",
           "make_lens", "LENSES",
           "DepthCamera", "MapPrior", "SENSORS", "register_sensor", "make_sensor"]
