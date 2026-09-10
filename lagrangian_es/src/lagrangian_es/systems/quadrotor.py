"""Full SE(3) quadrotor.

State: {p [...,3], v [...,3], R [...,3,3], om [...,3]}  -- world-frame position and
velocity, body-to-world rotation, body-frame angular rate.

Generalized force is u = (f, tau_x, tau_y, tau_z): a scalar thrust magnitude along
body z plus a body-frame torque.  n_force = 4, task_dim = 3 -- the plant is
underactuated, and `allocate` is where that shows up.

M is CONSTANT here: diag(m, Jx, Jy, Jz) in body coordinates.  `inv_mass` therefore
ignores its state argument.  That is the honest scope limit of the whitening claim
on this plant -- see `two_link_arm.py` for the case where it does not.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor

from .base import LagrangianSystem, State
from .so3 import rodrigues, vee


class QuadrotorSE3(LagrangianSystem):
    n_force = 4
    task_dim = 3
    allocator_dim = 6          # kR [3], kW [3]; +3 under yaw_mode='learned'
    dense_mass = False
    state_keys = ("p", "v", "R", "om")

    def __init__(
        self,
        m: float = 0.5,
        J: tuple = (5.0e-3, 5.0e-3, 9.0e-3),
        g: float = 9.81,
        thrust_ratio: float = 2.2,     # f_max / (m g)
        tau_max: float = 0.30,
        phi0: tuple = None,            # kR, kW priors; None = derived from the plant (see allocator_init)
        att_ratio: float = 5.0,        # the attitude loop's bandwidth over the position loop's
        att_zeta: float = 0.8,         # its damping ratio
        f_min_frac: float = 0.1,       # the thrust envelope keeps at least this fraction of f_max vertical
        yaw_budget_frac: float = 0.5,  # the Lagrangian yaw torque saturates smoothly at this fraction of tau_max
        # "world_x" pins the heading to a compass bearing; "learned" hands yaw
        # to the genome, which is free on this plant -- thrust is along body z,
        # so rotating about it does not disturb position tracking at all.
        yaw_mode: str = "world_x",
        # below this speed the heading holds a fixed bearing rather than chasing
        # a direction estimated from near-zero velocity
        yaw_speed_gate: float = 0.6,
        yaw_bearing_gate: float = 0.5,
        # how far the heading reference may lead the current heading (rad)
        yaw_slew: float = 0.25,
        # --- reset distribution
        reset_z: float = 0.5,
        reset_pos_noise: float = 0.05,
        reset_vel_noise: float = 0.05,
        reset_att_noise: float = 0.03,
        reset_yaw: float = 0.0,
        speed_limit: float = 0.0,
        reset_om_noise: float = 0.05,
        # --- liveness envelope
        z_floor: float = 0.05,
        p_max: float = 20.0,
        v_max: float = 30.0,
        om_max: float = 50.0,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        self.dtype, self.device = dtype, device
        self.m = float(m)
        self.g = float(g)
        self.Jvec = self._t(J)                       # [3]
        self.f_max = float(thrust_ratio) * self.m * self.g
        self.f_min = 0.0
        self.tau_max = float(tau_max)
        # `lagrangian`: no heading rule at all -- the reference is the current
        # heading, and yaw is driven by the torque -dV/dpsi the controller's own
        # potential produces through the body-fixed beams (plus the loop's
        # damping on the yaw rate); a task-level command still blends on top
        if yaw_mode not in ("world_x", "learned", "lagrangian"):
            raise ValueError(f"unknown yaw_mode {yaw_mode!r}")
        self.yaw_mode = yaw_mode
        self.yaw_speed_gate = float(yaw_speed_gate)
        # well under any standoff task's radius (ObserveTarget holds at 1.9 m),
        # so "hover and keep it in view" is untouched; just over the 0.25 m
        # arrival tolerance, so "parked on it" is not
        self.yaw_bearing_gate = float(yaw_bearing_gate)
        self.yaw_slew = float(yaw_slew)
        if yaw_mode == "learned":
            # three extra slots: where to LOOK.  Yaw is the one degree of freedom
            # a quadrotor has for free -- thrust acts along body z, so rotating
            # about it costs nothing in position tracking -- and pinning it to a
            # compass bearing throws that away.  Measured under `world_x`, the
            # body's forward axis sits 90.2 deg off the direction of travel and
            # the path lies inside an 80 deg field of view only 10% of the time,
            # so a body-mounted camera watches everything except where the
            # vehicle is going.
            self.allocator_dim = 9
        self.phi0 = None if phi0 is None else tuple(phi0)
        self.att_ratio, self.att_zeta, self.f_min_frac = float(att_ratio), float(att_zeta), float(f_min_frac)
        self.yaw_budget_frac = float(yaw_budget_frac)

        self.reset_z = float(reset_z)
        self.reset_pos_noise = float(reset_pos_noise)
        self.reset_vel_noise = float(reset_vel_noise)
        self.reset_att_noise = float(reset_att_noise)
        # half-range of a uniform random heading at reset; pi = any heading.  With a
        # forward-mounted fan, facing where you are going is a skill, not a given.
        self.reset_yaw = float(reset_yaw)
        self.speed_limit = float(speed_limit)   # an airspeed limit, like the thrust and torque clamps; 0 = none (`v_max` below is the liveness bound)
        self.reset_om_noise = float(reset_om_noise)

        self.z_floor = float(z_floor)
        self.p_max, self.v_max, self.om_max = float(p_max), float(v_max), float(om_max)

        self._e3 = self._t([0.0, 0.0, 1.0])
        self._e1 = self._t([1.0, 0.0, 0.0])
        self._hover = self.m * self.g
        # Folded out of the hot loop.  Each of these was a fresh tensor op every
        # step of every rollout; individually trivial, but the loop is dispatch
        # bound, so the count is what costs.
        self._g_vec = self._e3 * self.g                 # was rebuilt per step
        self._grav_wrench = self._e3 * self._hover      # was rebuilt per call
        self._inv_m = 1.0 / self.m                      # divide -> multiply
        self._inv_J = 1.0 / self.Jvec
        self._p_max2 = self.p_max ** 2                  # compare squares, no sqrt
        self._v_max2 = self.v_max ** 2
        self._om_max2 = self.om_max ** 2

    # --- simulation ---------------------------------------------------------
    def reset(self, B: int, gen: torch.Generator) -> State:
        kw = dict(generator=gen, dtype=self.dtype, device=self.device)
        p0 = self._e3 * self.reset_z
        p = p0 + self.reset_pos_noise * torch.randn(B, 3, **kw)
        v = self.reset_vel_noise * torch.randn(B, 3, **kw)
        om = self.reset_om_noise * torch.randn(B, 3, **kw)
        R = rodrigues(self.reset_att_noise * torch.randn(B, 3, **kw))
        if self.reset_yaw > 0.0:
            psi = (2.0 * torch.rand(B, generator=gen, dtype=self.dtype, device=self.device) - 1.0) * self.reset_yaw
            c, sn, z, o = torch.cos(psi), torch.sin(psi), torch.zeros_like(psi), torch.ones_like(psi)
            Rz = torch.stack([torch.stack([c, -sn, z], -1), torch.stack([sn, c, z], -1), torch.stack([z, z, o], -1)], -2)
            R = Rz @ R
        return {"p": p, "v": v, "R": R, "om": om}

    def step(self, s: State, u: Tensor, dt: float, params: Tensor = None) -> State:
        """Semi-implicit Euler on SE(3).  Pure: builds a fresh dict.

        `params` is unused: this plant declares no learned residual."""
        f = u[..., 0].clamp(self.f_min, self.f_max)          # safety clamp; the
        tau = u[..., 1:4].clamp(-self.tau_max, self.tau_max)  # allocator clamps too

        R, om = s["R"], s["om"]
        b3 = R[..., :, 2]                                     # body z in world
        acc = b3 * (f * self._inv_m)[..., None] - self._g_vec
        v = s["v"] + dt * acc
        if self.speed_limit > 0.0:
            # the airframe's speed limit, a physical clamp on the plant (user:
            # "set a max speed of 5 m/s"); smooth enough for the differentiable path
            sp = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
            v = v * (self.speed_limit / sp.clamp_min(1e-9)).clamp(max=1.0)
        p = s["p"] + dt * v

        Jom = om * self.Jvec
        om_dot = (tau - torch.cross(om, Jom, dim=-1)) * self._inv_J
        om_new = om + dt * om_dot
        R_new = R @ rodrigues(om_new * dt)
        return {"p": p, "v": v, "R": R_new, "om": om_new}

    def alive(self, s: State) -> Tensor:
        """Liveness, with the redundant work removed.

        The explicit `isfinite` checks on p, v and om were doing nothing the
        bounds were not already doing: a NaN or Inf component makes the squared
        norm NaN or Inf, and every comparison against it is False.  Only R needs
        one, because no bound is placed on it.  Squared norms also drop three
        square roots per step.
        """
        p, v, om, R = s["p"], s["v"], s["om"], s["R"]
        return (
            torch.isfinite(R).flatten(-2).all(-1)
            & (p[..., 2] > self.z_floor)
            & ((p * p).sum(-1) < self._p_max2)
            & ((v * v).sum(-1) < self._v_max2)
            & ((om * om).sum(-1) < self._om_max2)
        )

    # --- mechanics ----------------------------------------------------------
    def inv_mass(self, s: State) -> Tensor:
        """diag(1/m, 1/Jx, 1/Jy, 1/Jz), broadcast to the state's batch shape.

        Deliberately ignores `s`: for a rigid body M is constant in the body frame.
        The trajectory dependence of G on this plant therefore enters only through
        the controller Jacobian.
        """
        head = torch.cat([self._t([1.0 / self.m]), 1.0 / self.Jvec], dim=0)  # [4]
        shape = s["p"].shape[:-1] + (self.n_force,)
        return head.expand(shape)

    def gravity_force(self, s: State) -> Tensor:
        """m g e3 -- the world-frame force that holds the vehicle up."""
        return torch.zeros_like(s["p"]) + self._grav_wrench

    # --- the underactuation seam --------------------------------------------
    def allocate(self, F_des: Tensor, s: State, phi: Tensor,
                 goal: Optional[Tensor] = None, yaw: Optional[Tensor] = None,
                 yaw_gate: Optional[Tensor] = None, yaw_offset: Optional[Tensor] = None,
                 yaw_torque: Optional[Tensor] = None) -> Tensor:
        """Thrust along body z + a geometric SO(3) attitude loop.

        `phi` = (kR [3], kW [3]) raw; gains are used squared so they stay positive
        under unconstrained evolution.  Yaw reference is fixed at world x.

        vmap/jacrev safe: no branches, no in-place writes, every normalize floored.
        """
        R, om = s["R"], s["om"]
        kR = phi[..., 0:3] ** 2
        kW = phi[..., 3:6] ** 2

        # --- the thrust envelope ----------------------------------------------
        # A quadrotor produces at most f_max along its body z, so the forces it
        # can realise fill a ball of radius f_max -- and a force it cannot
        # realise must lose its HORIZONTAL part, not its vertical one, or the
        # body is pointed toward a direction it cannot hold up.  Measured
        # without this: the fresh potential asked for 24.6 N (f_max 10.8) with
        # 4.9 N vertical, the body was tilted 78-88 degrees, full thrust lifted
        # 2 N against 4.9 N of weight, and 30 of 32 flights were on the floor
        # inside two seconds on an EMPTY map.
        Fz = F_des[..., 2:3].clamp(self.f_min_frac * self.f_max, self.f_max)
        Fh = F_des[..., :2]
        room = (self.f_max ** 2 - Fz ** 2).clamp_min(0.0).sqrt()
        Fh = Fh * (room / torch.linalg.vector_norm(Fh, dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
        F_des = torch.cat([Fh, Fz], dim=-1)

        # --- thrust: project the desired force onto the current body z --------
        b3 = R[..., :, 2]
        f = (F_des * b3).sum(dim=-1).clamp(self.f_min, self.f_max)

        # --- desired attitude: body z along F_des, then choose where to LOOK --
        b3d = F_des / torch.linalg.vector_norm(F_des, dim=-1, keepdim=True).clamp_min(1e-6)
        if self.yaw_mode == "learned":
            # A heading blended from three horizontal cues, all available here:
            #   travel   where the vehicle is going now
            #   demand   where it is being accelerated -- i.e. where it is about
            #            to go, which is the ANTICIPATORY term
            #   lateral  travel rotated 90 deg, so the search can bias the look
            #            off-axis and scan into a turn before entering it
            w = phi[..., 6:9]
            vxy = torch.cat([s["v"][..., :2],
                             torch.zeros_like(s["v"][..., 2:])], dim=-1)
            sp0 = torch.linalg.vector_norm(vxy, dim=-1, keepdim=True)
            vh = vxy / sp0.clamp_min(1e-6)
            # TRAVEL faded out by speed: a direction estimated from millimetres
            # per second is well defined and meaningless, and chasing it spins
            # the heading.  Faded here rather than gating the whole blend, so the
            # other cues survive at hover -- which is the case that matters when
            # the job is to hold station and keep something in view.
            vgate = (sp0 / self.yaw_speed_gate).clamp(0.0, 1.0)
            # BEARING to the target: the only cue still defined at rest, and the
            # one that means "keep it in sight".
            if goal is not None:
                gxy = torch.cat([goal[..., :2] - s["p"][..., :2],
                                 torch.zeros_like(s["v"][..., 2:])], dim=-1)
                rg = torch.linalg.vector_norm(gxy, dim=-1, keepdim=True)
                gh = gxy / rg.clamp_min(1e-6)
                # RANGE-faded, for exactly the reason travel is speed-faded: a
                # bearing measured from the point you are standing ON is well
                # defined and meaningless.  Arrived, `gxy` is millimetres of
                # position error, `gh` swings through half turns between steps,
                # and the heading chases it -- measured at a mean 1.98 rad/s of
                # yaw rate while parked inside 0.30 m, reversing direction on
                # 3.9% of steps.  The comment that once stood here called this
                # "the only cue still defined at rest"; that is true of a target
                # you are holding station OFF, and false of one you are on.
                ggate = (rg / self.yaw_bearing_gate).clamp(0.0, 1.0)
            else:
                gh = torch.zeros_like(vh) + self._e1
                ggate = torch.ones_like(sp0)
            lat = torch.stack([-vh[..., 1], vh[..., 0],
                               torch.zeros_like(vh[..., 2])], dim=-1)
            # The rest state: with nowhere to go and nothing to look at, the
            # right reference is the heading already held.  Without this the
            # blend collapses to a normalised zero -- a direction made of pure
            # rounding -- which is the same failure as an ungated cue, reached
            # from the other side.
            b1 = R[..., :, 0]
            hxy = torch.cat([b1[..., :2], torch.zeros_like(b1[..., 2:])], dim=-1)
            hh = hxy / torch.linalg.vector_norm(hxy, dim=-1, keepdim=True).clamp_min(1e-6)
            look = (w[..., 0:1] * (ggate * gh) + w[..., 1:2] * (vgate * vh)
                    + w[..., 2:3] * (vgate * lat)
                    + (1.0 - ggate) * (1.0 - vgate) * hh)
            n = torch.linalg.vector_norm(look, dim=-1, keepdim=True)
            look = look / n.clamp_min(1e-6)
            if yaw_offset is not None:
                # the low level's own turning: the look-at rotated about world
                # z by an offset its learned heading head reads off the beams
                co, so = torch.cos(yaw_offset), torch.sin(yaw_offset)
                look = torch.stack([co * look[..., 0] - so * look[..., 1],
                                    so * look[..., 0] + co * look[..., 1], look[..., 2]], dim=-1)
            # Blend back to a FIXED bearing at low speed.  `vh` is a normalised
            # direction, so it is well defined but meaningless when barely
            # moving: it swings through large angles for millimetre-per-second
            # motion, the yaw reference spins, the attitude loop chases it and the
            # vehicle tumbles.  Measured with the guard keyed on the blend's norm
            # instead of on SPEED -- which is 1 for any nonzero v, so it never
            # fired -- the prior crashed 95% of episodes.
            # Cap how far the heading reference may sit from the CURRENT heading.
            # The attitude loop drives one combined SO(3) error, so a yaw
            # reference that keeps moving never lets it settle: measured
            # unclamped, tilt rose 4.7x (0.038 -> 0.178) and torque saturated
            # 23% of steps against 13%, while the gyroscopic coupling it was
            # blamed on stayed negligible (0.006 of a 0.30 N.m budget).  Bounding
            # the yaw error bounds its share of the torque, so pointing the camera
            # cannot outbid keeping the vehicle upright.
            cur = torch.atan2(R[..., 1, 0], R[..., 0, 0])
            des = torch.atan2(look[..., 1], look[..., 0])
            # every cue faded and the hold term degenerate (nose straight up):
            # hold, rather than steer to the direction of the rounding error
            valid = (n[..., 0] / 1e-3).clamp(0.0, 1.0)
            if yaw is not None:
                # A COMMANDED heading from the task-level layer, blended in by
                # `yaw_gate` (0 = the plant's own look-at, 1 = the command).
                # The blend starts from whatever reference is valid -- the
                # look-at where it has a norm, the current heading where it
                # does not -- and a command counts as validity in its own
                # right, or a learned agent whose look weights start at zero
                # would drop every command on the floor.  Same slew below, so
                # pointing the sensors can no more outbid staying upright than
                # the look-at could.
                g = torch.ones_like(cur) if yaw_gate is None else yaw_gate
                base = cur + valid * torch.atan2(torch.sin(des - cur), torch.cos(des - cur))
                dcmd = torch.atan2(torch.sin(yaw - base), torch.cos(yaw - base))
                des = base + g * dcmd
                valid = torch.maximum(valid, g)
            dpsi = torch.atan2(torch.sin(des - cur), torch.cos(des - cur))
            psi = cur + (valid * dpsi).clamp(-self.yaw_slew, self.yaw_slew)
            b1c = torch.stack([torch.cos(psi), torch.sin(psi),
                               torch.zeros_like(psi)], dim=-1)
        elif self.yaw_mode == "lagrangian":
            cur = torch.atan2(R[..., 1, 0], R[..., 0, 0])
            des = cur
            if yaw is not None:
                g = torch.ones_like(cur) if yaw_gate is None else yaw_gate
                dcmd = torch.atan2(torch.sin(yaw - cur), torch.cos(yaw - cur))
                des = cur + g * dcmd
            dpsi = torch.atan2(torch.sin(des - cur), torch.cos(des - cur))
            psi = cur + dpsi.clamp(-self.yaw_slew, self.yaw_slew)
            b1c = torch.stack([torch.cos(psi), torch.sin(psi), torch.zeros_like(psi)], dim=-1)
        elif yaw is not None:
            # No look-at of its own (yaw fixed at world x), but a heading has
            # been COMMANDED from the task-level layer: slew toward it from the
            # current heading, gated, under the same yaw slew.  With the gate
            # at zero this is world x exactly, so the bare plant is unchanged.
            cur = torch.atan2(R[..., 1, 0], R[..., 0, 0])
            g = torch.ones_like(cur) if yaw_gate is None else yaw_gate
            fixed = torch.zeros_like(cur)                       # world x
            dcmd = torch.atan2(torch.sin(yaw - fixed), torch.cos(yaw - fixed))
            des = fixed + g * dcmd
            dpsi = torch.atan2(torch.sin(des - cur), torch.cos(des - cur))
            psi = cur + dpsi.clamp(-self.yaw_slew, self.yaw_slew)
            b1c = torch.stack([torch.cos(psi), torch.sin(psi), torch.zeros_like(psi)], dim=-1)
        else:
            b1c = torch.zeros_like(b3d) + self._e1
        b2d = torch.cross(b3d, b1c, dim=-1)
        b2d = b2d / torch.linalg.vector_norm(b2d, dim=-1, keepdim=True).clamp_min(1e-6)
        b1d = torch.cross(b2d, b3d, dim=-1)
        Rd = torch.stack([b1d, b2d, b3d], dim=-1)            # columns

        # --- geometric attitude error (Lee et al.) ----------------------------
        RdT_R = Rd.transpose(-1, -2) @ R
        eR = 0.5 * vee(RdT_R - RdT_R.transpose(-1, -2))
        eW = om                                              # om_d = 0

        gyro = torch.cross(om, om * self.Jvec, dim=-1)       # feedforward
        tau = -kR * eR - kW * eW + gyro
        if yaw_torque is not None and self.yaw_mode == "lagrangian":
            # the potential's own turning, about body z (the third torque
            # channel); the loop's kW damps the yaw rate it produces.  Only
            # in the Lagrangian yaw mode: the other modes command a heading
            # through b1c, and adding the torque there fought that loop in
            # every existing configuration (review finding, Sept 9).
            #
            # SMOOTHLY budgeted, like the pull: the potential asks for tens of
            # N.m against a 0.3 N.m limit, so a hard clamp made yaw bang-bang
            # and the closed loop discontinuous -- a 1e-4 perturbation of the
            # genome flipped 13% of crash outcomes in corridors with this
            # torque and 3% without it.  tanh at half the torque limit keeps
            # the other half for the attitude loop's own damping and keeps
            # the map continuous.
            b = self.yaw_budget_frac * self.tau_max
            yaw_torque = b * torch.tanh(yaw_torque / b)
            tau = tau + torch.stack([torch.zeros_like(yaw_torque), torch.zeros_like(yaw_torque), yaw_torque], dim=-1)
        tau = tau.clamp(-self.tau_max, self.tau_max)
        return torch.cat([f[..., None], tau], dim=-1)

    # --- task-space accessors -----------------------------------------------
    def task_mass(self, s: State) -> Tensor:
        """m I_3 -- translational inertia is isotropic and CONSTANT, so kinetic
        shaping on this plant is exact rather than an IDA-PBC approximation."""
        eye = torch.eye(3, dtype=self.dtype, device=self.device)
        return (self.m * eye).expand(s["p"].shape[:-1] + (3, 3))

    def pull_budget(self) -> float:
        """The force a goal pull may ask for: HALF the horizontal room the
        rotors have at hover, sqrt(f_max^2 - (m g)^2) / 2.  The other half is
        the brake's.  A pull that takes the whole room -- the quadratic bowl
        asks 20 N at 10 m against 9.6 N of room -- leaves a braking term no
        authority at all: it must first cancel what the envelope was already
        discarding before it slows anything.  Measured on the 25%-buildings
        map: every crash at the speed limit, 1.0 s after takeoff, the brake
        never biting, the search flat for 40 iterations."""
        return 0.5 * float((self.f_max ** 2 - (self.m * self.g) ** 2) ** 0.5)

    def allocator_init(self) -> Tensor:
        """The attitude prior, derived from the plant rather than guessed.

        The position loop's prior is the unit bowl V = |e|^2 (force 2e), so its
        bandwidth is w_pos = sqrt(2/m); the attitude loop sits `att_ratio`
        times above it at damping `att_zeta`: kR_i = w^2 J_i, kW_i = 2 zeta w
        J_i, stored as square roots since the gains are used squared.  The old
        fixed prior (0.25, 0.10) gave 3.5 rad/s at damping 0.28 on this plant:
        a 60-degree command overshot to 95 degrees, past horizontal.  The base
        Lagrangian must fly on its own -- the perturbations are there to keep
        it from crashing, not to teach it to fly.  Pass `phi0` to override.

        Under `yaw_mode="learned"` the prior points straight at the target
        (1, 0, 0) -- the cue that survives at hover -- and the search is free to
        move weight onto travel (look where you are going) or lateral (scan into
        the turn) instead.
        """
        if self.phi0 is not None:
            phi = self._t(self.phi0)
        else:
            w_att = self.att_ratio * math.sqrt(2.0 / self.m)
            kR = (w_att ** 2) * self.Jvec
            kW = 2.0 * self.att_zeta * w_att * self.Jvec
            phi = torch.cat([kR.sqrt(), kW.sqrt()]).to(self.dtype)
        if self.yaw_mode == "learned":
            extra = torch.tensor([1.0, 0.0, 0.0], dtype=self.dtype,
                                 device=self.device)   # look AT the target
            phi = torch.cat([phi, extra])
        return phi

    def camera_pose(self, s: State):
        return s["p"], s["R"]

    def render_spec(self) -> dict:
        return {"dim": 3, "ground": 0.0, "scale": 1.0,
                "bodies": [{"type": "quadrotor", "size": [0.17, 0.17, 0.045],
                            "arm": 0.16, "rotor": 0.055}]}

    def render_poses(self, s: State) -> Tensor:
        R = s["R"].reshape(s["R"].shape[:-2] + (9,))
        return torch.cat([s["p"], R], dim=-1).unsqueeze(-2)      # [..., 1, 12]

    def task_position(self, s: State) -> Tensor:
        return s["p"]

    def task_velocity(self, s: State) -> Tensor:
        return s["v"]

    def nominal_state(self, x: Tensor, v: Tensor) -> State:
        """Level attitude, zero body rate, at position x with velocity v."""
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        v = torch.as_tensor(v, dtype=self.dtype, device=self.device)
        x, v = torch.broadcast_tensors(x, v)
        eye = torch.eye(3, dtype=self.dtype, device=self.device)
        return {
            "p": x.clone(),
            "v": v.clone(),
            "R": eye.expand(x.shape[:-1] + (3, 3)).clone(),
            "om": torch.zeros_like(x),
        }

    # --- cost hooks ---------------------------------------------------------
    def effort(self, u: Tensor, s: State) -> Tensor:
        """u is (f, tau) but gravity_force is a world-frame 3-vector, so the
        default implementation does not apply: the hover feedforward is m g in
        the thrust channel and zero in the torques."""
        du_f = u[..., 0] - self._hover
        du_t = u[..., 1:4]
        return du_f * du_f / self.m + (du_t * du_t / self.Jvec).sum(dim=-1)

    def shaping_cost(self, s: State) -> Tensor:
        """Tilt penalty: 1 - R[2,2], zero when level, 2 when inverted."""
        return 1.0 - s["R"][..., 2, 2]

    def sight_cost(self, s: State, goal: Tensor) -> Tensor:
        """1 - cos(angle) between body +x and the horizontal bearing to `goal`.

        0 looking straight at it, 1 broadside, 2 looking away.  Body +x is where
        a nose-mounted camera points, so this is the term that makes keeping the
        target in view worth something -- and yaw is free on this plant, so
        satisfying it costs nothing in tracking.  Degenerate directly overhead,
        where the horizontal bearing is undefined; the floored norm leaves the
        cost near zero there rather than swinging the heading around.
        """
        d = goal[..., :2] - s["p"][..., :2]
        n = torch.linalg.vector_norm(d, dim=-1, keepdim=True)
        dh = d / n.clamp_min(1e-6)
        fwd = s["R"][..., :2, 0]
        cos = (fwd * dh).sum(-1)
        gate = (n[..., 0] / 0.25).clamp(0.0, 1.0)      # ignore when almost on top
        return gate * (1.0 - cos)

    # --- diagnostics --------------------------------------------------------
    def saturation(self, u: Tensor, s: State) -> Tensor:
        tol = 1e-6
        f = u[..., 0]
        sat_f = (f <= self.f_min + tol) | (f >= self.f_max - tol)
        sat_t = u[..., 1:4].abs() >= self.tau_max - tol
        n = sat_f.to(u.dtype) + sat_t.to(u.dtype).sum(dim=-1)
        return n / self.n_force

    def describe(self) -> dict:
        d = super().describe()
        d.update(m=self.m, g=self.g, f_max=self.f_max, tau_max=self.tau_max,
                 Jx=float(self.Jvec[0]), Jz=float(self.Jvec[2]))
        return d
