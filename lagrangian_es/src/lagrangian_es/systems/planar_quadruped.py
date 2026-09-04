"""Planar quadruped -- a multi-body robot with ground contact.

Sagittal-plane ("robot dog") model.  Left/right legs project onto the same plane,
so the four legs share two hip attachment points while still moving independently.

    q = [x, z, pitch,  (hip, knee) x 4]        11 generalized coordinates
    u = 8 joint torques                        the base is UNACTUATED

Three things make this qualitatively harder than the quadrotor, and all three are
exactly what the abstractions were built for:

  * **M(q) is large, dense and strongly configuration-dependent.**  This is the
    plant where the strong form of the whitening claim lives; on the quadrotor M
    is constant in the body frame, so `G` could only ever vary through the
    controller Jacobian.
  * **Underactuation is severe.**  Three of eleven degrees of freedom have no
    actuator at all; the base is moved only through ground reaction forces.  The
    `allocate` seam becomes a contact-force distribution problem.
  * **Contact.**  The robot is defined by the forces it exchanges with the floor.
    Those forces are LAGRANGE MULTIPLIERS of holonomic constraints (see
    `holonomic.py`), solved with qddot from one KKT system, rather than penalty
    springs bolted onto the equations of motion.  The coupling force is then a
    real, reportable quantity instead of a spring constant's side effect.

    Joint coupling between links needs no constraint at all here: the model uses
    generalized (minimal) coordinates, so the joints are already implicit in q.
    `holonomic.py` carries the couplings minimal coordinates cannot express --
    contact, closed loops, and deliberate joint couplings such as gait symmetry.

Dynamics are assembled analytically rather than by autograd.  For a planar tree
every body's absolute angle is a fixed linear function of q, so with
u(phi) = (sin phi, -cos phi) and w(phi) = du/dphi = (cos phi, sin phi):

    p_i   = p_base + hip_i(theta) + sum_j C[i,j] u(phi_j)
    J_i   = [e_x, e_z, dhip/dtheta + ...] + sum_j C[i,j] Pm[j,:] w(phi_j)
    a_i   = -thetadot^2 hip_i - sum_j C[i,j] phidot_j^2 u(phi_j)      (at qddot = 0)

    M(q)  = sum_i m_i J_i^T J_i + I_i Pm_i^T Pm_i
    b     = sum_i m_i J_i^T (a_i + g e_z)

`b` is inverse dynamics evaluated at qddot = 0, which is exactly C(q,qdot)qdot +
G(q); the rotational part contributes nothing because angular acceleration is
linear in qddot.  Keeping this closed form matters for the method's central
claim: nothing in the dynamics is ever handed to autograd, so the search stays
gradient-free with respect to the plant.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .base import LagrangianSystem, State
from .holonomic import ConstraintStack, GroundContact, HolonomicConstraint

N_LEG = 4
N_Q = 3 + 2 * N_LEG          # 11
N_BODY = 1 + 2 * N_LEG       # trunk + thigh/shank per leg
N_ANG = N_BODY               # one absolute angle per body
N_PT = N_BODY + N_LEG        # body COMs, then feet


class PlanarQuadruped(LagrangianSystem):
    n_force = 2 * N_LEG      # 8 joint torques
    task_dim = 3             # base (x, z, pitch)
    allocator_dim = 4        # posture kp, kd, wrench ridge, contact sharpness
    dense_mass = True
    state_keys = ("q", "dq", "lam")
    n_feet = N_LEG

    def __init__(
        self,
        m_trunk: float = 6.0, l_trunk: float = 0.50,
        m_thigh: float = 0.80, l_thigh: float = 0.20,
        m_shank: float = 0.50, l_shank: float = 0.20,
        g: float = 9.81,
        hip_nom: float = 0.5, knee_nom: float = -1.0,
        tau_max: float = 40.0,
        mu: float = 0.8, v_slip: float = 0.02,
        contact_band: float = 0.01,             # a foot within this counts as loaded
        contact_compliance: float = 2.0e-6, contact_sharpness: float = 400.0,
        extra_constraints: tuple = (),
        learned_contact: bool = False,          # hybrid: constraint + learned penalty
        res_k_scale: float = 2.0e3, res_c_scale: float = 5.0e1,
        reset_noise: float = 0.02,
        dq_max: float = 40.0,
        dtype: torch.dtype = torch.float64, device: str = "cpu",
    ):
        self.dtype, self.device = dtype, device
        self.g = float(g)
        self.tau_max, self.dq_max = float(tau_max), float(dq_max)
        self.mu, self.v_slip = float(mu), float(v_slip)
        self.contact_band = float(contact_band)
        self.constraints = ConstraintStack(
            [GroundContact(compliance=contact_compliance, sharpness=contact_sharpness)]
            + list(extra_constraints))
        self.residual_dim = 4 if learned_contact else 0
        self.res_k_scale, self.res_c_scale = float(res_k_scale), float(res_c_scale)
        self.reset_noise = float(reset_noise)
        self.l_thigh, self.l_shank = float(l_thigh), float(l_shank)

        # --- bodies: [trunk, (thigh, shank) x 4] ---------------------------
        mass = [m_trunk] + [m_thigh, m_shank] * N_LEG
        inert = [m_trunk * l_trunk ** 2 / 12.0] + \
                [m_thigh * l_thigh ** 2 / 12.0, m_shank * l_shank ** 2 / 12.0] * N_LEG
        self.mass = self._t(mass)
        self.inertia = self._t(inert)
        self.total_mass = float(self.mass.sum())

        # hip attachment offset along the trunk: front pair +L/2, rear pair -L/2
        hx = [0.0] + sum(([s * l_trunk / 2.0] * 2 for s in (1, 1, -1, -1)), [])
        self.hip_x = self._t(hx + [s * l_trunk / 2.0 for s in (1, 1, -1, -1)])  # [N_PT]

        # --- Pm[j, m] = d(phi_j)/d(q_m), exact because phi is linear in q ---
        Pm = torch.zeros(N_ANG, N_Q, dtype=dtype, device=device)
        Pm[0, 2] = 1.0                                   # trunk angle = pitch
        for k in range(N_LEG):
            th_i, sh_i = 1 + 2 * k, 2 + 2 * k
            hip_q, knee_q = 3 + 2 * k, 4 + 2 * k
            Pm[th_i, 2] = Pm[th_i, hip_q] = 1.0
            Pm[sh_i, 2] = Pm[sh_i, hip_q] = Pm[sh_i, knee_q] = 1.0
        self.Pm = Pm

        # --- C[i, j] = length coefficient of u(phi_j) in point i's position -
        C = torch.zeros(N_PT, N_ANG, dtype=dtype, device=device)
        for k in range(N_LEG):
            th_i, sh_i, ft_i = 1 + 2 * k, 2 + 2 * k, N_BODY + k
            C[th_i, th_i] = l_thigh / 2.0                       # thigh COM
            C[sh_i, th_i] = l_thigh
            C[sh_i, sh_i] = l_shank / 2.0                       # shank COM
            C[ft_i, th_i] = l_thigh
            C[ft_i, sh_i] = l_shank                             # foot
        self.C = C

        self.n_rows = self.constraints.n_rows(self)
        self.q_nom = self._t([0.0, 0.0, 0.0] + [hip_nom, knee_nom] * N_LEG)
        self.stand_height = float(-self._points(self.q_nom[None])[0][0, N_BODY, 1])
        self._e_z = self._t([0.0, 1.0])

    # --- kinematics (one pass, shared by M, b and the constraint rows) -----
    def _kin(self, q: Tensor, dq: Tensor) -> dict:
        phi = torch.einsum("jm,...m->...j", self.Pm, q)
        dphi = torch.einsum("jm,...m->...j", self.Pm, dq)
        u = torch.stack([torch.sin(phi), -torch.cos(phi)], dim=-1)     # [..., N_ANG, 2]
        w = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)      # du/dphi
        th, dth = q[..., 2], dq[..., 2]
        rot = torch.stack([torch.cos(th), torch.sin(th)], dim=-1)[..., None, :]
        drot = torch.stack([-torch.sin(th), torch.cos(th)], dim=-1)[..., None, :]
        hip = self.hip_x[..., None] * rot                              # [..., N_PT, 2]
        dhip = self.hip_x[..., None] * drot

        pts = q[..., :2][..., None, :] + hip + torch.einsum("ij,...jd->...id", self.C, u)

        lead = q.shape[:-1]
        J = torch.zeros(lead + (N_PT, 2, N_Q), dtype=q.dtype, device=q.device)
        J[..., 0, 0] = 1.0
        J[..., 1, 1] = 1.0
        J = J + torch.einsum("ij,jm,...jd->...idm", self.C, self.Pm, w)
        J[..., 2] = J[..., 2] + dhip

        # Jdot qdot: the centripetal acceleration every point has at qddot = 0
        a = -(dth * dth)[..., None, None] * hip \
            - torch.einsum("ij,...j,...jd->...id", self.C, dphi * dphi, u)

        return {"phi": phi, "pts": pts, "J": J, "a": a,
                "Jf": J[..., N_BODY:, :, :], "pf": pts[..., N_BODY:, :],
                "af": a[..., N_BODY:, :]}

    def _points(self, q: Tensor):
        """Back-compatible accessor: point positions only."""
        k = self._kin(q, torch.zeros_like(q))
        return k["pts"], k["phi"], None

    def _jacobians(self, q: Tensor):
        k = self._kin(q, torch.zeros_like(q))
        return k["J"], k["phi"]

    # --- mass matrix and bias force ----------------------------------------
    def _M(self, q: Tensor, J: Tensor) -> Tensor:
        Jb = J[..., :N_BODY, :, :]
        M = torch.einsum("...idm,i,...idn->...mn", Jb, self.mass, Jb)
        return M + torch.einsum("im,i,in->mn", self.Pm, self.inertia, self.Pm)

    def _bias_from_kin(self, kin: dict) -> Tensor:
        """C(q,qdot) qdot + G(q): inverse dynamics evaluated at qddot = 0."""
        a = kin["a"][..., :N_BODY, :] + self.g * self._e_z
        Jb = kin["J"][..., :N_BODY, :, :]
        return torch.einsum("...idm,i,...id->...m", Jb, self.mass, a)

    def contact_residual(self, kin: dict, dq: Tensor, lam: Tensor,
                         params: Tensor) -> Tensor:
        """Learned penalty force correcting the constraint solve's model error.

        The constraint layer supplies the structural approximation -- the bulk of
        the ground reaction, exactly, as a multiplier.  But it is an approximation
        in four identifiable ways: finite compliance, Baumgarte position drift,
        friction lagged by one step, and no impact law at touchdown.  Those errors
        are systematic and state-dependent, which is exactly what a small learned
        penalty is good at absorbing -- and exactly what a multiplier cannot,
        since the multiplier is pinned by the constraint it enforces.

        So the two are complementary rather than redundant: the constraint gets
        the physics approximately right for free, and the penalty corrects the
        residual the approximation leaves behind.

        Parameterized as a normal spring-damper plus a friction correction,
        initialized at exactly zero so the hybrid model STARTS as the pure
        constraint model and can only learn a correction on top of it.
        """
        if params is None or self.residual_dim == 0:
            return torch.zeros_like(dq)
        k = (params[..., 0] ** 2) * self.res_k_scale
        c = (params[..., 1] ** 2) * self.res_c_scale
        dmu = params[..., 2]
        dv = params[..., 3] ** 2 + 1e-3

        Jf, pf = kin["Jf"], kin["pf"]
        vf = torch.einsum("...kdm,...m->...kd", Jf, dq)
        pen = (-pf[..., 1]).clamp_min(0.0)
        act = torch.sigmoid(-pf[..., 1] * 400.0)
        fz = act * (k[..., None] * pen - c[..., None] * vf[..., 1])
        fx = -dmu[..., None] * lam[..., :N_LEG].clamp_min(0.0) * torch.tanh(
            vf[..., 0] / dv[..., None])
        f = torch.stack([fx, fz], dim=-1)
        return torch.einsum("...kdm,...kd->...m", Jf, f)

    def _friction(self, kin: dict, dq: Tensor, lam: Tensor) -> Tensor:
        """Regularized Coulomb tangent, scaled by the previous step's NORMAL
        MULTIPLIER rather than by a penetration depth.

        Lagging the multiplier by one step is what keeps this a single linear
        solve; solving friction jointly with the normal force is a cone program,
        which is out of scope and which the search would gain nothing from.
        """
        Jf, vf = kin["Jf"], torch.einsum("...kdm,...m->...kd", kin["Jf"], dq)
        lam_n = lam[..., :N_LEG]                 # contact rows only; extra
        ft = -self.mu * lam_n.clamp_min(0.0) * torch.tanh(vf[..., 0] / self.v_slip)
        return torch.einsum("...km,...k->...m", Jf[..., 0, :], ft)

    # --- LagrangianSystem ---------------------------------------------------
    def reset(self, B: int, gen: torch.Generator) -> State:
        kw = dict(generator=gen, dtype=self.dtype, device=self.device)
        q = self.q_nom.expand(B, N_Q).clone()
        q[:, 1] = self.stand_height
        q = q + self.reset_noise * torch.randn(B, N_Q, **kw)
        dq = 0.5 * self.reset_noise * torch.randn(B, N_Q, **kw)
        lam = torch.full((B, self.n_rows), self.total_mass * self.g / N_LEG, **{
            k: v for k, v in kw.items() if k != "generator"})
        return {"q": q, "dq": dq, "lam": lam}

    def step(self, s: State, u: Tensor, dt: float, params: Tensor = None) -> State:
        """One constrained integration step.

        qddot and the constraint multipliers come out of a single KKT solve, so
        the ground reaction is the multiplier of the contact constraint rather
        than the output of a penalty spring.
        """
        q, dq, lam = s["q"], s["dq"], s["lam"]
        tau = u.clamp(-self.tau_max, self.tau_max)
        kin = self._kin(q, dq)
        M = self._M(q, kin["J"])

        gen_f = torch.zeros_like(q)
        gen_f[..., 3:] = tau                                    # the base is unactuated
        rhs = (gen_f - self._bias_from_kin(kin) + self._friction(kin, dq, lam)
               + self.contact_residual(kin, dq, lam, params))

        J, bias, act, eps, _ = self.constraints.assemble(self, q, dq, kin)
        ddq, lam_new = self.constraints.solve(M, rhs, J, bias, act, eps)

        # unilateral rows may push but not pull
        lam_new = torch.cat([lam_new[..., :N_LEG].clamp_min(0.0),
                             lam_new[..., N_LEG:]], dim=-1)
        dq_new = dq + dt * ddq
        return {"q": q + dt * dq_new, "dq": dq_new, "lam": lam_new}

    def alive(self, s: State) -> Tensor:
        q, dq = s["q"], s["dq"]
        finite = torch.isfinite(q).all(-1) & torch.isfinite(dq).all(-1)
        return (finite
                & (q[..., 1] > 0.10)                 # trunk off the floor
                & (q[..., 2].abs() < 1.0)            # not tipped over
                & (dq.norm(dim=-1) < self.dq_max))

    def inv_mass(self, s: State) -> Tensor:
        """The ACTUATED block of M(q)^-1, [..., 8, 8].

        The inverse inertia seen by a joint-torque perturbation.  Unlike the
        quadrotor's, this genuinely varies with configuration -- which is the
        whole reason this plant exists.
        """
        J, _ = self._jacobians(s["q"])
        Minv = torch.linalg.inv(self._M(s["q"], J))
        return Minv[..., 3:, 3:]

    def task_mass(self, s: State) -> Tensor:
        """Operational-space inertia of the base: (M^-1[0:3, 0:3])^-1."""
        J, _ = self._jacobians(s["q"])
        Minv = torch.linalg.inv(self._M(s["q"], J))
        return torch.linalg.inv(Minv[..., :3, :3])

    def gravity_force(self, s: State) -> Tensor:
        """The base wrench that holds the whole robot in static equilibrium.

        NOT simply (0, Mg, 0).  The base rows of the equations of motion read
        M qddot|base = (J^T lambda)|base - b|base, so equilibrium needs the
        contact wrench to equal the base-row bias -- which carries a pitch moment
        whenever the legs' mass sits off the trunk centreline.  At the nominal
        stance the legs lean forward, giving a real moment of a couple of newton
        metres that a constant (0, Mg, 0) feedforward simply ignores, leaving the
        trunk pitching at several rad/s^2 while the controller believes it is in
        equilibrium.

        Evaluated at zero velocity, so this is g(q) as the abstraction defines it
        and not a full inverse-dynamics term.
        """
        q = s["q"]
        return self._bias_from_kin(self._kin(q, torch.zeros_like(q)))[..., :3]

    def allocator_init(self) -> Tensor:
        # posture kp, kd (used squared), wrench ridge, contact band sharpness
        return self._t([2.5, 1.0, 0.5, 8.0])

    def potential_scale(self) -> float:
        """Closed-loop stiffness the shaped potential should start at, in task
        units.  m * omega_n^2 for an 11.2 kg robot at ~4 rad/s."""
        return 180.0

    def damping_scale(self) -> float:
        """2*zeta*sqrt(K*m) at zeta ~ 0.6 for an 11.2 kg robot at K = 180."""
        return 53.9

    def residual_init(self) -> Tensor:
        """Exactly zero: the hybrid starts as the pure constraint model."""
        return torch.zeros(self.residual_dim, dtype=self.dtype, device=self.device)

    def allocate(self, F_des: Tensor, s: State, phi: Tensor) -> Tensor:
        """Distribute a desired base wrench onto joint torques through the feet.

        This is the underactuation seam at its most literal: the base has no
        actuators, so every newton it feels arrives through a foot.  The desired
        wrench is shared among the feet currently loaded (least-norm, ridged),
        converted to joint torques by tau = -J_c^T f, and a joint-space posture
        term holds the unloaded legs.  Contact is weighted smoothly rather than
        switched, so the map stays jacrev-safe across touchdown.
        """
        q, dq = s["q"], s["dq"]
        kp = phi[..., 0] ** 2
        kd = phi[..., 1] ** 2
        ridge = phi[..., 2] ** 2 * 1e-2 + 1e-6
        sharp = phi[..., 3] ** 2 + 1e-3

        kin = self._kin(q, dq)
        Jf, pf = kin["Jf"], kin["pf"]
        # Smooth contact weight, centred on a BAND rather than on zero.
        # sigmoid(-h*k) equals 1/2 at h = 0 -- i.e. for a foot sitting exactly on
        # the ground, which is the normal standing case.  That halves the grasp
        # matrix and makes the least-norm solve return double the contact force
        # every foot actually needs, so commanding exactly gravity accelerates the
        # base upward at ~g.  Offsetting by the band puts a planted foot at ~1.
        c = torch.sigmoid((self.contact_band - pf[..., 1]) * sharp[..., None] * 20.0)

        r = pf - q[..., :2][..., None, :]                              # foot rel. base
        zero, one = torch.zeros_like(r[..., 0]), torch.ones_like(r[..., 0])
        # grasp matrix columns for (fx, fz) at each foot, weighted by contact
        gx = torch.stack([one, zero, -r[..., 1]], dim=-1) * c[..., None]
        gz = torch.stack([zero, one, r[..., 0]], dim=-1) * c[..., None]
        G = torch.cat([gx, gz], dim=-2)                                # [..., 8, 3]

        GGt = torch.einsum("...ki,...kj->...ij", G, G)
        eye3 = torch.eye(3, dtype=q.dtype, device=q.device)
        # Scale the ridge by the grasp matrix's own magnitude.  The force rows of
        # G G^T are O(n_feet) while the moment row is O(sum r^2) ~ 0.4 here, so an
        # ABSOLUTE ridge big enough to regularize the first swamps the second --
        # and the moment channel is the one that controls pitch.
        scale = (torch.diagonal(GGt, dim1=-2, dim2=-1).mean(-1) + 1e-9)
        lam = torch.linalg.solve(GGt + (ridge * scale)[..., None, None] * eye3,
                                 F_des.unsqueeze(-1)).squeeze(-1)
        fk = torch.einsum("...ki,...i->...k", G, lam)                  # [..., 8]
        f = torch.stack([fk[..., :N_LEG], fk[..., N_LEG:]], dim=-1)    # [..., 4, 2]

        tau_contact = -torch.einsum("...kdm,...kd->...m", Jf, f)[..., 3:]

        # Joint-space gravity/bias feedforward.  Static equilibrium of the
        # ACTUATED coordinates is tau = b - (J_c^T f), not just -(J_c^T f): the
        # legs have mass of their own, and without this term commanding exactly
        # gravity still leaves the base accelerating, because the torques hold up
        # the contact force but not the limbs producing it.  It is the same
        # feedforward every other plant gets through `gravity_force`, applied
        # where an articulated robot actually needs it.
        tau_ff = self._bias_from_kin(kin)[..., 3:]

        e_q = self.q_nom[3:] - q[..., 3:]
        tau_posture = kp[..., None] * e_q - kd[..., None] * dq[..., 3:]
        return (tau_ff + tau_contact + tau_posture).clamp(-self.tau_max, self.tau_max)

    # --- task space is the floating base ------------------------------------
    def task_position(self, s: State) -> Tensor:
        return s["q"][..., :3]

    def task_velocity(self, s: State) -> Tensor:
        return s["dq"][..., :3]

    def nominal_state(self, x: Tensor, v: Tensor) -> State:
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        v = torch.as_tensor(v, dtype=self.dtype, device=self.device)
        x, v = torch.broadcast_tensors(x, v)
        lead = x.shape[:-1]
        q = self.q_nom.expand(lead + (N_Q,)).clone()
        q = torch.cat([x, q[..., 3:]], dim=-1)
        dq = torch.cat([v, torch.zeros_like(q[..., 3:])], dim=-1)
        lam = torch.full(lead + (self.n_rows,), self.total_mass * self.g / N_LEG,
                         dtype=self.dtype, device=self.device)
        return {"q": q, "dq": dq, "lam": lam}

    # --- cost hooks ---------------------------------------------------------
    def effort(self, u: Tensor, s: State) -> Tensor:
        """tau^T M_act^-1 tau -- the base wrench has no feedforward to remove,
        because the base has no actuators."""
        Minv = self.inv_mass(s)
        return torch.einsum("...i,...ij,...j->...", u, Minv, u)

    def shaping_cost(self, s: State) -> Tensor:
        """Mild posture regularization only.

        Deliberately NOT a pitch penalty: pitch is one of the three task
        variables `BasePose` asks the robot to track, so penalizing it here would
        have the cost function fighting its own objective.
        """
        dev = s["q"][..., 3:] - self.q_nom[3:]
        return 0.05 * (dev * dev).sum(-1)

    def saturation(self, u: Tensor, s: State) -> Tensor:
        return (u.abs() >= self.tau_max - 1e-6).to(u.dtype).mean(dim=-1)

    def render_spec(self) -> dict:
        """Trunk box plus eight leg segments, drawn in the sagittal plane.

        The two hip attachment points carry two legs each -- the left/right pairs
        of the real robot projected onto the plane -- so the near pair is drawn
        lighter to keep them tellable apart.
        """
        bodies = [{"type": "box", "size": [0.50, 0.09], "shade": 1.0}]
        for k in range(N_LEG):
            shade = 1.0 if k % 2 == 0 else 0.62      # far pair / near pair
            bodies.append({"type": "segment", "size": [self.l_thigh, 0.045],
                           "shade": shade})
            bodies.append({"type": "segment", "size": [self.l_shank, 0.035],
                           "shade": shade})
        return {"dim": 2, "ground": 0.0, "scale": 0.75, "bodies": bodies}

    def render_poses(self, s: State) -> Tensor:
        """[..., 9, 3] as (x, z, angle).

        Link directions use u(phi) = (sin phi, -cos phi) -- phi = 0 points a leg
        straight down -- while the renderer draws along (cos a, sin a), so the
        leg angles are handed over as a = phi - pi/2.  The trunk's long axis is
        already its body x, so its angle is the pitch unchanged.
        """
        q = s["q"]
        kin = self._kin(q, torch.zeros_like(q))
        pts, phi = kin["pts"][..., :N_BODY, :], kin["phi"]
        ang = phi - 0.5 * torch.pi
        ang = torch.cat([q[..., 2:3], ang[..., 1:]], dim=-1)     # trunk keeps pitch
        return torch.cat([pts, ang[..., None]], dim=-1)

    def render_extras(self, s: State) -> dict:
        return {"feet": self.foot_positions(s), "contact": self.contact_forces(s)}

    def foot_positions(self, s: State) -> Tensor:
        """[..., 4, 2] -- for visualization and contact diagnostics."""
        return self._points(s["q"])[0][..., N_BODY:, :]

    def contact_forces(self, s: State) -> Tensor:
        """[..., 4] normal ground reaction -- literally the contact constraint's
        Lagrange multipliers, carried in the state."""
        return s["lam"][..., :N_LEG]

    def body_points(self, s: State) -> Tensor:
        return self._points(s["q"])[0]

    def describe(self) -> dict:
        d = super().describe()
        d.update(n_q=N_Q, n_legs=N_LEG, total_mass=self.total_mass,
                 stand_height=self.stand_height, tau_max=self.tau_max)
        return d
