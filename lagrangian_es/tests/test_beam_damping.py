"""The per-beam dissipation head: one damping axis per beam, weights learned.

Structure of the hand-designed `RangeDamper` -- R = 1/2 sum_i k_i h(-J_i.v)^2 --
with k_i produced by a small net on beam i's own range, shared across beams,
plus a learned isotropic floor.  What these pin: it can only remove energy,
whatever the weights; it is one-sided; its force lies along the beams that see
something; and its hand-written gradient matches autograd.
"""
import torch

from lagrangian_es.trainables.learned import LearnedShaping
from lagrangian_es.widen import to_beam_damping

D, N = 3, 24


def _term():
    return LearnedShaping(D, n_obs=N, hidden=16, damp_mode="beams")


def _inputs(B=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = 2.0 * torch.randn(B, D, generator=g, dtype=torch.float64)
    J = torch.randn(B, N, D, generator=g, dtype=torch.float64)
    J = J / J.norm(dim=-1, keepdim=True)
    rng = 6.0 * torch.rand(B, N, generator=g, dtype=torch.float64)
    return g, v, {"range": rng, "range/J": J}


def _R(term, theta, z, v, J):
    """The Rayleigh function itself, written out, for the gradient check."""
    H = term.BEAM_H
    wd = term._p(theta, "Wd", (3 * H + 1,))
    W1, b1, W2, b2 = wd[:H], wd[H:2 * H], wd[2 * H:3 * H], wd[3 * H:]
    s0 = torch.nn.functional.softplus(term._p(theta, "bd", (1,)))
    w = torch.nn.functional.softplus((torch.tanh(z.unsqueeze(-1) * W1 + b1) * W2).sum(-1) + b2)
    c = -(J * v.unsqueeze(-2)).sum(-1)
    h = 0.5 * (c + torch.sqrt(c * c + term.HINGE ** 2))
    return 0.5 * s0 * (v * v).sum(-1) + 0.5 * (w * h * h).sum(-1)


def test_slot_count_is_small():
    assert _term().dim == LearnedShaping(D, n_obs=N, hidden=16, damp_mode="full").dim - 225 + 20


def test_it_can_only_remove_energy():
    term = _term()
    g, v, obs = _inputs()
    for seed in range(5):
        theta = 2.0 * torch.randn(term.dim, generator=g, dtype=torch.float64)
        d = term._dRdv_beams(theta, term._read(obs), v, obs)
        assert float((d * v).sum(-1).min()) >= 0.0


def test_hand_written_gradient_matches_autograd():
    term = _term()
    g, v, obs = _inputs()
    theta = 0.7 * torch.randn(term.dim, generator=g, dtype=torch.float64)
    z = term._read(obs)
    vv = v.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(_R(term, theta, z, vv, obs["range/J"]).sum(), vv)
    got = term._dRdv_beams(theta, z, v, obs)
    assert torch.allclose(got, grad, atol=1e-10)


def test_receding_leaves_only_the_floor():
    """One-sided: moving away from every beam, the beam terms vanish up to the
    hinge's smoothing tail, which is O(eps^2 / |c|)."""
    term = _term()
    g, v, obs = _inputs()
    theta = torch.randn(term.dim, generator=g, dtype=torch.float64)
    vhat = v / v.norm(dim=-1, keepdim=True)
    obs["range/J"] = vhat.unsqueeze(-2).expand(-1, N, -1).clone()   # J along v
    d = term._dRdv_beams(theta, term._read(obs), v, obs)
    s0 = torch.nn.functional.softplus(term._p(theta, "bd", (1,)))
    tail = (d - s0 * v).norm(dim=-1)
    assert float(tail.max()) < 5e-2 * term.HINGE ** 2 / term.HINGE * 10


def test_closing_on_one_beam_pushes_back_along_it():
    term = _term()
    g, v, obs = _inputs(B=1)
    theta = torch.randn(term.dim, generator=g, dtype=torch.float64)
    J = torch.zeros(1, N, D, dtype=torch.float64)
    J[0, 0] = torch.tensor([1.0, 0.0, 0.0])           # beam 0: range grows toward +x,
                                                      # so it sees something at -x
    for i in range(1, N):
        J[0, i] = torch.tensor([-1.0, 0.0, 0.0])      # the rest see something at +x
    v = torch.tensor([[-3.0, 0.0, 0.0]], dtype=torch.float64)   # closing on beam 0,
                                                                 # receding from the rest
    obs["range/J"] = J
    d = term._dRdv_beams(theta, term._read(obs), v, obs)
    s0 = torch.nn.functional.softplus(term._p(theta, "bd", (1,)))
    beam_part = d - s0 * v
    # the bracket is subtracted from the force, so a NEGATIVE x here is a push
    # toward +x -- away from what beam 0 sees.  The receding beams contribute only
    # the hinge's O(eps^2) tail and cannot flip it.
    assert float(beam_part[0, 0]) < 0
    assert abs(float(beam_part[0, 1])) < 1e-12 and abs(float(beam_part[0, 2])) < 1e-12


def test_no_jacobian_means_floor_only():
    term = _term()
    g, v, obs = _inputs()
    theta = torch.randn(term.dim, generator=g, dtype=torch.float64)
    d = term._dRdv_beams(theta, term._read({"range": obs["range"]}), v,
                         {"range": obs["range"]})
    s0 = torch.nn.functional.softplus(term._p(theta, "bd", (1,)))
    assert torch.allclose(d, s0 * v)
    assert torch.allclose(term.damping(theta), s0 * torch.eye(D, dtype=torch.float64))


def test_transfer_keeps_the_trunk_and_sets_the_floor():
    full = LearnedShaping(D, n_obs=N, hidden=16, damp_mode="full")
    beams = _term()
    g = torch.Generator().manual_seed(3)
    theta = torch.cat([0.3 * torch.randn(full.dim, generator=g, dtype=torch.float64),
                       torch.arange(6, dtype=torch.float64)])
    got = to_beam_damping(theta, full, beams, floor=2.5, gen=g)
    assert got.shape[-1] == beams.dim + 6
    e = torch.randn(5, D, generator=g, dtype=torch.float64)
    obs = {"range": torch.rand(5, N, generator=g, dtype=torch.float64)}
    assert torch.equal(full.potential(theta[:full.dim], e, e, e, obs),
                       beams.potential(got[:beams.dim], e, e, e, obs))
    assert torch.allclose(beams.damping(got[:beams.dim]),
                          2.5 * torch.eye(D, dtype=torch.float64), atol=1e-9)
    assert torch.equal(got[beams.dim:], theta[full.dim:])
