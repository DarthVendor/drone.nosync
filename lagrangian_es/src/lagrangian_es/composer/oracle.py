"""An oracle composer for the city: a planner that hands out subgoals.

This is a TEACHER, not the behaviour.  It exists so the decisive question can be
asked before any transformer is trained: fed subgoals it can reach, does the
frozen low level traverse legs it otherwise cannot?  And so that the learned
composer has (context -> spec) pairs to warm-start from.  It decides only WHERE
the next subgoal sits; every gate and priority is left at identity, because
those are the learned layer's to set.

Planning is a distance field from the leg goal over an occupancy grid, cells
blocked where the scene's signed distance falls below a clearance margin.  The
subgoal is the point along the descent path whose path length from the vehicle
first reaches `reach` -- the low level's MEASURED competence radius, about 10 m
on this map at ~3% crash -- so the low level is only ever asked for a leg it is
known to fly.
"""
from __future__ import annotations

import heapq
from typing import Dict, Tuple

import torch
from torch import Tensor

from .base import COMPOSERS, Composer
from .spec import TaskSpec


class OracleSubgoal(Composer):
    kind = "oracle"

    def __init__(self, system, trainable, reach: float = 10.0, margin: float = 0.9,
                 cell: float = 0.5, every: int = 50, z_hold: bool = True,
                 los: bool = False, **kw):
        super().__init__(system, trainable)
        self.reach, self.margin, self.cell = float(reach), float(margin), float(cell)
        self.every, self.z_hold, self.los = int(every), bool(z_hold), bool(los)
        self._grid = None            # (x0, y0, nx, ny, free[nx, ny]) for the scene
        self._fields: Dict[Tuple[int, int], Tensor] = {}

    # --- the map ---------------------------------------------------------------
    def _occupancy(self, state):
        env = self.system.env
        span = float(env.span) if hasattr(env, "span") else 32.0
        x0 = y0 = -span - 2.0
        n = int((2 * span + 4.0) / self.cell) + 1
        xs = x0 + self.cell * torch.arange(n, dtype=torch.float64)
        gx, gy = torch.meshgrid(xs, xs, indexing="ij")
        z = torch.full_like(gx, 1.5)                       # cruise altitude
        pts = torch.stack([gx, gy, z], -1).reshape(-1, 3)
        # the scene is per episode in the state; sample against episode 0 and
        # tile the state to match the query batch
        one = {k: v[:1].expand(pts.shape[0], *v.shape[1:]) for k, v in state.items()
               if torch.is_tensor(v) and v.ndim and v.shape[0] >= 1}
        d = env.sdf(pts, one).reshape(n, n)
        self._grid = (x0, y0, n, d > self.margin)

    def _cell(self, p):
        x0, y0, n, _ = self._grid
        i = int(round((float(p[0]) - x0) / self.cell)); j = int(round((float(p[1]) - y0) / self.cell))
        return max(0, min(n - 1, i)), max(0, min(n - 1, j))

    def _field(self, goal_cell):
        """Dijkstra from the goal over free cells, 8-connected; cached per goal."""
        if goal_cell in self._fields:
            return self._fields[goal_cell]
        x0, y0, n, free = self._grid
        free = free.numpy()
        INF = float("inf")
        dist = torch.full((n, n), INF, dtype=torch.float64)
        gi, gj = goal_cell
        if not free[gi, gj]:                       # a goal inside the margin: nearest free cell
            best = None
            for r in range(1, 6):
                for di in range(-r, r + 1):
                    for dj in range(-r, r + 1):
                        a, b = gi + di, gj + dj
                        if 0 <= a < n and 0 <= b < n and free[a, b]:
                            if best is None or di * di + dj * dj < best[0]:
                                best = (di * di + dj * dj, a, b)
                if best: break
            if best: gi, gj = best[1], best[2]
        dist[gi, gj] = 0.0
        pq = [(0.0, gi, gj)]
        c = self.cell; diag = c * 2 ** 0.5
        while pq:
            d0, i, j = heapq.heappop(pq)
            if d0 > float(dist[i, j]):
                continue
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if not di and not dj:
                        continue
                    a, b = i + di, j + dj
                    if a < 0 or b < 0 or a >= n or b >= n or not free[a, b]:
                        continue
                    nd = d0 + (diag if di and dj else c)
                    if nd < float(dist[a, b]):
                        dist[a, b] = nd
                        heapq.heappush(pq, (nd, a, b))
        self._fields[goal_cell] = dist
        return dist

    def _subgoal(self, x, goal):
        """Walk the descent path from x toward the goal for `reach` metres."""
        gcell = self._cell(goal)
        dist = self._field(gcell)
        x0, y0, n, free = self._grid
        i, j = self._cell(x)
        if dist[i, j] == float("inf"):           # start inside the margin: hop to nearest free
            best = None
            for r in range(1, 8):
                for di in range(-r, r + 1):
                    for dj in range(-r, r + 1):
                        a, b = i + di, j + dj
                        if 0 <= a < n and 0 <= b < n and dist[a, b] < float("inf"):
                            if best is None or di * di + dj * dj < best[0]:
                                best = (di * di + dj * dj, a, b)
                if best: break
            if best is None:
                return goal
            i, j = best[1], best[2]
        # The tally starts at the snap offset and a step that would cross the
        # radius is not taken: `reach` is the low level's measured competence,
        # and a subgoal one metre past it is a leg it is not known to fly.
        walked = ((float(x[0]) - (x0 + self.cell * i)) ** 2
                  + (float(x[1]) - (y0 + self.cell * j)) ** 2) ** 0.5
        # `los`: stop at the last node the vehicle can SEE.  Measured on the
        # city at <= 20 m legs: a leg whose straight line crosses a building
        # is lost 91% of the time against 24% with line of sight, and a wider
        # margin only made it worse -- a path node round a corner is still an
        # occluded leg however short.  The subgoal the low level can execute is
        # the furthest visible one.
        last_vis = (i, j)
        while dist[i, j] > 0:
            nxt = None
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if 0 <= a < n and 0 <= b < n and (nxt is None or dist[a, b] < dist[nxt[0], nxt[1]]):
                        nxt = (a, b)
            step = self.cell * (2 ** 0.5 if (nxt[0] != i and nxt[1] != j) else 1.0)
            if walked + step > self.reach:
                break
            walked += step
            i, j = nxt
            if self.los:
                if self._visible(x, x0 + self.cell * i, y0 + self.cell * j, free, n, x0, y0):
                    last_vis = (i, j)
                else:
                    i, j = last_vis
                    break
        if dist[i, j] <= 0:
            # The walk reached the goal's cell: hand over the GOAL, not the
            # cell centre.  A centre can sit 0.35 m from the goal against an
            # arrival tolerance of 0.25 m, and a vehicle parked there never
            # registers -- measured as a 30% timeout rate on legs it flies
            # cleanly without the oracle.
            return goal
        px = x0 + self.cell * i; py = y0 + self.cell * j
        z = float(x[2]) if self.z_hold else float(goal[2])
        return torch.tensor([px, py, z], dtype=goal.dtype, device=goal.device)

    def _visible(self, x, px, py, free, n, x0, y0) -> bool:
        """Is the straight line from the vehicle to (px, py) clear of the margin?"""
        ax, ay = float(x[0]), float(x[1])
        L = ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
        k = max(2, int(L / (0.5 * self.cell)) + 1)
        for m in range(k + 1):
            t = m / k
            i = int(round((ax + t * (px - ax) - x0) / self.cell))
            j = int(round((ay + t * (py - ay) - y0) / self.cell))
            if i < 0 or j < 0 or i >= n or j >= n or not free[i, j]:
                return False
        return True

    # --- the interface ---------------------------------------------------------
    def reset(self, B):
        self._grid = None
        self._fields = {}

    def emit(self, ctx):
        x, goal, state = ctx["x"], ctx["goal"], ctx["state"]
        B = x.shape[0]
        if self._grid is None:
            self._occupancy(state)
        spec = TaskSpec.identity(B, self.d, self.n_terms, x.dtype, x.device)
        for b in range(B):
            if not bool(ctx["alive"][b]):
                continue
            sub = self._subgoal(x[b], goal[b])
            spec.delta[b] = sub - goal[b]
        return spec


COMPOSERS["oracle"] = OracleSubgoal
