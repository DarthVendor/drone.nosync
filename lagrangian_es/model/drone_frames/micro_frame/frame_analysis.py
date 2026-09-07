"""
Mass properties + render for the printed quad frame.

Outputs the PlantCfg numbers directly: total mass and the principal inertias
that the physics-whitened metric consumes as M^-1. These must come from the
real airframe, not from the prototype's placeholder constants.
"""
import numpy as np, trimesh, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

CAD = "/mnt/user-data/outputs/cad"
OUT = "/mnt/user-data/outputs"

PETG_DENSITY = 1270.0        # kg/m^3 solid
FILL_FACTOR  = 0.85          # thin walls -> mostly perimeter, near solid
OFF = 110.0 / 2 / np.sqrt(2) # motor x,y offset, mm

# (name, mass_g, x, y, z) in mm, frame origin at plate centre, plate bottom z=0
COMPONENTS = [
    ("motor FL",      5.2,  OFF,  OFF,  5.0),
    ("motor FR",      5.2, -OFF,  OFF,  5.0),
    ("motor RR",      5.2, -OFF, -OFF,  5.0),
    ("motor RL",      5.2,  OFF, -OFF,  5.0),
    ("prop FL",       0.9,  OFF,  OFF, 12.0),
    ("prop FR",       0.9, -OFF,  OFF, 12.0),
    ("prop RR",       0.9, -OFF, -OFF, 12.0),
    ("prop RL",       0.9,  OFF, -OFF, 12.0),
    ("Feather F405",  5.5,  0.0,  0.0,  8.0),
    ("DRV8833 #1",    2.5, 14.0, 15.0,  3.6),
    ("DRV8833 #2",    2.5,-14.0, 15.0,  3.6),
    ("LSM6DSOX",      1.2,  0.0,-14.0,  3.6),
    ("VL53L1X down",  1.5,  0.0,  0.0, -2.0),
    ("VL53L1X fwd",   1.5, 26.0,  0.0,  3.6),
    ("VL53L1X left",  1.5,  0.0, 20.0,  3.6),
    ("VL53L1X right", 1.5,  0.0,-20.0,  3.6),
    ("PMW3901 flow",  2.0, -8.0,  0.0, -2.0),
    ("LiPo 1S 500",  11.0,  0.0,  0.0, -6.0),
    ("wiring",        6.0,  0.0,  0.0,  0.0),
    ("hardware",      3.0,  0.0,  0.0,  2.0),
    ("prop guard x4", 4.8,  0.0,  0.0,  6.0),   # TPU rings, symmetric
    ("legs x4",       3.2,  0.0,  0.0, -8.0),
]


def mesh_properties(path, density, fill):
    m = trimesh.load(path)
    m.apply_scale(0.001)                       # mm -> m
    m.density = density * fill
    return m, m.mass, m.center_mass, m.moment_inertia


def parallel_axis(mass, com, ref):
    d = np.asarray(com) - np.asarray(ref)
    return mass * ((d @ d) * np.eye(3) - np.outer(d, d))


def analyse():
    frame, m_frame, com_frame, I_frame = mesh_properties(
        f"{CAD}/quad_frame.stl", PETG_DENSITY, FILL_FACTOR)

    items = [("printed frame", m_frame, np.array(com_frame))]
    for name, g, x, y, z in COMPONENTS:
        items.append((name, g / 1000.0, np.array([x, y, z]) / 1000.0))

    M = sum(m for _, m, _ in items)
    com = sum(m * c for _, m, c in items) / M

    # frame's own tensor about its centroid, then shift everything to system CoM
    I = I_frame + parallel_axis(m_frame, com_frame, com)
    for name, m, c in items[1:]:
        I = I + parallel_axis(m, c, com)       # point masses: no self-inertia

    eig, _ = np.linalg.eigh(I)

    # thrust: 8520 coreless with 65 mm prop, ~32 g static thrust each
    thrust_per_motor_N = 0.032 * 9.81
    twr = 4 * thrust_per_motor_N / (M * 9.81)

    return dict(
        frame_mass_g=round(m_frame * 1000, 1),
        frame_volume_cm3=round(frame.volume * 1e6, 1),
        total_mass_g=round(M * 1000, 1),
        com_mm=[round(v * 1000, 2) for v in com],
        inertia_diag=[float(f"{v:.3e}") for v in np.diag(I)],
        principal_inertia=[float(f"{v:.3e}") for v in eig],
        thrust_to_weight=round(twr, 2),
        plantcfg=dict(mass=round(M, 4),
                      inertia=[float(f"{v:.2e}") for v in np.diag(I)],
                      thrust_to_weight=round(twr, 2)),
    )


# ------------------------------------------------------------------ render
def shade(mesh, light=np.array([0.4, -0.7, 0.6])):
    light = light / np.linalg.norm(light)
    n = mesh.face_normals
    lum = np.clip(n @ light, 0, 1) * 0.75 + 0.25
    base = np.array([0.29, 0.42, 0.58])
    return np.clip(lum[:, None] * base + 0.10, 0, 1)


def render():
    fig = plt.figure(figsize=(12, 8.5))
    frame = trimesh.load(f"{CAD}/quad_frame.stl")
    guard = trimesh.load(f"{CAD}/prop_guard.stl")
    leg = trimesh.load(f"{CAD}/landing_leg.stl")

    views = [(24, -58, "Isometric"), (89, -90, "Top"), (4, -90, "Front")]
    for i, (el, az, title) in enumerate(views):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        tri = frame.vertices[frame.faces]
        ax.add_collection3d(Poly3DCollection(
            tri, facecolors=shade(frame), edgecolors="none"))
        r = 46
        ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r * .42, r * .42)
        ax.view_init(elev=el, azim=az)
        ax.set_box_aspect((1, 1, 0.42))
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()

    # assembly view: frame + guards + legs + component boxes
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    ax.add_collection3d(Poly3DCollection(
        frame.vertices[frame.faces], facecolors=shade(frame), edgecolors="none"))
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        g = guard.copy()
        ang = np.arctan2(sy, sx)          # tab points back toward the hub
        g.apply_transform(trimesh.transformations.rotation_matrix(ang, [0, 0, 1]))
        g.apply_translation([sx * OFF, sy * OFF, 4.0])
        ax.add_collection3d(Poly3DCollection(
            g.vertices[g.faces], facecolors=shade(g) * 0.8 + 0.12,
            edgecolors="none", alpha=0.5))
        l = leg.copy()
        l.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        l.apply_translation([sx * OFF, sy * OFF, 0])
        ax.add_collection3d(Poly3DCollection(
            l.vertices[l.faces], facecolors=shade(l) * 0.7 + 0.2, edgecolors="none"))
    r = 62
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r * .42, r * .42)
    ax.view_init(elev=22, azim=-58)
    ax.set_box_aspect((1, 1, 0.42))
    ax.set_title("Assembly: frame + TPU guards + legs", fontsize=11)
    ax.set_axis_off()

    fig.suptitle("Coreless quadrotor frame — 110 mm wheelbase, 65 mm props",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/frame_render.png", dpi=140)


if __name__ == "__main__":
    res = analyse()
    print(json.dumps(res, indent=2))
    render()
    print("rendered")
