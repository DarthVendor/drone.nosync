"""
Assembly render + mass-matrix analysis for the delivery quad.

The point of interest: with a gimballed payload the inertia tensor is a
function of the payload swing angle, so M(q) is genuinely state-dependent.
That is the case the whitening argument was originally scoped to require.
"""
import numpy as np, trimesh, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

CAD = "/mnt/user-data/outputs/cad_delivery"
OUT = "/mnt/user-data/outputs"

WB = 650.0
OFF = WB / 2 / np.sqrt(2)
PROP_R = 15 * 25.4 / 2
TUBE_OD = 25.0
CABLE_L = 0.13          # gimbal pivot to payload CoM, m


def shade(mesh, base, light=np.array([0.4, -0.7, 0.6])):
    light = light / np.linalg.norm(light)
    lum = np.clip(mesh.face_normals @ light, 0, 1) * 0.72 + 0.28
    return np.clip(lum[:, None] * np.array(base) + 0.08, 0, 1)


def cyl(p0, p1, r, sections=20):
    return trimesh.creation.cylinder(
        radius=r, segment=np.array([p0, p1]), sections=sections)


def assembly():
    """Return list of (mesh, colour, alpha) in mm."""
    hub = trimesh.load(f"{CAD}/hub_plate.stl")
    mount = trimesh.load(f"{CAD}/motor_mount.stl")
    ring = trimesh.load(f"{CAD}/gimbal_ring.stl")
    clamp = trimesh.load(f"{CAD}/arm_clamp.stl")
    parts = [(hub, (0.26, 0.38, 0.54), 1.0)]

    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        a = np.arctan2(sy, sx)
        d = np.array([np.cos(a), np.sin(a), 0.0])
        parts.append((cyl(d * 70, d * 322 + np.array([0, 0, 0]), TUBE_OD / 2),
                      (0.13, 0.13, 0.15), 1.0))          # carbon tube
        m = mount.copy()
        m.apply_translation([sx * OFF, sy * OFF, -8])
        parts.append((m, (0.26, 0.38, 0.54), 1.0))
        # motor body + prop disc
        parts.append((cyl([sx * OFF, sy * OFF, 27], [sx * OFF, sy * OFF, 55], 24),
                      (0.42, 0.42, 0.46), 1.0))
        parts.append((cyl([sx * OFF, sy * OFF, 58], [sx * OFF, sy * OFF, 61],
                          PROP_R, sections=48), (0.30, 0.34, 0.40), 0.28))
        c = clamp.copy()
        c.apply_transform(trimesh.transformations.rotation_matrix(a, [0, 0, 1]))
        c.apply_translation([85 * np.cos(a), 85 * np.sin(a), 4])
        parts.append((c, (0.30, 0.44, 0.60), 1.0))
        # landing gear
        parts.append((cyl([sx * 90, sy * 90, 0], [sx * 110, sy * 110, -150], 9),
                      (0.18, 0.20, 0.24), 1.0))

    r = ring.copy(); r.apply_translation([0, 0, -60])
    parts.append((r, (0.55, 0.42, 0.22), 1.0))
    # payload basket, hanging in the gimbal
    box = trimesh.creation.box(extents=[150, 150, 130])
    box.apply_translation([0, 0, -128])
    parts.append((box, (0.62, 0.47, 0.26), 0.45))
    # battery
    bat = trimesh.creation.box(extents=[150, 60, 45])
    bat.apply_translation([0, 0, 26])
    parts.append((bat, (0.20, 0.24, 0.30), 0.9))
    return parts


def inertia_vs_swing():
    """Body-frame inertia as the gimballed payload swings."""
    m_air, m_pay = 3.84, 1.50
    I_air = np.diag([0.105, 0.105, 0.190])       # airframe alone, kg m^2
    angles = np.linspace(0, 30, 31)
    out = []
    for deg in angles:
        th = np.radians(deg)
        # payload CoM offset from vehicle CoM: hangs below, swings in +x
        d = np.array([CABLE_L * np.sin(th), 0.0,
                      -0.06 - CABLE_L * np.cos(th)])
        I = I_air + m_pay * ((d @ d) * np.eye(3) - np.outer(d, d))
        out.append((deg, np.diag(I).copy(), I[0, 2]))
    return out


def render():
    parts = assembly()
    fig = plt.figure(figsize=(14, 10))
    views = [(20, -55, "Isometric — 650 mm wheelbase, 15\" props"),
             (89, -90, "Top — 79 mm prop tip clearance"),
             (2, -90, "Front — gimballed payload hangs clear of gear")]
    for i, (el, az, title) in enumerate(views):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        for mesh, col, al in parts:
            ax.add_collection3d(Poly3DCollection(
                mesh.vertices[mesh.faces], facecolors=shade(mesh, col),
                edgecolors="none", alpha=al))
        r = 350
        ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r * 0.6)
        ax.view_init(elev=el, azim=az); ax.set_box_aspect((1, 1, 0.8))
        ax.set_title(title, fontsize=10); ax.set_axis_off()

    ax = fig.add_subplot(2, 2, 4)
    data = inertia_vs_swing()
    deg = [d[0] for d in data]
    ax.plot(deg, [d[1][0] * 1e3 for d in data], lw=2, label="$I_{xx}$")
    ax.plot(deg, [d[1][1] * 1e3 for d in data], lw=2, ls="--", label="$I_{yy}$")
    ax.plot(deg, [abs(d[2]) * 1e3 for d in data], lw=2, label="$|I_{xz}|$ coupling")
    ax.set_xlabel("payload swing angle [deg]")
    ax.set_ylabel(r"inertia [$\times10^{-3}$ kg m$^2$]")
    ax.set_title("M(q) varies with payload swing", fontsize=10)
    ax.grid(alpha=0.3); ax.legend()

    fig.suptitle("Delivery quadrotor — 3.84 kg empty, 1.5 kg takeout, T/W 1.95",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/delivery_render.png", dpi=135)


if __name__ == "__main__":
    render()
    d = inertia_vs_swing()
    print(json.dumps({
        "Ixx_level": float(f"{d[0][1][0]:.4f}"),
        "Ixx_at_30deg": float(f"{d[-1][1][0]:.4f}"),
        "Ixz_coupling_at_30deg": float(f"{d[-1][2]:.4f}"),
        "pct_change_Ixx": round((d[-1][1][0] / d[0][1][0] - 1) * 100, 1),
    }, indent=2))
    print("rendered")
