"""
Sim export for the delivery quad.

STEP files do not import into Gazebo / PyBullet / Isaac / MuJoCo. This produces
what a simulator actually needs:

  - one merged visual mesh, in METRES, origin at the airframe centre of mass
  - convex collision primitives (never the full mesh -- it is 100x slower)
  - inertia tensors computed from the mass budget, not guessed
  - URDF (Gazebo / PyBullet / Isaac) and MJCF (MuJoCo)
  - 4 rotor frames + the 2-DOF cardan as real joints, so the payload swing
    dynamics are simulated rather than baked into a constant inertia
"""
import numpy as np, trimesh, os, json

CAD = "/mnt/user-data/outputs/cad_delivery"
OUT = "/mnt/user-data/outputs/sim"
os.makedirs(OUT, exist_ok=True)

WB = 0.650
OFF = WB / 2 / np.sqrt(2)          # 0.2298 m
PROP_R = 15 * 0.0254 / 2           # 0.1905 m
TUBE_OD = 0.025

# ---- mass budget (kg, positions in m, origin = hub plate centre, z up) ----
BODIES = [
    # name,                 mass,  x,      y,      z,     kind
    ("hub+electronics+PDB", 0.850,  0.0,    0.0,   0.010, "box"),
    ("battery",             1.200,  0.0,    0.0,   0.026, "box"),
    ("motor+esc+prop FL",   0.240,  OFF,    OFF,   0.055, "pt"),
    ("motor+esc+prop FR",   0.240, -OFF,    OFF,   0.055, "pt"),
    ("motor+esc+prop RR",   0.240, -OFF,   -OFF,   0.055, "pt"),
    ("motor+esc+prop RL",   0.240,  OFF,   -OFF,   0.055, "pt"),
    ("arm FL",              0.090,  OFF/2,  OFF/2, 0.004, "rod"),
    ("arm FR",              0.090, -OFF/2,  OFF/2, 0.004, "rod"),
    ("arm RR",              0.090, -OFF/2, -OFF/2, 0.004, "rod"),
    ("arm RL",              0.090,  OFF/2, -OFF/2, 0.004, "rod"),
    ("landing gear",        0.180,  0.0,    0.0,  -0.200, "box"),
    ("cardan + bracket",    0.300,  0.0,    0.0,  -0.060, "box"),
    ("winch + tether",      0.220,  0.0,    0.0,  -0.020, "box"),
]
PAYLOAD_MASS = 1.500               # box + takeout
BAY = (0.288, 0.248, 0.268)        # external box, m
CABLE = 0.130                      # cardan pivot -> payload CoM

ARM_L = 0.252                      # tube length, for rod inertia


def parallel_axis(m, c, ref):
    d = np.asarray(c) - np.asarray(ref)
    return m * ((d @ d) * np.eye(3) - np.outer(d, d))


def airframe_inertia():
    """Everything above the cardan pivot. Returns mass, CoM, I about CoM."""
    M = sum(b[1] for b in BODIES)
    com = sum(b[1] * np.array(b[2:5]) for b in BODIES) / M
    I = np.zeros((3, 3))
    for name, m, x, y, z, kind in BODIES:
        c = np.array([x, y, z])
        if kind == "box":                       # crude self-inertia
            e = np.array([0.18, 0.18, 0.05])
            I += np.diag(m / 12 * np.array([e[1]**2 + e[2]**2,
                                            e[0]**2 + e[2]**2,
                                            e[0]**2 + e[1]**2]))
        elif kind == "rod":                     # thin rod along its arm
            I += np.diag([m * ARM_L**2 / 12] * 2 + [m * ARM_L**2 / 12])
        I += parallel_axis(m, c, com)
    return M, com, I


def payload_inertia():
    m = PAYLOAD_MASS
    e = np.array(BAY)
    I = np.diag(m / 12 * np.array([e[1]**2 + e[2]**2,
                                   e[0]**2 + e[2]**2,
                                   e[0]**2 + e[1]**2]))
    return m, I


def build_mesh(com):
    """Merge printed parts + tubes + motors into one visual mesh, in metres."""
    parts = []
    hub = trimesh.load(f"{CAD}/hub_plate.stl"); parts.append(hub)
    mount = trimesh.load(f"{CAD}/motor_mount.stl")
    clamp = trimesh.load(f"{CAD}/arm_clamp.stl")
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        a = np.arctan2(sy, sx); d = np.array([np.cos(a), np.sin(a), 0.0])
        parts.append(trimesh.creation.cylinder(
            radius=12.5, segment=np.array([d * 70, d * 322])))
        m = mount.copy(); m.apply_translation([sx * OFF * 1000, sy * OFF * 1000, -8])
        parts.append(m)
        parts.append(trimesh.creation.cylinder(
            radius=24, segment=np.array([[sx * OFF * 1000, sy * OFF * 1000, 27],
                                         [sx * OFF * 1000, sy * OFF * 1000, 55]])))
        c = clamp.copy()
        c.apply_transform(trimesh.transformations.rotation_matrix(a, [0, 0, 1]))
        c.apply_translation([85 * np.cos(a), 85 * np.sin(a), 4])
        parts.append(c)
        parts.append(trimesh.creation.cylinder(
            radius=9, segment=np.array([[sx * 90, sy * 90, 0],
                                        [sx * 110, sy * 110, -150]])))
    for p in (f"{CAD}/cardan_outer.stl", f"{CAD}/tether_guide.stl"):
        m = trimesh.load(p); m.apply_translation([0, 0, -20]); parts.append(m)

    merged = trimesh.util.concatenate(parts)
    merged.apply_scale(0.001)                      # mm -> m
    merged.apply_translation(-com)                 # origin at CoM
    return merged


URDF = """<?xml version="1.0"?>
<!-- Delivery quadrotor, 650 mm wheelbase, 15in props, 1.5 kg gimballed payload.
     Units: metres, kg. base_link origin is at the airframe centre of mass.
     The cardan is modelled as two revolute joints so payload swing is
     simulated; do NOT collapse it into a fixed inertia. -->
<robot name="delivery_quad">

  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="{M:.4f}"/>
      <inertia ixx="{ixx:.6f}" iyy="{iyy:.6f}" izz="{izz:.6f}"
               ixy="{ixy:.6f}" ixz="{ixz:.6f}" iyz="{iyz:.6f}"/>
    </inertial>
    <visual>
      <geometry><mesh filename="package://delivery_quad/meshes/airframe.stl"/></geometry>
      <material name="carbon"><color rgba="0.20 0.22 0.26 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 {hub_z:.4f}"/>
      <geometry><box size="0.22 0.22 0.09"/></geometry>
    </collision>
{arm_collisions}  </link>

{rotors}
  <!-- passive cardan: gravity levels the payload, no actuator -->
  <link name="cardan_ring">
    <inertial><mass value="0.060"/>
      <inertia ixx="1e-4" iyy="1e-4" izz="1.8e-4" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="cardan_roll" type="revolute">
    <parent link="base_link"/><child link="cardan_ring"/>
    <origin xyz="0 0 {pivot_z:.4f}"/><axis xyz="1 0 0"/>
    <limit lower="-0.436" upper="0.436" effort="0" velocity="10"/>
    <dynamics damping="0.02" friction="0.01"/>
  </joint>

  <link name="payload">
    <inertial><origin xyz="0 0 {ncable:.4f}"/>
      <mass value="{pm:.4f}"/>
      <inertia ixx="{pxx:.6f}" iyy="{pyy:.6f}" izz="{pzz:.6f}"
               ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual><origin xyz="0 0 {ncable:.4f}"/>
      <geometry><box size="{bx} {by} {bz}"/></geometry>
      <material name="bay"><color rgba="0.62 0.47 0.26 1"/></material>
    </visual>
    <collision><origin xyz="0 0 {ncable:.4f}"/>
      <geometry><box size="{bx} {by} {bz}"/></geometry>
    </collision>
  </link>
  <joint name="cardan_pitch" type="revolute">
    <parent link="cardan_ring"/><child link="payload"/>
    <origin xyz="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-0.436" upper="0.436" effort="0" velocity="10"/>
    <dynamics damping="0.02" friction="0.01"/>
  </joint>
</robot>
"""

ROTOR = """  <link name="rotor_{i}">
    <inertial><mass value="0.030"/>
      <inertia ixx="1.0e-4" iyy="1.0e-4" izz="2.0e-4" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual><geometry><cylinder radius="{r:.4f}" length="0.004"/></geometry>
      <material name="prop"><color rgba="0.3 0.34 0.4 0.35"/></material></visual>
  </link>
  <joint name="rotor_{i}_joint" type="continuous">
    <parent link="base_link"/><child link="rotor_{i}"/>
    <origin xyz="{x:.4f} {y:.4f} {z:.4f}"/><axis xyz="0 0 1"/>
  </joint>
"""


def write_urdf(M, I, com):
    pm, PI = payload_inertia()
    rotors = "".join(
        ROTOR.format(i=i, r=PROP_R, x=sx * OFF - com[0],
                     y=sy * OFF - com[1], z=0.060 - com[2])
        for i, (sx, sy) in enumerate([(1, 1), (-1, 1), (-1, -1), (1, -1)]))
    arm_col = ""
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        a = np.arctan2(sy, sx)
        arm_col += (f'    <collision>\n'
                    f'      <origin xyz="{0.196*np.cos(a)-com[0]:.4f} '
                    f'{0.196*np.sin(a)-com[1]:.4f} {0.004-com[2]:.4f}" '
                    f'rpy="0 1.5708 {a:.4f}"/>\n'
                    f'      <geometry><cylinder radius="0.0125" length="0.252"/>'
                    f'</geometry>\n    </collision>\n')
    txt = URDF.format(
        M=M, ixx=I[0, 0], iyy=I[1, 1], izz=I[2, 2],
        ixy=I[0, 1], ixz=I[0, 2], iyz=I[1, 2],
        hub_z=0.010 - com[2], pivot_z=-0.060 - com[2],
        rotors=rotors, arm_collisions=arm_col,
        pm=pm, pxx=PI[0, 0], pyy=PI[1, 1], pzz=PI[2, 2],
        bx=BAY[0], by=BAY[1], bz=BAY[2], ncable=-CABLE)
    open(f"{OUT}/delivery_quad.urdf", "w").write(txt)


MJCF = """<mujoco model="delivery_quad">
  <compiler angle="radian" meshdir="meshes"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  <asset><mesh name="airframe" file="airframe.stl"/></asset>
  <worldbody>
    <body name="base_link" pos="0 0 0.5">
      <freejoint/>
      <inertial pos="0 0 0" mass="{M:.4f}" fullinertia="{ixx:.6f} {iyy:.6f} {izz:.6f} {ixy:.6f} {ixz:.6f} {iyz:.6f}"/>
      <geom type="mesh" mesh="airframe" contype="0" conaffinity="0" rgba="0.2 0.22 0.26 1"/>
      <geom type="box" size="0.11 0.11 0.045" pos="0 0 {hub_z:.4f}" rgba="0.2 0.22 0.26 0"/>
{sites}      <body name="cardan_ring" pos="0 0 {pivot_z:.4f}">
        <joint name="cardan_roll" type="hinge" axis="1 0 0" range="-0.436 0.436" damping="0.02"/>
        <inertial pos="0 0 0" mass="0.06" diaginertia="1e-4 1e-4 1.8e-4"/>
        <body name="payload" pos="0 0 0">
          <joint name="cardan_pitch" type="hinge" axis="0 1 0" range="-0.436 0.436" damping="0.02"/>
          <inertial pos="0 0 {ncable:.4f}" mass="{pm:.4f}" diaginertia="{pxx:.6f} {pyy:.6f} {pzz:.6f}"/>
          <geom type="box" size="{hx:.4f} {hy:.4f} {hz:.4f}" pos="0 0 {ncable:.4f}" rgba="0.62 0.47 0.26 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
{acts}  </actuator>
</mujoco>
"""


def write_mjcf(M, I, com):
    pm, PI = payload_inertia()
    sites, acts = "", ""
    for i, (sx, sy) in enumerate([(1, 1), (-1, 1), (-1, -1), (1, -1)]):
        sites += (f'      <site name="rotor{i}" pos="{sx*OFF-com[0]:.4f} '
                  f'{sy*OFF-com[1]:.4f} {0.060-com[2]:.4f}" size="0.01"/>\n')
        acts += (f'    <motor name="thrust{i}" site="rotor{i}" '
                 f'gear="0 0 1 0 0 {0.016 if (sx*sy)>0 else -0.016}" '
                 f'ctrlrange="0 26"/>\n')
    txt = MJCF.format(
        M=M, ixx=I[0, 0], iyy=I[1, 1], izz=I[2, 2],
        ixy=I[0, 1], ixz=I[0, 2], iyz=I[1, 2],
        hub_z=0.010 - com[2], pivot_z=-0.060 - com[2], ncable=-CABLE,
        pm=pm, pxx=PI[0, 0], pyy=PI[1, 1], pzz=PI[2, 2],
        hx=BAY[0] / 2, hy=BAY[1] / 2, hz=BAY[2] / 2, sites=sites, acts=acts)
    open(f"{OUT}/delivery_quad.xml", "w").write(txt)


if __name__ == "__main__":
    M, com, I = airframe_inertia()
    mesh = build_mesh(com)
    os.makedirs(f"{OUT}/meshes", exist_ok=True)
    mesh.export(f"{OUT}/meshes/airframe.stl")
    mesh.export(f"{OUT}/meshes/airframe.obj")
    write_urdf(M, I, com)
    write_mjcf(M, I, com)

    pm, PI = payload_inertia()
    report = {
        "airframe_mass_kg": round(M, 4),
        "payload_mass_kg": pm,
        "AUW_kg": round(M + pm, 3),
        "com_offset_from_hub_centre_m": [round(v, 5) for v in com],
        "airframe_inertia_diag": [float(f"{I[i,i]:.5f}") for i in range(3)],
        "max_offdiag": float(f"{np.abs(I - np.diag(np.diag(I))).max():.2e}"),
        "mesh_triangles": len(mesh.faces),
        "mesh_units": "metres, origin at airframe CoM",
        "mesh_bounds_m": [[round(v, 3) for v in mesh.bounds[0]],
                          [round(v, 3) for v in mesh.bounds[1]]],
    }
    print(json.dumps(report, indent=2))
    json.dump(report, open(f"{OUT}/properties.json", "w"), indent=2)
