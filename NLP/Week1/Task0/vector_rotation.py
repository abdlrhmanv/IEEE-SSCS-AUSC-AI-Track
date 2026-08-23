"""
Streamlit app: rotate a 3D vector around X, Y, or Z by angle θ.
Run: streamlit run vector_rotation.py
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — needed for 3D projection


def rotation_matrix_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])


def rotation_matrix_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ])


def rotation_matrix_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ])


ROTATIONS = {
    "X": rotation_matrix_x,
    "Y": rotation_matrix_y,
    "Z": rotation_matrix_z,
}


def rotate_vector(v: np.ndarray, axis: str, theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    R = ROTATIONS[axis](theta)
    return R @ v


st.set_page_config(page_title="3D Vector Rotation", layout="centered")
st.title("3D Vector Rotation")
st.write("Enter a 3D vector, choose a rotation axis (X / Y / Z), and set the angle θ.")

col1, col2, col3 = st.columns(3)
with col1:
    x = st.number_input("x", value=1.0, step=0.1, format="%.4f")
with col2:
    y = st.number_input("y", value=0.0, step=0.1, format="%.4f")
with col3:
    z = st.number_input("z", value=0.0, step=0.1, format="%.4f")

axis = st.selectbox("Rotation axis", options=["X", "Y", "Z"])
theta = st.slider("θ (degrees)", min_value=-180.0, max_value=180.0, value=45.0, step=1.0)

v = np.array([x, y, z], dtype=float)
v_rot = rotate_vector(v, axis, theta)

st.subheader("Result")
st.write(f"**Original vector:** `{v}`")
st.write(f"**Rotated vector (around {axis} by {theta:.1f}°):** `{v_rot}`")
st.write(f"**Rotation matrix R_{axis}:**")
st.code(np.array2string(ROTATIONS[axis](np.deg2rad(theta)), precision=4))

# 3D plot — right-handed XYZ with equal scale and clear axis labels
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")

lim = max(np.max(np.abs(v)), np.max(np.abs(v_rot)), 1.0) * 1.4

# Reference axes (right-handed: X right, Y depth, Z up)
for vec, color, name in [
    (np.array([lim, 0, 0]), "gray", "X"),
    (np.array([0, lim, 0]), "gray", "Y"),
    (np.array([0, 0, lim]), "gray", "Z"),
]:
    ax.plot([0, vec[0]], [0, vec[1]], [0, vec[2]], color=color, linewidth=1, linestyle="--")
    ax.text(vec[0] * 1.05, vec[1] * 1.05, vec[2] * 1.05, name, fontsize=12, fontweight="bold")

# Vectors as lines (more reliable than 3D quiver scaling)
ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color="C0", linewidth=2.5, label="Original")
ax.scatter([v[0]], [v[1]], [v[2]], color="C0", s=40)
ax.plot([0, v_rot[0]], [0, v_rot[1]], [0, v_rot[2]], color="C3", linewidth=2.5, label="Rotated")
ax.scatter([v_rot[0]], [v_rot[1]], [v_rot[2]], color="C3", s=40)

ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
ax.set_box_aspect((1, 1, 1))
ax.set_xticks([-lim, 0, lim])
ax.set_yticks([-lim, 0, lim])
ax.set_zticks([-lim, 0, lim])
ax.set_xlabel("X", labelpad=8)
ax.set_ylabel("Y", labelpad=8)
ax.set_zlabel("Z", labelpad=8)
ax.view_init(elev=20, azim=45)  # standard right-handed view
ax.legend(loc="upper left")
ax.set_title(f"Rotation around {axis}-axis by {theta:.0f}°")

st.pyplot(fig)
plt.close(fig)
