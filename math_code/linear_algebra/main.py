import numpy as np
import plotly.graph_objects as go

# ==========================================
# CONFIGURATION & TOGGLES
# ==========================================
HELICAL_MODE = True     # Sun translates along the Z-axis
MAX_TIME = 1.0          # Simulate 1 Earth Year
TIME_STEPS = 4000       # High resolution needed for the fast-moving Moon
Z_VELOCITY = 5.0        # Speed of the Sun along the Z-axis

# ==========================================
# CELESTIAL DATA (Visually Scaled)
# ==========================================
# Distances are artificially scaled so the Moon's orbit is distinguishable.
EARTH_R = 10.0          # Earth distance from Sun
EARTH_W = 2 * np.pi     # Earth angular velocity (1 orbit per year)
EARTH_INC = 0.0         # Earth inclination

MOON_R = 1.5            # Artificially inflated Moon distance from Earth
MOON_W = 2 * np.pi * (365.25 / 27.3)  # ~13.37 orbits per Earth year
MOON_INC = 5.14         # Moon's orbital inclination relative to Earth's plane

def get_rotation_matrix_x(angle_degrees):
    """Generates a 3x3 rotation matrix for the X-axis (inclination)."""
    theta = np.radians(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ])

# ==========================================
# VECTOR SPACE SETUP
# ==========================================
t = np.linspace(0, MAX_TIME, TIME_STEPS)
fig = go.Figure()

# ------------------------------------------
# 1. SUN VECTOR
# ------------------------------------------
sun_pos = np.zeros((3, TIME_STEPS))
if HELICAL_MODE:
    sun_pos[2, :] = Z_VELOCITY * t

fig.add_trace(go.Scatter3d(
    x=sun_pos[0, :], y=sun_pos[1, :], z=sun_pos[2, :],
    mode='lines', line=dict(color='#FFD700', width=6),
    name='Sun Path', legendgroup='Sun'
))
fig.add_trace(go.Scatter3d(
    x=[sun_pos[0, -1]], y=[sun_pos[1, -1]], z=[sun_pos[2, -1]],
    mode='markers', marker=dict(size=20, color='#FFD700'),
    name='Sun', legendgroup='Sun'
))

# ------------------------------------------
# 2. EARTH VECTOR
# ------------------------------------------
# Unrotated relative vector (flat planar orbit)
x_e = EARTH_R * np.cos(EARTH_W * t)
y_e = EARTH_R * np.sin(EARTH_W * t)
z_e = np.zeros_like(t)
earth_rel = np.vstack((x_e, y_e, z_e))

# Apply Rotation (Inclination) and Translation (Sun's position)
R_earth = get_rotation_matrix_x(EARTH_INC)
earth_abs = sun_pos + (R_earth @ earth_rel)

fig.add_trace(go.Scatter3d(
    x=earth_abs[0, :], y=earth_abs[1, :], z=earth_abs[2, :],
    mode='lines', line=dict(color='#1E90FF', width=3),
    name='Earth Orbit', legendgroup='Earth', showlegend=False
))
fig.add_trace(go.Scatter3d(
    x=[earth_abs[0, -1]], y=[earth_abs[1, -1]], z=[earth_abs[2, -1]],
    mode='markers', marker=dict(size=10, color='#1E90FF'),
    name='Earth', legendgroup='Earth'
))

# ------------------------------------------
# 3. MOON VECTOR (The Kinematic Chain)
# ------------------------------------------
# Unrotated relative vector (Moon relative to Earth)
x_m = MOON_R * np.cos(MOON_W * t)
y_m = MOON_R * np.sin(MOON_W * t)
z_m = np.zeros_like(t)
moon_rel = np.vstack((x_m, y_m, z_m))

# Apply Rotation (Moon's 5.14 deg inclination)
R_moon = get_rotation_matrix_x(MOON_INC)
rotated_moon_rel = R_moon @ moon_rel

# ** Crucial Step: The Moon's absolute position is its rotated relative 
# vector added to the Earth's absolute vector. **
moon_abs = earth_abs + rotated_moon_rel

# Plot Moon Trail
fig.add_trace(go.Scatter3d(
    x=moon_abs[0, :], y=moon_abs[1, :], z=moon_abs[2, :],
    mode='lines', line=dict(color='#D3D3D3', width=1),
    name='Moon Orbit', legendgroup='Moon', showlegend=False
))
# Plot Final Moon Position
fig.add_trace(go.Scatter3d(
    x=[moon_abs[0, -1]], y=[moon_abs[1, -1]], z=[moon_abs[2, -1]],
    mode='markers', marker=dict(size=4, color='#D3D3D3'),
    name='Moon', legendgroup='Moon'
))

# ==========================================
# VISUALIZATION STYLING
# ==========================================
title_text = "3D Sun-Earth-Moon Vector Kinematics"
if HELICAL_MODE:
    title_text += " (Helical Mode)"

fig.update_layout(
    title=title_text,
    template="plotly_dark",
    scene=dict(
        aspectmode='data',
        xaxis=dict(title='X Space', showgrid=True, zeroline=False, gridcolor='#333333'),
        yaxis=dict(title='Y Space', showgrid=True, zeroline=False, gridcolor='#333333'),
        zaxis=dict(title='Z Space', showgrid=True, zeroline=False, gridcolor='#333333'),
        bgcolor='black'
    ),
    legend=dict(
        yanchor="top", y=0.9,
        xanchor="left", x=0.05,
        bgcolor="rgba(0,0,0,0.5)"
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()