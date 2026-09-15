import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Link lengths
L1, L2, L3 = 1, 1, 1

# Initial joint angles at theta* (we ignore theta1 since it's 0 and stationary)
theta2_0 = np.pi / 4
theta3_0 = np.pi / 4
theta4_0 = np.pi / 4

# Null space velocity vector (scaled down for small movements)
# The Jacobian is a first-order (linear) approximation, so the null space 
# perfectly pins the end-effector only for instantaneous/small velocities!
v2 = 1.0
v3 = -(1 + np.sqrt(2))
v4 = (1 + np.sqrt(2))

# Setup the figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-1, 2)
ax.set_ylim(-1, 3)
ax.set_aspect('equal')
ax.set_xlabel('x1 (Horizontal)')
ax.set_ylabel('x3 (Vertical)')
ax.set_title('Null Space Self-Motion: Tip is Pinned, Elbows Swing')
ax.grid(True, linestyle='--', alpha=0.6)

# Line object for the robot arm
line, = ax.plot([], [], 'o-', lw=4, markersize=8, color='#FF5733', markerfacecolor='#2C3E50')
# Marker to highlight the stationary target
target, = ax.plot([0], [1 + np.sqrt(2)], 'kx', markersize=12, markeredgewidth=2)

def init():
    line.set_data([], [])
    return line, target

def animate(i):
    # Oscillate the time variable to swing back and forth
    t = np.sin(i * 0.1) * 0.15 
    
    # Calculate current joint angles using the null vector velocities
    t2 = theta2_0 + v2 * t
    t3 = theta3_0 + v3 * t
    t4 = theta4_0 + v4 * t
    
    # Forward Kinematics for the joints
    x0, y0 = 0, 0
    
    x1 = L1 * np.cos(t2)
    y1 = L1 * np.sin(t2)
    
    x2 = x1 + L2 * np.cos(t2 + t3)
    y2 = y1 + L2 * np.sin(t2 + t3)
    
    x3 = x2 + L3 * np.cos(t2 + t3 + t4)
    y3 = y2 + L3 * np.sin(t2 + t3 + t4)
    
    # Update the plot line
    line.set_data([x0, x1, x2, x3], [y0, y1, y2, y3])
    return line, target

# Run the animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=100, interval=50, blit=True)
plt.show()