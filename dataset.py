import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create folders
os.makedirs("dataset", exist_ok=True)
os.makedirs("picture", exist_ok=True)

# Parameters
num_samples = 80000  # Increased number of samples for better coverage
L1 = L2 = L3 = 1.0
m1 = m2 = m3 = 1.0
g = 9.81

# Joint angle limits with expanded θ3 and safe ranges for θ1, θ2
theta1 = np.random.uniform(-np.pi/2, np.pi/2, num_samples)        # Base: forward-facing half
theta2 = np.random.uniform(0, np.pi, num_samples)                 # Elbow-down
theta3 = np.random.uniform(-3*np.pi/4, 3*np.pi/4, num_samples)    # More flexible wrist

# Forward kinematics
x = L1*np.cos(theta1) + L2*np.cos(theta1+theta2) + L3*np.cos(theta1+theta2+theta3)
y = L1*np.sin(theta1) + L2*np.sin(theta1+theta2) + L3*np.sin(theta1+theta2+theta3)
r = np.sqrt(x**2 + y**2)

# Filter workspace within reach
valid_idx = (r >= 1.0) & (r <= 3.0)
x = x[valid_idx]
y = y[valid_idx]
theta1 = theta1[valid_idx]
theta2 = theta2[valid_idx]
theta3 = theta3[valid_idx]

# Compute sin and cos
sin_theta = np.column_stack([np.sin(theta1), np.sin(theta2), np.sin(theta3)])
cos_theta = np.column_stack([np.cos(theta1), np.cos(theta2), np.cos(theta3)])

# Theta sum and potential energy
theta_sum = theta1 + theta2 + theta3
h1 = (L1/2)*np.sin(theta1)
h2 = L1*np.sin(theta1) + (L2/2)*np.sin(theta1 + theta2)
h3 = L1*np.sin(theta1) + L2*np.sin(theta1 + theta2) + (L3/2)*np.sin(theta1 + theta2 + theta3)
U = m1*g*h1 + m2*g*h2 + m3*g*h3

# Create dataframe
data = pd.DataFrame({
    'x': x, 'y': y,
    'theta1': theta1, 'theta2': theta2, 'theta3': theta3,
    'sin_theta1': sin_theta[:, 0], 'sin_theta2': sin_theta[:, 1], 'sin_theta3': sin_theta[:, 2],
    'cos_theta1': cos_theta[:, 0], 'cos_theta2': cos_theta[:, 1], 'cos_theta3': cos_theta[:, 2],
    'theta_sum': theta_sum,
    'U': U
})

# Save dataset
csv_path = "dataset/robot_data_limited_expanded.csv"
data.to_csv(csv_path, index=False)
print(f"✅ Final enhanced dataset saved to {csv_path}")

# Plot distribution
r_filtered = np.sqrt(x**2 + y**2)
plt.figure(figsize=(10, 10))
plt.scatter(x, y, c=r_filtered, s=3, cmap='viridis', alpha=0.5)
plt.colorbar(label='Radial Distance (r)')
plt.gca().add_patch(plt.Circle((0, 0), 3, fill=False, linestyle='--', color='gray'))
plt.xlabel("X")
plt.ylabel("Y")
plt.title("End-Effector Distribution (Expanded Dataset)")
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.savefig("picture/limited_expanded_distribution.png", dpi=300)
plt.show()
