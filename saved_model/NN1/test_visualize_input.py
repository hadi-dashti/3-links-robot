import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import csv

# === Load model and scaler ===
model = load_model("saved_model/NN1/ik_model.h5", compile=False)
scaler_X = joblib.load("saved_model/NN1/input_scaler.pkl")

# === Define input (x, y) ===
x_input = float(input("Enter x target (recommended range -2.8 to 2.8): "))
y_input = float(input("Enter y target (recommended range -2.8 to 2.8): "))

# === Compute extra input features ===
r = np.sqrt(x_input**2 + y_input**2)
angle = np.arctan2(y_input, x_input)

# === Prepare input for prediction ===
input_point = np.array([[x_input, y_input, r, angle]])
input_scaled = scaler_X.transform(input_point)

# === Predict output (sin/cos of angles) ===
output = model.predict(input_scaled)
sin_theta = output[0, :3]
cos_theta = output[0, 3:]
theta = [math.atan2(sin_theta[i], cos_theta[i]) for i in range(3)]

# === Forward kinematics ===
def forward_kinematics(thetas, lengths=[1.0, 1.0, 1.0]):
    theta1, theta2, theta3 = thetas
    l1, l2, l3 = lengths
    joints = [
        [0, 0],
        [l1 * np.cos(theta1), l1 * np.sin(theta1)],
        [l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2),
         l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)],
        [l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2) + l3 * np.cos(theta1 + theta2 + theta3),
         l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2) + l3 * np.sin(theta1 + theta2 + theta3)]
    ]
    return np.array(joints)

joints = forward_kinematics(theta)
end_effector = joints[-1]
error = np.linalg.norm(end_effector - np.array([x_input, y_input]))

# === Visualization ===
plt.figure(figsize=(8, 8))
plt.plot(joints[:, 0], joints[:, 1], 'r--o', lw=3, markersize=10, label='Predicted Arm')
plt.scatter(x_input, y_input, c='lime', s=200, marker='*', label='Target (x,y)', edgecolor='black')
plt.gca().add_patch(patches.Circle((0, 0), 3, fill=False, linestyle='--', color='gray', alpha=0.5))
plt.title("Predicted Arm Configuration vs Target")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("saved_model/NN1/test_result_visual.png", dpi=300)
plt.show()

# === Save data directory ===
data_dir = "saved_model/NN1/data"
os.makedirs(data_dir, exist_ok=True)

# === Save joint angles ===
angles_path = os.path.join(data_dir, "angles_log.csv")
write_header = not os.path.exists(angles_path)

with open(angles_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if write_header:
        writer.writerow(["x_target", "y_target", "theta1", "theta2", "theta3"])
    writer.writerow([x_input, y_input, theta[0], theta[1], theta[2]])

# === Save error ===
error_path = os.path.join(data_dir, "error_log.csv")
write_header = not os.path.exists(error_path)

with open(error_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if write_header:
        writer.writerow(["x_target", "y_target", "end_effector_x", "end_effector_y", "error_m"])
    writer.writerow([x_input, y_input, end_effector[0], end_effector[1], error])

# === Print output ===
print("\nPredicted joint angles (radians):")
print("Theta1: %.4f" % theta[0])
print("Theta2: %.4f" % theta[1])
print("Theta3: %.4f" % theta[2])
print("End-Effector Error: %.4f meters" % error)
