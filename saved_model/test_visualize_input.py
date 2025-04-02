
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model


# === Load model and scaler ===
model = load_model("saved_model/ik_model.h5", compile=False)
scaler_X = joblib.load("saved_model/input_scaler.pkl")

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

# === Forward kinematics to compute joint positions ===
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

# === Visualize ===
joints = forward_kinematics(theta)

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
plt.savefig("saved_model/test_result_visual.png", dpi=300)
plt.show()

print("\nPredicted joint angles (radians):")
print("Theta1: %.4f" % theta[0])
print("Theta2: %.4f" % theta[1])
print("Theta3: %.4f" % theta[2])
