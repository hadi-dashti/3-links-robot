import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib
from tensorflow.keras.models import load_model
import tensorflow as tf

# === Enable dropout during inference ===
def predict_with_uncertainty(f_model, x, n_iter=10):
    f_model.training = True  # force dropout at inference
    predictions = [f_model(x, training=True).numpy() for _ in range(n_iter)]
    return np.array(predictions)

# === Load model and scaler ===
model = load_model("saved_model/ik_model.h5", compile=False)
scaler_X = joblib.load("saved_model/input_scaler.pkl")

# === Input: target (x, y)
x_input = float(input("Enter x target (recommended -2.8 to 2.8): "))
y_input = float(input("Enter y target (recommended -2.8 to 2.8): "))

# === Create input features
r = np.sqrt(x_input**2 + y_input**2)
angle = np.arctan2(y_input, x_input)
input_point = np.array([[x_input, y_input, r, angle]])
input_scaled = scaler_X.transform(input_point)

# === Get multiple predictions using dropout
predictions = predict_with_uncertainty(model, input_scaled, n_iter=15)

# === Forward kinematics
def forward_kinematics(theta, lengths=[1.0, 1.0, 1.0]):
    theta1, theta2, theta3 = theta
    l1, l2, l3 = lengths
    joints = [[0, 0]]
    joints.append([l1*np.cos(theta1), l1*np.sin(theta1)])
    joints.append([
        joints[1][0] + l2*np.cos(theta1 + theta2),
        joints[1][1] + l2*np.sin(theta1 + theta2)
    ])
    joints.append([
        joints[2][0] + l3*np.cos(theta1 + theta2 + theta3),
        joints[2][1] + l3*np.sin(theta1 + theta2 + theta3)
    ])
    return np.array(joints)

# === Potential energy calculation
def potential_energy(theta1, theta2, theta3):
    h1 = 0.5 * np.sin(theta1)
    h2 = np.sin(theta1) + 0.5 * np.sin(theta1 + theta2)
    h3 = np.sin(theta1) + np.sin(theta1 + theta2) + 0.5 * np.sin(theta1 + theta2 + theta3)
    return h1 + h2 + h3

# === Filter based on FK error and select by lowest energy
threshold = 0.2  # FK error threshold (meters)
best_theta = None
min_energy = float("inf")
best_joints = None
all_joints = []
passed = []

for pred in predictions:
    sin_theta = pred[0, :3]
    cos_theta = pred[0, 3:]
    theta = [math.atan2(sin_theta[i], cos_theta[i]) for i in range(3)]
    joints = forward_kinematics(theta)
    end_effector = joints[-1]
    fk_error = np.linalg.norm(end_effector - np.array([x_input, y_input]))
    energy = potential_energy(*theta)

    all_joints.append((joints, fk_error, energy, theta))

    if fk_error < threshold:
        passed.append((joints, fk_error, energy, theta))
        if energy < min_energy:
            min_energy = energy
            best_theta = theta
            best_joints = joints

# === Plot
plt.figure(figsize=(10, 10))
for (joints, fk_error, energy, _) in all_joints:
    plt.plot(joints[:, 0], joints[:, 1], 'gray', alpha=0.3)

if best_theta is not None:
    plt.plot(best_joints[:, 0], best_joints[:, 1], 'r--o', lw=2, label='Best Arm (Low U & Accurate)')
else:
    print("⚠ No prediction passed the FK accuracy threshold!")

plt.scatter(x_input, y_input, c='lime', marker='*', s=300, edgecolor='black', label='Target')
plt.gca().add_patch(patches.Circle((0, 0), 3, fill=False, linestyle='--', color='gray'))
plt.title("Dropout Predictions Filtered by Accuracy + Energy")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("saved_model/post_processing_filtered_energy.png", dpi=300)
plt.show()

# === Print results
if best_theta is not None:
    print("\n✅ Best predicted joint angles (radians):")
    print("Theta1: %.4f" % best_theta[0])
    print("Theta2: %.4f" % best_theta[1])
    print("Theta3: %.4f" % best_theta[2])
    print("Minimum Potential Energy: %.4f" % min_energy)
else:
    print("\n❌ No valid prediction found with FK error < %.2f m" % threshold)

