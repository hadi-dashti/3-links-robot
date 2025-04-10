import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib
from tensorflow.keras.models import load_model

# === Load model and scaler ===
model = load_model("saved_model/ik_model_dropout.h5", compile=False)
scaler_X = joblib.load("saved_model/input_scaler_dropout.pkl")

# === Predict with active Dropout + input noise ===
def predict_with_uncertainty(model, x_base, n_iter=70, input_noise_std=0.015):
    preds = []
    for _ in range(n_iter):
        noisy_input = x_base + np.random.normal(0, input_noise_std, x_base.shape)
        preds.append(model(noisy_input, training=True).numpy())
    return np.array(preds)

# === Get user input
x_input = float(input("Enter x target (recommended -2.8 to 2.8): "))
y_input = float(input("Enter y target (recommended -2.8 to 2.8): "))
r = np.sqrt(x_input**2 + y_input**2)
angle = np.arctan2(y_input, x_input)

# === Create noisy scaled input
input_point = np.array([[x_input, y_input, r, angle]])
input_scaled = scaler_X.transform(input_point)

# === Predict multiple outputs
samples = predict_with_uncertainty(model, input_scaled, n_iter=70)

# === FK function
def forward_kinematics(theta, lengths=[1.0, 1.0, 1.0]):
    theta1, theta2, theta3 = theta
    joints = [[0, 0]]
    joints.append([np.cos(theta1), np.sin(theta1)])
    joints.append([
        joints[1][0] + np.cos(theta1 + theta2),
        joints[1][1] + np.sin(theta1 + theta2)
    ])
    joints.append([
        joints[2][0] + np.cos(theta1 + theta2 + theta3),
        joints[2][1] + np.sin(theta1 + theta2 + theta3)
    ])
    return np.array(joints)

# === Potential energy function
def potential_energy(theta1, theta2, theta3):
    h1 = 0.5 * np.sin(theta1)
    h2 = np.sin(theta1) + 0.5 * np.sin(theta1 + theta2)
    h3 = np.sin(theta1) + np.sin(theta1 + theta2) + 0.5 * np.sin(theta1 + theta2 + theta3)
    return h1 + h2 + h3

# === Evaluate and select best
threshold = 0.2
best_theta = None
min_score = float("inf")
best_joints = None
all_joints = []

for pred in samples:
    sin_theta = pred[0, :3]
    cos_theta = pred[0, 3:]
    theta = [math.atan2(sin_theta[i], cos_theta[i]) for i in range(3)]

    joints = forward_kinematics(theta)
    ee = joints[-1]
    fk_error = np.linalg.norm(ee - np.array([x_input, y_input]))
    energy = potential_energy(*theta)
    score = fk_error + 0.1 * energy  # weighted scoring

    all_joints.append((joints, fk_error, energy, score, theta))

    if fk_error < threshold and score < min_score:
        best_theta = theta
        best_joints = joints
        min_score = score

# === Plotting
plt.figure(figsize=(10, 10))
for (joints, fk_error, energy, score, _) in all_joints:
    plt.plot(joints[:, 0], joints[:, 1], 'gray', alpha=0.25)

if best_theta is not None:
    plt.plot(best_joints[:, 0], best_joints[:, 1], 'r--o', lw=2, label='Best Candidate')
else:
    print("⚠ No prediction passed FK threshold.")

plt.scatter(x_input, y_input, c='lime', marker='*', s=300, edgecolor='black', label='Target')
plt.gca().add_patch(patches.Circle((0, 0), 3, fill=False, linestyle='--', color='gray'))
plt.title("Dropout Sampling with Input Noise - Best Candidate Selection")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("saved_model/dropout_diverse_result.png", dpi=300)
plt.show()

# === Print final result
if best_theta is not None:
    print("\n✅ Best joint angles (radians):")
    print("Theta1: %.4f" % best_theta[0])
    print("Theta2: %.4f" % best_theta[1])
    print("Theta3: %.4f" % best_theta[2])
    print("FK Error: %.4f m" % min_score)
    print("Potential Energy: %.4f" % potential_energy(*best_theta))
else:
    print("\n❌ No valid candidate found below FK error threshold.")
