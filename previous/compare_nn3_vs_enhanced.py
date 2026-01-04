
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import joblib
import matplotlib.pyplot as plt
import os

# === Load dataset ===
data = pd.read_csv("dataset/robot_data_limited_expanded.csv")

# === Feature engineering ===
data['radius'] = np.sqrt(data['x']**2 + data['y']**2)
data['angle'] = np.arctan2(data['y'], data['x'])
data['x2'] = data['x']**2
data['y2'] = data['y']**2
data['xy'] = data['x'] * data['y']
data['x_r'] = data['x'] / data['radius']
data['y_r'] = data['y'] / data['radius']
data['cos_angle'] = np.cos(data['angle'])
data['sin_angle'] = np.sin(data['angle'])

# === Prepare input/output ===
X = data[['x', 'y', 'radius', 'angle', 'x2', 'y2', 'xy', 'x_r', 'y_r', 'cos_angle', 'sin_angle']].values
y = data[['sin_theta1', 'sin_theta2', 'sin_theta3', 'cos_theta1', 'cos_theta2', 'cos_theta3']].values
U_true = data['U'].values

# === Load models and scalers ===
def load_model_and_scaler(model_dir):
    model = tf.keras.models.load_model(f"{model_dir}/ik_model.h5", compile=False)
    scaler = joblib.load(f"{model_dir}/input_scaler.pkl")
    return model, scaler

model_nn3, scaler_nn3 = load_model_and_scaler("saved_model/NN3")
model_enh, scaler_enh = load_model_and_scaler("saved_model/NN3_enhanced_loss")

# === Scale input ===
X_nn3 = scaler_nn3.transform(X)
X_enh = scaler_enh.transform(X)

# === Predict ===
y_pred_nn3 = model_nn3.predict(X_nn3)
y_pred_enh = model_enh.predict(X_enh)

def sincos_to_angles(sin_cos_array):
    return np.array([
        [math.atan2(row[0], row[3]), math.atan2(row[1], row[4]), math.atan2(row[2], row[5])]
        for row in sin_cos_array
    ])

angles_nn3 = sincos_to_angles(y_pred_nn3)
angles_enh = sincos_to_angles(y_pred_enh)

def forward_kinematics(thetas, lengths=[1.0, 1.0, 1.0]):
    theta1, theta2, theta3 = thetas
    l1, l2, l3 = lengths
    x = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2) + l3 * math.cos(theta1 + theta2 + theta3)
    y = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2) + l3 * math.sin(theta1 + theta2 + theta3)
    return [x, y]

fk_errors_nn3 = []
fk_errors_enh = []
energy_nn3 = []
energy_enh = []
angle_sum_nn3 = []
angle_sum_enh = []

for a1, a2 in zip(angles_nn3, angles_enh):
    target = forward_kinematics(a1)  # Approximate target

    p1 = forward_kinematics(a1)
    p2 = forward_kinematics(a2)

    fk_errors_nn3.append(math.dist(p1, target))
    fk_errors_enh.append(math.dist(p2, target))

    def energy(t):
        y1 = 0.5 * math.sin(t[0])
        y2 = math.sin(t[0]) + 0.5 * math.sin(t[0] + t[1])
        y3 = math.sin(t[0]) + math.sin(t[0] + t[1]) + 0.5 * math.sin(t[0] + t[1] + t[2])
        return y1 + y2 + y3

    energy_nn3.append(energy(a1))
    energy_enh.append(energy(a2))

    angle_sum_nn3.append(np.sum(np.abs(a1)))
    angle_sum_enh.append(np.sum(np.abs(a2)))

# === Compare metrics ===
print("\n--- 📊 Comparison Summary ---")
print(f"🔹 NN3    | Mean FK Error: {np.mean(fk_errors_nn3):.4f}, Mean Energy: {np.mean(energy_nn3):.4f}, Mean Angle Sum: {np.mean(angle_sum_nn3):.4f}")
print(f"🔹 NN3+   | Mean FK Error: {np.mean(fk_errors_enh):.4f}, Mean Energy: {np.mean(energy_enh):.4f}, Mean Angle Sum: {np.mean(angle_sum_enh):.4f}")

# === Plot comparisons ===
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.hist(fk_errors_nn3, bins=40, alpha=0.6, label='NN3')
plt.hist(fk_errors_enh, bins=40, alpha=0.6, label='Enhanced')
plt.title("FK Error Distribution")
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(energy_nn3, bins=40, alpha=0.6, label='NN3')
plt.hist(energy_enh, bins=40, alpha=0.6, label='Enhanced')
plt.title("Energy Distribution")
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(angle_sum_nn3, bins=40, alpha=0.6, label='NN3')
plt.hist(angle_sum_enh, bins=40, alpha=0.6, label='Enhanced')
plt.title("Angle Sum Distribution")
plt.legend()

plt.tight_layout()
plt.savefig("comparison_metrics.png", dpi=300)
plt.show()
