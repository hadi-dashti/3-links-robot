# NN3.py — نسخه‌ی تقویت‌شده‌ی مدل با ویژگی‌های پیشرفته برای کاهش بیشتر خطای مکان

# ✅ Mean End-Effector Position Error: 0.0036 meters
# 📈 Max End-Effector Position Error: 0.0588 meters

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import math
import joblib
import os

# === Load dataset ===
data = pd.read_csv("dataset/robot_data_limited_expanded.csv")

# === Fix types in case columns are strings ===
columns_to_convert = ['x', 'y', 'sin_theta1', 'sin_theta2', 'sin_theta3', 'cos_theta1', 'cos_theta2', 'cos_theta3']
for col in columns_to_convert:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna()

# === Feature engineering (پیشرفته) ===
data['radius'] = np.sqrt(data['x']**2 + data['y']**2)
data['angle'] = np.arctan2(data['y'], data['x'])

data['x2'] = data['x']**2
data['y2'] = data['y']**2
data['xy'] = data['x'] * data['y']
data['x_r'] = data['x'] / data['radius']
data['y_r'] = data['y'] / data['radius']
data['cos_angle'] = np.cos(data['angle'])
data['sin_angle'] = np.sin(data['angle'])

# === Input/output ===
X = data[['x', 'y', 'radius', 'angle', 'x2', 'y2', 'xy', 'x_r', 'y_r', 'cos_angle', 'sin_angle']].values
y = data[['sin_theta1', 'sin_theta2', 'sin_theta3', 'cos_theta1', 'cos_theta2', 'cos_theta3']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Scaling ===
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# === Custom Combined Loss Function with Energy & Angle Change ===
@tf.function
def combined_loss(y_true, y_pred):
    sin_pred, cos_pred = y_pred[:, :3], y_pred[:, 3:]
    theta_pred = tf.atan2(sin_pred, cos_pred)

    def batch_fk(thetas):
        theta1, theta2, theta3 = tf.unstack(thetas, axis=1)
        x = tf.cos(theta1) + tf.cos(theta1 + theta2) + tf.cos(theta1 + theta2 + theta3)
        y = tf.sin(theta1) + tf.sin(theta1 + theta2) + tf.sin(theta1 + theta2 + theta3)
        return tf.stack([x, y], axis=1)

    end_true = batch_fk(tf.atan2(y_true[:, :3], y_true[:, 3:]))
    end_pred = batch_fk(theta_pred)

    position_loss = tf.reduce_mean(tf.square(end_true - end_pred))

    # === Angle Change Loss (Assuming zero for now)
    last_angles = tf.zeros_like(theta_pred)  # Placeholder: you can modify this to real previous angles
    angle_change_loss = tf.reduce_mean(tf.square(theta_pred - last_angles))

    # === Potential Energy Loss (from column 'U')
    u_true = tf.expand_dims(y_true[:, -1], axis=1)  # U should be appended to y_true during training
    potential_energy_loss = tf.reduce_mean(tf.square(u_true))

    λ1 = 0.01  # weight for angle change
    λ2 = 0.01  # weight for energy
    return position_loss + λ1 * angle_change_loss + λ2 * potential_energy_loss

# === Model ===
def build_model():
    l2 = tf.keras.regularizers.l2(1e-4)
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(256, activation='swish', kernel_regularizer=l2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='swish', kernel_regularizer=l2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='swish', kernel_regularizer=l2),
        tf.keras.layers.Dense(6)
    ])

model = build_model()
model.compile(optimizer='adam', loss=combined_loss, metrics=['mae'])

# === Train ===
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=200,
    batch_size=128,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
    ],
    verbose=1
)

# === Save model ===
os.makedirs("saved_model/NN3_enhanced_loss", exist_ok=True)
model.save("saved_model/NN3_enhanced_loss/ik_model.h5")
joblib.dump(scaler_X, "saved_model/NN3_enhanced_loss/input_scaler.pkl")

# === Prediction & evaluation ===
y_pred = model.predict(X_test_scaled)

def sincos_to_angles(sin_cos_array):
    return np.array([
        [math.atan2(row[0], row[3]), math.atan2(row[1], row[4]), math.atan2(row[2], row[5])]
        for row in sin_cos_array
    ])

pred_angles = sincos_to_angles(y_pred)
true_angles = sincos_to_angles(y_test)

def forward_kinematics(thetas, lengths=[1.0, 1.0, 1.0]):
    theta1, theta2, theta3 = thetas
    l1, l2, l3 = lengths
    joints = [
        [0, 0],
        [l1 * math.cos(theta1), l1 * math.sin(theta1)],
        [0, 0],
        [0, 0]
    ]
    joints[2][0] = joints[1][0] + l2 * math.cos(theta1 + theta2)
    joints[2][1] = joints[1][1] + l2 * math.sin(theta1 + theta2)
    joints[3][0] = joints[2][0] + l3 * math.cos(theta1 + theta2 + theta3)
    joints[3][1] = joints[2][1] + l3 * math.sin(theta1 + theta2 + theta3)
    return joints

# === Error Metrics ===
def compute_fk_errors(true_angles, pred_angles):
    errors = []
    for t_true, t_pred in zip(true_angles, pred_angles):
        p1 = forward_kinematics(t_true)[-1]
        p2 = forward_kinematics(t_pred)[-1]
        dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        errors.append(dist)
    return errors

errors = compute_fk_errors(true_angles, pred_angles)

# ✅ چاپ فقط نتایج مهم
mean_error = np.mean(errors)
max_error = np.max(errors)
print(f"✅ Mean End-Effector Position Error: {mean_error:.4f} meters")
print(f"📈 Max End-Effector Position Error: {max_error:.4f} meters")

# === Plot prediction vs ground truth ===
os.makedirs("picture/NN3_enhanced_loss", exist_ok=True)
plt.figure(figsize=(10, 6))
plt.plot(errors, label="End-Effector Position Error", color='darkblue')
plt.xlabel("Sample")
plt.ylabel("Error (m)")
plt.title("End-Effector Prediction Error per Sample")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("picture/NN3_enhanced_loss/enhanced_loss_error_distribution.png", dpi=300)
plt.show()

# === Plot histogram of position errors ===
plt.figure(figsize=(12, 6))
plt.hist(errors, bins=50, color='skyblue', edgecolor='black')
plt.xlabel("Position Error (meters)")
plt.ylabel("Number of Samples")
plt.title("End-Effector Position Error Distribution (Enhanced Loss)")
plt.grid(True)
plt.tight_layout()
plt.savefig("picture/NN3_enhanced_loss/enhanced_loss_error_distribution.png", dpi=300)
plt.show()

# === Visualize 20 true vs predicted arms in one image ===
sample_indices = np.random.choice(len(pred_angles), size=20, replace=False)
fig, axs = plt.subplots(4, 5, figsize=(20, 16))
fig.suptitle("20 Sample Predictions: True vs Predicted Arms", fontsize=20)

for ax, idx in zip(axs.ravel(), sample_indices):
    joints_true = forward_kinematics(true_angles[idx])
    joints_pred = forward_kinematics(pred_angles[idx])

    ax.plot(*zip(*joints_true), marker='o', label='True')
    ax.plot(*zip(*joints_pred), marker='x', label='Predicted')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_title(f"Sample {idx}")
    ax.grid(True)
    ax.set_aspect('equal')
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("picture/NN3_enhanced_loss/20_samples_prediction_comparison.png", dpi=300)
plt.show()
