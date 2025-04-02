import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import math
import matplotlib.patches as patches
import joblib
import os

# === Load dataset ===
data = pd.read_csv("dataset/robot_data_limited_expanded.csv")

# === Feature engineering: x, y, radius and angle ===
data['radius'] = np.sqrt(data['x']**2 + data['y']**2)
data['angle'] = np.arctan2(data['y'], data['x'])

# === Input and output selection ===
X = data[['x', 'y', 'radius', 'angle']].values
y = data[['sin_theta1', 'sin_theta2', 'sin_theta3',
          'cos_theta1', 'cos_theta2', 'cos_theta3']].values

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Standardize inputs ===
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# === Forward kinematics function ===
def forward_kinematics_batch(thetas):
    x_end = []
    for row in thetas:
        theta1, theta2, theta3 = row
        x = np.cos(theta1) + np.cos(theta1 + theta2) + np.cos(theta1 + theta2 + theta3)
        y = np.sin(theta1) + np.sin(theta1 + theta2) + np.sin(theta1 + theta2 + theta3)
        x_end.append([x, y])
    return np.array(x_end)

# === Loss function with sin/cos + end-effector position error ===
@tf.function
def combined_loss(y_true, y_pred):
    sin_true, cos_true = y_true[:, :3], y_true[:, 3:]
    sin_pred, cos_pred = y_pred[:, :3], y_pred[:, 3:]
    
    # Standard MSE on sin and cos values
    mse_loss = tf.reduce_mean(tf.square(sin_true - sin_pred) + tf.square(cos_true - cos_pred))
    
    # Convert sin/cos to angles
    theta_true = tf.atan2(sin_true, cos_true)
    theta_pred = tf.atan2(sin_pred, cos_pred)
    
    # Compute FK for both
    def batch_fk(thetas):
        theta1, theta2, theta3 = tf.unstack(thetas, axis=1)
        x = tf.cos(theta1) + tf.cos(theta1 + theta2) + tf.cos(theta1 + theta2 + theta3)
        y = tf.sin(theta1) + tf.sin(theta1 + theta2) + tf.sin(theta1 + theta2 + theta3)
        return tf.stack([x, y], axis=1)
    
    end_true = batch_fk(theta_true)
    end_pred = batch_fk(theta_pred)
    
    fk_loss = tf.reduce_mean(tf.square(end_true - end_pred))
    
    # Combine losses
    total_loss = 0.7 * mse_loss + 0.3 * fk_loss
    return total_loss

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(128, activation='swish'),
        tf.keras.layers.Dropout(0.1),  # Dropout added to prevent overfitting
        tf.keras.layers.Dense(128, activation='swish'),
        tf.keras.layers.Dense(6)  # Outputs: sin and cos of 3 angles
    ])
    return model


model = build_model()
model.compile(optimizer='adam', loss=combined_loss, metrics=['mae'])

# === Train model ===
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=150,
    batch_size=128,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
    ],
    verbose=1
)

# === Save model and scaler ===
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/ik_model.h5")
joblib.dump(scaler_X, "saved_model/input_scaler.pkl")
print("✅ Model and scaler saved successfully.")

# === Prediction and evaluation ===
y_pred = model.predict(X_test_scaled)

def sincos_to_angles(sin_cos_array):
    return np.array([
        [math.atan2(row[0], row[3]), math.atan2(row[1], row[4]), math.atan2(row[2], row[5])]
        for row in sin_cos_array
    ])

pred_angles = sincos_to_angles(y_pred)

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

# === Visualize results for 20 samples ===
sample_indices = np.random.choice(len(X_test), 20, replace=False)
plt.figure(figsize=(25, 30))
link_lengths = [1.0, 1.0, 1.0]

for i, idx in enumerate(sample_indices):
    pred = pred_angles[idx]
    target = X_test[idx][:2]
    joints = forward_kinematics(pred, link_lengths)

    ax = plt.subplot(5, 4, i+1)
    plt.plot(joints[:, 0], joints[:, 1], 'r--o', lw=2, markersize=8, label='Predicted Arm')
    plt.scatter(target[0], target[1], c='lime', marker='*', s=400, label='Target (x,y)', edgecolor='black')
    plt.title(f"Sample {i+1}\\nTarget: ({target[0]:.2f}, {target[1]:.2f})", fontsize=10)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend(fontsize=8)
    workspace = patches.Circle((0, 0), 3, fill=False, linestyle='--', color='gray', alpha=0.3)
    ax.add_patch(workspace)

plt.tight_layout()
plt.savefig("picture/enhanced_loss_prediction_comparison.png", dpi=300)
plt.show()

# === Evaluate average and max end-effector error ===
end_effector_errors = []
for i in range(len(X_test)):
    pred_joint = forward_kinematics(pred_angles[i])[-1]
    target = X_test[i][:2]
    error = np.linalg.norm(pred_joint - target)
    end_effector_errors.append(error)

# Plot error histogram
plt.figure(figsize=(12, 6))
plt.hist(end_effector_errors, bins=50, color='skyblue', edgecolor='black')
plt.title('End-Effector Position Error Distribution (Enhanced Loss)')
plt.xlabel('Position Error (meters)')
plt.ylabel('Number of Samples')
plt.grid(True)
plt.tight_layout()
plt.savefig("picture/enhanced_loss_error_distribution.png", dpi=300)
plt.show()

# Final metrics
mean_error = np.mean(end_effector_errors)
max_error = np.max(end_effector_errors)
print(f"Mean End-Effector Position Error: {mean_error:.4f} meters")
print(f"Max End-Effector Position Error: {max_error:.4f} meters")
