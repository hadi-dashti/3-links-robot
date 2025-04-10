import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import math
import joblib
import os

# === Load dataset ===
data = pd.read_csv("dataset/robot_data_limited_expanded.csv")
data['radius'] = np.sqrt(data['x']**2 + data['y']**2)
data['angle'] = np.arctan2(data['y'], data['x'])

X = data[['x', 'y', 'radius', 'angle']].values
y = data[['sin_theta1', 'sin_theta2', 'sin_theta3',
          'cos_theta1', 'cos_theta2', 'cos_theta3']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# === FK function ===
def batch_fk(thetas):
    theta1, theta2, theta3 = tf.unstack(thetas, axis=1)
    x = tf.cos(theta1) + tf.cos(theta1 + theta2) + tf.cos(theta1 + theta2 + theta3)
    y = tf.sin(theta1) + tf.sin(theta1 + theta2) + tf.sin(theta1 + theta2 + theta3)
    return tf.stack([x, y], axis=1)

# === Combined loss ===
@tf.function
def combined_loss(y_true, y_pred):
    sin_true, cos_true = y_true[:, :3], y_true[:, 3:]
    sin_pred, cos_pred = y_pred[:, :3], y_pred[:, 3:]

    mse = tf.reduce_mean(tf.square(sin_true - sin_pred) + tf.square(cos_true - cos_pred))

    theta_true = tf.atan2(sin_true, cos_true)
    theta_pred = tf.atan2(sin_pred, cos_pred)

    fk_true = batch_fk(theta_true)
    fk_pred = batch_fk(theta_pred)
    fk_err = tf.reduce_mean(tf.square(fk_true - fk_pred))

    return 0.7 * mse + 0.3 * fk_err

# === Build model with Dropout ===
def build_model_with_dropout():
    inputs = tf.keras.layers.Input(shape=(X_train_scaled.shape[1],))
    x = tf.keras.layers.Dense(128, activation='swish')(inputs)
    x = tf.keras.layers.Dropout(0.2)(x, training=True)  # stays active during inference
    x = tf.keras.layers.Dense(128, activation='swish')(x)
    x = tf.keras.layers.Dropout(0.2)(x, training=True)
    outputs = tf.keras.layers.Dense(6)(x)
    return tf.keras.Model(inputs, outputs)

model = build_model_with_dropout()
model.compile(optimizer='adam', loss=combined_loss)

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
model.save("saved_model/ik_model_dropout.h5")
joblib.dump(scaler_X, "saved_model/input_scaler_dropout.pkl")
print("✅ Dropout-based model and scaler saved.")
