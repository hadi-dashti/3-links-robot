import numpy as np
import pandas as pd
import os

# === تنظیمات ===
os.makedirs("dataset", exist_ok=True)

num_points = 80000  # تعداد هدف‌ها در فضای کاری
num_candidates = 5  # تعداد کاندیدهایی که می‌خوایم برای هر هدف داشته باشیم
L1 = L2 = L3 = 1.0
max_reach = L1 + L2 + L3
min_reach = 1.0

# === تولید نقاط هدف ===
r = np.sqrt(np.random.uniform(min_reach**2, max_reach**2, num_points))
theta = np.random.uniform(0, 2 * np.pi, num_points)
x = r * np.cos(theta)
y = r * np.sin(theta)

records = []
for i in range(num_points):
    xi, yi = x[i], y[i]
    radius = r[i]
    angle = theta[i]

    found = 0
    attempts = 0
    while found < num_candidates and attempts < 50:  # تلاش‌های بیشتر برای پیدا کردن کاندید مناسب
        attempts += 1
        t1 = np.random.uniform(-np.pi, np.pi)
        t2 = np.random.uniform(-np.pi/2, np.pi/2)
        t3 = np.random.uniform(-np.pi/2, np.pi/2)

        x_fk = L1*np.cos(t1) + L2*np.cos(t1+t2) + L3*np.cos(t1+t2+t3)
        y_fk = L1*np.sin(t1) + L2*np.sin(t1+t2) + L3*np.sin(t1+t2+t3)

        if np.linalg.norm([x_fk - xi, y_fk - yi]) < 0.25:  # بازتر برای پذیرش
            h1 = 0.5 * np.sin(t1)
            h2 = np.sin(t1) + 0.5 * np.sin(t1 + t2)
            h3 = np.sin(t1) + np.sin(t1 + t2) + 0.5 * np.sin(t1 + t2 + t3)
            U = h1 + h2 + h3

            records.append([
                xi, yi, radius, angle,
                np.sin(t1), np.sin(t2), np.sin(t3),
                np.cos(t1), np.cos(t2), np.cos(t3),
                t1 + t2 + t3, U
            ])
            found += 1

# === ذخیره فایل CSV ===
columns = ['x', 'y', 'radius', 'angle',
           'sin_theta1', 'sin_theta2', 'sin_theta3',
           'cos_theta1', 'cos_theta2', 'cos_theta3',
           'theta_sum', 'U']

df = pd.DataFrame(records, columns=columns)
df.to_csv("dataset/multi_candidate_dataset.csv", index=False)

print("✅ Dataset saved to 'dataset/multi_candidate_dataset.csv'")
print("✅ Total records:", len(df))
