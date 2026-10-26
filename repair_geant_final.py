import numpy as np

INPUT = "geant_clean.npy"
OUTPUT = "geant_final.npy"

# Anomalous interval identified from the dataset inspection
START = 5137
END = 5148

x = np.load(INPUT)

print("Original shape:", x.shape)
print("Original max:", x.max())

# Keep the original file untouched
y = x.copy()

# Boundary matrices
before = x[START - 1]   # index 5136
after = x[END + 1]      # index 5149

# Linear interpolation for each anomalous timestep
num_steps = END - START + 1

for i in range(START, END + 1):
    alpha = (i - (START - 1)) / (num_steps + 1)
    y[i] = (1 - alpha) * before + alpha * after

np.save(OUTPUT, y)

print("\nSaved:", OUTPUT)
print("New shape:", y.shape)
print("New max:", y.max())
print("New median:", np.median(y))
print("New mean:", y.mean())

# Show the repaired region
print("\nRepaired region:")
step_max = y.reshape(len(y), -1).max(axis=1)

for i in range(START - 2, END + 3):
    print(i, round(float(step_max[i]), 2))
    