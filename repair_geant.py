import numpy as np

x = np.load("geant.npy")

print("Original shape:", x.shape)
print("Original max:", x.max())

# The known corrupted GÉANT time step
bad_index = 0

# Find the time step containing the extreme outlier
step_max = x.reshape(len(x), -1).max(axis=1)
bad_index = int(np.argmax(step_max))

print("Detected suspicious step:", bad_index)
print("Value:", step_max[bad_index])

# Replace it using neighboring matrices
if bad_index == 0 or bad_index == len(x) - 1:
    raise ValueError("Bad step is at dataset boundary.")

x[bad_index] = (x[bad_index - 1] + x[bad_index + 1]) / 2

np.save("geant_clean.npy", x)

print("Cleaned max:", x.max())
print("Saved: geant_clean.npy")