
import numpy as np

# Load binary data from vortex_000.raw
with open('vortex_004.raw', 'rb') as f:
    data = np.fromfile(f, dtype=np.float32)

print(data.shape, np.min(data), np.max(data))

# Normalize the array from 0 to 1
normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))

# Store the normalized data in a new file named normalized_vortex_000.raw
with open('normalized_vortex_004.raw', 'wb') as f:
    normalized_data.tofile(f)
