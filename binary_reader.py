import numpy as np

# Specify the file path
file_path = 'vortex.tfe'

# Read the binary file into a float32 array
float_array = np.fromfile(file_path, dtype=np.float32)

# Print the float32 array
print(min(float_array), max(float_array), len(float_array))
