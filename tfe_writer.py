import numpy as np
import json
from scipy import interpolate

# Define the array with 1024 elements, all set to 1024, and the rest set to 1
array = np.array([1024] + [0] * 1023, dtype=np.float32)

with open('preset.json', 'r') as json_file:
    data = json.load(json_file)

# print(data[0]["Points"])

max = max(data[0]["Points"])
print(max)
index_list = []
value_list = []
for i in range(len(data[0]["Points"])):
    if(i%4 == 0):
        index_list.append(data[0]["Points"][i] / max)
        value_list.append(data[0]["Points"][i+1])

print(index_list)
print(value_list)

x_y_coords = []

for i in range(len(index_list)):
    val = int(index_list[i] * 1022)
    x_y_coords.append([val, value_list[i]])

print(x_y_coords)

points = np.array(x_y_coords)

final_array = np.zeros(1023, dtype=np.float32)

for i in range(len(points)-1):
    start = points[i][0]
    end = points[i+1][0]
    m = (points[i+1][1] - points[i][1]) / (end - start)
    print(m)
    c = points[i][1] - m * start
    for j in range(int(start), int(end)):
        # print("mj + c: ", m * j + c)
        final_array[j] = m * j + c

for i in range(len(final_array)):
    array[i+1] = final_array[i]

with open('vortex.tfe', 'wb') as f:
    array.tofile(f)