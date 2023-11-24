import struct
import numpy as np
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkCommonCore import vtkFloatArray
from vtk import vtkXMLImageDataWriter
from vtkmodules.vtkCommonCore import vtkPoints
from vtk.util.numpy_support import numpy_to_vtk

# Parameters
x = 000
dimensions = (128, 128, 128)  # Replace with actual dimensions
data_type = vtkFloatArray().GetDataType()  # Assuming the data is of type 'float'

# Read the binary data from a file
input_binary_file_path = 'haha/normalized_vortex_004haha.mask'  # Replace with the actual file path
with open(input_binary_file_path, 'rb') as binary_file:
    binary_data = binary_file.read()

# Unpack the binary data into a flat list of values
num_values = len(binary_data) // 4  # float32 has 4 bytes
unpacked_data = struct.unpack('<' + 'f' * num_values, binary_data)

# Convert the flat list of values into a NumPy array
np_data = np.array(unpacked_data, dtype=np.float32).reshape(dimensions)

# Convert the numpy array to a VTK array
vtk_data = numpy_to_vtk(num_array=np_data.ravel(), deep=True, array_type=data_type)

# Create the vtkImageData object
image_data = vtkImageData()
image_data.SetDimensions(dimensions)
image_data.SetSpacing(1.0, 1.0, 1.0)  # Assuming unit spacing
image_data.GetPointData().SetScalars(vtk_data)

# Write to a .vti file
output_vti_file_path = 'haha/normalized_vortex_004.vti'  # Replace with the actual output path
writer = vtkXMLImageDataWriter()
writer.SetFileName(output_vti_file_path)
writer.SetInputData(image_data)
writer.Write()
