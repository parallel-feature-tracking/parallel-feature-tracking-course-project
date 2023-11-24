
from vtk import vtkXMLImageDataReader
import struct

li = ["000", "001", "002", "003", "004"]


for i in li:
    input_vti_file_path = '64/' + i + '.vti'
    output_binary_file_path = '64/normalized_64/vortex_' + i + '.raw'

    # Create a VTK reader for the VTI file
    reader = vtkXMLImageDataReader()
    reader.SetFileName(input_vti_file_path)
    reader.Update()

    # Get the 'vtkImageData' object from the reader
    image_data = reader.GetOutput()

    # Get the point data from 'vtkImageData'
    point_data = image_data.GetPointData()

    # Assuming that we are interested in the first array which is the scalar data
    data_array = point_data.GetArray(0)

    # Get the number of tuples and components to calculate the total number of values
    num_tuples = data_array.GetNumberOfTuples()
    num_components = data_array.GetNumberOfComponents()
    total_values = num_tuples * num_components

    # Create a buffer to store the binary data
    binary_data = bytearray(total_values * 4)  # float32 has 4 bytes

    # Extract the data and write it to the binary data buffer
    for i in range(num_tuples):
        for j in range(num_components):
            # Get the value at the current tuple and component
            value = data_array.GetComponent(i, j)
            # Pack the float into bytes using struct.pack and store it in the buffer
            binary_data[i*num_components*4 + j*4:i*num_components*4 + (j+1)*4] = struct.pack('<f', value)

    # Write the binary data to a file
    with open(output_binary_file_path, 'wb') as binary_file:
        binary_file.write(binary_data)
    
    print(f"Binary data extracted to {output_binary_file_path}")
