combined_binary_data = bytearray()

# Loop through the range of numbers, assuming 'x' goes from 0 to 7.
for x in range(8):
    # Construct the file name for each piece.
    file_name = f"vortex_001_{x}.mask"
    # Open and read the file's contents and append to the combined_binary_data bytearray.
    with open(file_name, 'rb') as file:
        binary_data = file.read()
        combined_binary_data += binary_data

# Write the combined binary data to a new binary file.
with open('combined_vortex_masks.raw', 'wb') as output_file:
    output_file.write(combined_binary_data)
