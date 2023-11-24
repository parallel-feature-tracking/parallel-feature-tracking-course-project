import xml.etree.ElementTree as ET
import csv
import os

def parse_xml_and_extract_mpi_times(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    mpi_times = []
    # Iterate over each task (which represents an MPI rank/process)
    for task in root.findall('task'):
        mpi_rank = task.get('mpi_rank')
        mpi_size = task.get('mpi_size')
        
        # Find the region containing MPI function times
        for region in task.findall('.//region'):
            # Iterate over each function entry within the region
            for func in region.findall('func'):
                func_name = func.get('name')
                func_count = func.get('count')
                func_time = func.text.strip()
                mpi_times.append({
                    'mpi_rank': mpi_rank,
                    'mpi_size': mpi_size,
                    'func_name': func_name,
                    'func_count': func_count,
                    'func_time': func_time
                })
    return mpi_times

def process_files_in_directory(directory_path):
    # Get the directory name which will be used to name the CSV
    base_name = os.path.basename(directory_path)
    csv_file_name = f"{base_name}.csv"
    csv_file_path = os.path.join(directory_path, csv_file_name)
    
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['mpi_rank', 'mpi_size', 'func_name', 'func_count', 'func_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process each XML file in the directory
        for file_name in os.listdir(directory_path):
            if file_name.endswith('.xml'):
                xml_file_path = os.path.join(directory_path, file_name)
                mpi_times = parse_xml_and_extract_mpi_times(xml_file_path)
                # Write the mpi_times to the CSV file
                for mpi_time in mpi_times:
                    writer.writerow(mpi_time)
            break;

    print(f"Data from directory {base_name} has been written to {csv_file_path}")

# Directory path that contains multiple XML files
directory_path = ''
process_files_in_directory(directory_path)
