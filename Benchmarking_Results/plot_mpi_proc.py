
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data
def load_data(file_paths):
    data = {}
    for key, path in file_paths.items():
        data[key] = pd.read_csv(path)
    return data

# Function to analyze MPI function times
def analyze_mpi_function_times(data):
    func_time_sum = data.groupby('func_name')['func_time'].sum().sort_values(ascending=False)
    return func_time_sum

# Function to create comparative bar plots for MPI function time across configurations
def create_mpi_time_comparison_plot(data_dict, title, file_name):
    # Preparing the data
    time_comparison_data = pd.DataFrame({config: data for config, data in data_dict.items()}).fillna(0)

    # Plotting
    plt.figure(figsize=(14, 8))
    time_comparison_data.plot(kind='bar')
    plt.title(title)
    plt.ylabel('Total Time Spent in MPI Functions (s)')
    plt.xlabel('MPI Functions')
    plt.xticks(rotation=45)
    plt.legend(title='Configuration (Dataset_Size_Processor_Count)')
    plt.tight_layout()

    # Saving the plot
    plt.savefig(file_name)
    plt.close()
    print(f"Plot saved as {file_name}")

# File paths
files_paths = {
    "64_8": "/path/to/64_8.csv",
    "128_8": "/path/to/128_8.csv",
    "128_16": "/path/to/128_16.csv",
    "256_8": "/path/to/256_8.csv",
    "256_16": "/path/to/256_16.cs",
}
# Load data
individual_data = load_data(files_paths)

# Analyzing MPI function times without removing outliers
mpi_function_times_analysis = {}
for key, data in individual_data.items():
    mpi_function_times_analysis[key] = analyze_mpi_function_times(data)

# Creating the comparative visualization with normal scale
create_mpi_time_comparison_plot(mpi_function_times_analysis, 
                                'Comparative Analysis of MPI Function Time Across Configurations',
                                'mpi_func_time_comparison.png')
