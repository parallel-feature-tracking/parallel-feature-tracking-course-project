import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_time_vs_num_procs(file_path, dataset_size):
    # Load the data
    data = pd.read_csv(file_path)
    # breakpoint()

    # Filter data based on the specified dataset size
    filtered_data = data[data['Datasize'] == dataset_size]

    # Convert Wallclock Time to seconds for easier analysis
    filtered_data['Wallclock Time (s)'] = filtered_data['Wallclock Time'].apply(
        lambda x: float(x.split()[0]) if 's' in x else float(x.split()[0]) * 60
    )
    # breakpoint()
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Processors', y='Wallclock Time (s)', data=filtered_data)
    plt.title(f'Wallclock Time vs Number of Processors for Dataset Size {dataset_size}')
    plt.xlabel('Number of Processors')
    plt.ylabel('Wallclock Time (s)')
    plt.savefig(f'wallclock_time_vs_num_procs_{dataset_size}.png')

# File path
file_path = '/'  # Replace with your file path

# Example usage
plot_time_vs_num_procs(file_path, dataset_size=256)  
