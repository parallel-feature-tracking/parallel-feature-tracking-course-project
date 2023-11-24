import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_time_vs_dataset_size(file_path, num_processors):
    # Load the data
    data = pd.read_csv(file_path)
    #sort the data by 'Datasize' and 'Processors'
    data = data.sort_values(by=['Datasize', 'Processors'])
    # Filter data based on the specified number of processors
    filtered_data = data[data['Processors'] == num_processors]

    # Convert Wallclock Time to seconds for easier analysis
    filtered_data['Wallclock Time (s)'] = filtered_data['Wallclock Time'].apply(
        lambda x: float(x.split()[0]) if 's' in x else float(x.split()[0]) * 60
    )

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Datasize', y='Wallclock Time (s)', data=filtered_data)
    plt.title(f'Wallclock Time vs Dataset Size for {num_processors} Processors')
    plt.xlabel('Dataset Size')
    plt.ylabel('Wallclock Time (s)')
    plt.savefig(f'wallclock_time_vs_dataset_size_{num_processors}.png')

# File path
file_path = ''  # Replace with your file path

# Example usage
plot_time_vs_dataset_size(file_path, num_processors=32)  # Replace 8 with your number of processors
