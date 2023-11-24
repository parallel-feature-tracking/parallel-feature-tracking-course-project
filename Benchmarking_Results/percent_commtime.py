import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ensure the plot appears inline in Jupyter Notebook
# %matplotlib inline  # Uncomment this line if you're using Jupyter Notebook

def plot_comm_percentage_vs_datasize_processor(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    
    # Convert 'Datasize' and 'Processors' to string for categorical plotting
    data['Datasize'] = data['Datasize'].astype(str)
    data['Processors'] = data['Processors'].astype(str)
    
    # sort the data by 'Datasize' and 'Processors'
    data = data.sort_values(by=['Datasize', 'Processors'])

    # Convert '%comm' from percentage string to float
    data['%comm'] = data['%comm'].str.rstrip('%').astype('float')

    # Group the data by 'Datasize' and 'Processors' and calculate the mean of '%comm'
    grouped_data = data.groupby(['Datasize', 'Processors'])['%comm'].mean().reset_index()
    grouped_data['%comm'] = grouped_data['%comm'].apply(lambda x: round(x, 2))
    breakpoint()
    # Create a bar plot to visualize %comm for each combination of 'Datasize' and 'Processors'
    plt.figure(figsize=(14, 7))
    barplot = sns.barplot(data=grouped_data, x='Processors',  hue='Datasize', y='%comm')

    # Customize the plot
    plt.title('Average Communication Time Percentage vs Dataset Size and Processors')
    plt.xlabel('Dataset Size')
    plt.ylabel('Average Communication Time Percentage (%)')
    plt.legend(title='Number of Processors')

    # # Display the values on the bars
    # for p in barplot.patches:
    #     barplot.annotate(format(p.get_height(), '.1f'), 
    #                      (p.get_x() + p.get_width() / 2., p.get_height()), 
    #                      ha = 'center', va = 'center', 
    #                      xytext = (0, 9), 
    #                      textcoords = 'offset points')

    # Save the figure
    # plt.tight_layout()  # Adjust the plot to make sure everything fits without overlapping
    plt.savefig('comm_percentage_vs_datasize_processor.png')

# Replace this path with the path to your CSV file
file_path = 'extracted_data.csv'

# Run the function
plot_comm_percentage_vs_datasize_processor(file_path)
