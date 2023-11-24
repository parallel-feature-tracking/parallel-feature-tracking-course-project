from bs4 import BeautifulSoup
import csv
import os

def extract_data_from_html(html_content, datasize, processors, x, y, z, iteration):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extracting details
    codename = soup.find_all('td', align='right')[0].text.strip()
    state = soup.find_all('td', align='right')[1].text.strip()
    username = soup.find_all('td', align='right')[2].text.strip()
    group = soup.find_all('td', align='right')[3].text.strip()
    host = soup.find_all('td', align='right')[4].text.strip()
    mpi_tasks = soup.find_all('td', align='right')[5].text.strip()
    start_time = soup.find_all('td', align='right')[6].text.strip()
    wallclock_time = soup.find_all('td', align='right')[7].text.strip()
    stop_time = soup.find_all('td', align='right')[8].text.strip()
    comm = soup.find_all('td', align='right')[9].text.strip()
    total_memory = soup.find_all('td', align='right')[10].text.strip()
    total_gflops = soup.find_all('td', align='right')[11].text.strip()
    
    # Extracting hostlist and ranks
    hostlist_table = soup.find_all('table')[-1]
    hosts = [row.find_all('td')[0].text.strip() for row in hostlist_table.find_all('tr')[1:]]
    ranks = [row.find_all('td')[1].text.strip() for row in hostlist_table.find_all('tr')[1:]]
    
    # Create a data tuple for each host and rank
    data = [(codename, state, username, group, host, mpi_tasks, start_time, wallclock_time, stop_time, comm, total_memory, total_gflops, datasize, processors, x, y, z, iteration, h, r) for h, r in zip(hosts, ranks)]
    return data

def main():
    # Path to the directory containing profiler report folders
    directory_path = './profiler_reports'
    
    # CSV file to save the extracted data
    csv_file = 'extracted_data.csv'
    
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(["Codename", "State", "Username", "Group", "Host", "MPI Tasks", "Start Time", "Wallclock Time", "Stop Time", "%comm", "Total Memory", "Total GFlops", "Datasize", "Processors", "X", "Y", "Z", "Iteration", "Hostlist", "Ranks"])
        
        # Walk through the directory
        for root, dirs, files in os.walk(directory_path):
            for folder in dirs:
                # Parse folder name to get parameters
                folder_parts = folder.split('_')
                # breakpoint()
                
                datasize, processors, x, y, z, iteration = folder_parts[0], folder_parts[1], folder_parts[2], folder_parts[3], folder_parts[4], folder_parts[5]
                
                # Construct path to the HTML file
                html_path = os.path.join(root, folder, 'index.html')
                if os.path.exists(html_path):
                    # Read and process the HTML file
                    with open(html_path, 'r') as html_file:
                        html_content = html_file.read()
                        data = extract_data_from_html(html_content, datasize, processors, x, y, z, iteration)
                        # Write the data to the CSV file
                        writer.writerows(data)

if __name__ == "__main__":
    main()
