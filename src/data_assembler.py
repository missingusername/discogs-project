import os
import pandas as pd
from tqdm import tqdm

def assemble_csv_files(folder_path, output_file):
    # Get a list of all CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # Initialize an empty DataFrame to store the combined data
    combined_data = pd.DataFrame()

    # Iterate through each CSV file and append its data to the combined DataFrame
    for file in tqdm(csv_files):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)
        combined_data = pd.concat([combined_data, data])

    # Write the combined data to the output file
    combined_data.to_csv(output_file, index=False)

folder_path = os.path.join('in', 'data_100k_chunks')
output_file = os.path.join('in', 'all_masters.csv')
assemble_csv_files(folder_path, output_file)
