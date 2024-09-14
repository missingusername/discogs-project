from pathlib import Path
from glob import glob
import pandas as pd
from tqdm import tqdm

def assemble_csv_files(folder_path: Path, output_file: Path) -> None:
    # Get a list of all CSV files in the folder
    csv_files = [file for file in Path(folder_path).glob('*.csv')]

    # Initialize an empty DataFrame to store the combined data
    combined_data = pd.DataFrame()

    # Iterate through each CSV file and append its data to the combined DataFrame
    for file in tqdm(csv_files):
        data = pd.read_csv(file)
        combined_data = pd.concat([combined_data, data])

    # Write the combined data to the output file
    combined_data.to_csv(output_file, index=False)

folder_path = Path('in') / 'data_100k_chunks'
output_file = Path('in') / 'all_masters.csv'
assemble_csv_files(folder_path, output_file)
