from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.logger_utils import get_logger

logger = get_logger(__name__)

def load_csv_as_df(input_csv_path: Path, progress_bar: bool = False) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    Parameters:
        input_csv_path (Path): The path to the input CSV file.
        progress_bar (bool, optional): Whether to display a progress bar during loading. Defaults to False.
    Returns:
        pd.DataFrame: The loaded CSV data as a pandas DataFrame.
    """
    logger.info(f"Trying to load CSV file: {input_csv_path}")
    if not input_csv_path.exists():
        raise FileNotFoundError(f"The file {input_csv_path} does not exist.")

    if progress_bar:
        # Initialize the progress bar in pandas
        tqdm.pandas()
        # Read the CSV file into a DataFrame with a customized progress bar
        chunks = pd.read_csv(input_csv_path, iterator=True, chunksize=1000)
        # Calc total num of chunks to set the progress bar length
        total_chunks = sum(1 for _ in chunks)
        chunks = pd.read_csv(
            input_csv_path, iterator=True, chunksize=1000
        )  # Reinitialize the iterator
        data = pd.concat(
            tqdm(chunks, desc="Processing CSV", ncols=100, total=total_chunks)
        )
        return data
    else:
        # Read the CSV file into a DataFrame
        logger.info(f"Successfully loaded CSV file: {input_csv_path.stem}")
        return pd.read_csv(input_csv_path)
    
def export_df_as_csv(df: pd.DataFrame, directory: Path, filename: str) -> None:
    """
    Export a pandas DataFrame as a CSV file.

    Parameters:
        df (pd.DataFrame): The DataFrame to be exported.
        directory (Path): The directory where the CSV file will be saved.
        filename (str): The name of the CSV file.

    Raises:
        PermissionError: If the function does not have permission to create the directory or write the file.
        OSError: If an OS error occurs when trying to create the directory.
        Exception: If an unexpected error occurs when trying to write to the file.
    """
    try:
        # Create the directory if it doesn't exist
        directory.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        logger.error(f"Permission denied when trying to create directory: {directory}")
        return
    except OSError as e:
        logger.error(f"OS error occurred when trying to create directory: {e}")
        return

    # Check if filename ends with .csv, if not, append it
    if not filename.endswith(".csv"):
        filename += ".csv"

    # Create the full file path
    file_path = directory / filename

    logger.info(f"Trying to export DataFrame to CSV: {file_path}")
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Successfully exported DataFrame to CSV: {file_path}")
    except PermissionError:
        logger.error(f"Permission denied when trying to write to file: {file_path}")
    except Exception as e:
        logger.error(f"Unexpected error occurred when trying to write to file: {e}")

def add_empty_column_to_df(df: pd.DataFrame, column_name: str, dtype: str) -> pd.DataFrame:
    """
    Add an empty column to the DataFrame with the specified name and data type.

    Parameters:
    df (pd.DataFrame): The DataFrame to which the column will be added.
    column_name (str): The name of the new column.
    dtype (str): The data type of the new column.

    Returns:
    pd.DataFrame: The DataFrame with the new column added.
    """
    df[column_name] = pd.Series(dtype=dtype)
    return df