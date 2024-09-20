import os
from pathlib import Path
from typing import List, Optional, Tuple
import shutil
import time

import discogs_client
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.logger_utils import get_logger
from utils.data_processing import load_csv_as_df, add_empty_column_to_df, export_df_as_csv

logger = get_logger(__name__)


def fetch_image_uri_by_master(
    df: pd.DataFrame,
    client: discogs_client.Client,
    chunk_size: int = None,
    process_unprocessed_only: bool = True,
    simulate_interruption: bool = False,
    interruption_after_chunks: int = 1,
    chunk_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    final_dir: Optional[Path] = None,
    output_file_name: str = "100_masters_with_image_uri.csv",
) -> pd.DataFrame:
    """
    Fetch image URIs for each master in the DataFrame and update the 'image_uri' column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing master IDs.
    client (discogs_client.Client): The Discogs client for making API calls.
    chunk_size (int, optional): The number of rows to process in each chunk. If None, process the entire DataFrame.
    process_unprocessed_only (bool, optional): Whether to process only unprocessed rows. Default is True.
    simulate_interruption (bool, optional): Whether to simulate an interruption. Default is False.
    interruption_after_chunks (int, optional): Number of chunks to process before simulating an interruption. Default is 1.
    chunk_dir (Optional[Path], optional): Directory to save chunk files. Default is None.
    output_dir (Optional[Path], optional): Directory to save the final output file. Default is None.

    Returns:
    pd.DataFrame: The DataFrame with the 'image_uri' column updated.
    """
    if chunk_dir is None:
        chunk_dir = Path(__file__).resolve().parents[0] / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[0] / "out"
        output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the 'image_uri' column exists
    if 'image_uri' not in df.columns:
        df = add_empty_column_to_df(df, "image_uri", "str")

    # Create a copy of the original DataFrame to preserve all rows
    original_df = df.copy()

    if process_unprocessed_only:
        df = filter_unprocessed_rows_base(df, [("image_uri", True)])

    if df.empty:
        logger.warning("The DataFrame is empty after filtering for unprocessed rows.")
        return original_df

    def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
        for row in tqdm(chunk.itertuples(), total=len(chunk), desc="Fetching Image URIs"):
            time.sleep(0.7)  # Sleep for 1 second to avoid rate limiting
            try:
                master_id = row.master_id  # Assuming the DataFrame has a column named 'Master_id'
                logger.debug(f"Processing master_id: {master_id}")  # Debugging statement
                master = client.master(master_id)
                image_uri = master.images[0]['uri'] if master.images else "Image not available"
                chunk.at[row.Index, 'image_uri'] = image_uri
            except Exception as e:
                logger.error(f"Error fetching image URI for master ID {master_id}: {e}")
                chunk.at[row.Index, 'image_uri'] = f"Error: {e}"
        return chunk

    processed_chunks = []
    try:
        if chunk_size:
            chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
            logger.info(f"Number of chunks created: {len(chunks)}")

            for i, chunk in enumerate(chunks):
                chunk_file = chunk_dir / f"chunk_{i}.csv"
                logger.info(f"Processing chunk {i+1}/{len(chunks)} with {len(chunk)} rows")
                processed_chunk = process_chunk(chunk)
                processed_chunk.to_csv(chunk_file, index=False)
                processed_chunks.append(processed_chunk)

                # Simulate an interruption
                if simulate_interruption and i + 1 >= interruption_after_chunks:
                    raise Exception("Simulated interruption")
        else:
            df = process_chunk(df)
            df.to_csv(chunk_dir / "chunk_0.csv", index=False)
            processed_chunks.append(df)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        # Combine chunks and update the input DataFrame
        combined_df = combine_chunk_files(chunk_dir, output_dir, save_combined=False, delete_chunks=True)
        original_df.set_index('master_id', inplace=True)
        combined_df.set_index('master_id', inplace=True)
        mask = combined_df['image_uri'].notna() & ~combined_df['image_uri'].str.contains("ERROR")
        indices_to_update = combined_df[mask].index.intersection(original_df.index)  # Ensure indices are common
        original_df.loc[indices_to_update, 'image_uri'] = combined_df.loc[indices_to_update, 'image_uri']
        original_df.reset_index(inplace=True)
    finally:
        if processed_chunks:
            combined_df = pd.concat(processed_chunks, ignore_index=True)
            original_df.set_index('master_id', inplace=True)
            combined_df.set_index('master_id', inplace=True)
            mask = combined_df['image_uri'].notna() & ~combined_df['image_uri'].str.contains("ERROR")
            indices_to_update = combined_df[mask].index.intersection(original_df.index)  # Ensure indices are common
            original_df.loc[indices_to_update, 'image_uri'] = combined_df.loc[indices_to_update, 'image_uri']
            original_df.reset_index(inplace=True)
            output_file_path = output_dir / output_file_name
            final_file_path = final_dir / output_file_name
            original_df.to_csv(output_file_path, index=False)
            move_file(output_file_path, final_dir)

    return original_df

def combine_chunk_files(chunk_dir: Path, output_dir: Path, save_combined: bool = False, delete_chunks: bool = True) -> pd.DataFrame:
    """
    Combine all chunk files into a single DataFrame.

    Parameters:
    chunk_dir (Path): Directory containing the chunk files.
    output_dir (Path): Directory to save the combined DataFrame.
    save_combined (bool, optional): Whether to save the combined DataFrame to a file. Default is True.
    delete_chunks (bool, optional): Whether to delete the chunk files after combining. Default is False.

    Returns:
    pd.DataFrame: The combined DataFrame.
    """
    chunk_files = sorted(chunk_dir.glob("chunk_*.csv"))
    combined_df = pd.concat([pd.read_csv(chunk_file) for chunk_file in chunk_files], ignore_index=True)

    if save_combined:
        combined_df.to_csv(output_dir / "combined_chunks.csv", index=False)

    if delete_chunks:
        for chunk_file in chunk_files:
            chunk_file.unlink()

    return combined_df

def filter_unprocessed_rows_base(
    df: pd.DataFrame, columns: List[Tuple[str, bool]]
) -> pd.DataFrame:
    conditions = [df[column[0]].isna() for column in columns]
    logger.debug(f"NaN conditions: {conditions}")

    conditions.extend(
        [df[column[0]].str.contains("ERROR", na=False) for column in columns if column[1]]
    )
    logger.debug(f"Extended conditions with 'ERROR': {conditions}")

    combined_conditions = np.logical_or.reduce(conditions)
    logger.debug(f"Combined conditions: {combined_conditions}")

    return df[combined_conditions]

def move_file(source_path: Path, destination_path: Path) -> None:
    """
    Move a file from the source directory to the destination directory.

    Parameters:
    source_path (Path): The path to the source file.
    destination_path (Path): The path to the destination directory.

    Raises:
    FileNotFoundError: If the source file does not exist.
    Exception: If there is an error during the move process.
    """
    try:
        if not source_path.exists():
            raise FileNotFoundError(f"The source file {source_path} does not exist.")

        # Ensure the destination directory exists
        destination_path.mkdir(parents=True, exist_ok=True)

        # Move the file
        shutil.move(str(source_path), str(destination_path / source_path.name))
        print(f"File moved from {source_path} to {destination_path}")
    except Exception as e:
        print(f"Error moving file from {source_path} to {destination_path}: {e}")

def main():
    # Load environment variables from .env file from root directory
    load_dotenv()
    
    # Load the CSV file into a DataFrame
    # input_csv_path = Path(__file__).resolve().parents[1] / "in" / "100_discogs_masters.csv"
    input_dir = Path(__file__).resolve().parents[1] / "in"
    input_csv_path = input_dir / "100_discogs_masters.csv"

    output_path = Path(__file__).resolve().parents[1] / "out"
    df = load_csv_as_df(input_csv_path)

    # Initialize the Discogs client with user token
    user_token = os.getenv("DISCOGS_API_KEY")
    client = discogs_client.Client(user_agent='jl-prototyping/0.1', user_token=user_token)

    df = fetch_image_uri_by_master(
        df,
        client,
        chunk_size=16,
        process_unprocessed_only=True,
        simulate_interruption=False,
        interruption_after_chunks=2,
        output_dir=output_path,
        final_dir=input_dir
    )

if __name__ == "__main__":
    main()