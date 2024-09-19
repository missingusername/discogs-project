"""
This script interacts with the Discogs API to fetch image URIs for a list of master releases.

Authentication is loaded from a .env file in the root directory. An empty column in initialized in the DataFrame to store the image URIs.

The script fetches image URIs for each master release in the DataFrame using the Discogs API and updates the 'Image_uri' column.

The resulting DataFrame is then exported as a CSV file.
"""

import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import discogs_client
from dotenv import load_dotenv
import time

from utils.logger_utils import get_logger
from utils.data_processing import load_csv_as_df, add_empty_column_to_df, export_df_as_csv

logger = get_logger(__name__)


def fetch_image_uri_by_master(df: pd.DataFrame, client: discogs_client.Client) -> pd.DataFrame:
    """
    Fetch image URIs for each master in the DataFrame and update the 'Image_uri' column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing master IDs.
    client (discogs_client.Client): The Discogs client for making API calls.

    Returns:
    pd.DataFrame: The DataFrame with the 'Image_uri' column updated.
    """
    for row in tqdm(df.itertuples(), total=len(df), desc="Fetching Image URIs"):
        time.sleep(1)  # Sleep for 1 second to avoid rate limiting
        try:
            master_id = row.master_id  # Assuming the DataFrame has a column named 'Master_id'
            master = client.master(master_id)
            image_uri = master.images[0]['uri'] if master.images else "Image not available"
            df.at[row.Index, 'Image_uri'] = image_uri
        except Exception as e:
            logger.error(f"Error fetching image URI for master ID {master_id}: {e}")
            df.at[row.Index, 'Image_uri'] = None
    return df

def main():
    # Load environment variables from .env file from root directory
    load_dotenv()
    
    # Load the CSV file into a DataFrame
    input_csv_path = Path("in") / "100_discogs_masters.csv"
    df = load_csv_as_df(input_csv_path)

    # Add an empty column to the DataFrame to store the image URIs
    df = add_empty_column_to_df(df, "Image_uri", "str")

    # Initialize the Discogs client with user token
    user_token = os.getenv("DISCOGS_API_KEY")
    print(user_token)
    client = discogs_client.Client(user_agent='jl-prototyping/0.1', user_token=user_token)

    # Fetch image URIs for each master in the DataFrame
    df = fetch_image_uri_by_master(df, client)

    # Export result
    export_df_as_csv(df, Path("out"), "100_masters_with_image_uri")

if __name__ == "__main__":
    main()
