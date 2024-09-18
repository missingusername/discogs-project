"""
This script generates various statistics from a CSV file containing music data and saves the results to CSV and JSON files.
"""

import pandas as pd
from collections import Counter
import ast
import os
from tqdm import tqdm
import json

# Function to generate statistics from the CSV
def generate_csv_stats(input_csv):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv)

    # Initialize counters for genres, styles, years, and artists
    genre_counter = Counter()
    style_counter = Counter()
    year_counter = Counter()
    artist_counter = Counter()

    # 2D dictionaries to track genre and style distributions across time
    genre_by_year = {}
    style_by_year = {}

    # Iterate through each row in the DataFrame
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Count genres
        genres = ast.literal_eval(row['genres'])  # Convert string representation of list to actual list
        genre_counter.update(genres)

        # Count styles
        styles = ast.literal_eval(row['styles'])  # Convert string representation of list to actual list
        style_counter.update(styles)

        # Count years
        year = row['year']
        year_counter[year] += 1

        # Track genres by year
        if year not in genre_by_year:
            genre_by_year[year] = Counter()
        genre_by_year[year].update(genres)

        # Track styles by year
        if year not in style_by_year:
            style_by_year[year] = Counter()
        style_by_year[year].update(styles)

        # Count artists
        artists = ast.literal_eval(row['artist_names'])  # Convert string representation of list to actual list
        artist_counter.update(artists)

    # Create DataFrames for each stat
    genre_df = pd.DataFrame(genre_counter.items(), columns=['Genre', 'Frequency']).sort_values(by='Frequency', ascending=False)
    style_df = pd.DataFrame(style_counter.items(), columns=['Style', 'Frequency']).sort_values(by='Frequency', ascending=False)
    year_df = pd.DataFrame(year_counter.items(), columns=['Year', 'Frequency']).sort_values(by='Year', ascending=False)
    artist_df = pd.DataFrame(artist_counter.items(), columns=['Artist', 'Frequency']).sort_values(by='Frequency', ascending=False)

    # Get the file name being analyzed
    file_name = os.path.basename(input_csv)

    # Create the "stats" folder if it doesn't exist
    output_folder = os.path.join('out', 'stats', file_name)
    os.makedirs(output_folder, exist_ok=True)

    # Export the stats to CSV files
    genre_df.to_csv(os.path.join(output_folder, 'genre_stats.csv'), index=False)
    style_df.to_csv(os.path.join(output_folder, 'style_stats.csv'), index=False)
    year_df.to_csv(os.path.join(output_folder, 'year_stats.csv'), index=False)
    artist_df.to_csv(os.path.join(output_folder, 'artist_stats.csv'), index=False)

    # Dump the 2D dictionaries to a JSON file
    with open(os.path.join(output_folder, 'genre_by_year.json'), 'w') as f:
        json.dump(genre_by_year, f)

    with open(os.path.join(output_folder, 'style_by_year.json'), 'w') as f:
        json.dump(style_by_year, f)

    print(f"Statistics CSV files and JSON files generated in {output_folder}")

def main():
    input_csv = 'in/random_samples/random_sample_10k.csv'
    generate_csv_stats(input_csv)

if __name__ == '__main__':
    main()
