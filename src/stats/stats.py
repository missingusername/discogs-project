"""
This script generates various statistics from a CSV file containing music data and saves the results to CSV and JSON files.
"""
from collections import Counter
import pandas as pd
import ast
import os
import json

# Function to generate statistics from the CSV
def generate_csv_stats(input_csv):
    print("Loading CSV file...")
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv)
    print("CSV file loaded.")

    # Initialize counters for genres, styles, years, and artists
    genre_counter = Counter()
    style_counter = Counter()
    year_counter = Counter()
    artist_counter = Counter()

    # 2D dictionaries to track genre and style distributions across time
    genre_by_year = {}
    style_by_year = {}

    print("Processing columns...")
    # Explode the lists in 'genres', 'styles', and 'artist_names' columns
    df['genres'] = df['genres'].apply(ast.literal_eval)
    df['styles'] = df['styles'].apply(ast.literal_eval)
    df['artist_names'] = df['artist_names'].apply(ast.literal_eval)

    # Update counters
    genre_counter.update(df['genres'].explode())
    style_counter.update(df['styles'].explode())
    year_counter.update(df['year'])
    artist_counter.update(df['artist_names'].explode())

    print("Tracking genres and styles by year...")
    # Track genres and styles by year
    for year, group in df.groupby('year'):
        genre_by_year[year] = Counter(group['genres'].explode())
        style_by_year[year] = Counter(group['styles'].explode())

    print("Creating DataFrames for each stat...")
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

    print("Exporting stats to CSV files...")
    # Export the stats to CSV files
    genre_df.to_csv(os.path.join(output_folder, 'genre_stats.csv'), index=False)
    style_df.to_csv(os.path.join(output_folder, 'style_stats.csv'), index=False)
    year_df.to_csv(os.path.join(output_folder, 'year_stats.csv'), index=False)
    artist_df.to_csv(os.path.join(output_folder, 'artist_stats.csv'), index=False)

    print("Dumping 2D dictionaries to JSON files...")
    # Dump the 2D dictionaries to a JSON file
    with open(os.path.join(output_folder, 'genre_by_year.json'), 'w') as f:
        json.dump(genre_by_year, f)

    with open(os.path.join(output_folder, 'style_by_year.json'), 'w') as f:
        json.dump(style_by_year, f)

    print(f"Statistics CSV files and JSON files generated in {output_folder}")

def main():
    input_csv = 'in/all_masters.csv'
    generate_csv_stats(input_csv)

if __name__ == '__main__':
    main()
