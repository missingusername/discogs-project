import pandas as pd
from collections import Counter
import ast
import os
from tqdm import tqdm

# Function to generate statistics from the CSV
def generate_csv_stats(input_csv):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv)

    # Initialize counters for genres, styles, years, and artists
    genre_counter = Counter()
    style_counter = Counter()
    year_counter = Counter()
    artist_counter = Counter()

    # Iterate through each row in the DataFrame
    for _, row in tqdm(df.iterrows()):
        # Count genres
        genres = ast.literal_eval(row['genres'])  # Convert string representation of list to actual list
        genre_counter.update(genres)
        
        # Count styles
        styles = ast.literal_eval(row['styles'])  # Convert string representation of list to actual list
        style_counter.update(styles)
        
        # Count years
        year = row['year']
        year_counter[year] += 1
        
        # Count artists
        artists = ast.literal_eval(row['artist_names'])  # Convert string representation of list to actual list
        artist_counter.update(artists)
    
    # Create DataFrames for each stat
    genre_df = pd.DataFrame(genre_counter.items(), columns=['Genre', 'Frequency']).sort_values(by='Frequency', ascending=False)
    style_df = pd.DataFrame(style_counter.items(), columns=['Style', 'Frequency']).sort_values(by='Frequency', ascending=False)
    year_df = pd.DataFrame(year_counter.items(), columns=['Year', 'Frequency']).sort_values(by='Frequency', ascending=False)
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

input_csv = 'in/all_masters.csv'
generate_csv_stats(input_csv)
