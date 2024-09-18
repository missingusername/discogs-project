"""
This script generates visualizations for Discogs data statistics from CSV and JSON files.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import json

# Function to create bar charts
def create_bar_chart(data, column, output_path, title, xlabel, ylabel, rotation=0):
    plt.figure(figsize=(10, 6))
    plt.bar(data[column], data['Frequency'], color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Function to create line plots for genre/style distribution across time
def create_line_plots(data_by_year, title, xlabel, ylabel, output_path, remove_outliers=False):
    plt.figure(figsize=(12, 8))

    # Collect all unique genres/styles
    labels = set()
    for year_data in data_by_year.values():
        labels.update(year_data.keys())
    labels = sorted(labels)  # Sort them alphabetically for consistent plotting

    # Create a dictionary with years as x-axis and counts for each genre/style as y-axis
    genre_years = sorted(data_by_year.keys(),reverse=True)  # Sort years in ascending order

    if remove_outliers:
        genre_years = genre_years[:-1] # Remove year 0 if it exists

    for label in labels:
        counts_per_year = []
        for year in genre_years:
            # Get count for the genre/style in the current year or 0 if it doesn't exist
            counts_per_year.append(data_by_year[year].get(label, 0))
        
        # Plot each genre/style as a separate line
        plt.plot(genre_years, counts_per_year, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig(output_path)
    plt.close()


# Function to generate all visualizations
def main():
    stats_folder = 'out/stats/random_sample_10k.csv'
    
    # Load CSV files
    genre_df = pd.read_csv(os.path.join(stats_folder, 'genre_stats.csv'))
    style_df = pd.read_csv(os.path.join(stats_folder, 'style_stats.csv'))
    year_df = pd.read_csv(os.path.join(stats_folder, 'year_stats.csv'))

    # Bar charts
    create_bar_chart(genre_df, 'Genre', os.path.join(stats_folder, 'genre_distribution.png'),
                     'Genre Distribution', 'Genre', 'Frequency', rotation=90)
    create_bar_chart(style_df, 'Style', os.path.join(stats_folder, 'style_distribution.png'),
                     'Style Distribution', 'Style', 'Frequency', rotation=90)
    create_bar_chart(year_df, 'Year', os.path.join(stats_folder, 'year_distribution.png'),
                     'Year Distribution (Newest to Oldest)', 'Year', 'Frequency', rotation=90)

    # Load JSON files for genres and styles by year
    with open(os.path.join(stats_folder, 'genre_by_year.json'), 'r') as f:
        genre_by_year = json.load(f)
    
    with open(os.path.join(stats_folder, 'style_by_year.json'), 'r') as f:
        style_by_year = json.load(f)

    remove_outliers = True #remove year 0
    # Line plots
    create_line_plots(genre_by_year, 'Genre Distribution Across Time', 'Year', 'Frequency',
                      os.path.join(stats_folder, 'genre_time_distribution.png'), remove_outliers)
    create_line_plots(style_by_year, 'Style Distribution Across Time', 'Year', 'Frequency',
                      os.path.join(stats_folder, 'style_time_distribution.png'), remove_outliers)

if __name__ == '__main__':
    main()
