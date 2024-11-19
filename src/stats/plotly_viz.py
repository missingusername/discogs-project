"""
This script generates visualizations for Discogs data statistics from CSV and JSON files using Plotly.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json

# Function to create bar charts
def create_bar_chart(data, column, output_path, title, xlabel, ylabel):
    fig = px.bar(data, x=column, y='Frequency', title=title, labels={column: xlabel, 'Frequency': ylabel})
    fig.update_layout(xaxis_tickangle=-90)
    fig.write_html(output_path)

# Function to create line plots for genre/style distribution across time
def create_line_plots(data_by_year, title, xlabel, ylabel, output_path, remove_outliers=False):
    # Collect all unique genres/styles
    labels = set()
    for year_data in data_by_year.values():
        labels.update(year_data.keys())
    labels = sorted(labels)  # Sort them alphabetically for consistent plotting

    # Create a dictionary with years as x-axis and counts for each genre/style as y-axis
    genre_years = sorted(data_by_year.keys(), reverse=True)  # Sort years in ascending order

    if remove_outliers:
        genre_years = genre_years[:-1]  # Remove year 0 if it exists

    fig = go.Figure()

    for label in labels:
        counts_per_year = []
        for year in genre_years:
            # Get count for the genre/style in the current year or 0 if it doesn't exist
            counts_per_year.append(data_by_year[year].get(label, 0))
        
        # Add each genre/style as a separate line
        fig.add_trace(go.Scatter(x=genre_years, y=counts_per_year, mode='lines', name=label))

    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    fig.write_html(output_path)

# Function to generate all visualizations
def main():
    stats_folder = 'out/stats/all_masters.csv'
    
    # Load CSV files
    genre_df = pd.read_csv(os.path.join(stats_folder, 'genre_stats.csv'))
    style_df = pd.read_csv(os.path.join(stats_folder, 'style_stats.csv'))
    year_df = pd.read_csv(os.path.join(stats_folder, 'year_stats.csv'))

    # Replace year 0 with "No Year Given"
    year_df['Year'] = year_df['Year'].replace(0, 'No Year Given')
    year_df['Year'] = year_df['Year'].astype(str)

    # Bar charts
    create_bar_chart(genre_df, 'Genre', os.path.join(stats_folder, 'genre_distribution.html'),
                     'Genre Distribution', 'Genre', 'Frequency')
    create_bar_chart(style_df, 'Style', os.path.join(stats_folder, 'style_distribution.html'),
                     'Style Distribution', 'Style', 'Frequency')
    create_bar_chart(year_df, 'Year', os.path.join(stats_folder, 'year_distribution.html'),
                     'Year Distribution (Newest to Oldest)', 'Year', 'Frequency')

    # Load JSON files for genres and styles by year
    with open(os.path.join(stats_folder, 'genre_by_year.json'), 'r') as f:
        genre_by_year = json.load(f)
    
    with open(os.path.join(stats_folder, 'style_by_year.json'), 'r') as f:
        style_by_year = json.load(f)

    remove_outliers = True  # Remove year 0
    # Line plots
    create_line_plots(genre_by_year, 'Genre Distribution Across Time', 'Year', 'Frequency',
                      os.path.join(stats_folder, 'genre_time_distribution.html'), remove_outliers)
    create_line_plots(style_by_year, 'Style Distribution Across Time', 'Year', 'Frequency',
                      os.path.join(stats_folder, 'style_time_distribution.html'), remove_outliers)

if __name__ == '__main__':
    main()
