"""
This script generates a randomized sample from a CSV file and saves it to a new CSV file.
"""

import pandas as pd

# Function to load CSV and generate a randomized sample
def generate_random_sample(csv_file, sample_size):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Check if sample_size is greater than the number of rows in the CSV
    if sample_size > len(df):
        print("Sample size exceeds the total number of entries in the CSV.")
        return None
    
    # Generate a randomized sample of the CSV
    sample_df = df.sample(n=sample_size, random_state=None).reset_index(drop=True)
    
    return sample_df

# Define the CSV file path and desired sample size
csv_file = 'in/all_masters.csv'  # Replace with your CSV file path
sample_size = 100  # Replace with the desired number of entries in the sample

# Generate and save the sample
sample = generate_random_sample(csv_file, sample_size)
sample.to_csv('in/random_sample_100.csv', index=False)  # Save the sample to a new CSV file
