import ast  # For safely evaluating string representations of lists
from pathlib import Path

import pandas as pd
import plotly.express as px

csv_path = Path(__file__).resolve().parents[1] / "output" / "albums_with_embeddings.csv"
# Load CSV data
df = pd.read_csv(csv_path)

base_path = Path(__file__).resolve().parents[1]
input_csv = base_path / "output" / "albums_with_embeddings.csv"
output_csv = base_path / "output" / "albums_with_embeddings_sample.csv"

# Load and sample data
df = pd.read_csv(input_csv)
sampled_df = df.sample(n=10, random_state=42)  # Set random_state for reproducibility

# Save sampled data
sampled_df.to_csv(output_csv, index=False)
print(f"Saved {len(sampled_df)} rows to {output_csv}")

# Convert string representation of lists to actual lists in Genres column
df["genres"] = df["genres"].apply(ast.literal_eval)

# Filter for rows containing "Electronic" in Genres
electronic_df = df[df["genres"].apply(lambda x: "Rock" in x)]

# Create 2D scatter plot with filtered data
fig = px.scatter(
    electronic_df,
    x="x",
    y="y",
    title="2D Coordinate Visualization - Electronic Music",
)

# Customize layout
fig.update_layout(
    xaxis_title="X Coordinate",
    yaxis_title="Y Coordinate",
    template="plotly_white",
    showlegend=False,
)

# Add hover information
fig.update_traces(
    marker=dict(size=8), hovertemplate="X: %{x}<br>Y: %{y}<extra></extra>"
)

# Show plot
fig.show()
