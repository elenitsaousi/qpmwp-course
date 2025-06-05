import pandas as pd

# Load your data
jkp_data = pd.read_parquet("/Users/elenetsaouse/Downloads/jkp_data.parquet")

# Pick a date for testing
date = "1999-01-01"

# Filter
filtered = jkp_data.loc[
    (jkp_data.index.get_level_values("date") < date) &
    (jkp_data.index.get_level_values("date") >= pd.to_datetime(date) - pd.Timedelta(days=365))
]

# Output
print("Filtered shape:", filtered.shape)
print("Available columns:", filtered.columns.tolist())
print(filtered.head())
