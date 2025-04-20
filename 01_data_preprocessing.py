# 1. Load raw dataset from data/sugar_consumption_dataset.csv
# 2. Drop duplicate rows
# 3. Handle missing values:
#    - Impute all numeric columns with their mean
#    - Fill all categorical columns with 'Unknown'
# 4. One‑hot encode "Continent" and "Region"
# 5. Drop "Country_Code"
# 6. Save processed DataFrame to outputs/processed_data.csv
