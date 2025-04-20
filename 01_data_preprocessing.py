import os
import pandas as pd
from sklearn.impute import SimpleImputer

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df = df.drop_duplicates()
    # Separate cols
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    # Impute numeric
    num_imputer = SimpleImputer(strategy='mean')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    # Fill categoricals
    df[cat_cols] = df[cat_cols].fillna('Unknown')
    # One‑hot encode
    df = pd.get_dummies(df, columns=['Continent','Region'], drop_first=True)
    # Drop code column
    if 'Country_Code' in df.columns:
        df = df.drop(columns=['Country_Code'])
    return df

def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def main():
    raw_path = 'data/sugar_consumption_dataset.csv'
    out_path = 'outputs/processed_data.csv'
    df = load_data(raw_path)
    df_proc = preprocess(df)
    save_data(df_proc, out_path)

if __name__ == '__main__':
    main()
