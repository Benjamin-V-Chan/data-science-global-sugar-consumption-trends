import os
import pandas as pd
import statsmodels.formula.api as smf

def load_data(path):
    return pd.read_csv(path)

def run_regression(df):
    df['Year'] = df['Year'].astype(int)
    formula = (
        'Per_Capita_Sugar_Consumption ~ Gov_Tax + Gov_Subsidies + '
        'Education_Campaign + GDP_Per_Capita + Urbanization_Rate + C(Year)'
    )
    model = smf.ols(formula=formula, data=df).fit()
    return model

def save_summary(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(model.summary().as_text())

def main():
    df = load_data('outputs/processed_data.csv')
    model = run_regression(df)
    save_summary(model, 'outputs/policy_regression_summary.txt')

if __name__ == '__main__':
    main()
