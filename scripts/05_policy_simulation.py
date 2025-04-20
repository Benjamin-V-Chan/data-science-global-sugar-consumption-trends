import os
import pandas as pd
import joblib

def load_data(path):
    return pd.read_csv(path)

def simulate(df, model, tax_delta=0.05):
    sim = df.copy()
    sim['Gov_Tax'] = sim['Gov_Tax'] + tax_delta
    features = sim.drop(columns=['Country','Year','Per_Capita_Sugar_Consumption','Total_Sugar_Consumption','Avg_Daily_Sugar_Consumption'])
    sim['Predicted_Original'] = model.predict(features)
    # revert tax for new prediction
    sim['Gov_Tax'] -= tax_delta
    sim['Predicted_Simulated'] = model.predict(features.assign(Gov_Tax=sim['Gov_Tax']+tax_delta))
    return sim[['Country','Year','Per_Capita_Sugar_Consumption','Predicted_Original','Predicted_Simulated']]

def save_sim(sim_df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sim_df.to_csv(path, index=False)

def main():
    df = load_data('outputs/processed_data.csv')
    model = joblib.load('outputs/models/rf_model.pkl')
    sim_df = simulate(df, model)
    save_sim(sim_df, 'outputs/simulation_results.csv')

if __name__ == '__main__':
    main()
