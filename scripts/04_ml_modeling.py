import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_data(path):
    return pd.read_csv(path)

def prepare_xy(df):
    y = df['Per_Capita_Sugar_Consumption']
    X = df.drop(columns=['Country','Year','Per_Capita_Sugar_Consumption','Total_Sugar_Consumption','Avg_Daily_Sugar_Intake'])
    return X, y

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2  = r2_score(y_test, preds)
    return rmse, r2

def save_artifacts(model, metrics, model_path, metrics_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    with open(metrics_path, 'w') as f:
        f.write(f'RMSE: {metrics[0]:.4f}\nR2: {metrics[1]:.4f}')

def main():
    df = load_data('outputs/processed_data.csv')
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    save_artifacts(model, metrics, 'outputs/models/rf_model.pkl', 'outputs/models/metrics.txt')

if __name__ == '__main__':
    main()
