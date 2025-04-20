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
