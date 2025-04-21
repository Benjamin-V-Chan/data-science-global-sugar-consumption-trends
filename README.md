# data-science-global-sugar-consumption-trends

## Project Overview
This project analyzes global sugar consumption trends across 200+ countries from 1960 to 2023 using a comprehensive dataset that includes economic indicators, agricultural metrics, public health statistics, and policy interventions. Through data preprocessing, exploratory data analysis (EDA), policy regression, machine learning modeling, and policy simulations, the project uncovers the relationships between sugar intake and socioeconomic factors, assesses the impact of government interventions, and builds predictive models to support future policy planning.

## Folder Structure
```
project-root/
├── data/
│   └── sugar_consumption_dataset.csv
├── scripts/
│   ├── 01_data_preprocessing.py
│   ├── 02_exploratory_analysis.py
│   ├── 03_policy_impact_analysis.py
│   ├── 04_ml_modeling.py
│   └── 05_policy_simulation.py
└── outputs/
    ├── processed_data.csv
    ├── summary_stats.csv
    ├── figures/
    │   ├── corr_heatmap.png
    │   ├── daily_intake_by_continent.png
    │   └── consumption_vs_gdp.png
    ├── policy_regression_summary.txt
    ├── models/
    │   ├── rf_model.pkl
    │   └── metrics.txt
    └── simulation_results.csv
```

## Usage
1. Setup the Project:

Clone the repository.
Ensure you have Python installed.
Install required dependencies using the requirements.txt file.
```bash
pip install -r requirements.txt
```

2. Run Data Preprocessing:
```bash
python scripts/01_data_preprocessing.py
```

3. Perform Exploratory Data Analysis:
```bash
python scripts/02_exploratory_analysis.py
```

4. Run Policy Impact Regression Analysis:
```bash
python scripts/03_policy_impact_analysis.py
```

5. Train Machine Learning Model:
```bash
python scripts/04_ml_modeling.py
```

6. Simulate Policy Intervention (e.g., Increase Gov_Tax):
```bash
python scripts/05_policy_simulation.py
```

## Requirements
- Python 3.8+
- pandas
- matplotlib
- scikit-learn
- statsmodels
- joblib

## Acknowledgments
**dataset name:** Global Sugar Consumption Trends (1960–2023)  
**dataset author:** Akshay Kumar  
**dataset source:** https://www.kaggle.com/datasets/ak0212/global-sugar-consumption-trends-19602023

