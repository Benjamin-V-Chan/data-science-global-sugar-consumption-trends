import os
import pandas as pd
import matplotlib.pyplot as plt

def load_data(path):
    return pd.read_csv(path)

def summary_stats(df):
    stats = df.describe()
    out = 'outputs/summary_stats.csv'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    stats.to_csv(out)

def plot_corr(df):
    corr = df.corr()
    fig, ax = plt.subplots()
    ax.imshow(corr, aspect='auto')
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    plt.tight_layout()
    os.makedirs('outputs/figures', exist_ok=True)
    fig.savefig('outputs/figures/corr_heatmap.png')
