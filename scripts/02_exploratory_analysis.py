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

def plot_time_series(df):
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    conts = [c for c in df.columns if c.startswith('Continent_')] + ['Continent_Unknown']
    # revert one-hot for plotting (or use original)
    # here assume original Continent kept
    df_orig = pd.read_csv('data/sugar_consumption_dataset.csv')
    fig, ax = plt.subplots()
    for cont in df_orig['Continent'].unique():
        grp = df_orig[df_orig['Continent']==cont]
        ts = grp.groupby('Year')['Avg_Daily_Sugar_Intake'].mean()
        ts.plot(ax=ax, label=cont)
    ax.legend()
    plt.tight_layout()
    fig.savefig('outputs/figures/daily_intake_by_continent.png')

def plot_scatter(df):
    fig, ax = plt.subplots()
    ax.scatter(df['GDP_Per_Capita'], df['Per_Capita_Sugar_Consumption'], alpha=0.5)
    ax.set_xlabel('GDP per Capita')
    ax.set_ylabel('Per Capita Sugar Consumption')
    plt.tight_layout()
    fig.savefig('outputs/figures/consumption_vs_gdp.png')

def main():
    df = load_data('outputs/processed_data.csv')
    summary_stats(df)
    plot_corr(df)
    plot_time_series(df)
    plot_scatter(df)

if __name__ == '__main__':
    main()
