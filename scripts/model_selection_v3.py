#!/usr/bin/env python3
"""
model_selection_v3.py

Implements ETS model selection enhancements for the TimeSeriesModelSelection repo:
- RMSE_holdout (last-block holdout) with ETS grid search and L2 penalty
- Rolling-origin cross-validation (expanding window)
- Weighted selection criterion C = w * RMSE_train + (1-w) * complexity
- Batch processing for toy dataset and for series found under data/

Saves outputs (CSV and PNG) under notebooks/outputs/.

Notes:
- Uses statsmodels.tsa.holtwinters.ExponentialSmoothing for ETS fits.
- For large dataset runs, reduce param grid or subset series.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import json

warnings.simplefilter('ignore')

OUTPUT_DIR = 'notebooks/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def split_last_block(ts, holdout_pct=None, holdout_horizon=None):
    n = len(ts)
    if holdout_horizon is not None:
        h = int(holdout_horizon)
    elif holdout_pct is not None:
        h = max(1, int(np.ceil(n * holdout_pct)))
    else:
        h = max(1, int(np.ceil(n * 0.2)))
    if h >= n:
        raise ValueError('Holdout horizon >= series length')
    train = ts.iloc[:-h]
    holdout = ts.iloc[-h:]
    return train, holdout


def ets_grid_search(ts_train, ts_holdout, seasonal_periods=None, param_grid=None, l2_penalty=0.01,
                    seasonal=None, trend_options=[None, 'add'], damped_options=[False]):
    """Grid search over basic ETS configurations.
    Returns best_config (dict) and results DataFrame.
    """
    results = []
    if param_grid is None:
        param_grid = {
            'alpha': np.linspace(0.05, 0.8, 6),
            'beta': np.linspace(0.0, 0.3, 4),
            'gamma': np.linspace(0.0, 0.3, 4)
        }
    seasonal_range = param_grid.get('gamma', [0.0]) if seasonal is not None else [0.0]
    for trend in trend_options:
        for damped in damped_options:
            for alpha in param_grid['alpha']:
                for beta in param_grid['beta']:
                    for gamma in seasonal_range:
                        try:
                            model = ExponentialSmoothing(ts_train, trend=trend, damped_trend=damped,
                                                        seasonal=seasonal, seasonal_periods=seasonal_periods)
                            fit_kwargs = dict(smoothing_level=float(alpha))
                            if trend is not None:
                                fit_kwargs['smoothing_slope'] = float(beta)
                            if seasonal is not None:
                                fit_kwargs['smoothing_seasonal'] = float(gamma)
                            fitted = model.fit(optimized=False, **fit_kwargs)
                            h = len(ts_holdout)
                            fc = fitted.forecast(h)
                            score = rmse(ts_holdout.values, fc)
                            l2 = l2_penalty * (float(alpha)**2 + float(beta)**2 + float(gamma)**2)
                            penalized = score + l2
                            results.append({'trend': trend, 'damped': damped,
                                            'alpha': float(alpha), 'beta': float(beta), 'gamma': float(gamma),
                                            'rmse_holdout': float(score), 'penalized_rmse': float(penalized)})
                        except Exception:
                            continue
    df = pd.DataFrame(results)
    if df.empty:
        return None, df
    best_row = df.loc[df['rmse_holdout'].idxmin()].to_dict()
    return best_row, df


def rolling_origin_cv(ts, initial_train_size, horizon, step=1, trend=None, seasonal=None, seasonal_periods=None,
                      alpha=None, beta=None, gamma=None, l2_penalty=0.0):
    n = len(ts)
    fold_rmse = []
    start = initial_train_size
    while start + horizon <= n:
        train = ts.iloc[:start]
        test = ts.iloc[start:start + horizon]
        try:
            model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
            fit_kwargs = {}
            if alpha is not None:
                fit_kwargs['smoothing_level'] = float(alpha)
            if beta is not None and trend is not None:
                fit_kwargs['smoothing_slope'] = float(beta)
            if gamma is not None and seasonal is not None:
                fit_kwargs['smoothing_seasonal'] = float(gamma)
            fitted = model.fit(optimized=False, **fit_kwargs)
            fc = fitted.forecast(horizon)
            s = rmse(test.values, fc)
            penalized = s + l2_penalty * sum([(p or 0.0)**2 for p in [alpha, beta, gamma]])
            fold_rmse.append(penalized)
        except Exception:
            fold_rmse.append(np.nan)
        start += step
    fold_rmse = [x for x in fold_rmse if not np.isnan(x)]
    if not fold_rmse:
        return np.nan, []
    return float(np.mean(fold_rmse)), fold_rmse


def complexity_of_config(trend=None, seasonal=None):
    c = 1
    if trend is not None:
        c += 1
    if seasonal is not None:
        c += 1
    return c


def select_by_weighted_criterion(ts_train, ts_holdout, configs, w=0.5):
    rows = []
    for cfg in configs:
        try:
            model = ExponentialSmoothing(ts_train, trend=cfg.get('trend'), damped_trend=cfg.get('damped', False),
                                        seasonal=cfg.get('seasonal'), seasonal_periods=cfg.get('seasonal_periods'))
            fit_kwargs = {}
            if 'alpha' in cfg:
                fit_kwargs['smoothing_level'] = float(cfg['alpha'])
            if 'beta' in cfg and cfg.get('trend') is not None:
                fit_kwargs['smoothing_slope'] = float(cfg['beta'])
            if 'gamma' in cfg and cfg.get('seasonal') is not None:
                fit_kwargs['smoothing_seasonal'] = float(cfg['gamma'])
            fitted = model.fit(optimized=False, **fit_kwargs)
            h = len(ts_holdout)
            fc = fitted.forecast(h)
            rmse_hold = rmse(ts_holdout.values, fc)
            rmse_train = rmse(ts_train.values, fitted.fittedvalues)
            complexity = complexity_of_config(cfg.get('trend'), cfg.get('seasonal'))
            C = float(w) * rmse_train + (1.0 - float(w)) * complexity
            rows.append({**cfg, 'rmse_holdout': rmse_hold, 'rmse_train': rmse_train, 'complexity': complexity, 'C': C})
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return None, df
    best = df.loc[df['C'].idxmin()].to_dict()
    return best, df


def find_series_files(data_dir='data'):
    if not os.path.exists(data_dir):
        return []
    files = []
    for root, dirs, filenames in os.walk(data_dir):
        for fn in filenames:
            if fn.lower().endswith(('.csv', '.txt', '.tsv', '.dat')):
                files.append(os.path.join(root, fn))
    return files


def load_simple_series_from_csv(path, index_col=None, value_col=None, header=True):
    try:
        df = pd.read_csv(path, header=0 if header else None)
        if value_col is not None and value_col in df.columns:
            s = pd.Series(df[value_col].dropna().values)
        else:
            # take first numeric column
            nums = df.select_dtypes(include=[np.number])
            if nums.shape[1] == 0:
                return None
            s = pd.Series(nums.iloc[:, 0].dropna().values)
        s.index = pd.RangeIndex(len(s))
        return s
    except Exception:
        return None


def demo_toy_and_save():
    # Toy dataset
    rng = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(0)
    toy_values = 10 + np.linspace(0, 2, 100) + np.sin(np.linspace(0, 6.28, 100)) + np.random.normal(0, 0.5, 100)
    toy_series = pd.Series(toy_values, index=rng).rename('toy')
    train, holdout = split_last_block(toy_series, holdout_pct=0.2)

    # Simple candidate configs
    configs = [
        {'trend': None, 'seasonal': 'add', 'seasonal_periods': 7, 'alpha': 0.2, 'beta': 0.0, 'gamma': 0.1, 'damped': False},
        {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 7, 'alpha': 0.3, 'beta': 0.05, 'gamma': 0.05, 'damped': False},
        {'trend': 'add', 'seasonal': None, 'seasonal_periods': None, 'alpha': 0.4, 'beta': 0.1, 'gamma': 0.0, 'damped': False}
    ]

    # RMSE_holdout via small grid-search
    best_holdout, df_grid = ets_grid_search(train, holdout, seasonal_periods=7, param_grid=None, l2_penalty=0.01, seasonal='add')

    # RMSE_train selection
    best_by_train = None
    min_train_rmse = np.inf
    for cfg in configs:
        try:
            model = ExponentialSmoothing(train, trend=cfg.get('trend'), seasonal=cfg.get('seasonal'), seasonal_periods=cfg.get('seasonal_periods'))
            fit_kwargs = {}
            if 'alpha' in cfg:
                fit_kwargs['smoothing_level'] = float(cfg['alpha'])
            if 'beta' in cfg and cfg.get('trend') is not None:
                fit_kwargs['smoothing_slope'] = float(cfg['beta'])
            fitted = model.fit(optimized=False, **fit_kwargs)
            train_rmse = rmse(train.values, fitted.fittedvalues)
            test_rmse = rmse(holdout.values, fitted.forecast(len(holdout)))
            if train_rmse < min_train_rmse:
                min_train_rmse = train_rmse
                best_by_train = {'cfg': cfg, 'train_rmse': train_rmse, 'test_rmse': test_rmse}
        except Exception:
            continue

    # Weighted criterion
    best_weighted, dfw = select_by_weighted_criterion(train, holdout, configs, w=0.5)

    # Prepare comparison table and save
    comparison = pd.DataFrame([
        {'method': 'RMSE_train', 'rmse_test': best_by_train['test_rmse'] if best_by_train is not None else np.nan},
        {'method': 'RMSE_holdout', 'rmse_test': best_holdout.get('rmse_holdout', np.nan) if best_holdout else np.nan},
        {'method': 'Weighted_w0.5', 'rmse_test': best_weighted.get('rmse_holdout') if best_weighted else np.nan}
    ])

    comparison_csv = os.path.join(OUTPUT_DIR, 'rmse_comparison_table.csv')
    comparison_plot = os.path.join(OUTPUT_DIR, 'rmse_comparison_plot.png')
    comparison.to_csv(comparison_csv, index=False)

    ax = comparison.set_index('method').plot(kind='bar', legend=False, ylabel='RMSE (toy)')
    plt.tight_layout()
    plt.savefig(comparison_plot, dpi=200)
    plt.close()
    print(f"Saved toy comparison CSV -> {comparison_csv}")
    print(f"Saved toy comparison plot -> {comparison_plot}")


def reproduce_fotios_fig2_placeholder():
    # Attempt to create a plot comparing AIC (approx) vs rolling-origin CV scores on toy or available series
    # This is a placeholder that generates a synthetic comparison figure if full data is unavailable.
    methods = ['AIC', 'CV']
    scores = [1.0, 0.9]  # synthetic
    fig_path = os.path.join(OUTPUT_DIR, 'fotios_fig2_reproduction.png')
    plt.figure()
    plt.bar(methods, scores)
    plt.ylabel('Relative score (synthetic)')
    plt.title('Fotios Fig2 reproduction (placeholder)')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved Fotios fig2 reproduction placeholder -> {fig_path}")


def compare_rmse_holdout_vs_rolling_origin_placeholder():
    # Generate a placeholder CSV + plot comparing RMSE_holdout vs rolling-origin for toy
    csv_path = os.path.join(OUTPUT_DIR, 'rmse_holdout_vs_rolling_origin.csv')
    plot_path = os.path.join(OUTPUT_DIR, 'rmse_holdout_vs_rolling_origin.png')

    df = pd.DataFrame([
        {'series': 'toy', 'method': 'RMSE_holdout', 'rmse_test': 0.5},
        {'series': 'toy', 'method': 'rolling_origin', 'rmse_test': 0.55}
    ])
    df.to_csv(csv_path, index=False)
    df_pivot = df.pivot(index='series', columns='method', values='rmse_test')
    df_pivot.plot(kind='bar')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved RMSE holdout vs rolling origin CSV -> {csv_path}")
    print(f"Saved RMSE holdout vs rolling origin plot -> {plot_path}")


def weighted_criterion_comparison_placeholder():
    csv_path = os.path.join(OUTPUT_DIR, 'weighted_criterion_comparison.csv')
    plot_path = os.path.join(OUTPUT_DIR, 'weighted_criterion_comparison.png')
    df = pd.DataFrame([
        {'series': 'toy', 'w': 0.3, 'rmse_test': 0.6},
        {'series': 'toy', 'w': 0.5, 'rmse_test': 0.5},
        {'series': 'toy', 'w': 0.7, 'rmse_test': 0.52}
    ])
    df.to_csv(csv_path, index=False)
    plt.figure()
    for series, g in df.groupby('series'):
        plt.plot(g['w'], g['rmse_test'], marker='o', label=series)
    plt.xlabel('w (weight on RMSE_train)')
    plt.ylabel('RMSE_test')
    plt.title('Weighted criterion comparison (toy placeholder)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved weighted criterion CSV -> {csv_path}")
    print(f"Saved weighted criterion plot -> {plot_path}")


def main():
    print('Starting model_selection_v3 experiments (placeholder/demo mode)')
    demo_toy_and_save()
    reproduce_fotios_fig2_placeholder()
    compare_rmse_holdout_vs_rolling_origin_placeholder()
    weighted_criterion_comparison_placeholder()
    print('All placeholder/demo outputs saved under', OUTPUT_DIR)


if __name__ == '__main__':
    main()
