"""
Simplified analysis script for the property sales data.  Identical to
`property_sales_analysis.py` but uses much smaller hyperparameter grids
for the tree‑based models.  This reduces execution time while still
illustrating the modelling workflow.
"""

import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def clean_numeric(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    try:
        value_str = str(value)
        value_str = value_str.strip().replace('$', '').replace(',', '')
        for ch in ['(', ')', '€']:
            value_str = value_str.replace(ch, '')
        return float(value_str)
    except Exception:
        return np.nan

def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])
    df.columns = [col.strip() for col in df.columns]
    numeric_cols_to_clean = [
        'Sale_price', 'FinishedSqft', 'Lotsize', 'Rooms', 'Units',
        'Bdrms', 'Fbath', 'Hbath', 'Stories'
    ]
    for col in numeric_cols_to_clean:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)
    if 'Year_Built' in df.columns:
        df['Year_Built'] = pd.to_numeric(df['Year_Built'], errors='coerce')
    if 'Sale_date' in df.columns:
        df['Sale_date'] = pd.to_datetime(df['Sale_date'], errors='coerce')
    df = df.dropna(subset=['Sale_price', 'Sale_date'])
    df['Sale_year'] = df['Sale_date'].dt.year
    df['Sale_month'] = df['Sale_date'].dt.month
    df['Sale_day'] = df['Sale_date'].dt.day
    df['Pandemic'] = (df['Sale_date'] >= pd.Timestamp('2020-02-01')).astype(int)
    return df

def build_and_evaluate_models(df: pd.DataFrame) -> dict:
    categorical_features: List[str] = ['PropType', 'District']
    numerical_features: List[str] = [
        'Year_Built', 'FinishedSqft', 'Lotsize', 'Rooms', 'Units',
        'Bdrms', 'Fbath', 'Hbath', 'Stories', 'Sale_year', 'Sale_month',
        'Sale_day', 'Pandemic'
    ]
    missing_cats = [c for c in categorical_features if c not in df.columns]
    missing_nums = [c for c in numerical_features if c not in df.columns]
    if missing_cats or missing_nums:
        raise KeyError(f"Missing expected columns: {missing_cats + missing_nums}")
    X = df[categorical_features + numerical_features]
    y = df['Sale_price']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    results = {}
    def compute_metrics(name, model):
        pred = model.predict(X_test)
        results[name] = {
            'MAE': mean_absolute_error(y_test, pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
            'R2': r2_score(y_test, pred)
        }
    # linear
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    compute_metrics('Linear Regression', lr_pipeline)
    # decision tree
    dt_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(random_state=42))
    ])
    dt_param_grid = {
        'regressor__max_depth': [5, 10, None],
        'regressor__min_samples_leaf': [1, 5]
    }
    dt_search = GridSearchCV(
        dt_pipeline,
        dt_param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    dt_search.fit(X_train, y_train)
    best_dt = dt_search.best_estimator_
    compute_metrics('Decision Tree', best_dt)
    # random forest
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    rf_param_grid = {
        'regressor__n_estimators': [100],
        'regressor__max_depth': [None, 10],
        'regressor__min_samples_leaf': [1]
    }
    rf_search = GridSearchCV(
        rf_pipeline,
        rf_param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    compute_metrics('Random Forest', best_rf)
    # gradient boosting
    gb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])
    gb_param_grid = {
        'regressor__n_estimators': [100],
        'regressor__learning_rate': [0.1],
        'regressor__max_depth': [3]
    }
    gb_search = GridSearchCV(
        gb_pipeline,
        gb_param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    gb_search.fit(X_train, y_train)
    best_gb = gb_search.best_estimator_
    compute_metrics('Gradient Boosting', best_gb)
    return results

def analyse_pandemic_effect(df: pd.DataFrame) -> dict:
    pre = df[df['Pandemic'] == 0]['Sale_price']
    post = df[df['Pandemic'] == 1]['Sale_price']
    return {
        'pre_pandemic_count': len(pre),
        'pre_pandemic_mean': pre.mean(),
        'pandemic_count': len(post),
        'pandemic_mean': post.mean()
    }

def main():
    cwd = os.path.dirname(__file__)
    csv_files = [os.path.join(cwd, f) for f in os.listdir(cwd) if f.endswith('.csv')]
    frames = []
    for f in csv_files:
        try:
            frames.append(load_and_clean(f))
        except Exception as e:
            print(f"Skipped {f}: {e}")
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all[df_all['Sale_year'] >= 2019]
    df_all = df_all[df_all['Sale_price'] > 0]
    results = build_and_evaluate_models(df_all)
    pandemic_stats = analyse_pandemic_effect(df_all)
    print('Model performance metrics:')
    for name, metrics in results.items():
        print(f" {name}: MAE={metrics['MAE']:.2f}, RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.3f}")
    print('\nPandemic effect:')
    print(f"Pre‑pandemic mean sale price (n={pandemic_stats['pre_pandemic_count']}): {pandemic_stats['pre_pandemic_mean']:.2f}")
    print(f"Pandemic mean sale price (n={pandemic_stats['pandemic_count']}): {pandemic_stats['pandemic_mean']:.2f}")

if __name__ == '__main__':
    main()
