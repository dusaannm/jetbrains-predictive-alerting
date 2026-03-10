import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Configuration
WINDOW_SIZE = 30
HORIZON = 15
ALERT_PROBABILITY_THRESHOLD = 0.28  # Precision-Recall optimization


def engineer_features(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Target labeling
    df['target'] = df['incident'].shift(-1).rolling(window=HORIZON, min_periods=1).max()

    # 1. Temporal Features (Seasonality)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # 2. Advanced Rolling Metrics
    df['cpu_rolling_mean'] = df['cpu_usage'].rolling(window=WINDOW_SIZE).mean()
    df['cpu_volatility'] = df['cpu_usage'].rolling(window=WINDOW_SIZE).std()  # Variance check
    df['cpu_max'] = df['cpu_usage'].rolling(window=WINDOW_SIZE).max()

    # 3. Lag Features (Trend detection)
    for lag in [5, 10, 15]:
        df[f'cpu_lag_{lag}'] = df['cpu_usage'].shift(lag)

    # 4. Rate of Change (Momentum)
    df['cpu_delta'] = df['cpu_usage'].diff(periods=5)

    return df.dropna()


def train_production_model():
    df = engineer_features('server_metrics.csv')

    # Chronological Split (No shuffling for time-series!)
    split_idx = int(len(df) * 0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    feature_cols = [
        'hour', 'day_of_week', 'cpu_rolling_mean', 'cpu_volatility',
        'cpu_max', 'cpu_lag_5', 'cpu_lag_10', 'cpu_delta'
    ]

    X_train, y_train = train[feature_cols], train['target']
    X_test, y_test = test[feature_cols], test['target']

    # XGBoost with specific parameters to prevent overfitting
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=3,  # Handling imbalanced data (incidents are rare)
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # Threshold tuning for early warning system
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= ALERT_PROBABILITY_THRESHOLD).astype(int)

    print("\n=== Production-Ready Model Evaluation ===")
    print(f"Metrics extracted from {len(feature_cols)} features.")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    train_production_model()