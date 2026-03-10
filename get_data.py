import pandas as pd

# Source: Numenta Anomaly Benchmark (NAB) - Real AWS EC2 CloudWatch metrics
DATA_URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv"


def fetch_and_prepare():
    print(f"Downloading metrics from {DATA_URL}...")
    df = pd.read_csv(DATA_URL)

    df.columns = ['timestamp', 'cpu_usage']
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Dynamic thresholding: identifying the top 5% of usage as critical incidents
    threshold = df['cpu_usage'].quantile(0.95)
    df['incident'] = (df['cpu_usage'] > threshold).astype(int)

    # Simulated memory usage to provide a multi-feature dataset
    df['memory_usage'] = df['cpu_usage'] * 0.6 + 25.0

    df.to_csv('server_metrics.csv', index=False)
    print(f"Ingestion complete. Detected {df['incident'].sum()} incident points at >{threshold:.2f}% CPU.")


if __name__ == "__main__":
    fetch_and_prepare()