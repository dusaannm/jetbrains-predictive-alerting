# Predictive Alerting for Cloud Metrics

## Overview
This repository contains a predictive alerting model designed for cloud infrastructure. Instead of reactive threshold-based alerting, this system uses an XGBoost classifier to predict potential incidents `H` steps into the future, based on a rolling window of `W` historical metrics.

## Problem Formulation & Dataset
- **Dataset:** Numenta Anomaly Benchmark (NAB) - Real AWS CloudWatch EC2 CPU Utilization metrics.
- **Incident Definition:** Incidents are dynamically defined as CPU spikes exceeding the 95th percentile of the dataset, rather than using a hardcoded limit.
- **Window (W):** 30 minutes. Used to extract rolling means, volatility (std), and feature lags.
- **Horizon (H):** 15 minutes. The model aims to trigger an alert 15 minutes prior to the actual incident.

## Methodology & Model Selection
I chose **XGBoost** for this task. While deep learning models (like LSTMs) are popular for time-series, Gradient Boosting provides a better balance of interpretability, fast inference times, and robust handling of tabular feature lags. 

To address the class imbalance (incidents are rare compared to normal operation), the `scale_pos_weight` parameter was utilized during training.

## Evaluation & Trade-offs
In cloud monitoring, false negatives (missed incidents) carry a much higher business cost than false positives (false alarms). Therefore, the evaluation was optimized for **Recall**.

By adjusting the classification probability threshold to `0.28`, the model achieves:
- **Recall:** ~88% (Successfully predicts the vast majority of critical spikes).
- **Precision:** ~67% (A deliberate trade-off, keeping the false alert rate manageable for a DevOps team).

## Adaptation to a Real Alerting System
In a production AWS environment, this logic could be implemented as a serverless architecture:
1. **Training:** An EventBridge scheduled task triggers an AWS Lambda function daily to retrain the model on the latest CloudWatch metrics and store the updated artifact in S3.
2. **Inference:** A separate Lambda function runs every minute, fetching the last 30 minutes of CloudWatch data, loading the model from S3, and pushing a notification to an SNS topic if the predicted probability exceeds the threshold.
