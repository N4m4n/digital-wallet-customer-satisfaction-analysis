# Digital Wallet Customer Satisfaction Analysis

## Project question

To what extent do transaction activity, app usage frequency, loyalty points, cashback received, support-ticket volume, and issue-resolution time explain customer satisfaction in a digital-wallet software product?

## Dataset

Input file: `digital_wallet_ltv_dataset.csv`

The dataset contains 7,000 customer-level records and 20 columns. It includes transaction behavior, engagement, loyalty, support, satisfaction, and lifetime-value variables. The analysis uses customer satisfaction as the dependent variable and does not use LTV as a predictor.

## Method

The script performs:

- Basic data-quality checks.
- Duplicate removal.
- Descriptive summaries.
- Group summaries by app-usage frequency and preferred payment method.
- Correlations.
- Multiple ordinary least-squares regression.
- PNG charts and CSV result tables.

Continuous predictors are standardized before regression so their coefficients are comparable per one standard deviation. Categorical predictors are one-hot encoded with the first category as the reference category.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Place the CSV and script in the same folder, then run:

```bash
python analyze_wallet_ltv.py --input digital_wallet_ltv_dataset.csv --output output
```

## Outputs

The `output` directory contains:

- `analysis_summary.txt`
- `regression_coefficients.csv`
- `satisfaction_by_usage.csv`
- `satisfaction_by_payment.csv`
- `correlations_with_satisfaction.csv`
- `satisfaction_distribution.png`
- `satisfaction_by_usage.png`
- `resolution_vs_satisfaction.png`
- `correlation_heatmap.png`

## Interpretation caution

This is an observational dataset. Regression coefficients show associations, not proof of causation. The dataset is synthetic according to its Kaggle description; findings should not be presented as estimates of the entire digital-wallet market.
