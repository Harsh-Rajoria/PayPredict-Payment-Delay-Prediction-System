import pandas as pd
import numpy as np

# Define the exact feature order your model expects
MODEL_FEATURE_ORDER = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    'avg_payment_delay', 'max_payment_delay', 'payment_volatility',
    'payment_ratio', 'credit_utilization', 'payment_trend',
    'avg_bill_amt', 'avg_pay_amt', 'bill_variance', 'utilization'
]

def preprocess_input(df):
    df = df.copy()

    # categorical encoding
    for c in ['SEX', 'EDUCATION', 'MARRIAGE']:
        if c in df.columns:
            df[c] = df[c].astype('category').cat.codes

    # compute engineered features
    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7) if f'BILL_AMT{i}' in df.columns]
    pay_cols  = [f'PAY_AMT{i}' for i in range(1, 7) if f'PAY_AMT{i}' in df.columns]

    if bill_cols:
        df['avg_bill_amt'] = df[bill_cols].mean(axis=1)
        df['bill_variance'] = df[bill_cols].var(axis=1)
        df['utilization'] = df['avg_bill_amt'] / (df['LIMIT_BAL'] + 1)
    else:
        df['avg_bill_amt'] = df['bill_variance'] = df['utilization'] = 0

    if pay_cols:
        df['avg_pay_amt'] = df[pay_cols].mean(axis=1)
        df['payment_ratio'] = df['avg_pay_amt'] / (df['avg_bill_amt'] + 1)
    else:
        df['avg_pay_amt'] = df['payment_ratio'] = 0

    # add missing engineered features
    engineered = [
        'avg_payment_delay', 'max_payment_delay', 'payment_volatility',
        'credit_utilization', 'payment_trend'
    ]
    for col in engineered:
        if col not in df.columns:
            df[col] = 0

    # Clean data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # ✅ Ensure exact feature order match
    for col in MODEL_FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0  # Add if missing (safety net)
    df = df[MODEL_FEATURE_ORDER]

    return df