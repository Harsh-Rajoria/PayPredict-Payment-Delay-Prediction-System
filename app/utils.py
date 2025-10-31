import pandas as pd
import numpy as np

# expected training features (from your model training script)
TRAIN_FEATURES = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
    'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
    'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6',
    'avg_bill_amt','avg_pay_amt','payment_ratio','bill_variance','utilization'
]

def preprocess_input(df):
    df = df.copy()
    for c in ['SEX', 'EDUCATION', 'MARRIAGE']:
        if c in df.columns:
            df[c] = df[c].astype('category').cat.codes

    bill_cols = [f'BILL_AMT{i}' for i in range(1,7)]
    pay_cols = [f'PAY_AMT{i}' for i in range(1,7)]

    df['avg_bill_amt'] = df[bill_cols].mean(axis=1)
    df['avg_pay_amt'] = df[pay_cols].mean(axis=1)
    df['payment_ratio'] = df['avg_pay_amt'] / (df['avg_bill_amt'] + 1)
    df['bill_variance'] = df[bill_cols].var(axis=1)
    df['utilization'] = df['avg_bill_amt'] / (df['LIMIT_BAL'] + 1)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # ensure all training features present
    for c in TRAIN_FEATURES:
        if c not in df.columns:
            df[c] = 0

    return df[TRAIN_FEATURES]