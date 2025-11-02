import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("/Users/harsha/Desktop/AI_Workshop/AIML_Projects/PayPredict/data/paypredict_cleaned.csv")

# Detect target column
target_col = None
for col in df.columns:
    if col.lower() in ["default", "label", "target", "default_payment_next_month"]:
        target_col = col
        break

if not target_col:
    raise ValueError("❌ No valid target column found in dataset!")

print(f"🎯 Target column detected: {target_col}")

# Check class distribution
print("\nBefore balancing:")
print(df[target_col].value_counts())

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Split before SMOTE to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Merge balanced data
balanced_df = pd.concat([pd.DataFrame(X_train_bal, columns=X.columns), pd.Series(y_train_bal, name=target_col)], axis=1)

print("\nAfter SMOTE balancing:")
print(balanced_df[target_col].value_counts())

# Save balanced dataset
balanced_df.to_csv("data/paypredict_balanced.csv", index=False)
print("\n✅ Balanced dataset saved as: data/paypredict_balanced.csv")