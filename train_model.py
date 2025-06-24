import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# 1. Generate dummy churn dataset
np.random.seed(0)
n = 1000
df = pd.DataFrame({
    'tenure': np.random.randint(0, 72, n),
    'monthly_charges': np.random.uniform(20, 120, n),
})
df['churn'] = ((df.tenure < 12) & (df.monthly_charges > 80)).astype(int)

# 2. Train model
X = df[['tenure', 'monthly_charges']]
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train)

# 3. Save model
joblib.dump(model, 'churn_model.joblib')

# (Optional) Check accuracy
print("Accuracy:", model.score(X_test, y_test))
