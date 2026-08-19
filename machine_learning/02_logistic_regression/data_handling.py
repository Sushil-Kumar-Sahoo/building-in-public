import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNNImputer
from sklearn.linear_model import LinearRegression

# 1. Load dataset
df = pd.read_csv('diabetes.csv')

# 2. Replace 0s with NaN for invalid physiological columns
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

# 3. SPLIT DATASET FIRST
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# STEP A: Impute BMI using [Age, Pregnancies]
# ---------------------------------------------------------
knn_bmi = KNNImputer(n_neighbors=5)

# Fit on train, transform both
X_train['BMI'] = knn_bmi.fit_transform(X_train[['Age', 'Pregnancies', 'BMI']])[:, 2]
X_test['BMI'] = knn_bmi.transform(X_test[['Age', 'Pregnancies', 'BMI']])[:, 2]

# ---------------------------------------------------------
# STEP B: Impute SkinThickness using [BMI, Age]
# ---------------------------------------------------------
# Filter rows where SkinThickness is NOT NaN in training set to fit the regression
train_skin_known = X_train.dropna(subset=['SkinThickness'])

lr_skin = LinearRegression()
lr_skin.fit(train_skin_known[['BMI', 'Age']], train_skin_known['SkinThickness'])

# Predict missing SkinThickness for Train
train_skin_missing = X_train['SkinThickness'].isna()
X_train.loc[train_skin_missing, 'SkinThickness'] = lr_skin.predict(
    X_train.loc[train_skin_missing, ['BMI', 'Age']]
)

# Predict missing SkinThickness for Test using the SAME fitted model
test_skin_missing = X_test['SkinThickness'].isna()
X_test.loc[test_skin_missing, 'SkinThickness'] = lr_skin.predict(
    X_test.loc[test_skin_missing, ['BMI', 'Age']]
)

# ---------------------------------------------------------
# STEP C: Impute Glucose using KNN
# ---------------------------------------------------------
knn_glucose = KNNImputer(n_neighbors=5)

X_train['Glucose'] = knn_glucose.fit_transform(X_train[['Age', 'BMI', 'Pregnancies', 'Glucose']])[:, 3]
X_test['Glucose'] = knn_glucose.transform(X_test[['Age', 'BMI', 'Pregnancies', 'Glucose']])[:, 3]

# ---------------------------------------------------------
# STEP D: Impute Insulin using KNN
# ---------------------------------------------------------
knn_insulin = KNNImputer(n_neighbors=5)

insulin_cols = ['Glucose', 'BMI', 'Age', 'SkinThickness', 'Insulin']
X_train['Insulin'] = knn_insulin.fit_transform(X_train[insulin_cols])[:, 4]
X_test['Insulin'] = knn_insulin.transform(X_test[insulin_cols])[:, 4]