import numpy as np
import pandas as pd

# 1. Use a DIFFERENT seed so the future data is completely new and unseen
np.random.seed(99) 
n_future_samples = 20000 

# Informative features
X1_future = np.random.normal(loc=0, scale=1, size=n_future_samples)
X2_future = np.random.normal(loc=0, scale=1, size=n_future_samples)

# Multicollinear features (correlated with X1 and X2)
X3_future = X1_future + np.random.normal(0, 0.05, size=n_future_samples)
X4_future = X2_future + np.random.normal(0, 0.05, size=n_future_samples)

# Pure noise features (irrelevant)
X5_future = np.random.normal(loc=0, scale=1, size=n_future_samples)
X6_future = np.random.normal(loc=0, scale=1, size=n_future_samples)

# Target: True relation depends ONLY on X1 and X2 + noise
noise_future = np.random.normal(0, 0.5, size=n_future_samples)
y_future = (3.5 * X1_future) + (-2.0 * X2_future) + noise_future

# Assemble DataFrame with the EXACT same column names as your training data
X_test = pd.DataFrame({
    'X1_real_1': X1_future, 
    'X2_real_2': X2_future, 
    'X3_corr_1': X3_future, 
    'X4_corr_2': X4_future, 
    'X5_noise_1': X5_future, 
    'X6_noise_2': X6_future
},)
X_test.name = 'unseen_x_test'
y_unseen_test = pd.Series(y_future, name='target')