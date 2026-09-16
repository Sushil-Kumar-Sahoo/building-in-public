import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 100000

# Informative features
X1 = np.random.normal(loc=0, scale=1, size=n_samples)
X2 = np.random.normal(loc=0, scale=1, size=n_samples)

# Multicollinear features (correlated with X1 and X2)
X3 = X1 + np.random.normal(0, 0.05, size=n_samples)
X4 = X2 + np.random.normal(0, 0.05, size=n_samples)

# Pure noise features (irrelevant)
X5 = np.random.normal(loc=0, scale=1, size=n_samples)
X6 = np.random.normal(loc=0, scale=1, size=n_samples)

# Target: True relation depends ONLY on X1 and X2 + noise
noise = np.random.normal(0, 0.5, size=n_samples)
y = (3.5 * X1) + (-2.0 * X2) + noise

# Assemble DataFrame
X = pd.DataFrame({'X1_real_1': X1, 'X2_real_2': X2, 
                  'X3_corr_1': X3, 'X4_corr_2': X4, 
                  'X5_noise_1': X5, 'X6_noise_2': X6})

