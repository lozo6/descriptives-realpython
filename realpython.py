import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd

x = [8.0, 1, 2.5, 4, 28.0]
x_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0]
x
# [8.0, 1, 2.5, 4, 28.0]
x_with_nan
# [8.0, 1, 2.5, nan, 4, 28.0]

math.isnan(np.nan), np.isnan(math.nan)
# (True, True)

# math.isnan(y_with_nan[3]), np.isnan(y_with_nan[3])
# (True, True)

y, y_with_nan = np.array(x), np.array(x_with_nan)
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
print(y)
print(y_with_nan)
print(z)
print(z_with_nan)

# normal way of getting mean/average
mean_ = sum(x) / len(x)
print(mean_)

# using statistics module to calculate mean/average
mean_ = statistics.mean(x)
print(mean_)

# if any values have nan in data, output will be nan
mean_ = statistics.mean(x_with_nan)
print(mean_)

# numpy can also calculate mean/average
mean_ = np.mean(y)
print(mean_)

# .mean() function built into pandas
mean_ = y.mean()
print(mean_)

# you can ignore nan using the method .nanmean()
print(np.nanmean(y_with_nan))

# pandas module ignores nan default
mean_ = z_with_nan.mean()
print(mean_)

x = [8.0, 1, 2.5, 4, 28.0]
w = [0.1, 0.2, 0.3, 0.25, 0.15]