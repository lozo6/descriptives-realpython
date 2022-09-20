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

## Measures of Central Tendency ##

### MEAN ###
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

### WEIGHTED MEAN ###
# The weighted mean, also called the weighted arithmetic mean or weighted average, is a
# generalization of the arithmetic mean that enables you to define the relative contribution of
# each data point to the result.

# You can implement the weighted mean in pure Python by combining sum() with either range() or zip():
x = [8.0, 1, 2.5, 4, 28.0]
w = [0.1, 0.2, 0.3, 0.25, 0.15]
wmean = sum(w[i] * x[i] for i in range(len(x))) / sum(w)
print(wmean)
wmean = sum(x_ * w_ for (x_, w_) in zip(x, w)) / sum(w)
print(wmean)

# However, if you have large datasets, then NumPy is likely to provide a better solution. 
# You can use np.average() to get the weighted mean of NumPy arrays or Pandas Series:
y, z, w = np.array(x), pd.Series(x), np.array(w)
wmean = np.average(y, weights=w)
print(wmean)
wmean = np.average(z, weights=w)
print(wmean)

### HARMONIC MEAN ###
# The harmonic mean is the reciprocal of the mean of the reciprocals of all items in the
# dataset: 𝑛 / Σᵢ(1/𝑥ᵢ), where 𝑖 = 1, 2, …, 𝑛 and 𝑛 is the number of items in the dataset 𝑥. One
# variant of the pure Python implementation of the harmonic mean is this

hmean = len(x) / sum(1 / item for item in x)
print(hmean)

# You can also calculate this measure with statistics.harmonic_mean()
hmean = statistics.harmonic_mean(x)
print(hmean)

# The example above shows one implementation of statistics.harmonic_mean(). If you
# have a nan value in a dataset, then it’ll return nan. If there’s at least one 0, then it’ll return 0.
# If you provide at least one negative number, then you’ll get statistics.StatisticsError:

statistics.harmonic_mean(x_with_nan)
# nan
statistics.harmonic_mean([1, 0, 2])
# 0
# statistics.harmonic_mean([1, 2, -2])  # Raises StatisticsError

# A third way to calculate the harmonic mean is to use scipy.stats.hmean()
scipy.stats.hmean(y)
# 2.7613412228796843
scipy.stats.hmean(z)
# 2.7613412228796843

### GEOMETRIC MEAN ###
# The geometric mean is the 𝑛-th root of the product of all 𝑛 elements 𝑥ᵢ in a dataset 𝑥:
# ⁿ√(Πᵢ𝑥ᵢ), where 𝑖 = 1, 2, …, 𝑛. The following figure illustrates the arithmetic, harmonic, and
# geometric means of a dataset

# You can implement the geometric mean in pure Python like this
gmean = 1
for item in x:
    gmean *= item
gmean **= 1 / len(x)
print(gmean)
# 4.677885674856041

# Python 3.8 introduced statistics.geometric_mean(), which converts all values to 
# floating-point numbers and returns their geometric mean:
gmean = statistics.geometric_mean(x)
print(gmean) # currently using Python 3.9.1

# If you pass data with nan values, then statistics.geometric_mean() will behave like most 
# similar functions and return nan:
gmean = statistics.geometric_mean(x_with_nan)
print(gmean)

# You can also get the geometric mean with scipy.stats.gmean()
scipy.stats.gmean(y)
# 4.67788567485604
scipy.stats.gmean(z)
# 4.67788567485604

### MEDIAN ###

# The sample median is the middle element of a sorted dataset. The dataset can be sorted in
# increasing or decreasing order. If the number of elements 𝑛 of the dataset is odd, then the
# median is the value at the middle position: 0.5(𝑛 + 1). If 𝑛 is even, then the median is the
# arithmetic mean of the two values in the middle, that is, the items at the positions 0.5𝑛 and 0.5𝑛 + 1.

# For example, if you have the data points 2, 4, 1, 8, and 9, then the median value is 4, which is
# in the middle of the sorted dataset (1, 2, 4, 8, 9). If the data points are 2, 4, 1, and 8, then the
# median is 3, which is the average of the two middle elements of the sorted sequence (2 and 4)

# Here is one of many possible pure Python implementations of the median
n = len(x)
if n % 2:
    median_ = sorted(x)[round(0.5*(n-1))]
else:
    x_ord, index = sorted(x), round(0.5 * n)
    median_ = 0.5 * (x_ord[index-1] + x_ord[index])
print(median_)

# Two most important steps of this implementation are as follows:
# Sorting the elements of the dataset
# Finding the middle element(s) in the sorted dataset

# You can get the median with statistics.median():
median_ = statistics.median(x)
print(median_)
median_ = statistics.median(x[:-1]) # x without the last item 28.0, is [1, 2.5, 4, 8.0]
print(median_)

# median_low() and median_high() are two more functions related to the median in the
# Python statistics library. They always return an element from the dataset:
# If the number of elements is odd, then there’s a single middle value, so these
#   functions behave just like median().
# If the number of elements is even, then there are two middle values. In this case,
#   median_low() returns the lower and median_high() the higher middle value.
statistics.median_low(x[:-1])
# 2.5
statistics.median_high(x[:-1])
# 4

# Unlike most other functions from the Python statistics library, median(), median_low(),
# and median_high() don’t return nan when there are nan values among the data points:
statistics.median(x_with_nan)
# 6.0
statistics.median_low(x_with_nan)
# 4
statistics.median_high(x_with_nan)
# 8.0

### MODE ###

# The sample mode is the value in the dataset that occurs most frequently. If there isn’t a
# single such value, then the set is multimodal since it has multiple modal values. For
# example, in the set that contains the points 2, 3, 2, 8, and 12, the number 2 is the mode
# because it occurs twice, unlike the other items that occur only once.
u = [2, 3, 2, 8, 12]
mode_ = max((u.count(item), item) for item in set(u))[1]
print(f'This is the mode, {mode_}')

# You can obtain the mode with statistics.mode() and statistics.multimode():
mode_ = statistics.mode(u)
print(f'This is the mode, {mode_}, using .mode()')
mode_ = statistics.multimode(u)
print(f'This is the mode, {mode_}, using .multimode()')

# As you can see, mode() returned a single value, while multimode() returned the list that 
# contains the result. This isn’t the only difference between the two functions, though. If 
# there’s more than one modal value, then mode() raises StatisticsError, while 
# multimode() returns the list with all modes:
v = [12, 15, 12, 15, 21, 15, 12]
# statistics.mode(v)  # Raises StatisticsError
statistics.multimode(v)

# statistics.mode() and statistics.multimode() handle nan values as regular values and can return nan as the modal value:
statistics.mode([2, math.nan, 2])
# 2
statistics.multimode([2, math.nan, 2])
# [2]
statistics.mode([2, math.nan, 0, math.nan, 5])
# nan
statistics.multimode([2, math.nan, 0, math.nan, 5])
# [nan]

# Note: statistics.multimode() is introduced in Python 3.8.

#You can also get the mode with scipy.stats.mode():
u, v = np.array(u), np.array(v)
mode_ = scipy.stats.mode(u)
mode_
# ModeResult(mode=array([2]), count=array([2]))
mode_ = scipy.stats.mode(v)
mode_
# ModeResult(mode=array([12]), count=array([3]))

mode_.mode
# array([12])
mode_.count
# array([3])

# Pandas Series objects have the method .mode() that handles multimodal values well and
# ignores nan values by default:
u, v, w = pd.Series(u), pd.Series(v), pd.Series([2, 2, math.nan])
u.mode()
# 0    2
# dtype: int64
v.mode()
# 0    12
# 1    15
# dtype: int64
w.mode()
# 0    2.0
# dtype: float64

## Measures of Variability ##

### VARIANCE ###
# The sample variance quantifies the spread of the data. It shows numerically how far the 
# data points are from the mean. You can express the sample variance of the dataset 𝑥 with 𝑛
# elements mathematically as 𝑠² = Σᵢ(𝑥ᵢ − mean(𝑥))² / (𝑛 − 1), where 𝑖 = 1, 2, …, 𝑛 and mean(𝑥)
# is the sample mean of 𝑥. If you want to understand deeper why you divide the sum with 𝑛 − 1
# instead of 𝑛, then you can dive deeper into Bessel’s correction.

# Here’s how you can calculate the sample variance with pure Python:
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
print(f'This is variance using pure Python, {var_}')