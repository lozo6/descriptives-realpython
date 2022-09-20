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
print(mode_)
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

# This approach is sufficient and calculates the sample variance well. However, the shorter
# and more elegant solution is to call the existing function statistics.variance():
var_ = statistics.variance(x)
print(f'This is the variance using Statistics module, {var_}')

# If you have nan values among your data, then statistics.variance() will return nan:
statistics.variance(x_with_nan)
# nan

# You can also calculate the sample variance with NumPy. You should use the function
# np.var() or the corresponding method .var():
var_ = np.var(y, ddof=1)
var_
# 123.19999999999999
var_ = y.var(ddof=1)
var_
# 123.19999999999999

# It’s very important to specify the parameter ddof=1. That’s how you set the delta degrees of
# freedom to 1. This parameter allows the proper calculation of 𝑠², with (𝑛 − 1) in the
# denominator instead of 𝑛.

# If you have nan values in the dataset, then np.var() and .var() will return nan:
np.var(y_with_nan, ddof=1)
# nan
y_with_nan.var(ddof=1)
# nan

# This is consistent with np.mean() and np.average(). If you want to skip nan values, then you should use np.nanvar():
np.nanvar(y_with_nan, ddof=1) #np.nanvar() ignores nan values. It also needs you to specify ddof=1.
# 123.19999999999999

# pd.Series objects have the method .var() that skips nan values by default:
z.var(ddof=1)
# 123.19999999999999
z_with_nan.var(ddof=1)
# 123.19999999999999

# It also has the parameter ddof, but its default value is 1, so you can omit it. If you want a
# different behavior related to nan values, then use the optional parameter skipna.

# You calculate the population variance similarly to the sample variance. However, you have 
# to use 𝑛 in the denominator instead of 𝑛 − 1: Σᵢ(𝑥ᵢ − mean(𝑥))² / 𝑛. In this case, 𝑛 is the 
# number of items in the entire population. You can get the population variance similar to the 
# sample variance, with the following differences:
#   Replace (n - 1) with n in the pure Python implementation.
#   Use statistics.pvariance() instead of statistics.variance().
#   Specify the parameter ddof=0 if you use NumPy or Pandas. In NumPy, you can omit 
#       ddof because its default value is 0.

### STANDARD DEVIATION ###
# The sample standard deviation is another measure of data spread. It’s connected to the
# sample variance, as standard deviation, 𝑠, is the positive square root of the sample variance.
# The standard deviation is often more convenient than the variance because it has the same
# unit as the data points. Once you get the variance, you can calculate the standard deviation
# with pure Python:

std_ = var_ ** 0.5
std_
# 11.099549540409285

# Although this solution works, you can also use statistics.stdev():
std_ = statistics.stdev(x)
std_
# 11.099549540409287

# You can get the standard deviation with NumPy in almost the same way. You can use the
# function std() and the corresponding method .std() to calculate the standard deviation. If
# there are nan values in the dataset, then they’ll return nan. To ignore nan values, you should 
# use np.nanstd(). You use std(), .std(), and nanstd() from NumPy as you would use 
# var(), .var(), and nanvar():
np.std(y, ddof=1)
# 11.099549540409285
y.std(ddof=1)
# 11.099549540409285
np.std(y_with_nan, ddof=1)
# nan
y_with_nan.std(ddof=1)
# nan
np.nanstd(y_with_nan, ddof=1)
# 11.099549540409285

# Don’t forget to set the delta degrees of freedom to 1!

# pd.Series objects also have the method .std() that skips nan by default:
z.std(ddof=1)
# 11.099549540409285
z_with_nan.std(ddof=1)
# 11.099549540409285

# The parameter ddof defaults to 1, so you can omit it. Again, if you want to treat nan values
# differently, then apply the parameter skipna.

### SKEWNESS ###

# The sample skewness measures the asymmetry of a data sample.

# There are several mathematical definitions of skewness. One common expression to
# calculate the skewness of the dataset 𝑥 with 𝑛 elements is (𝑛² / ((𝑛 − 1)(𝑛 − 2))) (Σᵢ(𝑥ᵢ − mean(𝑥))³ / (𝑛𝑠³)). 
# A simpler expression is Σᵢ(𝑥ᵢ − mean(𝑥))³ 𝑛 / ((𝑛 − 1)(𝑛 − 2)𝑠³), where 𝑖 = 1, 2, …, 𝑛 and mean(𝑥) 
# is the sample mean of 𝑥. The skewness defined like this is called the 
# adjusted Fisher-Pearson standardized moment coefficient.

# Once you’ve calculated the size of your dataset n, the sample mean mean_, and the standard 
# deviation std_, you can get the sample skewness with pure Python:
x = [8.0, 1, 2.5, 4, 28.0]
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
std_ = var_ ** 0.5
skew_ = (sum((item - mean_)**3 for item in x) * n / ((n - 1) * (n - 2) * std_**3))
skew_
# 1.9470432273905929 skew is positive, so x has a right-side tail

# You can also calculate the sample skewness with scipy.stats.skew():
y, y_with_nan = np.array(x), np.array(x_with_nan)
scipy.stats.skew(y, bias=False)
# 1.9470432273905927
# scipy.stats.skew(y_with_nan, bias=False)
# nan

# The obtained result is the same as the pure Python implementation. The parameter bias is
# set to False to enable the corrections for statistical bias. The optional parameter
# nan_policy can take the values 'propagate', 'raise', or 'omit'. It allows you to control
# how you’ll handle nan values.

# Pandas Series objects have the method .skew() that also returns the skewness of a dataset:
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
z.skew()
# 1.9470432273905924
z_with_nan.skew()
# 1.9470432273905924

# Like other methods, .skew() ignores nan values by default, 
# because of the default value of the optional parameter skipna.

### PERCENTILES ###

# The sample 𝑝 percentile is the element in the dataset such that 𝑝% of the elements in the
# dataset are less than or equal to that value. Also, (100 − 𝑝)% of the elements are greater than
# or equal to that value. If there are two such elements in the dataset, then the sample 𝑝
# percentile is their arithmetic mean. Each dataset has three quartiles, which are the
# percentiles that divide the dataset into four parts:
#   The first quartile is the sample 25th percentile. It divides roughly 25% of the smallest 
#       items from the rest of the dataset.
#   The second quartile is the sample 50th percentile or the median. Approximately 25%
#       of the items lie between the first and second quartiles and another 25% between the 
#       second and third quartiles.
#   The third quartile is the sample 75th percentile. It divides roughly 25% of the largest
#       items from the rest of the dataset.

# Each part has approximately the same number of items. If you want to divide your data into
# several intervals, then you can use statistics.quantiles():
x = [-5.0, -1.1, 0.1, 2.0, 8.0, 12.8, 21.0, 25.8, 41.0]
statistics.quantiles(x, n=2)
# [8.0]
statistics.quantiles(x, n=4, method='inclusive')
# [0.1, 8.0, 21.0]

# You can also use np.percentile() to determine any sample percentile in your dataset. For
# example, this is how you can find the 5th and 95th percentiles:
y = np.array(x)
np.percentile(y, 5)
# -3.44
np.percentile(y, 95)
# 34.919999999999995

# percentile() takes several arguments. You have to provide the dataset as the first 
# argument and the percentile value as the second. The dataset can be in the form of a NumPy 
# array, list, tuple, or similar data structure. The percentile can be a number between 0 and 
# 100 like in the example above, but it can also be a sequence of numbers:
np.percentile(y, [25, 50, 75]) # calculates the 25th, 50th and 75th percentile
# array([ 0.1,  8. , 21. ])
np.median(y)
# 8.0

# If you want to ignore nan values, then use np.nanpercentile() instead:
y_with_nan = np.insert(y, 2, np.nan)
y_with_nan
# array([-5. , -1.1,  nan,  0.1,  2. ,  8. , 12.8, 21. , 25.8, 41. ])
np.nanpercentile(y_with_nan, [25, 50, 75])
# array([ 0.1,  8. , 21. ])

# NumPy also offers you very similar functionality in quantile() and nanquantile(). If you
# use them, then you’ll need to provide the quantile values as the numbers between 0 and 1
# instead of percentiles:
np.quantile(y, 0.05)
# -3.44
np.quantile(y, 0.95)
# 34.919999999999995
np.quantile(y, [0.25, 0.5, 0.75])
# array([ 0.1,  8. , 21. ])
np.nanquantile(y_with_nan, [0.25, 0.5, 0.75])
# array([ 0.1,  8. , 21. ])

# pd.Series objects have the method .quantile():
z, z_with_nan = pd.Series(y), pd.Series(y_with_nan)
z.quantile(0.05)
# -3.44
z.quantile(0.95)
# 34.919999999999995
z.quantile([0.25, 0.5, 0.75])
# 0.25     0.1
# 0.50     8.0
# 0.75    21.0
# dtype: float64
z_with_nan.quantile([0.25, 0.5, 0.75])
# 0.25     0.1
# 0.50     8.0
# 0.75    21.0
# dtype: float64

### RANGES ###

# The range of data is the difference between the maximum and minimum element in the 
# dataset. You can get it with the function np.ptp():
np.ptp(y)
# 46.0
np.ptp(z)
# 46.0
np.ptp(y_with_nan)
# nan
np.ptp(z_with_nan)
# 46.0

# Alternatively, you can use built-in Python, NumPy, or Pandas functions and methods 
# to calculate the maxima and minima of sequences:
#   max() and min() from the Python standard library
#   amax() and amin() from NumPy
#   nanmax() and nanmin() from NumPy to ignore nan values
#   .max() and .min() from NumPy
#   .max() and .min() from Pandas to ignore nan values by default

np.amax(y) - np.amin(y)
# 46.0
np.nanmax(y_with_nan) - np.nanmin(y_with_nan)
# 46.0
y.max() - y.min()
# 46.0
z.max() - z.min()
# 46.0
z_with_nan.max() - z_with_nan.min()
# 46.0

# The interquartile range is the difference between the first and third quartile.
# Once you calculate the quartiles, you can take their difference:
quartiles = np.quantile(y, [0.25, 0.75])
quartiles[1] - quartiles[0]
# 20.9
quartiles = z.quantile([0.25, 0.75])
quartiles[0.75] - quartiles[0.25]
# 20.9

## Summary of Descriptive Statistics ##

# SciPy and Pandas offer useful routines to quickly get descriptive statistics with a single 
# function or method call. You can use scipy.stats.describe() like this:
result = scipy.stats.describe(y, ddof=1, bias=False)
print(result)

# describe() returns an object that holds the following descriptive statistics:
#   nobs: the number of observations or elements in your dataset
#   minmax: the tuple with the minimum and maximum values of your dataset
#   mean: the mean of your dataset
#   variance: the variance of your dataset
#   skewness: the skewness of your dataset
#   kurtosis: the kurtosis of your dataset

result.nobs
# 9
result.minmax[0]  # Min
# -5.0
result.minmax[1]  # Max
# 41.0
result.mean
# 11.622222222222222
result.variance
# 228.75194444444446
result.skewness
# 0.9249043136685094
result.kurtosis
# 0.14770623629658886

# Pandas has similar, if not better, functionality. Series objects have the method .describe():
result = z.describe()
result
# count     9.000000
# mean     11.622222
# std      15.124548
# min      -5.000000
# 25%       0.100000
# 50%       8.000000
# 75%      21.000000
# max      41.000000
# dtype: float64

# It returns a new Series that holds the following:
#   count: the number of elements in your dataset
#   mean: the mean of your dataset
#   std: the standard deviation of your dataset
#   min and max: the minimum and maximum values of your dataset
#   25%, 50%, and 75%: the quartiles of your dataset

# If you want the resulting Series object to contain other percentiles, then you should specify
# the value of the optional parameter percentiles. You can access each item of result with its label:
result['mean']
# 11.622222222222222
result['std']
# 15.12454774346805
result['min']
# -5.0
result['max']
# 41.0
result['25%']
# 0.1
result['50%']
# 8.0
result['75%']
# 21.0

## Measures of Correlation Between Pairs of Data ##

# The two statistics that measure the correlation between datasets are covariance and the
# correlation coefficient. Let’s define some data to work with these measures. You’ll create
# two Python lists and use them to get corresponding NumPy arrays and Pandas Series:
x = list(range(-10, 11))
y = [0, 2, 2, 2, 2, 3, 3, 6, 7, 4, 7, 6, 6, 9, 4, 5, 5, 10, 11, 12, 14]
x_, y_ = np.array(x), np.array(y)
x__, y__ = pd.Series(x_), pd.Series(y_)

## COVARIANCE ##

# The sample covariance is a measure that quantifies the strength and direction of a
# relationship between a pair of variables:

n = len(x)
mean_x, mean_y = sum(x) / n, sum(y) / n
cov_xy = (sum((x[k] - mean_x) * (y[k] - mean_y) for k in range(n)) / (n - 1))
cov_xy
# 19.95

# NumPy has the function cov() that returns the covariance matrix:
cov_matrix = np.cov(x_, y_)
cov_matrix
# array([[38.5       , 19.95      ],
#        [19.95      , 13.91428571]])

# As you can see, the variances of x and y are equal to cov_matrix[0, 0] and 
# cov_matrix[1, 1], respectively.
x_.var(ddof=1)
# 38.5
y_.var(ddof=1)
# 13.914285714285711

cov_xy = cov_matrix[0, 1]
cov_xy
# 19.95
cov_xy = cov_matrix[1, 0]
cov_xy
# 19.95

# Pandas Series have the method .cov() that you can use to calculate the covariance:
cov_xy = x__.cov(y__)
cov_xy
# 19.95
cov_xy = y__.cov(x__)
cov_xy
# 19.95

## CORRELATION COEFFICIENT ##

# The value 𝑟 > 0 indicates positive correlation.
# The value 𝑟 < 0 indicates negative correlation.
# The value r = 1 is the maximum possible value of 𝑟. It corresponds to a perfect positive 
#   linear relationship between variables.
# The value r = −1 is the minimum possible value of 𝑟. It corresponds to a perfect 
#   negative linear relationship between variables.
# The value r ≈ 0, or when 𝑟 is around zero, means that the correlation between 
#   variables is weak.

var_x = sum((item - mean_x)**2 for item in x) / (n - 1)
var_y = sum((item - mean_y)**2 for item in y) / (n - 1)
std_x, std_y = var_x ** 0.5, var_y ** 0.5
r = cov_xy / (std_x * std_y)
r # 0.861950005631606

# scipy.stats has the routine pearsonr() that calculates the correlation coefficient and the 𝑝-value:
r, p = scipy.stats.pearsonr(x_, y_)
r # 0.861950005631606
p # 5.122760847201171e-07

# Similar to the case of the covariance matrix, you can apply np.corrcoef() with x_ and y_ as 
# the arguments and get the correlation coefficient matrix:
corr_matrix = np.corrcoef(x_, y_)
corr_matrix
# array([[1.        , 0.86195001],
    #    [0.86195001, 1.        ]])

# The upper-left element is the correlation coefficient between x_ and x_. The lower-right 
# element is the correlation coefficient between y_ and y_. Their values are equal to 1.0. The 
# other two elements are equal and represent the actual correlation coefficient between x_ and y_:
r = corr_matrix[0, 1]
r # 0.8619500056316061
r = corr_matrix[1, 0]
r # 0.861950005631606

# You can get the correlation coefficient with scipy.stats.linregress():
scipy.stats.linregress(x_, y_)
# LinregressResult(slope=0.5181818181818181, intercept=5.714285714285714, rvalue=0.861950005631606, pvalue=5.122760847201164e-07, stderr=0.06992387660074979)

# linregress() takes x_ and y_, performs linear regression, and returns the results. slope
# and intercept define the equation of the regression line, while rvalue is the correlation
# coefficient. To access particular values from the result of linregress(), including the 
# correlation coefficient, use dot notation:
result = scipy.stats.linregress(x_, y_)
r = result.rvalue
r # 0.861950005631606

# Pandas Series have the method .corr() for calculating the correlation coefficient:
r = x__.corr(y__)
r # 0.8619500056316061
r = y__.corr(x__)
r # 0.861950005631606