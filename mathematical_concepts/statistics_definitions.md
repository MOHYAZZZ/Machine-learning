# Statistics Definitions

This document provides definitions and examples for the key statistical concepts used in our `statistics_calculator.py` script.

## Mean

The mean, often referred to as the average, is calculated by adding up all the numbers in the dataset and then dividing by the count of numbers in the dataset.

For example, the mean of [1, 2, 3, 4, 5] is (1+2+3+4+5)/5 = 3.

## Median

The median is the middle number in a sorted, ascending or descending, list of numbers. If the list contains an even number of observations, the median is the average of the two middle numbers.

For example, the median of [1, 2, 3, 4, 5] is 3. If we had [1, 2, 3, 4], the median would be (2+3)/2 = 2.5.

## Mode

The mode is the value that appears most frequently in a data set. A set of data may have one mode, more than one mode, or no mode at all.

For example, the mode of [1, 1, 2, 3, 4] is 1, because 1 appears more often than any other number.

## Sample Variance

Variance is a measure of how spread out a data set is. It is calculated as the average squared deviation of each number from the mean of a data set.

For example, for the dataset [1, 2, 3, 4, 5], the mean is 3. The squared deviations are (1-3)²=4, (2-3)²=1, (3-3)²=0, (4-3)²=1, (5-3)²=4. So, the variance is (4+1+0+1+4)/(5-1) = 2.5.

## Sample Standard Deviation

The standard deviation is a measure of the amount of variation or dispersion of a set of values. It is the square root of the variance.

For example, for the dataset [1, 2, 3, 4, 5] the variance is 2.5 (calculated above), so the standard deviation is √2.5 = 1.5811.

## Confidence Interval for the Mean

A confidence interval for the mean is an estimate of the interval in which the population mean is likely to be located, given a certain level of confidence (for example, 95%). It's based on the sample mean and the standard deviation.

For example, for a large enough dataset with a mean of 3 and a standard deviation of 1.5811, the 95% confidence interval would be approximately from 3 - 1.96*(1.5811/√n) to 3 + 1.96*(1.5811/√n), where n is the size of the dataset.
