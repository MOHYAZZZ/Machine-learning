# This script contains a function 'get_statistics' which takes in a list of numbers
# and returns a dictionary containing the following statistics about the numbers:
# the mean, median, mode, sample variance, sample standard deviation,
# and a 95% confidence interval for the mean.

# This function does not use any libraries and is implemented purely using base Python.
# It is designed to operate under the assumption of large samples from a population
# (enough to use a Z-score of 1.96) and a normal distribution.

def get_statistics(input_list):
    n = len(input_list)

    # Calculate mean
    total_sum = sum(input_list)
    mean = total_sum / n

    # Calculate median
    sorted_list = sorted(input_list)
    if n % 2 == 0:
        median = (sorted_list[n // 2 - 1] + sorted_list[n // 2]) / 2
    else:
        median = sorted_list[n // 2]

    # Calculate mode
    freq_dict = {}
    for num in input_list:
        if num in freq_dict:
            freq_dict[num] += 1
        else:
            freq_dict[num] = 1
    mode = max(freq_dict, key=freq_dict.get)

    # Calculate variance and standard deviation
    sum_diff_sq = sum((xi - mean) ** 2 for xi in input_list)
    variance = sum_diff_sq / (n - 1)
    std_dev = variance ** 0.5

    # Calculate 95% confidence interval for the mean
    margin_error = 1.96 * (std_dev / (n ** 0.5))
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error

    return {
        "mean": round(mean, 4),
        "median": round(median, 4),
        "mode": round(mode, 4),
        "sample_variance": round(variance, 4),
        "sample_standard_deviation": round(std_dev, 4),
        "mean_confidence_interval": [round(ci_lower, 4), round(ci_upper, 4)]
    }
