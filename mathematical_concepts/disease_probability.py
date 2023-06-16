"""

This Python script includes a function to calculate the probability that an individual has or does not have a disease given the result of a test, 
taking into account the accuracy of the test and the prevalence of the disease in the population. 
It uses the concept of Bayes' theorem, considering the test's sensitivity and specificity (assumed to be equal to the given accuracy). 

"""

def probability_of_disease(accuracy, prevalence):
    sensitivity = accuracy
    specificity = accuracy

    # Compute Positive Predictive Value (PPV)
    PPV = (sensitivity * prevalence) / ((sensitivity * prevalence) + ((1 - specificity) * (1 - prevalence)))

    # Compute Negative Predictive Value (NPV)
    NPV = (specificity * (1 - prevalence)) / (((1 - sensitivity) * prevalence) + (specificity * (1 - prevalence)))

    PPV = round(PPV * 100, 4)
    NPV = round(NPV * 100, 4)

    return [PPV, NPV]

# Test Cases
print(probability_of_disease(0.95, 0.03))  
print(probability_of_disease(0.80, 0.10))  
print(probability_of_disease(0.60, 0.45))  
