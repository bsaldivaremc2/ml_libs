def get_confidence_interval(std,n,degrees_of_freedom,confidence_level=0.95):
    """
    Returns confidence interval margin to add and substract from the mean.
    std: standard deviation
    n: sample size
    degrees_of_freedom: degrees of freedom, n-1
    Confidence_level: Confidence level
    Reference: 
    https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/confidence-interval/
    
    Example:
    get_confidence_interval(0.5,10,9,confidence_level=0.95)
    returns: 0.357
    
    """
    from scipy.stats import t
    import numpy as np
    tv = abs(t.ppf((1-confidence_level)/2,degrees_of_freedom))
    return tv * std/(n**0.5)