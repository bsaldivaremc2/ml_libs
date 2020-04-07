import numpy as np

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

def anova_2_df(idf1,idf2):
    """
    Return a dataframe of n variables X p_value, valid and value.
    Input: two dataframes that hold two groups. Rows: samples. Columns variables.
    """
    import scipy.stats as statsx
    import pandas as pd
    _ = statsx.f_oneway(idf1.values,idf2.values)
    pvalue = _.pvalue.round(4)
    cols = list(idf1.columns)
    valid_p = _.pvalue<0.05
    head = dict(zip(cols,valid_p))
    tail = dict(zip(cols,pvalue))
    metrics_label = 'metrics'
    odf = pd.concat([pd.DataFrame(columns=[metrics_label],data=['pvalue < 0.05','pvalue']),pd.DataFrame([head,tail])],1).set_index(metrics_label).transpose()
    return odf.copy()

def paired_t_test(inp1,inp2,round_digits=4):
    from scipy import stats as st
    _ = st.ttest_rel(inp1,inp2)
    pv = _.pvalue
    pvv = (pv<0.05)
    pv = (np.expand_dims(pv,-1)).round(round_digits)
    pvv = np.expand_dims(pvv,-1)
    o = pd.concat([pd.DataFrame(data=pv,columns=['pvalue']),
        pd.DataFrame(data=pvv,columns=['pvalue<0.05'])],1)
    return o.copy()

def get_one_prob(inp):
    return (inp.sum()/inp.flatten().shape[0])

def random_one_zero_density(reference_matrix):
    import numpy as np
    """
    Returns a matrix with the same ones and zeros' probability than reference_matrix.
    The returned matrix has the same shape as the reference_matrix.
    """
    inp = reference_matrix.copy()
    shape = inp.shape
    v = inp.flatten()
    n = v.shape[0]
    one_prob = v.sum()/n
    zero_prob = 1 - one_prob
    return np.random.choice([0,1],size=shape,p=[zero_prob,one_prob]).copy()
