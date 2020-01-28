import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from scipy import stats
from numpy.random import seed
import matplotlib.pyplot as plt

from ml_libs.util import utils

def get_min_per_group_df(idf,class_col='Grup '):
    """
    Gets the minimum number of samples in a group given a group column of a DataFrame.
    """
    class_vals = list(set(idf[class_col].values))
    min_n_group = idf.shape[0]
    for cv in class_vals :
        gn = idf[idf[class_col]==cv].shape[0]
        min_n_group = min(min_n_group,gn)
    return min_n_group

def get_train_test_ids(idf,test_split=0.2,class_col='Grup ',id_col = 'Animal',min_per_group=7):
    """
    Returns the index for training and testing taking in mind the number of samples per group
    in the class_col. It will use the 'id_col' to identify each individual
    """
    test_n = int(np.ceil(min_per_group*test_split))
    train_n = min_per_group - test_n
    class_vals = list(set(idf[class_col].values))
    all_train_ids = []
    all_test_ids = []
    for cv in class_vals :
        gx = idf[idf[class_col]==cv]
        gxl = list(gx[id_col].values)
        test_ids = list(np.random.choice(gxl,test_n,replace=False))
        train_ids = list(filter(lambda x: x not in test_ids,gxl))
        train_ids = list(np.random.choice(train_ids,train_n,replace=False))
        all_train_ids.extend(train_ids)
        all_test_ids.extend(test_ids)
    if sum(list(map(lambda x: int(x in all_train_ids), all_test_ids))) > 0:
        print("Error. An id got from test into train")
        return None
    return all_train_ids[:],all_test_ids[:]

def get_train_test(idf,train_test_ids_list,id_col = 'Animal'):
    return [ pd.concat([ idf[idf[id_col]== idx] for idx in ids],0).reset_index(drop=True) for ids in train_test_ids_list ]

def binary_tp_tn_fp_fn(y_true,y_pred):
    """
    returns true positive, true negative, false positive and false negative
    Input example:
    >y_true = np.array([0,0,1,1,1,0,1,0,0])
    >y_pred = np.array([1,0,1,0,1,0,0,1,1])
    >binary_tp_tn_fp_fn(y_true,y_pred)
    returned: (2, 2, 3, 2)
    """
    tp = np.sum(y_true*y_pred)
    y_true_ = np.abs(y_true-1)
    y_pred_ = np.abs(y_pred-1)
    tn = np.sum(y_true_*y_pred_)
    fp = np.sum(y_pred) - tp
    fn = np.sum(y_pred_) -tn
    return tp,tn,fp,fn

def binary_sens_spec(tp,tn,fp,fn,epsilon=1e-8):
    """
    Returns sensibility and specificity
    """
    sens = tp/(tp+fn+epsilon)
    spec = tn/(tn+fp+epsilon)
    return sens, spec

def get_binary_sens_spec(y_true,y_pred,epsilon=1e-8,round_n = 3):
    """
    Calculate sensitivity and specificity given ground truth and prediction for
    binary classification
    """
    sens, spec =  binary_sens_spec(*binary_tp_tn_fp_fn(y_true,y_pred),epsilon)
    return round(sens,round_n),round(spec,round_n)

def get_binary_sensitivity(y_true,y_pred,epsilon=1e-8,round_n = 3):
    """
    Calculate sensitivity given ground truth and prediction for
    binary classification
    """
    return get_binary_sens_spec(y_true,y_pred,epsilon,round_n)[0]

def get_binary_specificity(y_true,y_pred,epsilon=1e-8,round_n = 3):
    """
    Calculate specificity given ground truth and prediction for
    binary classification
    """
    return get_binary_sens_spec(y_true,y_pred,epsilon,round_n)[1]



def get_metrics(base, pred):
    from sklearn import metrics
    """
    MAPE: on progress.
    """
    output_metrics = {}
    output_metrics['correlation'] = np.corrcoef(base, pred)[0, 1]
    output_metrics['mae'] = metrics.mean_absolute_error(base, pred)
    output_metrics['mse'] = metrics.mean_squared_error(base, pred)
    output_metrics['rmse'] = output_metrics['mse'] ** 0.5
    output_metrics['r2'] = metrics.r2_score(base, pred)
    return output_metrics.copy()


def cv_metrics(X, y, reg, regxk={}, kfold=5):
    from sklearn import metrics
    """
    MAPE: on progress.
    """
    kf = KFold(n_splits=kfold, shuffle=True)
    output_metrics = {'mae': [], 'corr': [], 'rmse': [], 'r2': [], 'mse': [], 'sum_absolute_error': []}
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        r = reg(**regxk)
        r.fit(X_train, y_train)
        pred_test = r.predict(X_test)
        output_metrics['sum_absolute_error'].append(np.sum(np.abs(y_test - pred_test)))
        output_metrics['corr'].append(np.corrcoef(pred_test, y_test)[0, 1])
        output_metrics['mae'].append(metrics.mean_absolute_error(y_test, pred_test))
        output_metrics['mse'].append(metrics.mean_squared_error(y_test, pred_test))
        output_metrics['rmse'].append(output_metrics['mse'][-1] ** 0.5)
        output_metrics['r2'].append(metrics.r2_score(y_test, pred_test))
    return output_metrics.copy(), {'regresor': r, 'X': {'train': X_train.copy(), 'test': X_test.copy()},
                                   'y': {'train': y_train.copy(), 'test': y_test.copy()}}

def plot_and_report(idf,idata,figsize=(8,8),dpi=80,ititle="",ixl="",iyl=""):
    y_pred = idata['regresor'].predict(idata['X']['test'])
    y_test = idata['y']['test']
    fig = plt.figure(figsize=figsize,dpi=dpi)
    plt.title(ititle)
    plt.xlabel(ixl)
    plt.ylabel(iyl)
    plt.scatter(y_test, y_pred)
    plt.show()
    rowx = idf.head(1)
    for col in idf.columns:
        print(col+":",rowx[col].values[0])

def get_top_corr(ix,iy,get_top_cc = 10):
    ccs = []
    for cx in range(ix.shape[1]):
        cc = abs(np.corrcoef(ix[:,cx],iy)[0,1])
        ccs.append(cc)
    cc_max_min_index = list(np.argsort(ccs))[::-1]
    top_cc_index = cc_max_min_index[:min(get_top_cc,ix.shape[1])]
    print(top_cc_index)
    top_cc_vals = [ ccs[ci] for ci in top_cc_index]
    print("Top {} correlation values {}".format(get_top_cc,top_cc_vals))
    return ix[:,top_cc_index].copy()

"""
def cv_metrics_stratified_class(X, Y, stratus_list,clf, clfk={}, kfold=5,shuffle=True):
    #https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
    from sklearn.model_selection import StratifiedKFold
    from sklearn import metrics
    sk_metrics
    skf = StratifiedKFold(n_splits=kfold,shuffle=shuffle)
    skf.get_n_splits(X, stratus_list)
    output_metrics = {'roc_auc_score': []}
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        r = clf(**clfk)
        r.fit(X_train, y_train)
        pred_test = r.predict(X_test)
        pred_prob = r.predict_proba(X_test)[:,1]
        output_metrics['roc_auc_score'].append(metrics.roc_auc_score(y_test,pred_prob))
    return output_metrics.copy()
"""

def cv_metrics_stratified_class(X, Y, stratus_list,clf, clfk={}, kfold=5,shuffle=True,
                               report_metrics=['roc_auc_score','auc','f1_score','sensitivity','specificity']):
    #https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
    from sklearn.model_selection import StratifiedKFold
    from sklearn import metrics
    calc_metrics = {'roc_auc_score':metrics.roc_auc_score,
                  'auc':metrics.auc,
                  'f1_score':metrics.f1_score,
                  'sensitivity':get_binary_sensitivity,
                  'specificity':get_binary_specificity
                 }
    skf = StratifiedKFold(n_splits=kfold,shuffle=shuffle)
    skf.get_n_splits(X, stratus_list)
    output_metrics = {}
    for m in report_metrics:
        output_metrics[m]=[]
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        r = clf(**clfk)
        r.fit(X_train, y_train)
        pred_test = r.predict(X_test)
        pred_prob = r.predict_proba(X_test)[:,1]
        for m in report_metrics:
            if m in ['roc_auc_score']:
                output_metrics[m].append(calc_metrics[m](y_test,pred_prob))
            else:
                output_metrics[m].append(calc_metrics[m](y_test,pred_test))
    return output_metrics.copy()

def metrics_stats(ilist,rn=3):
    output={
        'mean': np.mean(ilist).round(rn),
        'std': np.std(ilist).round(rn),
        'max': np.max(ilist).round(rn),
        'min': np.min(ilist).round(rn),
    }
    return output.copy()


def get_grid_hyperparams(hyperparams_lists_dict,hyperparams_types_dict):
    """
    Returns list of dicts with hyperparameter combinations
    EXAMPLE:
    hyperparams_lists_dict = {
        'hidden_layer_sizes':['(2048,2048)','(1024,1024)'],
        'learning_rate':['adaptive'],
         'learning_rate_init':10.0**(-np.arange(1,6,1))
        }
    hyperparams_types_dict  = {'hidden_layer_sizes':'tuple','learning_rate':'str','learning_rate_init':'float'}

    """
    hps = hyperparams_lists_dict.copy()
    hps_types = hyperparams_types_dict.copy()
    #Sort
    hp_ks = list(hps.keys()) #Only keys
    hp_vs = [ hps[k] for k in hp_ks] #Only values
    hp_ts = [hps_types[k] for k in hp_ks]
    #Creat basic grid
    grid = np.array(np.meshgrid(*hp_vs))
    grid = np.vstack([ gx.flatten() for gx in grid ])
    cols = grid.shape[1]
    output = []
    for col in range(cols):
        vals = list(grid[:,col])
        vals = list(map(lambda v,t:utils.val_to_type(v,t),vals,hp_ts))
        clfk = dict(zip(hp_ks,vals))
        output.append(clfk.copy())
    return output.copy()


def grid_search_own_metrics_class_stratified(ix,iy,stratus_list,
                                             clf,get_grid_hyperparams_kargs,
                                         regressor_name="Regressor",
                                         show_progress_percentage=0.1, kfold=5,shuffle=True,
                                            sort_report_by='roc_auc_score'):
    clfks = get_grid_hyperparams(**get_grid_hyperparams_kargs)
    report_data = []
    hpks = list(clfks[0].keys())
    cols = len(clfks)
    progress_int = max(int(round(cols*show_progress_percentage,0)),1)
    print("Total number of evaluations:{}".format(cols))
    for col,clfk in enumerate(clfks):
        metrics = cv_metrics_stratified_class(ix, iy.flatten(), stratus_list=stratus_list,
                                              clf=clf, clfk=clfk, kfold=kfold,shuffle=shuffle)
        metrics_report = {'name':regressor_name}
        for m in metrics.keys():
            stats = metrics_stats(metrics[m],rn=3)
            for sk in stats.keys():
                metrics_report[m+"_"+sk]=stats[sk]
        metrics_report.update(clfk)
        report_data.append(metrics_report.copy())
        if col%progress_int==0:
            progress = round(100*col/cols,0)
            print("{} %".format(progress))
    print("100.0 %")
    odf = pd.DataFrame(report_data[:])
    odf = odf.sort_values(by=[sort_report_by+"_mean"],ascending=False)
    mean_cols = list(filter(lambda x: "mean" in x,list(odf.columns)))
    ocols = ['name']
    ocols.extend(hpks)
    ocols.extend(mean_cols)
    ocols.extend(list(filter(lambda x: x not in ocols,odf.columns)))
    odf = odf[ocols].reset_index(drop=True)
    return odf.copy()
