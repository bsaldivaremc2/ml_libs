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

def cv_metrics_stratified_class(X, Y, stratus_list,clf, clfk={}, kfold=5,shuffle=True,
                               report_metrics=['matthews_corr_coef','roc_auc_score','f1_score','sensitivity','specificity'],
                               norm=False):
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
    x = X.copy()
    if len(X.shape)>2:
        x=np.random.random((X.shape[0],1))
    skf.get_n_splits(x, stratus_list)
    output_metrics = {}
    for m in report_metrics:
        output_metrics[m]=[]
    y=Y.copy()
    if len(Y.shape)>1:
        y=Y[:,0]
    for train_index, test_index in skf.split(x, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        if len(y_test.shape)>1:
            y_test = y_test.argmax(1)
        r = clf(**clfk)
        if norm==True:
            n_mean = X_train.mean(0)
            n_std = X_train.std(0)
            def z_score(ix,im,istd):
                return (ix-im)/istd
            X_train, X_test = z_score(X_train,n_mean,n_std),z_score(X_test,n_mean,n_std)
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
                                            sort_report_by='roc_auc_score',metrics_kargs={
                                            'kfold':5,'shuffle':True,'report_metrics':['matthews_corr_coef','roc_auc_score','f1_score','sensitivity','specificity']
                                            }):
    clfks = get_grid_hyperparams(**get_grid_hyperparams_kargs)
    report_data = []
    hpks = list(clfks[0].keys())
    cols = len(clfks)
    progress_int = max(int(round(cols*show_progress_percentage,0)),1)
    print("Total number of evaluations:{}".format(cols))
    for col,clfk in enumerate(clfks):
        metrics = cv_metrics_stratified_class(ix, iy.flatten(), stratus_list=stratus_list,
                                              clf=clf, clfk=clfk,**metrics_kargs)
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


def cv_metrics_stratified_class_report(X, Y, stratus_list,clf, clfk={}, kfold=5,shuffle=True,
                               report_metrics=['matthews_corr_coef','roc_auc_score','f1_score','sensitivity','specificity'],
                                      regressor_name='Regressor',sort_report_by='roc_auc_score',norm=False):
    report_data = []
    metrics = cv_metrics_stratified_class(X, Y, stratus_list=stratus_list,
                                              clf=clf, clfk=clfk,report_metrics=report_metrics,norm=norm)
    metrics_report = {'name':regressor_name}
    for m in metrics.keys():
        stats = metrics_stats(metrics[m],rn=3)
        for sk in stats.keys():
            metrics_report[m+"_"+sk]=stats[sk]
    metrics_report.update(clfk)
    report_data.append(metrics_report.copy())
    odf = pd.DataFrame(report_data[:])
    odf = odf.sort_values(by=[sort_report_by+"_mean"],ascending=False)
    mean_cols = list(filter(lambda x: "mean" in x,list(odf.columns)))
    ocols = ['name']
    ocols.extend(clfk)
    ocols.extend(mean_cols)
    ocols.extend(list(filter(lambda x: x not in ocols,odf.columns)))
    odf = odf[ocols].reset_index(drop=True)
    return odf.copy()


def cv_metrics_stratified_class_with_indexes(X, Y, indexes,clf, clfk={}, kfold=5,shuffle=True,
                               report_metrics=['matthews_corr_coef','roc_auc_score','f1_score','sensitivity','specificity'],
                               norm=False):
    #https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
    from sklearn import metrics
    calc_metrics = {'roc_auc_score':metrics.roc_auc_score,
                  'auc':metrics.auc,
                  'f1_score':metrics.f1_score,
                  'sensitivity':get_binary_sensitivity,
                  'specificity':get_binary_specificity
                 }
    output_metrics = {}
    for m in report_metrics:
        output_metrics[m]=[]
    for train_index, test_index in indexes:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        if len(y_test.shape)>1:
            y_test = y_test.argmax(1)
        r = clf(**clfk)
        if norm==True:
            n_mean = X_train.mean(0)
            n_std = X_train.std(0)
            def z_score(ix,im,istd):
                return (ix-im)/istd
            X_train, X_test = z_score(X_train,n_mean,n_std),z_score(X_test,n_mean,n_std)
        r.fit(X_train, y_train)
        pred_test = r.predict(X_test)
        pred_prob = r.predict_proba(X_test)[:,1]
        for m in report_metrics:
            if m in ['roc_auc_score']:
                output_metrics[m].append(calc_metrics[m](y_test,pred_prob))
            else:
                output_metrics[m].append(calc_metrics[m](y_test,pred_test))
    return output_metrics.copy()



def cv_metrics_stratified_class_report_with_indexes(X, Y, indexes,clf, clfk={}, kfold=5,shuffle=True,
                               report_metrics=['matthews_corr_coef','roc_auc_score','f1_score','sensitivity','specificity'],
                                      regressor_name='Regressor',sort_report_by='roc_auc_score',norm=False):
    report_data = []
    metrics = cv_metrics_stratified_class_with_indexes(X, Y, indexes=indexes,
                                              clf=clf, clfk=clfk,report_metrics=report_metrics,norm=norm)
    metrics_report = {'name':regressor_name}
    for m in metrics.keys():
        stats = metrics_stats(metrics[m],rn=3)
        for sk in stats.keys():
            metrics_report[m+"_"+sk]=stats[sk]
    metrics_report.update(clfk)
    report_data.append(metrics_report.copy())
    odf = pd.DataFrame(report_data[:])
    odf = odf.sort_values(by=[sort_report_by+"_mean"],ascending=False)
    mean_cols = list(filter(lambda x: "mean" in x,list(odf.columns)))
    ocols = ['name']
    ocols.extend(clfk)
    ocols.extend(mean_cols)
    ocols.extend(list(filter(lambda x: x not in ocols,odf.columns)))
    odf = odf[ocols].reset_index(drop=True)
    return odf.copy()

def get_train_test_indexes(X,Y,stratus_list,kfold=5,shuffle=True):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=kfold,shuffle=shuffle)
    x = X.copy()
    if len(X.shape)>2:
        x=np.random.random((X.shape[0],1))
    skf.get_n_splits(x, stratus_list)
    y=Y.copy()
    if len(Y.shape)>1:
        y=Y[:,0]
    train_indexes,test_indexes = [],[]
    for train_index, test_index in skf.split(x, y):
        train_indexes.append(train_index.copy())
        test_indexes.append(test_index.copy())
    return train_indexes[:],test_indexes[:]

def get_rfe_best_cols_with_indexes(iX,iY,iclf,iclfk,indexes,metric_to_improve='roc_auc_score_mean',
                                  cv_kargs={'report_metrics':['matthews_corr_coef','roc_auc_score','f1_score','sensitivity','specificity'],
                             'kfold':5,'shuffle':True ,'norm':False}):
    from sklearn.feature_selection import RFE
    performance = 0
    sort_report_by = "_".join(metric_to_improve.split("_")[:-1])
    est = iclf(**iclfk)
    selector = RFE(est, n_features_to_select=1, step=1, verbose=0)
    selector.fit(iX,iY)
    ranking = selector.ranking_
    for feats in range(1,iX.shape[1]):
    #for feats in range(1,3):
        valid_cols = ranking<=feats
        Xrfe = iX[:,valid_cols]
        report = cv_metrics_stratified_class_report_with_indexes(Xrfe, iY,zip(*indexes),
                                                                     iclf, iclfk, regressor_name='Regressor',
                                                                     sort_report_by=sort_report_by,**cv_kargs)
        performance_ = report[metric_to_improve].values[0]
        if performance_ > performance:
            performance = performance_
            print("Feats: {} . {} Performance: {}".format(feats,metric_to_improve,performance))
            best_cols = valid_cols
    return best_cols.copy()


def transform_x_train_test(ix_train,ix_test,iy_train,iy_test,
                           transform=None,iclf=None,iclfk=None,features_top_n = None,vector=None):
    x_tr,x_ts,y_tr,y_ts = ix_train.copy(),ix_test.copy(),iy_train.copy(),iy_test.copy()
    total_features = x_tr.shape[1]
    if type(features_top_n)==int:
        total_features = min(max(1,features_top_n),total_features)
    if type(transform)==str:
        if transform=='PCA':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=total_features)
            pca.fit(x_tr)
            x_tr = pca.transform(x_tr)
            x_ts = pca.transform(x_ts)
        elif transform=='PLS':
            from sklearn.cross_decomposition import PLSRegression
            pls = PLSRegression(n_components=total_features)
            pls.fit(x_tr,y_tr)
            x_tr = pls.transform(x_tr)
            x_ts = pls.transform(x_ts)
        elif transform=='RFE':
            from sklearn.feature_selection import RFE
            r = iclf(**iclfk)
            selector = RFE(r, n_features_to_select=1, step=1, verbose=0)
            selector.fit(x_tr,y_tr)
            ranking = selector.ranking_
            ranking = np.argsort(ranking)[:total_features]
            x_tr = x_tr[:,ranking]
            x_ts = x_ts[:,ranking]
            if type(vector)!=type(None):#
              xv = vector[ranking,:]#
              x_tr_, x_ts_ = x_tr.copy(),x_ts.copy()
              x_tr = np.dot(x_tr,xv)#
              x_ts = np.dot(x_ts,xv)#
              x_tr = np.hstack([x_tr_,x_tr])
              x_ts = np.hstack([x_ts_,x_ts])
    return x_tr.copy(),x_ts.copy(),y_tr.copy(),y_ts.copy()


def z_score(ix,im,istd):
    return (ix-im)/istd

def norm_z_score(ix_train,ix_test):
    n_mean = ix_train.mean(0)
    n_std = ix_train.std(0)
    ox_train, ox_test = z_score(ix_train,n_mean,n_std),z_score(ix_test,n_mean,n_std)
    return ox_train.copy(),ox_test.copy()

def fit_and_get_metrics(ix_train,ix_test,iy_train,iy_test,iclf,iclfk,
                report_metrics=['matthews_corr_coef','roc_auc_score','f1_score','sensitivity','specificity'],
                       scores_idic={}):
    #https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
    from sklearn import metrics
    calc_metrics = {'roc_auc_score':metrics.roc_auc_score,
                  'auc':metrics.auc,
                  'f1_score':metrics.f1_score,
                  'sensitivity':get_binary_sensitivity,
                  'specificity':get_binary_specificity,
                  'matthews_corr_coef':metrics.matthews_corrcoef,
                  'accuracy':metrics.accuracy_score
                 }
    r = iclf(**iclfk)
    r.fit(ix_train, iy_train)
    pred_test = r.predict(ix_test)
    pred_prob = r.predict_proba(ix_test)[:,1]
    tmp_scores = scores_idic.copy()
    for m in report_metrics:
        tmp_metric = tmp_scores.get(m,[])
        if m in ['roc_auc_score']:
            tmp_metric.append(calc_metrics[m](iy_test,pred_prob))
        else:
            tmp_metric.append(calc_metrics[m](iy_test,pred_test))
        tmp_scores[m] = tmp_metric
    return tmp_scores.copy()
    #output_metrics['F'+str(feature_number)] = tmp_scores

def transform_and_join(iXs,iy,train_index,test_index,transformations,features_top_ns,
                       iclf,iclfk,joint_transformation=None,dot_product=False,vectors=[]):
    X_trains,X_tests = [],[]
    if len(vectors)==0:
      vectors = [None for _ in range(len(iXs))]
    y_train, y_test = iy[train_index].copy(), iy[test_index].copy()
    for X,transform,features_top_n,vector in zip(iXs,transformations,features_top_ns,vectors):
        X_train, X_test = X[train_index].copy(), X[test_index].copy()
        if type(transform)==str:
            X_train, X_test, y_train, y_test = transform_x_train_test(X_train, X_test, y_train, y_test,
                           transform,iclf,iclfk,features_top_n,vector)
        X_trains.append(X_train.copy())
        X_tests.append(X_test.copy())
    if dot_product:
        failed_dot = False
        if len(X_trains)==2:
            if X_trains[0].shape[1]==X_trains[1].shape[0]:
                X_train,X_test = np.dot(*X_trains),np.dot(*X_test)
            else:
                failed_dot=True
        else:
            failed_dot=True
        if failed_dot:
            print("Failed dot product, joining")
    X_train, X_test = np.hstack(X_trains),np.hstack(X_tests)
    if type(joint_transformation)==str:
        X_train, X_test, y_train, y_test = transform_x_train_test(X_train, X_test, y_train, y_test,
                           transform=transform,iclf=iclf,iclfk=iclfk)
    return X_train.copy(),X_test.copy(),y_train.copy(),y_test.copy()

def cv_metrics_stratified_class_with_indexes_and_transform(X, Y, indexes,iclf, iclfk={}, transform=None, kfold=5,shuffle=True,
                               report_metrics=['matthews_corr_coef','roc_auc_score','f1_score','sensitivity','specificity'],
                               norm=False,calc_stats=True,report_name='CLF',sort_metric = 'roc_auc_score_min',
                                                          transformations=[],features_top_ns=[],X_names=[],vectors=[],vector=None,
                                                          allow_x_list_size=1):
    output_objs = {}
    output_metrics = {}
    stats_df = []
    report_name_sufix = ''
    report_name_sufix_xs = ''
    conditions = [type(X)==list,len(transformations)>allow_x_list_size,
                  len(features_top_ns)==len(transformations),len(X_names)==len(transformations)]#
    multiple_x = utils.validate_multiple_conditions(conditions)#
    for train_index, test_index in indexes:
        if multiple_x:
            X_train, X_test,y_train, y_test = transform_and_join(X,Y,train_index,test_index,
                                                                 transformations,features_top_ns,
                       iclf,iclfk,joint_transformation=None,vectors=vectors)
            report_name_sufix_xs = [ xn+"_"+tr+"_"+str(feats) for xn,tr,feats in zip(X_names,transformations,features_top_ns)]
            report_name_sufix_xs = " & ".join(report_name_sufix_xs)
        else:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
        total_features = X_train.shape[-1]
        number_of_features = total_features
        if type(transform)==str:
            report_name_sufix = report_name_sufix_xs+"_"+transform
            number_of_features = 1
            X_train, X_test, y_train, y_test = transform_x_train_test(X_train, X_test, y_train, y_test,
                           transform=transform,iclf=iclf,iclfk=iclfk,vector=vector)
        if len(y_test.shape)>1:
            y_test = y_test.argmax(1)
        if norm==True:
            X_train,X_test = norm_z_score(X_train,X_test)
        start_feature_number = number_of_features
        end_feature_number = total_features + 1
        for feature_number in range(start_feature_number,end_feature_number):
            tmp_scores = output_metrics.get('F'+str(feature_number),{})
            X_train_ = X_train[:,:feature_number]
            X_test_ = X_test[:,:feature_number]
            tmp_scores = fit_and_get_metrics(X_train_,X_test_,y_train,y_test,iclf,iclfk,
                                                 report_metrics,tmp_scores)
            output_metrics['F'+str(feature_number)] = tmp_scores
    if calc_stats==True:
        for fn in range(number_of_features,total_features+1):
            fk = 'F'+str(fn)
            metrics = output_metrics[fk]
            metrics_report = {'Name':report_name+report_name_sufix,'Number of Variables':fn}
            for m in metrics.keys():
                stats = metrics_stats(metrics[m],rn=3)
                for sk in stats.keys():
                    metrics_report[m+"_"+sk]=stats[sk]
            stats_df.append(metrics_report)
        stats_df = pd.DataFrame(stats_df).sort_values(by=[sort_metric,'Number of Variables'],ascending=[False,True]).reset_index(drop=True)
    return output_metrics.copy(),stats_df.copy()


def get_RFE_consistent_priority(iX,iy,train_indexes,iclf,iclfk):
  from sklearn.feature_selection import RFE
  rankings = []
  for train_index in train_indexes:
    x_tr = iX[train_index].copy()
    y_tr = iy[train_index].copy()
    r = iclf(**iclfk)
    selector = RFE(r, n_features_to_select=1, step=1, verbose=0)
    selector.fit(x_tr,y_tr)
    ranking = selector.ranking_
    rankings.append(ranking.copy())
  rankings = np.vstack(rankings).sum(0)
  ranking = np.argsort(rankings)
  return ranking

def cv_metrics_df_with_indexes(X, Y, train_indexes, test_indexes,iclf, iclfk={},
                               report_metrics=['matthews_corr_coef','roc_auc_score','f1_score','sensitivity','specificity'],
                               norm=False,calc_stats=True,report_name='CLF',sort_metric = 'matthews_corr_coef_min'):
    output_objs = {}
    output_metrics = {}
    stats_df = []
    report_name_sufix = ''
    total_features = X.shape[-1]
    for train_index, test_index in zip(train_indexes,test_indexes):
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = Y[train_index], Y[test_index]
      if len(y_test.shape)>1:
        y_test = y_test.argmax(1)
      if norm==True:
        X_train,X_test = norm_z_score(X_train,X_test)
      start_feature_number = 1
      end_feature_number = total_features + 1
      for feature_number in range(start_feature_number,end_feature_number):
        tmp_scores = output_metrics.get('F'+str(feature_number),{})
        X_train_ = X_train[:,:feature_number]
        X_test_ = X_test[:,:feature_number]
        tmp_scores = fit_and_get_metrics(X_train_,X_test_,y_train,y_test,iclf,iclfk,report_metrics,tmp_scores)
        output_metrics['F'+str(feature_number)] = tmp_scores
    if calc_stats==True:
        number_of_features = 1
        for fn in range(number_of_features,total_features+1):
            fk = 'F'+str(fn)
            metrics = output_metrics[fk]
            metrics_report = {'Name':report_name+report_name_sufix,'Number of Variables':fn}
            for m in metrics.keys():
                stats = metrics_stats(metrics[m],rn=3)
                for sk in stats.keys():
                    metrics_report[m+"_"+sk]=stats[sk]
            stats_df.append(metrics_report)
        stats_df = pd.DataFrame(stats_df).sort_values(by=[sort_metric,'Number of Variables'],ascending=[False,True]).reset_index(drop=True)
    return output_metrics.copy(),stats_df.copy()


def get_metrics_class(y_true,y_pred,
                report_metrics=['matthews_corr_coef','roc_auc_score','f1_score','sensitivity','specificity','accuracy'],
                       scores_idic={}):
    #https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
    from sklearn import metrics
    calc_metrics = {'roc_auc_score':metrics.roc_auc_score,
                  'auc':metrics.auc,
                  'f1_score':metrics.f1_score,
                  'sensitivity':get_binary_sensitivity,
                  'specificity':get_binary_specificity,
                  'matthews_corr_coef':metrics.matthews_corrcoef,
                  'accuracy':metrics.accuracy_score
                 }
    iy_test = y_true
    pred_test = y_pred.round()
    pred_prob = y_pred
    tmp_scores = scores_idic.copy()
    for m in report_metrics:
        tmp_metric = tmp_scores.get(m,[])
        if m in ['roc_auc_score']:
            tmp_metric.append(calc_metrics[m](iy_test,pred_prob))
        else:
            tmp_metric.append(calc_metrics[m](iy_test,pred_test))
        tmp_scores[m] = tmp_metric
    return tmp_scores.copy()
