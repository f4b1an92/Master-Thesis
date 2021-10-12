import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
from itertools import product
from tqdm import tqdm

import torch 
import torch.nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import TensorDataset, DataLoader

import tenseal as ts
import random
import time
import pickle

from sklearn.metrics import auc,roc_auc_score,log_loss,f1_score,brier_score_loss,precision_recall_curve

os.path.realpath(__file__)
main_path = os.getcwd()

# Load config
with open(main_path + '\\config_general.yml','r') as f:
    config = yaml.safe_load(f)

sys.path.append(main_path + '\\code\\modules')

from feature_gen import *
from nonfed_models import *
from vfed_models import *
from hc_fun import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 300)
pd.options.display.float_format = '{:,.4f}'.format

input_path = main_path + '\\cleaned_datasets\\'
save_path =  main_path + '\\models\\'
result_path =  main_path +'\\results\\FL Results\\'
param_path = main_path + '\\results\\Non-FL Results\\Best Params Vertical\\'


random_state = config['random_state']
train_size = config['vtrain_size']


#================================================================
# 1. Load Data Sets
#================================================================
print("1. Previous & Current Applications")
fg_v1 = Feature_Generator(
    data_path =  input_path
    ,file_name = ['application_train_cleaned','prev_app_cleaned']
    ,n_jobs =  -1
    ,random_state = random_state # super important to be equal for all other nfm-objects!!! Otherwise modeling doesn't work!
    ,train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_mean','ext_source_2','ext_source_3']
    ,drop_target = False
    ,vertical_lg_host = False
)


print("2. Bureau Data")
fg_v2 = Feature_Generator(
    data_path =  input_path,
    file_name = 'bureau_data_cleaned',
    n_jobs =  -1,
    random_state = random_state, # super important to be equal for all other nfm-objects!!! Otherwise modeling doesn't work!
    train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_source_1','ext_source_3'] # 'ext_source_2'
    ,drop_target = True
)


print("3. Installment Payments Data")
fg_v3 = Feature_Generator(
    data_path =  input_path,
    file_name = 'installments_payments_cleaned',
    n_jobs =  -1,
    random_state = random_state, # super important to be equal for all other nfm-objects!!! Otherwise modeling doesn't work!
    train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_source_1','ext_source_2']
    ,drop_target = True
)


print("4. Full Data Set")
fg_vfull = Feature_Generator(
    data_path =  input_path,
    file_name = 'full_data_cleaned',
    n_jobs =  -1,
    random_state = random_state, # super important to be equal for all other nfm-objects!!! Otherwise modeling doesn't work!
    train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_mean']
)



# Benchmark 1: Train local model for the label holder only (current & previous applications)
#=============================================================================================
# Load parameters
with open(param_path + 'vfl_best_params_rf_curr.pickle','rb') as handle:
    v1_param = pickle.load(handle)

other_params = {
    'random_state':random_state,
    'criterion':'entropy',
    'max_features':'sqrt',
    'verbose':0,
    'n_jobs':-1
}

local_rf = RandomForestClassifier(**{**v1_param,**other_params})

local_rf.fit(fg_v1.data[0],fg_v1.data[2])

# Predictions
y_pred_train = local_rf.predict_proba(fg_v1.data[0])[:,1] 
y_pred_test = local_rf.predict_proba(fg_v1.data[1])[:,1]

# Evaluations
metrics_rf_v1 = {
    'LOGLOSS':{'train':{},'test':{}},
    'ROCAUC':{'train':{},'test':{}},
    'PRAUC':{'train':{},'test':{}},
    'BRIER':{'train':{},'test':{}}
}

log_loss_train = log_loss(fg_v1.data[2],y_pred_train,eps=0.000001)
log_loss_test = log_loss(fg_v1.data[3],y_pred_test,eps=0.000001)
metrics_rf_v1['LOGLOSS']['train'] = log_loss_train
metrics_rf_v1['LOGLOSS']['test'] = log_loss_test

roc_auc_train = roc_auc_score(fg_v1.data[2],y_pred_train)
roc_auc_test = roc_auc_score(fg_v1.data[3],y_pred_test)
metrics_rf_v1['ROCAUC']['train'] = roc_auc_train
metrics_rf_v1['ROCAUC']['test'] = roc_auc_test

precision_train, recall_train, _ = precision_recall_curve(fg_v1.data[2],y_pred_train)
pr_auc_train = auc(recall_train,precision_train)
precision_test, recall_test, _ = precision_recall_curve(fg_v1.data[3],y_pred_test)
pr_auc_test = auc(recall_test,precision_test)
metrics_rf_v1['PRAUC']['train'] = pr_auc_train
metrics_rf_v1['PRAUC']['test'] = pr_auc_test

brier_train = brier_score_loss(fg_v1.data[2],y_pred_train)
brier_test = brier_score_loss(fg_v1.data[3],y_pred_test)
metrics_rf_v1['BRIER']['train'] = brier_train
metrics_rf_v1['BRIER']['test'] = brier_test

print(f'Log-Loss on train data: {log_loss(fg_v1.data[2],y_pred_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_v1.data[3],y_pred_test,eps=0.000001):.4f}')
print(f'ROC-AUC on train data: {roc_auc_score(fg_v1.data[2],y_pred_train):.4f}')
print(f'ROC-AUC on test data: {roc_auc_score(fg_v1.data[3],y_pred_test):.4f}')
print(80*"#")
print("\t\t END OF RANDOM FOREST FITTING for PREV. & CURRENT APPLICATIONS DATA")
print(80*"#")



# Benchmark 2: Train global model using the data of all pariticipants directly (should theoretically result in the optimal outcome)
#====================================================================================================================================
# Load parameters
with open(param_path + 'vfl_best_params_rf_full.pickle','rb') as handle:
    vfull_param = pickle.load(handle)

other_params = {
    'random_state':random_state,
    'criterion':'entropy',
    'max_features':'sqrt',
    'verbose':0,
    'n_jobs':-1
}

global_rf = RandomForestClassifier(**{**vfull_param,**other_params})

global_rf.fit(fg_vfull.data[0],fg_vfull.data[2])

# Predictions
y_pred_train = global_rf.predict_proba(fg_vfull.data[0])[:,1] 
y_pred_test = global_rf.predict_proba(fg_vfull.data[1])[:,1]

# Evaluations
metrics_rf_vfull = {
    'LOGLOSS':{'train':{},'test':{}},
    'ROCAUC':{'train':{},'test':{}},
    'PRAUC':{'train':{},'test':{}},
    'BRIER':{'train':{},'test':{}}
}

log_loss_train = log_loss(fg_vfull.data[2],y_pred_train,eps=0.000001)
log_loss_test = log_loss(fg_vfull.data[3],y_pred_test,eps=0.000001)
metrics_rf_vfull['LOGLOSS']['train'] = log_loss_train
metrics_rf_vfull['LOGLOSS']['test'] = log_loss_test

roc_auc_train = roc_auc_score(fg_vfull.data[2],y_pred_train)
roc_auc_test = roc_auc_score(fg_vfull.data[3],y_pred_test)
metrics_rf_vfull['ROCAUC']['train'] = roc_auc_train
metrics_rf_vfull['ROCAUC']['test'] = roc_auc_test

precision_train, recall_train, _ = precision_recall_curve(fg_vfull.data[2],y_pred_train)
pr_auc_train = auc(recall_train,precision_train)
precision_test, recall_test, _ = precision_recall_curve(fg_vfull.data[3],y_pred_test)
pr_auc_test = auc(recall_test,precision_test)
metrics_rf_vfull['PRAUC']['train'] = pr_auc_train
metrics_rf_vfull['PRAUC']['test'] = pr_auc_test

brier_train = brier_score_loss(fg_vfull.data[2],y_pred_train)
brier_test = brier_score_loss(fg_vfull.data[3],y_pred_test)
metrics_rf_vfull['BRIER']['train'] = brier_train
metrics_rf_vfull['BRIER']['test'] = brier_test


print(f'Log-Loss on train data: {log_loss(fg_v1.data[2],y_pred_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_v1.data[3],y_pred_test,eps=0.000001):.4f}')
print(f'ROC-AUC on train data: {roc_auc_score(fg_v1.data[2],y_pred_train):.4f}')
print(f'ROC-AUC on test data: {roc_auc_score(fg_v1.data[3],y_pred_test):.4f}')
print(80*"#")
print("\t\t END OF RANDOM FOREST FITTING for FULL DATA")
print(80*"#")



# Vertical Federated Forest
#=============================    
X_train = [
    fg_v1.data[0].reset_index(),
    fg_v2.data[0].reset_index(),
    fg_v3.data[0].reset_index()
]
y_train = fg_v1.data[2]

X_test = [
    fg_v1.data[1].reset_index(),
    fg_v2.data[1].reset_index(),
    fg_v3.data[1].reset_index()
]
y_test = fg_v1.data[3]


arg_dict = {
    'num_parties':3,
    'target_values': [0,1],
    'id_col_name':'index',
    'n_estimators':vfull_param['n_estimators'],
    'max_depth':vfull_param['max_depth'],
    'max_features':'sqrt',
    'max_samples':vfull_param['max_samples'],
    'min_samples_split':vfull_param['min_samples_split'],
    'min_impurity_decrease':vfull_param['min_impurity_decrease'],
    'crit':'entropy',
    'random_state':random_state
}
    
fed_forest = FederatedForest(**arg_dict)
fed_forest.fit_forest(X_train,y_train)    
 
yhat_train = fed_forest.predict(X_train,pred_type='prob',drop_idx=True)
yhat_test = fed_forest.predict(X_test,pred_type='prob',drop_idx=True)

# Evaluations
metrics_vff = {
    'LOGLOSS':{'train':{},'test':{}},
    'ROCAUC':{'train':{},'test':{}},
    'PRAUC':{'train':{},'test':{}},
    'BRIER':{'train':{},'test':{}}
}

log_loss_train = log_loss(y_train.sort_index(),yhat_train[:,1],eps=0.000001)
log_loss_test = log_loss(y_test.sort_index(),yhat_test[:,1],eps=0.000001)
metrics_vff['LOGLOSS']['train'] = log_loss_train
metrics_vff['LOGLOSS']['test'] = log_loss_test

roc_auc_train = roc_auc_score(y_train.sort_index(),yhat_train[:,1])
roc_auc_test = roc_auc_score(y_test.sort_index(),yhat_test[:,1])
metrics_vff['ROCAUC']['train'] = roc_auc_train
metrics_vff['ROCAUC']['test'] = roc_auc_test

precision_train, recall_train, _ = precision_recall_curve(y_train.sort_index(),yhat_train[:,1])
pr_auc_train = auc(recall_train,precision_train)
precision_test, recall_test, _ = precision_recall_curve(y_test.sort_index(),yhat_test[:,1])
pr_auc_test = auc(recall_test,precision_test)
metrics_vff['PRAUC']['train'] = pr_auc_train
metrics_vff['PRAUC']['test'] = pr_auc_test

brier_train = brier_score_loss(y_train.sort_index(),yhat_train[:,1])
brier_test = brier_score_loss(y_test.sort_index(),yhat_test[:,1])
metrics_vff['BRIER']['train'] = brier_train
metrics_vff['BRIER']['test'] = brier_test


print(f'Log-Loss on train data: {log_loss(y_train.sort_index(),yhat_train[:,1],eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(y_test.sort_index(),yhat_test[:,1],eps=0.000001):.4f}')
print(f'ROC-AUC on train data: {roc_auc_score(y_train.sort_index(),yhat_train[:,1]):.4f}')
print(f'ROC-AUC on test data: {roc_auc_score(y_test.sort_index(),yhat_test[:,1]):.4f}')
print(80*"#")
print("\t\t END OF FEDERATED FOREST FITTING")
print(80*"#")



# Put results together in one dataframe
modeltypes = ['Random Forest','Random Forest','Federated Forest']
fl = [0,0,1]
dc = ['Label Holder','All','All']
df_list = []
for i,ds in enumerate([metrics_rf_v1, metrics_rf_vfull, metrics_vff]):
    df1 = pd.DataFrame({'Model':[modeltypes[i]],'FL':[fl[i]],'Data_Contributors':[dc[i]]})

    df_dict = {}
    for m in ds.keys():
        if m == 'F1':
            pass
        else:
            for dstype in ds[m].keys():
                col = '_'.join([m,dstype])
                df_dict[col] = [ds[m][dstype]]
                
    df2 = pd.DataFrame(df_dict)    
    df_list.append(pd.concat([df1,df2],axis=1))

df_final = pd.concat(df_list,axis=0).reset_index(drop=True)

# Save results
df_final.to_csv(result_path + 'vfl_ff_results.csv',index=False)


