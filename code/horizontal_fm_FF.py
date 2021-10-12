import os
import sys
import yaml
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from itertools import chain
from itertools import product
from functools import partial
from functools import reduce

import pickle
import yaml
import torch 
import torch.nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim
from torch.utils.data import TensorDataset, DataLoader
from hyperopt import Trials, fmin, hp, tpe

import tenseal as ts
import random
import time
from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.metrics import auc,roc_auc_score,log_loss,f1_score,brier_score_loss,precision_recall_curve
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

os.path.realpath(__file__)
main_path = os.getcwd()

# Load config
with open(main_path + '\\config_general.yml','r') as f:
    config = yaml.safe_load(f)

sys.path.append(main_path + '\\code\\modules')

from feature_gen import *
from hfed_models import *


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 300)
pd.options.display.float_format = '{:,.4f}'.format

input_path = main_path + '\\cleaned_datasets\\'
save_path =  main_path + '\\models\\'
result_path = main_path + '\\results\\FL Results\\'
param_path = main_path + '\\results\\Non-FL Results\\Best Params Horizontal\\'


train_size =  config['htrain_size']


#=======================================================
# 1. IID-Data
#=======================================================
start = time.time()
# Load Data
print("Horizontal Split No. 1")
fg_h1 = Feature_Generator(
    data_path =  input_path
    ,file_name = 'horizontal_iid_data1'
    ,n_jobs =  -1
    ,train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_mean']
    ,drop_target = False
)


print("Horizontal Split No. 2")
fg_h2 = Feature_Generator(
    data_path =  input_path,
    file_name = 'horizontal_iid_data2',
    n_jobs =  -1,
    train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_mean']
    ,drop_target = False
)


print("Horizontal Split No. 3")
fg_h3 = Feature_Generator(
    data_path =  input_path,
    file_name = 'horizontal_iid_data3',
    n_jobs =  -1,
    train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_mean']
    ,drop_target = False
)

# Create dataframe for storing the results together
dsplit = ['8-2-1']
fl = [0,1] #[0,1]
noniid = [0] #[0,1]
y_dist = ['7.83% - 7.85% - 7.90%']
model_number = [1,2,3]
epoch_list = ['-']
batches_list = ['-']
scores = [[i] for i in np.repeat(np.nan,12)]

result_df = pd.DataFrame(
    list(product(dsplit,fl,noniid,y_dist,epoch_list,batches_list,model_number,*scores)),
    columns = [
        'Data_Split','FL','Non-IID','y_Dist','Epochs','Batches','Participant',
        'ROC_AUC_Train','ROC_AUC_Test','ROC_AUC_OOS',
        'LOGLOSS_Train','LOGLOSS_Test','LOGLOSS_OOS',
        'PRAUC_Train','PRAUC_Test','PRAUC_OOS',
        'BRIER_Train','BRIER_Test','BRIER_OOS' 
    ]
)

# Create input objects for the model
data_dict = {
    'm1':{
        'fed_xtrain_data':fg_h1.data[0],
        'fed_xtest_data':fg_h1.data[1],
        'fed_ytrain_data':fg_h1.data[2],
        'fed_ytest_data':fg_h1.data[3],
    },
    'm2':{
        'fed_xtrain_data':fg_h2.data[0],
        'fed_xtest_data':fg_h2.data[1],
        'fed_ytrain_data':fg_h2.data[2],
        'fed_ytest_data':fg_h2.data[3],
    },
    'm3':{
        'fed_xtrain_data':fg_h3.data[0],
        'fed_xtest_data':fg_h3.data[1],
        'fed_ytrain_data':fg_h3.data[2],
        'fed_ytest_data':fg_h3.data[3]
    }
}

    

# Load optimal hyperparameters for each participant
params = []
for i in range(1,4):
    with open(param_path + f'hfl_best_params_rfh{i}.pickle','rb') as handle:
        h_param = pickle.load(handle)
        params.append(h_param)



arg_dict = {
    'data':data_dict,
    'params':params,
    'random_state':2021
}


# Instantiate and fit the models (local models + horizontal federated forest)
hfl_ff = HFL_FedForest(**arg_dict)
hfl_ff.train()

# Load additional out-of-sample test data
# Prepare test data
data_test = pd.read_parquet(input_path + 'horizontal_iid_global.parquet') 
data_ext = pd.read_parquet(input_path + 'ext_sources.parquet').drop(labels='target',axis=1)

data_test = data_test.merge(data_ext,how='left')
y_test = data_test['target']
data_test.drop(labels=['sk_id_curr','ext_mean','target'],axis=1,inplace=True)
del data_ext

# Make predictions and evaluate
hfl_ff.evaluate() # creates evaluation scores for 'regular' train and test sets
hfl_ff.evaluate(data_test={'fed_xtest_data':data_test,'fed_ytest_data':y_test}) # creates evaluation scores for 'out-of-sample' set

# Store the results in a 
for mid in result_df['Participant']:
    for idx,t in enumerate(['','_fl']):
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'LOGLOSS_Train'] = hfl_ff.LOGLOSS[f'm{mid}'][f'train{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'LOGLOSS_Test'] = hfl_ff.LOGLOSS[f'm{mid}'][f'test{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'LOGLOSS_OOS'] = hfl_ff.LOGLOSS[f'm{mid}'][f'oos{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'ROC_AUC_Train'] = hfl_ff.ROCAUC[f'm{mid}'][f'train{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'ROC_AUC_Test'] = hfl_ff.ROCAUC[f'm{mid}'][f'test{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'ROC_AUC_OOS'] = hfl_ff.ROCAUC[f'm{mid}'][f'oos{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'PRAUC_Train'] = hfl_ff.PRAUC[f'm{mid}'][f'train{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'PRAUC_Test'] = hfl_ff.PRAUC[f'm{mid}'][f'test{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'PRAUC_OOS'] = hfl_ff.PRAUC[f'm{mid}'][f'oos{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'BRIER_Train'] = hfl_ff.BRIER[f'm{mid}'][f'train{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'BRIER_Test'] = hfl_ff.BRIER[f'm{mid}'][f'test{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'BRIER_OOS'] = hfl_ff.BRIER[f'm{mid}'][f'oos{t}']
           
result_df.to_csv(result_path + 'hfl_FF_results_iid_821.csv')

end = time.time()
print(f"Total Execution Time for IID Data: {(end-start)/60:.2f} min.")




#=======================================================
# 2. Non-IID-Data
#=======================================================
start = time.time()
# Load Data
print("Horizontal Split No. 1")
fg_h1 = Feature_Generator(
    data_path =  input_path
    ,file_name = 'horizontal_noniid_data1'
    ,n_jobs =  -1
    ,train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_mean']
    ,drop_target = False
)


print("Horizontal Split No. 2")
fg_h2 = Feature_Generator(
    data_path =  input_path,
    file_name = 'horizontal_noniid_data2',
    n_jobs =  -1,
    train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_mean']
    ,drop_target = False
)


print("Horizontal Split No. 3")
fg_h3 = Feature_Generator(
    data_path =  input_path,
    file_name = 'horizontal_noniid_data3',
    n_jobs =  -1,
    train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_mean']
    ,drop_target = False
)

# Create dataframe for storing the results together
dsplit = ['8-2-1']
fl = [0,1] #[0,1]
noniid = [1] #[0,1]
y_dist = ['1.9% - 15% - 30%']
model_number = [1,2,3]
epoch_list = ['-']
batches_list = ['-']
scores = [[i] for i in np.repeat(np.nan,12)]

result_df = pd.DataFrame(
    list(product(dsplit,fl,noniid,y_dist,epoch_list,batches_list,model_number,*scores)),
    columns = [
        'Data_Split','FL','Non-IID','y_Dist','Epochs','Batches','Participant',
        'ROC_AUC_Train','ROC_AUC_Test','ROC_AUC_OOS',
        'LOGLOSS_Train','LOGLOSS_Test','LOGLOSS_OOS',
        'PRAUC_Train','PRAUC_Test','PRAUC_OOS',
        'BRIER_Train','BRIER_Test','BRIER_OOS' 
    ]
)

# Create input objects for the model
data_dict = {
    'm1':{
        'fed_xtrain_data':fg_h1.data[0],
        'fed_xtest_data':fg_h1.data[1],
        'fed_ytrain_data':fg_h1.data[2],
        'fed_ytest_data':fg_h1.data[3],
    },
    'm2':{
        'fed_xtrain_data':fg_h2.data[0],
        'fed_xtest_data':fg_h2.data[1],
        'fed_ytrain_data':fg_h2.data[2],
        'fed_ytest_data':fg_h2.data[3],
    },
    'm3':{
        'fed_xtrain_data':fg_h3.data[0],
        'fed_xtest_data':fg_h3.data[1],
        'fed_ytrain_data':fg_h3.data[2],
        'fed_ytest_data':fg_h3.data[3]
    }
}

# Load optimal hyperparameters for each participant
params = []
for i in range(1,4):
    with open(param_path + f'hfl_best_params_rfh{i}.pickle','rb') as handle:
        h_param = pickle.load(handle)
        params.append(h_param)
  
    
arg_dict = {
    'data':data_dict,
    'params':params,
    'random_state':2021
}


# Instantiate and fit the models (local models + horizontal federated forest)
hfl_ff = HFL_FedForest(**arg_dict)
hfl_ff.train()

# Load additional out-of-sample test data
# Prepare test data
data_test = pd.read_parquet(input_path + 'horizontal_noniid_global.parquet') 
data_ext = pd.read_parquet(input_path + 'ext_sources.parquet').drop(labels='target',axis=1)

data_test = data_test.merge(data_ext,how='left')
y_test = data_test['target']
data_test.drop(labels=['sk_id_curr','ext_mean','target'],axis=1,inplace=True)
del data_ext

# Make predictions and evaluate
hfl_ff.evaluate() # creates evaluation scores for 'regular' train and test sets
hfl_ff.evaluate(data_test={'fed_xtest_data':data_test,'fed_ytest_data':y_test}) # creates evaluation scores for 'out-of-sample' set


# Store the results in a 
for mid in result_df['Participant']:
    for idx,t in enumerate(['','_fl']):
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'LOGLOSS_Train'] = hfl_ff.LOGLOSS[f'm{mid}'][f'train{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'LOGLOSS_Test'] = hfl_ff.LOGLOSS[f'm{mid}'][f'test{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'LOGLOSS_OOS'] = hfl_ff.LOGLOSS[f'm{mid}'][f'oos{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'ROC_AUC_Train'] = hfl_ff.ROCAUC[f'm{mid}'][f'train{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'ROC_AUC_Test'] = hfl_ff.ROCAUC[f'm{mid}'][f'test{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'ROC_AUC_OOS'] = hfl_ff.ROCAUC[f'm{mid}'][f'oos{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'PRAUC_Train'] = hfl_ff.PRAUC[f'm{mid}'][f'train{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'PRAUC_Test'] = hfl_ff.PRAUC[f'm{mid}'][f'test{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'PRAUC_OOS'] = hfl_ff.PRAUC[f'm{mid}'][f'oos{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'BRIER_Train'] = hfl_ff.BRIER[f'm{mid}'][f'train{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'BRIER_Test'] = hfl_ff.BRIER[f'm{mid}'][f'test{t}']
        result_df.loc[(result_df['Participant']==mid) & (result_df['FL']==idx),'BRIER_OOS'] = hfl_ff.BRIER[f'm{mid}'][f'oos{t}']
           
result_df.to_csv(result_path + 'hfl_FF_results_noniid_821.csv')

end = time.time()
print(f"Total Execution Time for Non-IID Data: {(end-start)/60:.2f} min.")