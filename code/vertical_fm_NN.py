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
model_path =  main_path + '\\models\\'
result_path =  main_path + '\\results\\FL Results\\'
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
    ,vars_to_drop = ['ext_source_1','ext_source_3'] 
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
with open(param_path + 'vfl_best_params_nn_curr.pickle','rb') as handle:
    v1_param = pickle.load(handle)

arg_dict = {
    'layer_sizes': [fg_v1.scaled_data[0].shape[1],*[v1_param[f'layer_size{i}'] for i in range(1,v1_param['num_hidden_layers']+1)],1],
    'drop_rate':v1_param['drop_rate'],
    'weight_decay':v1_param['weight_decay'],
    'momentum':v1_param['momentum'],
    'lr':v1_param['lr'],
    'random_state':random_state
}

# Fit final model
nfm = NonFed_Models()

nn_model = nfm.NN(**arg_dict)

trainer_nn = nfm.Trainer(
    model = nn_model,
    data = fg_v1.scaled_data,
    opti = 'SGD'
)
trainer_nn.train(
    num_epochs = v1_param['epochs'],
    batch_size = v1_param['batch_size'],
    save_model = False,
    save_path = model_path,
    model_name = 'neuralnet_prev_curr_appdata'
)

# Predictions
yhat_train = nn_model.predict(fg_v1.scaled_data[0])
yhat_test = nn_model.predict(fg_v1.scaled_data[1])

# Evaluations
metrics_nn_prev = {
    'LOGLOSS':trainer_nn.LOGLOSS_epoch,
    'ROCAUC':trainer_nn.ROCAUC_epoch,
    'PRAUC':trainer_nn.PRAUC_epoch,
    'BRIER':trainer_nn.BRIER_epoch
}


print(f'Log-Loss on train data: {log_loss(fg_v1.data[2],yhat_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_v1.data[3],yhat_test,eps=0.000001):.4f}')
print(f"ROC-AUC on TRAIN data set: {roc_auc_score(fg_v1.scaled_data[2],yhat_train):.4f}")
print(f"ROC-AUC on TEST data set: {roc_auc_score(fg_v1.scaled_data[3],yhat_test):.4f}")
print(80*"#")
print("\t\t END OF NEURAL NETWORK FITTING for CURRENT & PREV. APPLICATIONS")
print(80*"#")




# Benchmark 2: Train global model using the data of all pariticipants directly (should theoretically result in the optimal outcome)
#====================================================================================================================================
# Load parameters
with open(param_path + 'vfl_best_params_nn_full.pickle','rb') as handle:
    vfull_param = pickle.load(handle)


arg_dict = {
    'layer_sizes': [fg_vfull.scaled_data[0].shape[1],*[vfull_param[f'layer_size{i}'] for i in range(1,vfull_param['num_hidden_layers']+1)],1],
    'drop_rate':vfull_param['drop_rate'],
    'weight_decay':vfull_param['weight_decay'],
    'momentum':vfull_param['momentum'],
    'lr':vfull_param['lr'],
    'random_state':random_state
}

# Fit final model
nfm = NonFed_Models()

nn_model = nfm.NN(**arg_dict)

trainer_nn = nfm.Trainer(
    model = nn_model,
    data = fg_vfull.scaled_data,
    opti = 'SGD'
)

trainer_nn.train(
    num_epochs = vfull_param['epochs'],
    batch_size = vfull_param['batch_size'],
    save_model = False,
    save_path = model_path,
    model_name = 'neuralnet_prev_curr_appdata'
)

# Predictions
yhat_train = nn_model.predict(fg_vfull.scaled_data[0])
yhat_test = nn_model.predict(fg_vfull.scaled_data[1])

# Evaluations
metrics_nn_full = {
    'LOGLOSS':trainer_nn.LOGLOSS_epoch,
    'ROCAUC':trainer_nn.ROCAUC_epoch,
    'PRAUC':trainer_nn.PRAUC_epoch,
    'BRIER':trainer_nn.BRIER_epoch
}


print(f'Log-Loss on train data: {log_loss(fg_vfull.data[2],yhat_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_vfull.data[3],yhat_test,eps=0.000001):.4f}')
print(f"ROC-AUC on TRAIN data set: {roc_auc_score(fg_vfull.scaled_data[2],yhat_train):.4f}")
print(f"ROC-AUC on TEST data set: {roc_auc_score(fg_vfull.scaled_data[3],yhat_test):.4f}")
print(80*"#")
print("\t\t END OF NEURAL NETWORK FITTING for FULL DATA")
print(80*"#")




# Split Neural Networks
#========================================
# Load parameters
with open(param_path + 'vfl_best_params_nn_full.pickle','rb') as handle:
    vfull_param = pickle.load(handle)

arg_dict = {
    'drop_rate':vfull_param['drop_rate'],
    'weight_decay':vfull_param['weight_decay'],
    'momentum':vfull_param['momentum'],
    'lr':vfull_param['lr'],
    'init':'kaiming',
    'bias':True,
    'random_state':random_state
}


# Select structure for splitNN:
# -> this structure was choosen arbitrarily 
# -> further hyperparameter(s)
# Participant 1: gets a NN with an input and one hidden layer (no output layer) -> hiddenlayer size = int((fg_v1.data.shape[1] / count(all features)) * vfull_param['layer_size1']) 
# Participant 2: gets a NN with an input and one hidden layer (no output layer) -> hiddenlayer size = int((fg_v2.data.shape[1] / count(all features)) * vfull_param['layer_size1']) 
# Participant 3: gets a NN with an input and one hidden layer (no output layer) -> hiddenlayer size = int((fg_v3.data.shape[1] / count(all features)) * vfull_param['layer_size1']) 
# Coordinator:   gets a NN with an input layer (HL outputs of all 3 pariticpant models) and two hidden layers (no output layer) 
# Participant 1: as labelholder, P1 also gets the final layer of the splitNN
hlsize1 = round((fg_v1.data[0].shape[1] / sum([i.data[0].shape[1] for i in [fg_v1,fg_v2,fg_v3]])) * vfull_param['layer_size1'])
hlsize2 = round((fg_v2.data[0].shape[1] / sum([i.data[0].shape[1] for i in [fg_v1,fg_v2,fg_v3]])) * vfull_param['layer_size1'])
hlsize3 = round((fg_v3.data[0].shape[1] / sum([i.data[0].shape[1] for i in [fg_v1,fg_v2,fg_v3]])) * vfull_param['layer_size1'])

model1 = SplitNN(layer_sizes=[fg_v1.scaled_data[0].shape[1],hlsize1],**arg_dict)
model2 = SplitNN(layer_sizes=[fg_v2.scaled_data[0].shape[1],hlsize2],**arg_dict)
model3 = SplitNN(layer_sizes=[fg_v3.scaled_data[0].shape[1],hlsize3],**arg_dict)
model4 = SplitNN(layer_sizes=[sum([hlsize1,hlsize2,hlsize3]),vfull_param['layer_size2'],vfull_param['layer_size3'],vfull_param['layer_size4']],**arg_dict)
model5 = SplitNN(layer_sizes=[vfull_param['layer_size4'],1],**arg_dict)

model_dict = {
    'm1':model1,
    'm2':model2,
    'm3':model3,
    'm4':model4,
    'm5':model5
}

data_dict = {
    'm1':{
        'fed_xtrain_data':fg_v1.scaled_data[0],
        'fed_xtest_data':fg_v1.scaled_data[1],
        'fed_ytrain_data':fg_v1.scaled_data[2],
        'fed_ytest_data':fg_v1.scaled_data[3],
    },
    'm2':{
        'fed_xtrain_data':fg_v2.scaled_data[0],
        'fed_xtest_data':fg_v2.scaled_data[1],
    },
    'm3':{
        'fed_xtrain_data':fg_v3.scaled_data[0],
        'fed_xtest_data':fg_v3.scaled_data[1],
    }
}
         
arg_dict_trainer = {
    'data':data_dict,
    'models':model_dict,
    'poly_mod_degree':16384,
    'coeff_mod_bit_sizes': [60,60,60,60],
    'global_scale': 2**60,
    'opti':'SGD',
    'verbose':1,
    'random_state':random_state
    
}        
        
        
split_trainer = SplitNN_Trainer(**arg_dict_trainer)

split_trainer.train(num_epochs= vfull_param['epochs'] ,batch_size=vfull_param['batch_size'],encryption=True,red='mean',progress_freq=1,stop_criteria=None)     

# Make predictions
coord_output_train = split_trainer.ensemble_forward(bat_ids=None,encryption=False,train_set=True) # 'encryption' was set to False b/c encrypting the entire test set after the ensemble forward pass lead to memory shortage brought the kernel to crash on my machine
coord_output_test = split_trainer.ensemble_forward(bat_ids=None,encryption=False,train_set=False) # 'encryption' was set to False b/c encrypting the entire test set after the ensemble forward pass lead to memory shortage & brought the kernel to crash on my machine      


y_pred_train = model_dict['m5'].predict(coord_output_train).detach().numpy()
y_pred_test = model_dict['m5'].predict(coord_output_test).detach().numpy()                


# Evaluations
metrics_vnn = {
    'LOGLOSS':{'train':{},'test':{}},
    'ROCAUC':{'train':{},'test':{}},
    'PRAUC':{'train':{},'test':{}},
    'BRIER':{'train':{},'test':{}}
}

log_loss_train = log_loss(fg_v1.data[2],y_pred_train,eps=0.000001)
log_loss_test = log_loss(fg_v1.data[3],y_pred_test,eps=0.000001)
metrics_vnn['LOGLOSS']['train'] = log_loss_train
metrics_vnn['LOGLOSS']['test'] = log_loss_test

roc_auc_train = roc_auc_score(fg_v1.data[2],y_pred_train)
roc_auc_test = roc_auc_score(fg_v1.data[3],y_pred_test)
metrics_vnn['ROCAUC']['train'] = roc_auc_train
metrics_vnn['ROCAUC']['test'] = roc_auc_test

precision_train, recall_train, _ = precision_recall_curve(fg_v1.data[2],y_pred_train)
pr_auc_train = auc(recall_train,precision_train)
precision_test, recall_test, _ = precision_recall_curve(fg_v1.data[3],y_pred_test)
pr_auc_test = auc(recall_test,precision_test)
metrics_vnn['PRAUC']['train'] = pr_auc_train
metrics_vnn['PRAUC']['test'] = pr_auc_test

brier_train = brier_score_loss(fg_v1.data[2],y_pred_train)
brier_test = brier_score_loss(fg_v1.data[3],y_pred_test)
metrics_vnn['BRIER']['train'] = brier_train
metrics_vnn['BRIER']['test'] = brier_test



# Put results together in one dataframe
epoch_list_v1 = [key for key in metrics_nn_prev['LOGLOSS']['train'].keys()]
epoch_list_vfull = [key for key in metrics_nn_full['LOGLOSS']['train'].keys()]
epoch_max = [epoch_list_v1[-1],epoch_list_vfull[-1]]

modeltypes = ['Neural Network','Neural Network']
fl = [0,0]
dc = ['Label Holder','All']
df_list = []
for i,ds in enumerate([metrics_nn_prev, metrics_nn_full]):
    df1 = pd.DataFrame({'Model':[modeltypes[i]],'FL':[fl[i]],'Data_Contributors':[dc[i]]})

    df_dict = {}
    for m in ds.keys():
        if m == 'F1':
            pass
        else:
            for dstype in ds[m].keys():
                col = '_'.join([m,dstype])
                df_dict[col] = [ds[m][dstype][epoch_max[i]]]
                
    df2 = pd.DataFrame(df_dict)    
    df_list.append(pd.concat([df1,df2],axis=1))

df_final = pd.concat(df_list,axis=0).reset_index(drop=True)



# Put results together in one dataframe
modeltypes = ['SplitNN']
fl = [1]
dc = ['All']
df_list = []

df1 = pd.DataFrame({'Model':[modeltypes[0]],'FL':[fl[0]],'Data_Contributors':[dc[0]]})

df_dict = {}
for m in metrics_vnn.keys():
    if m == 'F1':
        pass
    else:
        for dstype in metrics_vnn[m].keys():
            col = '_'.join([m,dstype])
            df_dict[col] = [metrics_vnn[m][dstype]]
            
df2 = pd.DataFrame(df_dict)    

df_vnn = pd.concat([df1,df2],axis=1)

df_final = pd.concat([df_final,df_vnn],axis=0).reset_index(drop=True)

# Save results
df_final.to_csv(result_path + 'vfl_splitnn_results.csv',index=False)
