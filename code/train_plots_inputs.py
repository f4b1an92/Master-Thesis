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

sys.path.append(main_path + '\\code\\modules')

from feature_gen import *
from nonfed_models import *
from hfed_models import *
from hc_fun import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 300)
pd.options.display.float_format = '{:,.4f}'.format


input_path = main_path + '\\cleaned_datasets\\'
model_path =  main_path + '\\models\\'
result_path =  main_path + '\\results\\'
plot_path = main_path + '\\results\\Plot Data\\'
param_path = main_path + '\\results\\Non-FL Results\\Best Params Horizontal\\'

# Load config
with open(main_path + '/config_general.yml','r') as f:
    config = yaml.safe_load(f)

random_state = config['random_state']
train_size = config['vtrain_size']
ep = [36,12,6]
bat = [1400,700,350]

for j in range(len(ep)):
    #=======================================================
    # 1. IID-Data
    #=======================================================
    print("Horizontal Split No. 1")
    fg_h1 = Feature_Generator(
        data_path =  input_path,
        file_name = 'horizontal_iid_data1',
        n_jobs =  -1,
        train_size = train_size
        ,all_ext_sources = True
        ,vars_to_drop = ['ext_mean']
    )
    
    
    print("Horizontal Split No. 2")
    fg_h2 = Feature_Generator(
        data_path =  input_path,
        file_name = 'horizontal_iid_data2',
        n_jobs =  -1,
        train_size = train_size
        ,all_ext_sources = True
        ,vars_to_drop = ['ext_mean']
    )
    
    
    print("Horizontal Split No. 3")
    fg_h3 = Feature_Generator(
        data_path =  input_path,
        file_name = 'horizontal_iid_data3',
        n_jobs =  -1,
        train_size = train_size
        ,all_ext_sources = True
        ,vars_to_drop = ['ext_mean']
    )
    
    # LOGISTIC REGRESSION
    # Load parameters
    param_list = []
    for i in range(1,4):
        with open(param_path + f'hfl_best_params_lgh{i}.pickle','rb') as handle:
            h_param = pickle.load(handle)
            param_list.append(h_param)
    
    
    # Without FedAvg
    model1 = FedLogReg(input_size=fg_h1.scaled_data[0].shape[1],momentum=param_list[0]['momentum'],lr=param_list[0]['lr'],random_state=config['random_state'])
    model2 = FedLogReg(input_size=fg_h2.scaled_data[0].shape[1],momentum=param_list[1]['momentum'],lr=param_list[1]['lr'],random_state=config['random_state'])
    model3 = FedLogReg(input_size=fg_h3.scaled_data[0].shape[1],momentum=param_list[2]['momentum'],lr=param_list[2]['lr'],random_state=config['random_state'])
    
    model_dict = {
        'm1':model1,
        'm2':model2,
        'm3':model3
    }
    
    data_dict = {
        'm1':{
            'fed_xtrain_data':fg_h1.scaled_data[0],
            'fed_xtest_data':fg_h1.scaled_data[1],
            'fed_ytrain_data':fg_h1.scaled_data[2],
            'fed_ytest_data':fg_h1.scaled_data[3],
        },
        'm2':{
            'fed_xtrain_data':fg_h2.scaled_data[0],
            'fed_xtest_data':fg_h2.scaled_data[1],
            'fed_ytrain_data':fg_h2.scaled_data[2],
            'fed_ytest_data':fg_h2.scaled_data[3],
        },
        'm3':{
            'fed_xtrain_data':fg_h3.scaled_data[0],
            'fed_xtest_data':fg_h3.scaled_data[1],
            'fed_ytrain_data':fg_h3.scaled_data[2],
            'fed_ytest_data':fg_h3.scaled_data[3]
        }
    }
    
    # With loop
    fed_plain_trainer = FLP_Trainer_WOA(models=model_dict,data=data_dict,opti='SGD',verbose=0)
    fed_plain_trainer.train(num_epochs = ep[j],batch_size = bat[j])
    
    with open(result_path + f'bs{bat[j]}_hlg_trainer_WOA_iid.pickle','wb') as handle:
        pickle.dump(fed_plain_trainer.LOGLOSS_batch,handle)
    
    
    # With FedAvg
    model1 = FedLogReg(input_size=fg_h1.scaled_data[0].shape[1],momentum=param_list[0]['momentum'],lr=param_list[0]['lr'],random_state=2021)
    model2 = FedLogReg(input_size=fg_h2.scaled_data[0].shape[1],momentum=param_list[1]['momentum'],lr=param_list[1]['lr'],random_state=2021)
    model3 = FedLogReg(input_size=fg_h3.scaled_data[0].shape[1],momentum=param_list[2]['momentum'],lr=param_list[2]['lr'],random_state=2021)
    
    model_dict = {
        'm1':model1,
        'm2':model2,
        'm3':model3
    }
    
    data_dict = {
        'm1':{
            'fed_xtrain_data':fg_h1.scaled_data[0],
            'fed_xtest_data':fg_h1.scaled_data[1],
            'fed_ytrain_data':fg_h1.scaled_data[2],
            'fed_ytest_data':fg_h1.scaled_data[3],
        },
        'm2':{
            'fed_xtrain_data':fg_h2.scaled_data[0],
            'fed_xtest_data':fg_h2.scaled_data[1],
            'fed_ytrain_data':fg_h2.scaled_data[2],
            'fed_ytest_data':fg_h2.scaled_data[3],
        },
        'm3':{
            'fed_xtrain_data':fg_h3.scaled_data[0],
            'fed_xtest_data':fg_h3.scaled_data[1],
            'fed_ytrain_data':fg_h3.scaled_data[2],
            'fed_ytest_data':fg_h3.scaled_data[3]
        }
    }
    
    arg_dict = {
        'models' : model_dict,
        'data' : data_dict,
        'poly_mod_degree' : 16384, 
        'coeff_mod_bit_sizes' : [60,60,60,60],
        'global_scale' : 2 ** 60,
        'opti' : 'SGD',
        'verbose': 0
    }
    fed_crypt_trainer = FLC_Trainer_CKKS(**arg_dict)
    fed_crypt_trainer.train(num_epochs = ep[j],batch_size = bat[j])
    
    with open(result_path + f'bs{bat[j]}_hlg_trainer_FedAvg_iid.pickle','wb') as handle:
        pickle.dump(fed_crypt_trainer.LOGLOSS_batch,handle)
    
    
    # NEURAL NETWORK
    # Load parameters
    param_list = []
    for i in range(1,4):
        with open(param_path + f'hfl_best_params_nnh{i}.pickle','rb') as handle:
            h_param = pickle.load(handle)
            param_list.append(h_param)
    
    # Without FedAvg
    arg_dict = {
        'layer_sizes': [208,*[param_list[0][f'layer_size{i}'] for i in range(1,param_list[0]['num_hidden_layers']+1)],1],
        'drop_rate':param_list[0]['drop_rate'],
        'momentum':param_list[0]['momentum'],
        'lr':param_list[0]['lr'],
        'random_state':2021
    }
    
    model1 = FedNN(**arg_dict)
    model2 = FedNN(**arg_dict)
    model3 = FedNN(**arg_dict)
    
    model_dict = {
        'm1':model1,
        'm2':model2,
        'm3':model3
    }
    
    data_dict = {
        'm1':{
            'fed_xtrain_data':fg_h1.scaled_data[0],
            'fed_xtest_data':fg_h1.scaled_data[1],
            'fed_ytrain_data':fg_h1.scaled_data[2],
            'fed_ytest_data':fg_h1.scaled_data[3],
        },
        'm2':{
            'fed_xtrain_data':fg_h2.scaled_data[0],
            'fed_xtest_data':fg_h2.scaled_data[1],
            'fed_ytrain_data':fg_h2.scaled_data[2],
            'fed_ytest_data':fg_h2.scaled_data[3],
        },
        'm3':{
            'fed_xtrain_data':fg_h3.scaled_data[0],
            'fed_xtest_data':fg_h3.scaled_data[1],
            'fed_ytrain_data':fg_h3.scaled_data[2],
            'fed_ytest_data':fg_h3.scaled_data[3]
        }
    }
    
    # With loop
    fed_plain_trainer = FLP_Trainer_WOA(models=model_dict,data=data_dict,opti='SGD',verbose=0)
    fed_plain_trainer.train(num_epochs = ep[j],batch_size = bat[j])
    
    with open(result_path + f'bs{bat[j]}_hNN_trainer_WOA_iid.pickle','wb') as handle:
        pickle.dump(fed_plain_trainer.LOGLOSS_batch,handle)
    
    
    # With FedAvg
    arg_dict = {
        'layer_sizes': [208,*[param_list[0][f'layer_size{i}'] for i in range(1,param_list[0]['num_hidden_layers']+1)],1],
        'drop_rate':param_list[0]['drop_rate'],
        'momentum':param_list[0]['momentum'],
        'lr':param_list[0]['lr'],
        'random_state':2021
    }
    
    
    model1 = FedNN(**arg_dict)
    model2 = FedNN(**arg_dict)
    model3 = FedNN(**arg_dict)
    
    model_dict = {
        'm1':model1,
        'm2':model2,
        'm3':model3
    }
    
    data_dict = {
        'm1':{
            'fed_xtrain_data':fg_h1.scaled_data[0],
            'fed_xtest_data':fg_h1.scaled_data[1],
            'fed_ytrain_data':fg_h1.scaled_data[2],
            'fed_ytest_data':fg_h1.scaled_data[3],
        },
        'm2':{
            'fed_xtrain_data':fg_h2.scaled_data[0],
            'fed_xtest_data':fg_h2.scaled_data[1],
            'fed_ytrain_data':fg_h2.scaled_data[2],
            'fed_ytest_data':fg_h2.scaled_data[3],
        },
        'm3':{
            'fed_xtrain_data':fg_h3.scaled_data[0],
            'fed_xtest_data':fg_h3.scaled_data[1],
            'fed_ytrain_data':fg_h3.scaled_data[2],
            'fed_ytest_data':fg_h3.scaled_data[3]
        }
    }
    
    arg_dict = {
        'models' : model_dict,
        'data' : data_dict,
        'poly_mod_degree' : 32_768, 
        'coeff_mod_bit_sizes' : [60,60,60,60],
        'global_scale' : 2 ** 60,
        'opti' : 'SGD',
        'verbose': 0
    }
    
    # With loop
    fed_crypt_trainer = FLC_Trainer_CKKS(**arg_dict)
    fed_crypt_trainer.train(num_epochs = ep[j],batch_size = bat[j])
    
    with open(result_path + f'bs{bat[j]}_hNN_trainer_FedAvg_iid.pickle','wb') as handle:
        pickle.dump(fed_crypt_trainer.LOGLOSS_batch,handle)
    
    
    
    
    
    
    
    #=======================================================
    # 2. Non-IID
    #=======================================================
    print("Horizontal Split No. 1")
    fg_h1 = Feature_Generator(
        data_path =  input_path,
        file_name = 'horizontal_noniid_data1',
        n_jobs =  -1,
        train_size = train_size
        ,all_ext_sources = True
        ,vars_to_drop = ['ext_mean']
    )
    
    
    print("Horizontal Split No. 2")
    fg_h2 = Feature_Generator(
        data_path =  input_path,
        file_name = 'horizontal_noniid_data2',
        n_jobs =  -1,
        train_size = train_size
        ,all_ext_sources = True
        ,vars_to_drop = ['ext_mean']
    )
    
    
    print("Horizontal Split No. 3")
    fg_h3 = Feature_Generator(
        data_path =  input_path,
        file_name = 'horizontal_noniid_data3',
        n_jobs =  -1,
        train_size = train_size
        ,all_ext_sources = True
        ,vars_to_drop = ['ext_mean']
    )
    
    
    # LOGISTIC REGRESSION
    # Load parameters
    param_list = []
    for i in range(1,4):
        with open(param_path + f'hfl_best_params_lgh{i}.pickle','rb') as handle:
            h_param = pickle.load(handle)
            param_list.append(h_param)
            
            
    # Without FedAvg
    model1 = FedLogReg(input_size=fg_h1.scaled_data[0].shape[1],momentum=param_list[0]['momentum'],lr=param_list[0]['lr'],random_state=2021)
    model2 = FedLogReg(input_size=fg_h2.scaled_data[0].shape[1],momentum=param_list[2]['momentum'],lr=param_list[1]['lr'],random_state=2021)
    model3 = FedLogReg(input_size=fg_h3.scaled_data[0].shape[1],momentum=param_list[2]['momentum'],lr=param_list[2]['lr'],random_state=2021)
    
    model_dict = {
        'm1':model1,
        'm2':model2,
        'm3':model3
    }
    
    data_dict = {
        'm1':{
            'fed_xtrain_data':fg_h1.scaled_data[0],
            'fed_xtest_data':fg_h1.scaled_data[1],
            'fed_ytrain_data':fg_h1.scaled_data[2],
            'fed_ytest_data':fg_h1.scaled_data[3],
        },
        'm2':{
            'fed_xtrain_data':fg_h2.scaled_data[0],
            'fed_xtest_data':fg_h2.scaled_data[1],
            'fed_ytrain_data':fg_h2.scaled_data[2],
            'fed_ytest_data':fg_h2.scaled_data[3],
        },
        'm3':{
            'fed_xtrain_data':fg_h3.scaled_data[0],
            'fed_xtest_data':fg_h3.scaled_data[1],
            'fed_ytrain_data':fg_h3.scaled_data[2],
            'fed_ytest_data':fg_h3.scaled_data[3]
        }
    }
    
    # With loop
    fed_plain_trainer = FLP_Trainer_WOA(models=model_dict,data=data_dict,opti='SGD',verbose=0)
    fed_plain_trainer.train(num_epochs = ep[j],batch_size = bat[j])
    
    
    with open(result_path + f'bs{bat[j]}_hlg_trainer_WOA_noniid.pickle','wb') as handle:
        pickle.dump(fed_plain_trainer.LOGLOSS_batch,handle)
        
        
    # With FedAvg
    model1 = FedLogReg(input_size=fg_h1.scaled_data[0].shape[1],momentum=param_list[0]['momentum'],lr=param_list[0]['lr'],random_state=2021)
    model2 = FedLogReg(input_size=fg_h2.scaled_data[0].shape[1],momentum=param_list[0]['momentum'],lr=param_list[0]['lr'],random_state=2021)
    model3 = FedLogReg(input_size=fg_h3.scaled_data[0].shape[1],momentum=param_list[0]['momentum'],lr=param_list[0]['lr'],random_state=2021)
    
    model_dict = {
        'm1':model1,
        'm2':model2,
        'm3':model3
    }
    
    data_dict = {
        'm1':{
            'fed_xtrain_data':fg_h1.scaled_data[0],
            'fed_xtest_data':fg_h1.scaled_data[1],
            'fed_ytrain_data':fg_h1.scaled_data[2],
            'fed_ytest_data':fg_h1.scaled_data[3],
        },
        'm2':{
            'fed_xtrain_data':fg_h2.scaled_data[0],
            'fed_xtest_data':fg_h2.scaled_data[1],
            'fed_ytrain_data':fg_h2.scaled_data[2],
            'fed_ytest_data':fg_h2.scaled_data[3],
        },
        'm3':{
            'fed_xtrain_data':fg_h3.scaled_data[0],
            'fed_xtest_data':fg_h3.scaled_data[1],
            'fed_ytrain_data':fg_h3.scaled_data[2],
            'fed_ytest_data':fg_h3.scaled_data[3]
        }
    }
    
    arg_dict = {
        'models' : model_dict,
        'data' : data_dict,
        'poly_mod_degree' : 16384, 
        'coeff_mod_bit_sizes' : [60,60,60,60],
        'global_scale' : 2 ** 60,
        'opti' : 'SGD',
        'verbose': 0
    }
    fed_crypt_trainer = FLC_Trainer_CKKS(**arg_dict)
    fed_crypt_trainer.train(num_epochs = ep[j],batch_size = bat[j])
    
    with open(result_path + f'bs{bat[j]}_hlg_trainer_FedAvg_noniid.pickle','wb') as handle:
        pickle.dump(fed_crypt_trainer.LOGLOSS_batch,handle)
        
        
        
        
    # NEURAL NETWORK
    # Load parameters
    param_list = []
    for i in range(1,4):
        with open(param_path + f'hfl_best_params_nnh{i}.pickle','rb') as handle:
            h_param = pickle.load(handle)
            param_list.append(h_param)
            
    # Without FedAvg
    arg_dict = {
        'layer_sizes': [208,*[param_list[0][f'layer_size{i}'] for i in range(1,param_list[0]['num_hidden_layers']+1)],1],
        'drop_rate':param_list[0]['drop_rate'],
        'momentum':param_list[0]['momentum'],
        'lr':param_list[0]['lr'],
        'random_state':2021
    }
    
    
    model1 = FedNN(**arg_dict)
    model2 = FedNN(**arg_dict)
    model3 = FedNN(**arg_dict)
    
    model_dict = {
        'm1':model1,
        'm2':model2,
        'm3':model3
    }
    
    data_dict = {
        'm1':{
            'fed_xtrain_data':fg_h1.scaled_data[0],
            'fed_xtest_data':fg_h1.scaled_data[1],
            'fed_ytrain_data':fg_h1.scaled_data[2],
            'fed_ytest_data':fg_h1.scaled_data[3],
        },
        'm2':{
            'fed_xtrain_data':fg_h2.scaled_data[0],
            'fed_xtest_data':fg_h2.scaled_data[1],
            'fed_ytrain_data':fg_h2.scaled_data[2],
            'fed_ytest_data':fg_h2.scaled_data[3],
        },
        'm3':{
            'fed_xtrain_data':fg_h3.scaled_data[0],
            'fed_xtest_data':fg_h3.scaled_data[1],
            'fed_ytrain_data':fg_h3.scaled_data[2],
            'fed_ytest_data':fg_h3.scaled_data[3]
        }
    }
    
    # With loop
    fed_plain_trainer = FLP_Trainer_WOA(models=model_dict,data=data_dict,opti='SGD',verbose=0)
    fed_plain_trainer.train(num_epochs = ep[j],batch_size = bat[j])
    
    with open(result_path + f'bs{bat[j]}_hNN_trainer_WOA_noniid.pickle','wb') as handle:
        pickle.dump(fed_plain_trainer.LOGLOSS_batch,handle)
        
        
    
    # With FedAvg
    arg_dict = {
        'layer_sizes': [208,*[param_list[0][f'layer_size{i}'] for i in range(1,param_list[0]['num_hidden_layers']+1)],1],
        'drop_rate':param_list[0]['drop_rate'],
        'momentum':param_list[0]['momentum'],
        'lr':param_list[0]['lr'],
        'random_state':2021
    }
    
    
    model1 = FedNN(**arg_dict)
    model2 = FedNN(**arg_dict)
    model3 = FedNN(**arg_dict)
    
    model_dict = {
        'm1':model1,
        'm2':model2,
        'm3':model3
    }
    
    data_dict = {
        'm1':{
            'fed_xtrain_data':fg_h1.scaled_data[0],
            'fed_xtest_data':fg_h1.scaled_data[1],
            'fed_ytrain_data':fg_h1.scaled_data[2],
            'fed_ytest_data':fg_h1.scaled_data[3],
        },
        'm2':{
            'fed_xtrain_data':fg_h2.scaled_data[0],
            'fed_xtest_data':fg_h2.scaled_data[1],
            'fed_ytrain_data':fg_h2.scaled_data[2],
            'fed_ytest_data':fg_h2.scaled_data[3],
        },
        'm3':{
            'fed_xtrain_data':fg_h3.scaled_data[0],
            'fed_xtest_data':fg_h3.scaled_data[1],
            'fed_ytrain_data':fg_h3.scaled_data[2],
            'fed_ytest_data':fg_h3.scaled_data[3]
        }
    }
    
    arg_dict = {
        'models' : model_dict,
        'data' : data_dict,
        'poly_mod_degree' : 32_768, 
        'coeff_mod_bit_sizes' : [60,60,60,60],
        'global_scale' : 2 ** 60,
        'opti' : 'SGD',
        'verbose': 0
    }
    
    # With loop
    fed_crypt_trainer = FLC_Trainer_CKKS(**arg_dict)
    fed_crypt_trainer.train(num_epochs = ep[j],batch_size = bat[j])
    
    with open(result_path + f'bs{bat[j]}_hNN_trainer_FedAvg_noniid.pickle','wb') as handle:
        pickle.dump(fed_crypt_trainer.LOGLOSS_batch,handle)
        
        
        
# Create the acutal plots
# Extract data
bs = bat[::-1]
mod = ['lg','NN']
mod_type = ['WOA','FedAvg']
data_type = ['iid','noniid']
inp_list = [bs,mod,mod_type,data_type]

combis = list(product(*inp_list))

data_list_m2 = []
data_list_m3 = []

for c in combis:
    with open(result_path + f'bs{c[0]}_h{c[1]}_trainer_{c[2]}_{c[3]}.pickle','rb') as handle:
        tmp = pickle.load(handle)
    
    m2_epoch_list = []
    for val in tmp['m2'].values():
        m2_epoch_list.extend(val)
    
    m3_epoch_list = []
    for val in tmp['m3'].values():
        m3_epoch_list.extend(val)
    
    m2_data = pd.DataFrame({'client':'m2','Batch Size':c[0],'model':c[1],'Model Type':c[2],'data_dist':c[3],'loss':m2_epoch_list})
    data_list_m2.append(m2_data)
    
    m3_data = pd.DataFrame({'client':'m3','Batch Size':c[0],'model':c[1],'Model Type':c[2],'data_dist':c[3],'loss':m3_epoch_list})
    data_list_m3.append(m3_data)
    
df_m2 = pd.concat(data_list_m2,axis=0)
df_m3 = pd.concat(data_list_m3,axis=0)

df = pd.concat([df_m2,df_m3],axis=0)


# Prepare final format for plots
combis = list(product(['m2','m3'],data_type))

data_lg = {}
data_nn = {}

for c in combis:
    data_lg[c[0]] = {
        c[1]:df.loc[(df['model']=='lg') & (df['client']==c[0]) & (df['data_dist']==c[1]), :]
    }
    
    data_nn[c[0]] = {
        c[1]:df.loc[(df['model']=='NN') & (df['client']==c[0]) & (df['data_dist']==c[1]), :]
    }
    
clients = ['m2','m3']
ddist = ['iid','noniid']

data_lg = {c:{} for c in clients} 
data_nn = {c:{} for c in clients}

for c in clients:
    for d in ddist:
        # Logistic Regression data
        tmp = df.loc[(df['model']=='lg') & (df['client']==c) & (df['data_dist']==d), :].reset_index()
        max_rows = tmp.groupby('Batch Size')['index'].max().min()

        tmp = tmp.loc[tmp['index']<=max_rows,:]
        data_lg[c][d] = tmp

        # Neural Network data
        tmp = df.loc[(df['model']=='NN') & (df['client']==c) & (df['data_dist']==d), :].reset_index()
        max_rows = tmp.groupby('Batch Size')['index'].max().min()

        tmp = tmp.loc[tmp['index']<=max_rows,:]
        data_nn[c][d] = tmp



# Plot & Save the results           
# 1. Log. Regression (two batch sizes)
arg_dict = {
    'data':data_lg,
    'fsize':(16,10),
    'bsizes':[700,1400],
    'colors':['red','blue','orange','green'], # lime & purple for plot with three lines
    'save':True,
    'save_path':plot_path,
    'file_name': "progress_plot_lg2"
}
progress_plot(**arg_dict)


# 2. Neural Network (two batch sizes)
arg_dict = {
    'data':data_nn,
    'fsize':(16,10),
    'bsizes':[700,1400],
    'colors':['red','blue','orange','green'],
    'save':True,
    'save_path':plot_path,
    'file_name':"progress_plot_nn2"
}
progress_plot(**arg_dict)


# 3. Log. Regression (three batch sizes)
arg_dict = {
    'data':data_lg,
    'fsize':(16,10),
    'bsizes':[350,700,1400],
    'colors':['red','blue','lime','orange','green','purple'], # lime & purple for plot with three lines
    'save':True,
    'save_path':plot_path,
    'file_name': "progress_plot_lg3"
}
progress_plot(**arg_dict)


# 2. Neural Network (two batch sizes)
arg_dict = {
    'data':data_nn,
    'fsize':(16,10),
    'bsizes':[350,700,1400],
    'colors':['red','blue','lime','orange','green','purple'],
    'save':True,
    'save_path':plot_path,
    'file_name':"progress_plot_nn3"
}
progress_plot(**arg_dict)