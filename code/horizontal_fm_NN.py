import os
import sys
import yaml
import numpy as np
import pandas as pd
from itertools import chain
from itertools import product

import torch 
import torch.nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import TensorDataset, DataLoader

import tenseal as ts
import random
import time
import pickle
from tqdm import tqdm
from sklearn.metrics import auc,roc_auc_score,log_loss,f1_score,brier_score_loss,precision_recall_curve

os.path.realpath(__file__)
main_path = os.getcwd()

# Load config
with open(main_path + '\\config_general.yml','r') as f:
    config = yaml.safe_load(f)

sys.path.append(main_path + '\\code\\modules')

from feature_gen import *
from hfed_models import *
from hc_fun import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 300)
pd.options.display.float_format = '{:,.4f}'.format

input_path = main_path + '\\cleaned_datasets\\'
model_path =  main_path + '\\models\\'
result_path =  main_path + '\\results\\FL Results\\'
param_path = main_path + '\\results\\Non-FL Results\\Best Params Horizontal\\'

train_size =  config['htrain_size']

#=======================================================
# 1. IID-Data
#=======================================================
print("Horizontal Split No. 1")
nfm_h1 = Feature_Generator(
    data_path =  input_path
    ,file_name = 'horizontal_iid_data1'
    ,n_jobs =  -1
    ,train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_mean']
    ,drop_target = False
)


print("Horizontal Split No. 2")
nfm_h2 = Feature_Generator(
    data_path =  input_path,
    file_name = 'horizontal_iid_data2',
    n_jobs =  -1,
    train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_mean']
    ,drop_target = False
)


print("Horizontal Split No. 3")
nfm_h3 = Feature_Generator(
    data_path =  input_path,
    file_name = 'horizontal_iid_data3',
    n_jobs =  -1,
    train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_mean']
    ,drop_target = False
)


# Load parameters
param_list = []
for i in range(1,4):
    with open(param_path + f'hfl_best_params_nnh{i}.pickle','rb') as handle:
        h_param = pickle.load(handle)
        param_list.append(h_param)


# 1.1. IID WithOUT Aggregation
dsplit = ['8-2-1']
fl = [0] #[0,1]
noniid = [0] #[0,1]
y_dist = ['7.83% - 7.85% - 7.90%']
model_number = [1,2,3]
epoch_list = [10,5,3,1]
batches_list = [1024,512,256,128,64,32,16]
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


start = time.time()
for ep_idx,ep in enumerate(epoch_list):
    for bat in tqdm(batches_list):
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
                'fed_xtrain_data':nfm_h1.scaled_data[0],
                'fed_xtest_data':nfm_h1.scaled_data[1],
                'fed_ytrain_data':nfm_h1.scaled_data[2],
                'fed_ytest_data':nfm_h1.scaled_data[3],
            },
            'm2':{
                'fed_xtrain_data':nfm_h2.scaled_data[0],
                'fed_xtest_data':nfm_h2.scaled_data[1],
                'fed_ytrain_data':nfm_h2.scaled_data[2],
                'fed_ytest_data':nfm_h2.scaled_data[3],
            },
            'm3':{
                'fed_xtrain_data':nfm_h3.scaled_data[0],
                'fed_xtest_data':nfm_h3.scaled_data[1],
                'fed_ytrain_data':nfm_h3.scaled_data[2],
                'fed_ytest_data':nfm_h3.scaled_data[3]
            }
        }

        # With loop
        fed_plain_trainer = FLP_Trainer_WOA(models=model_dict,data=data_dict,opti='SGD',verbose=0)
        fed_plain_trainer.train(num_epochs = ep,batch_size = bat)
        
        train_results_rocauc = [fed_plain_trainer.ROCAUC_epoch[mod]['train'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        test_results_rocauc = [fed_plain_trainer.ROCAUC_epoch[mod]['test'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        train_results_logloss = [fed_plain_trainer.LOGLOSS_epoch[mod]['train'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        test_results_logloss = [fed_plain_trainer.LOGLOSS_epoch[mod]['test'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        train_results_prauc = [fed_plain_trainer.PRAUC_epoch[mod]['train'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        test_results_prauc = [fed_plain_trainer.PRAUC_epoch[mod]['test'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        train_results_brier = [fed_plain_trainer.BRIER_epoch[mod]['train'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        test_results_brier = [fed_plain_trainer.BRIER_epoch[mod]['test'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        
        # Prepare test data
        data_test = pd.read_parquet(input_path + 'horizontal_iid_global.parquet') 
        data_ext = pd.read_parquet(input_path + 'ext_sources.parquet').drop(labels='target',axis=1)

        data_test = data_test.merge(data_ext,how='left')
        data_test.drop(labels=['ext_mean'],axis=1,inplace=True)
        del data_ext

        diff_list = [col for col in data_test.columns if col not in ['sk_id_curr','target']]
        scale_list = [col for col in diff_list if len(np.unique(data_test[col])) > 2]
        unscale_list = [col for col in data_test.columns if col not in scale_list]

        # Scale using the appropriate scaler
        roc_auc_oos_results = []
        log_loss_oos_results = []
        pr_auc_oos_results = []
        brier_oos_results = []
        
        for i,model in enumerate(model_dict.values()):
            if i == 0:
                scaled_data = nfm_h1.scaler.transform(data_test.loc[:,nfm_h1.scale_list])
                scaled_data = pd.DataFrame(scaled_data,columns=nfm_h1.scale_list)
                scaled_data = pd.concat([data_test.loc[:,nfm_h1.unscale_list],scaled_data],axis=1).sort_index(axis=1)
                scaled_data = torch.tensor(scaled_data.loc[:,nfm_h1.diff_list].values,dtype=torch.float32)
            elif i == 1:
                scaled_data = nfm_h2.scaler.transform(data_test.loc[:,nfm_h2.scale_list])
                scaled_data = pd.DataFrame(scaled_data,columns=nfm_h2.scale_list)
                scaled_data = pd.concat([data_test.loc[:,nfm_h2.unscale_list],scaled_data],axis=1).sort_index(axis=1)
                scaled_data = torch.tensor(scaled_data.loc[:,nfm_h2.diff_list].values,dtype=torch.float32)

            else:
                scaled_data = nfm_h3.scaler.transform(data_test.loc[:,nfm_h3.scale_list])
                scaled_data = pd.DataFrame(scaled_data,columns=nfm_h3.scale_list)
                scaled_data = pd.concat([data_test.loc[:,nfm_h3.unscale_list],scaled_data],axis=1).sort_index(axis=1)
                scaled_data = torch.tensor(scaled_data.loc[:,nfm_h3.diff_list].values,dtype=torch.float32)


            # Make predictions and evaluate 
            yhat = model.predict(scaled_data)
            
            roc_auc_score_oos = roc_auc_score(data_test['target'],yhat)
            log_loss_oos = log_loss(data_test['target'],yhat,eps=0.000001)
            precision_oos, recall_oos, _ = precision_recall_curve(data_test['target'],yhat) 
            prauc_oos = auc(recall_oos,precision_oos)
            brier_oos = brier_score_loss(data_test['target'],yhat)

            roc_auc_oos_results.append(roc_auc_score_oos)
            log_loss_oos_results.append(log_loss_oos)
            pr_auc_oos_results.append(prauc_oos)
            brier_oos_results.append(brier_oos)
            
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'ROC_AUC_Train'] = train_results_rocauc
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'ROC_AUC_Test'] = test_results_rocauc
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'ROC_AUC_OOS'] = roc_auc_oos_results
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'LOGLOSS_Train'] = train_results_logloss
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'LOGLOSS_Test'] = test_results_logloss
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'LOGLOSS_OOS'] = log_loss_oos_results
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'PRAUC_Train'] = train_results_prauc
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'PRAUC_Test'] = test_results_prauc
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'PRAUC_OOS'] = pr_auc_oos_results
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'BRIER_Train'] = train_results_brier
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'BRIER_Test'] = test_results_brier
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'BRIER_OOS'] = brier_oos_results
        
    print(80 * '#')
    print(f"Iteration {ep_idx+1}/{len(epoch_list)} done.")
    print(80 * '#')
end = time.time()
print(f"Total Execution Time 1.1.: {(end-start)/60:.2f} min.")

result_df.to_parquet(result_path + 'hfl_NN_results11_821.parquet')



# 1.2. IID WITH Aggregation
dsplit = ['8-2-1']
fl = [1] #[0,1]
noniid = [0] #[0,1]
y_dist = ['7.83% - 7.85% - 7.90%']
model_number = [1,2,3]
epoch_list = [10,5,3,1]
batches_list = [1024,512,256,128,64,32,16]
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

start = time.time()
for ep_idx,ep in enumerate(epoch_list):
    for bat in tqdm(batches_list):
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
                'fed_xtrain_data':nfm_h1.scaled_data[0],
                'fed_xtest_data':nfm_h1.scaled_data[1],
                'fed_ytrain_data':nfm_h1.scaled_data[2],
                'fed_ytest_data':nfm_h1.scaled_data[3],
            },
            'm2':{
                'fed_xtrain_data':nfm_h2.scaled_data[0],
                'fed_xtest_data':nfm_h2.scaled_data[1],
                'fed_ytrain_data':nfm_h2.scaled_data[2],
                'fed_ytest_data':nfm_h2.scaled_data[3],
            },
            'm3':{
                'fed_xtrain_data':nfm_h3.scaled_data[0],
                'fed_xtest_data':nfm_h3.scaled_data[1],
                'fed_ytrain_data':nfm_h3.scaled_data[2],
                'fed_ytest_data':nfm_h3.scaled_data[3]
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
        fed_crypt_trainer.train(num_epochs = ep,batch_size = bat)
        
        train_results_rocauc = [fed_crypt_trainer.ROCAUC_epoch_agg[mod]['train'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        test_results_rocauc = [fed_crypt_trainer.ROCAUC_epoch_agg[mod]['test'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        train_results_logloss = [fed_crypt_trainer.LOGLOSS_epoch_agg[mod]['train'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        test_results_logloss = [fed_crypt_trainer.LOGLOSS_epoch_agg[mod]['test'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        train_results_prauc = [fed_crypt_trainer.PRAUC_epoch_agg[mod]['train'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        test_results_prauc = [fed_crypt_trainer.PRAUC_epoch_agg[mod]['test'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        train_results_brier = [fed_crypt_trainer.BRIER_epoch_agg[mod]['train'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        test_results_brier = [fed_crypt_trainer.BRIER_epoch_agg[mod]['test'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        
        # Prepare test data
        data_test = pd.read_parquet(input_path + 'horizontal_iid_global.parquet') 
        data_ext = pd.read_parquet(input_path + 'ext_sources.parquet').drop(labels='target',axis=1)

        data_test = data_test.merge(data_ext,how='left')
        data_test.drop(labels=['ext_mean'],axis=1,inplace=True)
        del data_ext

        diff_list = [col for col in data_test.columns if col not in ['sk_id_curr','target']]
        scale_list = [col for col in diff_list if len(np.unique(data_test[col])) > 2]
        unscale_list = [col for col in data_test.columns if col not in scale_list]

        # Scale using the appropriate scaler
        roc_auc_oos_results = []
        log_loss_oos_results = []
        pr_auc_oos_results = []
        brier_oos_results = []
        
        for i,model in enumerate(model_dict.values()):
            if i == 0:
                scaled_data = nfm_h1.scaler.transform(data_test.loc[:,nfm_h1.scale_list])
                scaled_data = pd.DataFrame(scaled_data,columns=nfm_h1.scale_list)
                scaled_data = pd.concat([data_test.loc[:,nfm_h1.unscale_list],scaled_data],axis=1).sort_index(axis=1)
                scaled_data = torch.tensor(scaled_data.loc[:,nfm_h1.diff_list].values,dtype=torch.float32)
            elif i == 1:
                scaled_data = nfm_h2.scaler.transform(data_test.loc[:,nfm_h2.scale_list])
                scaled_data = pd.DataFrame(scaled_data,columns=nfm_h2.scale_list)
                scaled_data = pd.concat([data_test.loc[:,nfm_h2.unscale_list],scaled_data],axis=1).sort_index(axis=1)
                scaled_data = torch.tensor(scaled_data.loc[:,nfm_h2.diff_list].values,dtype=torch.float32)

            else:
                scaled_data = nfm_h3.scaler.transform(data_test.loc[:,nfm_h3.scale_list])
                scaled_data = pd.DataFrame(scaled_data,columns=nfm_h3.scale_list)
                scaled_data = pd.concat([data_test.loc[:,nfm_h3.unscale_list],scaled_data],axis=1).sort_index(axis=1)
                scaled_data = torch.tensor(scaled_data.loc[:,nfm_h3.diff_list].values,dtype=torch.float32)


            # Make predictions and evaluate 
            yhat = model.predict(scaled_data)
            
            roc_auc_score_oos = roc_auc_score(data_test['target'],yhat)
            log_loss_oos = log_loss(data_test['target'],yhat,eps=0.000001)
            precision_oos, recall_oos, _ = precision_recall_curve(data_test['target'],yhat) 
            prauc_oos = auc(recall_oos,precision_oos)
            brier_oos = brier_score_loss(data_test['target'],yhat)

            roc_auc_oos_results.append(roc_auc_score_oos)
            log_loss_oos_results.append(log_loss_oos)
            pr_auc_oos_results.append(prauc_oos)
            brier_oos_results.append(brier_oos)
            
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'ROC_AUC_Train'] = train_results_rocauc
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'ROC_AUC_Test'] = test_results_rocauc
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'ROC_AUC_OOS'] = roc_auc_oos_results
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'LOGLOSS_Train'] = train_results_logloss
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'LOGLOSS_Test'] = test_results_logloss
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'LOGLOSS_OOS'] = log_loss_oos_results
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'PRAUC_Train'] = train_results_prauc
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'PRAUC_Test'] = test_results_prauc
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'PRAUC_OOS'] = pr_auc_oos_results
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'BRIER_Train'] = train_results_brier
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'BRIER_Test'] = test_results_brier
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'BRIER_OOS'] = brier_oos_results
        
    print(80 * '#')
    print(f"Iteration {ep_idx+1}/{len(epoch_list)} done.")
    print(80 * '#')

end = time.time()
print(f"Total Execution Time 1.2.: {(end-start)/60:.2f} min.")

result_df.to_parquet(result_path + 'hfl_NN_results12_821.parquet')



#=======================================================
# 2. Non-IID
#=======================================================
print("Horizontal Split No. 1")
nfm_h1 = Feature_Generator(
    data_path =  input_path,
    file_name = 'horizontal_noniid_data1',
    n_jobs =  -1,
    train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_mean']
)


print("Horizontal Split No. 2")
nfm_h2 = Feature_Generator(
    data_path =  input_path,
    file_name = 'horizontal_noniid_data2',
    n_jobs =  -1,
    train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_mean']
)


print("Horizontal Split No. 3")
nfm_h3 = Feature_Generator(
    data_path =  input_path,
    file_name = 'horizontal_noniid_data3',
    n_jobs =  -1,
    train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_mean']
)


# Load parameters
param_list = []
for i in range(1,4):
    with open(param_path + f'hfl_best_params_nnh{i}.pickle','rb') as handle:
        h_param = pickle.load(handle)
        param_list.append(h_param)


# 2.1. Non-IID WithOUT Aggregation
dsplit = ['8-2-1']
fl = [0] #[0,1]
noniid = [1] #[0,1]
y_dist = ['1.9% - 15% - 30%']
model_number = [1,2,3]
epoch_list = [10,5,3,1]
batches_list = [1024,512,256,128,64,32,16]
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

start = time.time()
for ep_idx,ep in enumerate(epoch_list):
    for bat in tqdm(batches_list):
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
                'fed_xtrain_data':nfm_h1.scaled_data[0],
                'fed_xtest_data':nfm_h1.scaled_data[1],
                'fed_ytrain_data':nfm_h1.scaled_data[2],
                'fed_ytest_data':nfm_h1.scaled_data[3],
            },
            'm2':{
                'fed_xtrain_data':nfm_h2.scaled_data[0],
                'fed_xtest_data':nfm_h2.scaled_data[1],
                'fed_ytrain_data':nfm_h2.scaled_data[2],
                'fed_ytest_data':nfm_h2.scaled_data[3],
            },
            'm3':{
                'fed_xtrain_data':nfm_h3.scaled_data[0],
                'fed_xtest_data':nfm_h3.scaled_data[1],
                'fed_ytrain_data':nfm_h3.scaled_data[2],
                'fed_ytest_data':nfm_h3.scaled_data[3]
            }
        }

        # With loop
        fed_plain_trainer = FLP_Trainer_WOA(models=model_dict,data=data_dict,opti='SGD',verbose=0)
        fed_plain_trainer.train(num_epochs = ep,batch_size = bat)
        
        train_results_rocauc = [fed_plain_trainer.ROCAUC_epoch[mod]['train'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        test_results_rocauc = [fed_plain_trainer.ROCAUC_epoch[mod]['test'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        train_results_logloss = [fed_plain_trainer.LOGLOSS_epoch[mod]['train'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        test_results_logloss = [fed_plain_trainer.LOGLOSS_epoch[mod]['test'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        train_results_prauc = [fed_plain_trainer.PRAUC_epoch[mod]['train'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        test_results_prauc = [fed_plain_trainer.PRAUC_epoch[mod]['test'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        train_results_brier = [fed_plain_trainer.BRIER_epoch[mod]['train'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        test_results_brier = [fed_plain_trainer.BRIER_epoch[mod]['test'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        
        # Prepare test data
        data_test = pd.read_parquet(input_path + 'horizontal_noniid_global.parquet') 
        data_ext = pd.read_parquet(input_path + 'ext_sources.parquet').drop(labels='target',axis=1)

        data_test = data_test.merge(data_ext,how='left')
        data_test.drop(labels=['ext_mean'],axis=1,inplace=True)
        del data_ext

        diff_list = [col for col in data_test.columns if col not in ['sk_id_curr','target']]
        scale_list = [col for col in diff_list if len(np.unique(data_test[col])) > 2]
        unscale_list = [col for col in data_test.columns if col not in scale_list]

        # Scale using the appropriate scaler
        roc_auc_oos_results = []
        log_loss_oos_results = []
        pr_auc_oos_results = []
        brier_oos_results = []
        
        for i,model in enumerate(model_dict.values()):
            if i == 0:
                scaled_data = nfm_h1.scaler.transform(data_test.loc[:,nfm_h1.scale_list])
                scaled_data = pd.DataFrame(scaled_data,columns=nfm_h1.scale_list)
                scaled_data = pd.concat([data_test.loc[:,nfm_h1.unscale_list],scaled_data],axis=1).sort_index(axis=1)
                scaled_data = torch.tensor(scaled_data.loc[:,nfm_h1.diff_list].values,dtype=torch.float32)
            elif i == 1:
                scaled_data = nfm_h2.scaler.transform(data_test.loc[:,nfm_h2.scale_list])
                scaled_data = pd.DataFrame(scaled_data,columns=nfm_h2.scale_list)
                scaled_data = pd.concat([data_test.loc[:,nfm_h2.unscale_list],scaled_data],axis=1).sort_index(axis=1)
                scaled_data = torch.tensor(scaled_data.loc[:,nfm_h2.diff_list].values,dtype=torch.float32)

            else:
                scaled_data = nfm_h3.scaler.transform(data_test.loc[:,nfm_h3.scale_list])
                scaled_data = pd.DataFrame(scaled_data,columns=nfm_h3.scale_list)
                scaled_data = pd.concat([data_test.loc[:,nfm_h3.unscale_list],scaled_data],axis=1).sort_index(axis=1)
                scaled_data = torch.tensor(scaled_data.loc[:,nfm_h3.diff_list].values,dtype=torch.float32)


            # Make predictions and evaluate 
            yhat = model.predict(scaled_data)
            
            roc_auc_score_oos = roc_auc_score(data_test['target'],yhat)
            log_loss_oos = log_loss(data_test['target'],yhat,eps=0.000001)
            precision_oos, recall_oos, _ = precision_recall_curve(data_test['target'],yhat) 
            prauc_oos = auc(recall_oos,precision_oos)
            brier_oos = brier_score_loss(data_test['target'],yhat)

            roc_auc_oos_results.append(roc_auc_score_oos)
            log_loss_oos_results.append(log_loss_oos)
            pr_auc_oos_results.append(prauc_oos)
            brier_oos_results.append(brier_oos)
            
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'ROC_AUC_Train'] = train_results_rocauc
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'ROC_AUC_Test'] = test_results_rocauc
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'ROC_AUC_OOS'] = roc_auc_oos_results
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'LOGLOSS_Train'] = train_results_logloss
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'LOGLOSS_Test'] = test_results_logloss
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'LOGLOSS_OOS'] = log_loss_oos_results
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'PRAUC_Train'] = train_results_prauc
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'PRAUC_Test'] = test_results_prauc
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'PRAUC_OOS'] = pr_auc_oos_results
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'BRIER_Train'] = train_results_brier
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'BRIER_Test'] = test_results_brier
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'BRIER_OOS'] = brier_oos_results
        
    print(80 * '#')
    print(f"Iteration {ep_idx+1}/{len(epoch_list)} done.")
    print(80 * '#')

end = time.time()
print(f"Total Execution Time 2.1.: {(end-start)/60:.2f} min.") 

result_df.to_parquet(result_path + 'hfl_NN_results21_821.parquet')



# 2.2. Non-IID WITH Aggregation
dsplit = ['8-2-1']
fl = [1] #[0,1]
noniid = [1] #[0,1]
y_dist = ['1.9% - 15% - 30%']
model_number = [1,2,3]
epoch_list = [10,5,3,1]
batches_list = [1024,512,256,128,64,32,16]
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

start = time.time()
for ep_idx,ep in enumerate(epoch_list):
    for bat in tqdm(batches_list):
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
                'fed_xtrain_data':nfm_h1.scaled_data[0],
                'fed_xtest_data':nfm_h1.scaled_data[1],
                'fed_ytrain_data':nfm_h1.scaled_data[2],
                'fed_ytest_data':nfm_h1.scaled_data[3],
            },
            'm2':{
                'fed_xtrain_data':nfm_h2.scaled_data[0],
                'fed_xtest_data':nfm_h2.scaled_data[1],
                'fed_ytrain_data':nfm_h2.scaled_data[2],
                'fed_ytest_data':nfm_h2.scaled_data[3],
            },
            'm3':{
                'fed_xtrain_data':nfm_h3.scaled_data[0],
                'fed_xtest_data':nfm_h3.scaled_data[1],
                'fed_ytrain_data':nfm_h3.scaled_data[2],
                'fed_ytest_data':nfm_h3.scaled_data[3]
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
        fed_crypt_trainer.train(num_epochs = ep,batch_size = bat)
        
        train_results_rocauc = [fed_crypt_trainer.ROCAUC_epoch_agg[mod]['train'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        test_results_rocauc = [fed_crypt_trainer.ROCAUC_epoch_agg[mod]['test'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        train_results_logloss = [fed_crypt_trainer.LOGLOSS_epoch_agg[mod]['train'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        test_results_logloss = [fed_crypt_trainer.LOGLOSS_epoch_agg[mod]['test'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        train_results_prauc = [fed_crypt_trainer.PRAUC_epoch_agg[mod]['train'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        test_results_prauc = [fed_crypt_trainer.PRAUC_epoch_agg[mod]['test'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        train_results_brier = [fed_crypt_trainer.BRIER_epoch_agg[mod]['train'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        test_results_brier = [fed_crypt_trainer.BRIER_epoch_agg[mod]['test'][f"epoch{ep}"] for mod in ['m1','m2','m3']]
        
        # Prepare test data
        data_test = pd.read_parquet(input_path + 'horizontal_noniid_global.parquet') 
        data_ext = pd.read_parquet(input_path + 'ext_sources.parquet').drop(labels='target',axis=1)

        data_test = data_test.merge(data_ext,how='left')
        data_test.drop(labels=['ext_mean'],axis=1,inplace=True)
        del data_ext

        diff_list = [col for col in data_test.columns if col not in ['sk_id_curr','target']]
        scale_list = [col for col in diff_list if len(np.unique(data_test[col])) > 2]
        unscale_list = [col for col in data_test.columns if col not in scale_list]

        # Scale using the appropriate scaler
        roc_auc_oos_results = []
        log_loss_oos_results = []
        pr_auc_oos_results = []
        brier_oos_results = []
        
        for i,model in enumerate(model_dict.values()):
            if i == 0:
                scaled_data = nfm_h1.scaler.transform(data_test.loc[:,nfm_h1.scale_list])
                scaled_data = pd.DataFrame(scaled_data,columns=nfm_h1.scale_list)
                scaled_data = pd.concat([data_test.loc[:,nfm_h1.unscale_list],scaled_data],axis=1).sort_index(axis=1)
                scaled_data = torch.tensor(scaled_data.loc[:,nfm_h1.diff_list].values,dtype=torch.float32)
            elif i == 1:
                scaled_data = nfm_h2.scaler.transform(data_test.loc[:,nfm_h2.scale_list])
                scaled_data = pd.DataFrame(scaled_data,columns=nfm_h2.scale_list)
                scaled_data = pd.concat([data_test.loc[:,nfm_h2.unscale_list],scaled_data],axis=1).sort_index(axis=1)
                scaled_data = torch.tensor(scaled_data.loc[:,nfm_h2.diff_list].values,dtype=torch.float32)

            else:
                scaled_data = nfm_h3.scaler.transform(data_test.loc[:,nfm_h3.scale_list])
                scaled_data = pd.DataFrame(scaled_data,columns=nfm_h3.scale_list)
                scaled_data = pd.concat([data_test.loc[:,nfm_h3.unscale_list],scaled_data],axis=1).sort_index(axis=1)
                scaled_data = torch.tensor(scaled_data.loc[:,nfm_h3.diff_list].values,dtype=torch.float32)


            # Make predictions and evaluate 
            yhat = model.predict(scaled_data)
            
            roc_auc_score_oos = roc_auc_score(data_test['target'],yhat)
            log_loss_oos = log_loss(data_test['target'],yhat,eps=0.000001)
            precision_oos, recall_oos, _ = precision_recall_curve(data_test['target'],yhat) 
            prauc_oos = auc(recall_oos,precision_oos)
            brier_oos = brier_score_loss(data_test['target'],yhat)
            
            roc_auc_oos_results.append(roc_auc_score_oos)
            log_loss_oos_results.append(log_loss_oos)
            pr_auc_oos_results.append(prauc_oos)
            brier_oos_results.append(brier_oos)
            
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'ROC_AUC_Train'] = train_results_rocauc
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'ROC_AUC_Test'] = test_results_rocauc
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'ROC_AUC_OOS'] = roc_auc_oos_results
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'LOGLOSS_Train'] = train_results_logloss
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'LOGLOSS_Test'] = test_results_logloss
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'LOGLOSS_OOS'] = log_loss_oos_results
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'PRAUC_Train'] = train_results_prauc
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'PRAUC_Test'] = test_results_prauc
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'PRAUC_OOS'] = pr_auc_oos_results
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'BRIER_Train'] = train_results_brier
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'BRIER_Test'] = test_results_brier
        result_df.loc[(result_df['Epochs']==ep) & (result_df['Batches']==bat),'BRIER_OOS'] = brier_oos_results
        
    print(80 * '#')
    print(f"Iteration {ep_idx+1}/{len(epoch_list)} done.")
    print(80 * '#')
    
result_df.to_parquet(result_path + 'hfl_NN_results22_821.parquet')

end = time.time()
print(f"Total Execution Time 2.2.: {(end-start)/60:.2f} min.")



# Reload Results and fuse to one csv-file
result_df_list = []

for i in range(1,3):
    for j in range(1,3):
        result_df_list.append(pd.read_parquet(result_path + f'hfl_NN_results{i}{j}_821.parquet'))
        
result_df = pd.concat(result_df_list,axis=0).reset_index(drop=True)
result_df.to_csv(result_path + 'hfl_NN_results_total.csv',sep = ',',index=False)