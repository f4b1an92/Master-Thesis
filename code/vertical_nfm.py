import os
import sys
import yaml
import numpy as np
import pandas as pd
import re
import pickle as pkl
from functools import partial
from sklearn.metrics import auc,roc_auc_score,log_loss,f1_score,brier_score_loss,precision_recall_curve

from hyperopt import Trials, fmin, hp, tpe

os.path.realpath(__file__)
main_path = os.getcwd()

# Load config
with open(main_path + '\\config_general.yml','r') as f:
    config = yaml.safe_load(f)

sys.path.append(main_path + '\\code\\modules')


from feature_gen import *
from nonfed_models import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 300)
pd.options.display.float_format = '{:,.4f}'.format

#path = 'C:/Users/Fabian/Desktop/Studium/2.) HU Berlin/6.) SS_21/Master Thesis/Federated Learning for Credit Risk/Available Data Sets/1.) Home Credit Default Risk'
input_path = main_path + '\\cleaned_datasets\\'
model_path =  main_path + '\\models\\'
result_path = main_path + '\\results\\Non-FL Results\\'
param_path = main_path + '\\results\\Non-FL Results\\Best Params Vertical\\'

# Global Parameters
#-----------------------
random_state = config['random_state']
train_size = config['vtrain_size']
train_size_full = config['vtrain_size']
n_splits = config['n_splits']
max_evals = config['max_evals']


#=============================================================
#  1. Full Data Set
#=============================================================
print("\n\n\t\t 1. Full Data Set \n\n")
fg_vfull = Feature_Generator(
    data_path =  input_path,
    file_name = 'full_data_cleaned',
    n_jobs =  -1,
    train_size = train_size_full
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_mean']
)

nfm = NonFed_Models()

params = {
    'lr':hp.uniform('lr',0.0001,0.01),
    'momentum': hp.uniform('momentum',0.6,0.99),
    'num_epochs': hp.quniform('num_epochs',2,5,1),
    'batch_size': hp.quniform('batch_size',32,512,8)
}

objective_func = partial(nfm.optimize_func_LogReg,data=fg_vfull.scaled_data,random_state=random_state)

trials = Trials()

best_hyperparams = fmin(
    fn = objective_func,
    algo = tpe.suggest,
    max_evals = max_evals,
    space = params,
    trials = trials,
    rstate = np.random.RandomState(random_state)
)

# Extracting best hyperparameters
new_param_dict = {}

for key,val in best_hyperparams.items():
    if key in ['num_epochs','batch_size']:
        new_param_dict[key] = int(val)
    else:
        new_param_dict[key] = val
        
# Save best hyperparameters
with open(param_path + 'vfl_best_params_lg_full.pickle','wb') as handle:
    pkl.dump(new_param_dict,handle)

     
lg_model = nfm.LogReg(
    input_size = fg_vfull.scaled_data[0].shape[1],momentum = new_param_dict['momentum'], lr = new_param_dict['lr']
)

trainer = nfm.Trainer(
    model = lg_model,
    data = fg_vfull.scaled_data,
    opti = 'SGD'
)

trainer.train(
    num_epochs = new_param_dict['num_epochs'],
    batch_size = new_param_dict['batch_size'],
    save_model = True,
    save_path = model_path,
    model_name = 'log_reg_fulldata'
)

# Predictions
yhat_train = lg_model.predict(fg_vfull.scaled_data[0])
yhat_test = lg_model.predict(fg_vfull.scaled_data[1])

# Evaluations
metrics_logreg_full = {
    'LOGLOSS':trainer.LOGLOSS_epoch,
    'ROCAUC':trainer.ROCAUC_epoch,
    'PRAUC':trainer.PRAUC_epoch,
    'F1':trainer.F1_epoch,
    'BRIER':trainer.BRIER_epoch
}


print(f'Log-Loss on train data: {log_loss(fg_vfull.data[2],yhat_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_vfull.data[3],yhat_test,eps=0.000001):.4f}')
print(f"ROC-AUC on TRAIN data set: {roc_auc_score(fg_vfull.scaled_data[2],yhat_train):.4f}")
print(f"ROC-AUC on TEST data set: {roc_auc_score(fg_vfull.scaled_data[3],yhat_test):.4f}")
print(80*"#")
print("\t\t END OF LOGISTIC REGRESSION FITTING for FULL DATA")
print(80*"#")



# 1.2. Neural Networks
# Parameter Definition
params = {
    'lr':hp.uniform('lr',0.0001,0.01),
    'momentum': hp.uniform('momentum',0.6,0.99),
    'weight_decay':hp.uniform('weight_decay',0,0.2),
    'drop_rate':hp.uniform('drop_rate',0,0.5),
    'num_hidden_layers': hp.quniform('num_hidden_layers',3,4,1),
    'layer_size1':hp.quniform('layer_size1',128,512,8),
    'layer_size2':hp.quniform('layer_size2',64,256,8),
    'layer_size3':hp.quniform('layer_size3',25,150,25),
    'layer_size4':hp.quniform('layer_size4',8,64,8),
    'epochs':hp.quniform('epochs',2,5,1),
    'batch_size':hp.quniform('batch_size',64,512,8)
}

objective_func = partial(nfm.optimize_func_NN,data=fg_vfull.scaled_data,random_state=random_state,opti='SGD')

trials = Trials()

best_hyperparams = fmin(
    fn = objective_func,
    algo = tpe.suggest,
    space = params,
    max_evals = max_evals,
    trials = trials,
    rstate = np.random.RandomState(random_state)
)

# Extract best hyperparameters
new_param_dict = {}
for key,val in best_hyperparams.items():
    if key in ['lr','momentum','weight_decay','drop_rate']:
        new_param_dict[key] = val
    else:
        new_param_dict[key] = int(val)

# Save best hyperparameters
with open(param_path + 'vfl_best_params_nn_full.pickle','wb') as handle:
    pkl.dump(new_param_dict,handle)
    
    
# Fit final model
ls_list = [new_param_dict[f"layer_size{i+1}"] for i in range(new_param_dict['num_hidden_layers'])]

nn_model = nfm.NN(
    layer_sizes = [fg_vfull.scaled_data[0].shape[1],*ls_list,1],
    drop_rate = new_param_dict['drop_rate'],
    momentum = new_param_dict['momentum'],
    weight_decay = new_param_dict['weight_decay'],
    lr = new_param_dict['lr']
)

trainer_nn = nfm.Trainer(
    model = nn_model,
    data = fg_vfull.scaled_data,
    opti = 'SGD'
)
trainer_nn.train(
    num_epochs = new_param_dict['epochs'],
    batch_size = new_param_dict['batch_size'],
    save_model = True,
    save_path = model_path,
    model_name = 'neuralnet_fulldata'
)

# Predictions
yhat_train = nn_model.predict(fg_vfull.scaled_data[0])
yhat_test = nn_model.predict(fg_vfull.scaled_data[1])

# Evaluations
metrics_nn_full = {
    'LOGLOSS':trainer_nn.LOGLOSS_epoch,
    'ROCAUC':trainer_nn.ROCAUC_epoch,
    'PRAUC':trainer_nn.PRAUC_epoch,
    'F1':trainer_nn.F1_epoch,
    'BRIER':trainer_nn.BRIER_epoch
}


print(f'Log-Loss on train data: {log_loss(fg_vfull.data[2],yhat_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_vfull.data[3],yhat_test,eps=0.000001):.4f}')
print(f"ROC-AUC on TRAIN data set: {roc_auc_score(fg_vfull.scaled_data[2],yhat_train):.4f}")
print(f"ROC-AUC on TEST data set: {roc_auc_score(fg_vfull.scaled_data[3],yhat_test):.4f}")
print(80*"#")
print("\t\t END OF NEURAL NETWORK FITTING for FULL DATA")
print(80*"#")




# 1.3. Random Forest
# Parameter Definition
# Fitting final model
params = {
    'n_estimators':hp.quniform('n_estimators',200,1000,25),
    'max_depth':hp.quniform('max_depth',3,8,1),
    'max_samples':hp.uniform('max_samples',0.2,0.5),
    'min_samples_split':hp.quniform('min_samples_split', 10,200,10),
    'min_impurity_decrease':hp.uniform('min_impurity_decrease',0.000001,0.0001)
}

# Hyperparameter Optimization
fmin_objective = partial(
    nfm.optimize_func_RF,
    x=fg_vfull.data[0],
    y=fg_vfull.data[2],
    n_splits=n_splits,
    shuffle=True,
    random_state=random_state,
    n_jobs=-1,
    verbose = 0
)

trials = Trials()

best_hyperparams = fmin(
    fn = fmin_objective,
    space = params,
    algo = tpe.suggest,
    max_evals = max_evals,
    trials = trials,
    rstate = np.random.RandomState(random_state)
)

# Extracting best hyperparameters
new_param_dict = {}

for key,val in best_hyperparams.items():
    if key in ['n_estimators','max_depth','min_samples_split']:
        new_param_dict[key] = int(val)
    else:
        new_param_dict[key] = val
        
# Save best hyperparameters
with open(param_path + 'vfl_best_params_rf_full.pickle','wb') as handle:
    pkl.dump(new_param_dict,handle)


# Fitting final model
other_params = {
    'random_state':random_state,
    'criterion':'entropy',
    'max_features':'sqrt',
    'verbose':0,
    'n_jobs':-1
}

hyperopt_model = RandomForestClassifier(**{**new_param_dict,**other_params})

hyperopt_model.fit(fg_vfull.data[0],fg_vfull.data[2])

# Predictions
y_pred_train = hyperopt_model.predict_proba(fg_vfull.data[0])[:,1] 
y_pred_test = hyperopt_model.predict_proba(fg_vfull.data[1])[:,1]

# Evaluations
metrics_rf_full = {
    'LOGLOSS':{'train':{},'test':{}},
    'ROCAUC':{'train':{},'test':{}},
    'PRAUC':{'train':{},'test':{}},
    'F1':{'train':{},'test':{}},
    'BRIER':{'train':{},'test':{}}
}

log_loss_train = log_loss(fg_vfull.data[2],y_pred_train,eps=0.000001)
log_loss_test = log_loss(fg_vfull.data[3],y_pred_test,eps=0.000001)
metrics_rf_full['LOGLOSS']['train'] = log_loss_train
metrics_rf_full['LOGLOSS']['test'] = log_loss_test

roc_auc_train = roc_auc_score(fg_vfull.data[2],y_pred_train)
roc_auc_test = roc_auc_score(fg_vfull.data[3],y_pred_test)
metrics_rf_full['ROCAUC']['train'] = roc_auc_train
metrics_rf_full['ROCAUC']['test'] = roc_auc_test

precision_train, recall_train, _ = precision_recall_curve(fg_vfull.data[2],y_pred_train)
pr_auc_train = auc(recall_train,precision_train)
precision_test, recall_test, _ = precision_recall_curve(fg_vfull.data[3],y_pred_test)
pr_auc_test = auc(recall_test,precision_test)
metrics_rf_full['PRAUC']['train'] = pr_auc_train
metrics_rf_full['PRAUC']['test'] = pr_auc_test

f1_score_train = f1_score(fg_vfull.data[2],[1 if i >= 0.5 else 0 for i in y_pred_train])
f1_score_test = f1_score(fg_vfull.data[3],[1 if i >= 0.5 else 0 for i in y_pred_test])
metrics_rf_full['F1']['train'] = f1_score_train
metrics_rf_full['F1']['test'] = f1_score_test

brier_train = brier_score_loss(fg_vfull.data[2],y_pred_train)
brier_test = brier_score_loss(fg_vfull.data[3],y_pred_test)
metrics_rf_full['BRIER']['train'] = brier_train
metrics_rf_full['BRIER']['test'] = brier_test



print(f'Log-Loss on train data: {log_loss(fg_vfull.data[2],y_pred_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_vfull.data[3],y_pred_test,eps=0.000001):.4f}')
print(f'ROC-AUC on train data: {roc_auc_score(fg_vfull.data[2],y_pred_train):.4f}')
print(f'ROC-AUC on test data: {roc_auc_score(fg_vfull.data[3],y_pred_test):.4f}')
print(80*"#")
print("\t\t END OF RANDOM FOREST FITTING for FULL DATA")
print(80*"#")



#==============================================================
#  2. Previous and Current Applications
#==============================================================
print("\n\n\t\t 2. Previous & Current Applications \n\n")
fg_v2 = Feature_Generator(
    data_path =  input_path,
    file_name = ['application_train_cleaned','prev_app_cleaned'],
    n_jobs =  -1,
    random_state = random_state + 1,
    train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_mean','ext_source_2','ext_source_3']
)

nfm = NonFed_Models()

# 2.1. Logistic Regression
params = {
    'lr':hp.uniform('lr',0.0001,0.01),
    'momentum': hp.uniform('momentum',0.6,0.99),
    'num_epochs': hp.quniform('num_epochs',2,5,1),
    'batch_size': hp.quniform('batch_size',32,512,8)
}

objective_func = partial(nfm.optimize_func_LogReg,data=fg_v2.scaled_data,random_state=random_state)

trials = Trials()

best_hyperparams = fmin(
    fn = objective_func,
    algo = tpe.suggest,
    space = params,
    max_evals = max_evals,
    trials = trials,
    rstate = np.random.RandomState(random_state)
)

# Extract best hyperparams
new_param_dict =  {}
for key,val in best_hyperparams.items():
    if key in ['num_epochs','batch_size']:
        new_param_dict[key] = int(val)
    else:
        new_param_dict[key] = val
        
        
# Save best hyperparameters
with open(param_path + 'vfl_best_params_lg_curr.pickle','wb') as handle:
    pkl.dump(new_param_dict,handle)


# Train Final Model
lg_model = nfm.LogReg(
    input_size = fg_v2.scaled_data[0].shape[1],momentum = new_param_dict['momentum'], lr = new_param_dict['lr']
)

trainer = nfm.Trainer(
    model = lg_model,
    data = fg_v2.scaled_data,
    opti = 'SGD'
)

trainer.train(
    num_epochs = new_param_dict['num_epochs'],
    batch_size = new_param_dict['batch_size'],
    save_model = True,
    save_path = model_path,
    model_name = 'log_reg_prev_curr_appdata'
)

# Predictions
yhat_train = lg_model.predict(fg_v2.scaled_data[0])
yhat_test = lg_model.predict(fg_v2.scaled_data[1])

# Evaluations
metrics_logreg_prev = {
    'LOGLOSS':trainer.LOGLOSS_epoch,
    'ROCAUC':trainer.ROCAUC_epoch,
    'PRAUC':trainer.PRAUC_epoch,
    'F1':trainer.F1_epoch,
    'BRIER':trainer.BRIER_epoch
}

print(f'Log-Loss on train data: {log_loss(fg_v2.data[2],yhat_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_v2.data[3],yhat_test,eps=0.000001):.4f}')
print(f"ROC-AUC on TRAIN data set: {roc_auc_score(fg_v2.scaled_data[2],yhat_train):.4f}")
print(f"ROC-AUC on TEST data set: {roc_auc_score(fg_v2.scaled_data[3],yhat_test):.4f}")
print(80*"#")
print("\t\t END OF LOGISTIC REGRESSION FITTING for CURRENT & PREV. APPLICATIONS")
print(80*"#")

#with open(result_path + '/Non-FL Results/Best Params Vertical/lg_best_params_prev_curr.parquet','wb') as handle:
#    pkl.dump(best_hyperparams,handle)



# 2.2. Neural Networks
params = {
    'lr':hp.uniform('lr',0.0001,0.01),
    'momentum': hp.uniform('momentum',0.6,0.99),
    'weight_decay':hp.uniform('weight_decay',0,0.2),
    'drop_rate':hp.uniform('drop_rate',0,0.5),
    'num_hidden_layers': hp.quniform('num_hidden_layers',2,4,1),
    'layer_size1':hp.quniform('layer_size1',128,512,8),
    'layer_size2':hp.quniform('layer_size2',64,256,8),
    'layer_size3':hp.quniform('layer_size3',25,150,25),
    'layer_size4':hp.quniform('layer_size4',8,64,8),
    'epochs':hp.quniform('epochs',2,5,1),
    'batch_size':hp.quniform('batch_size',64,512,8)
}

objective_func = partial(nfm.optimize_func_NN,data=fg_v2.scaled_data,random_state=random_state,opti='SGD')

trials = Trials()

best_hyperparams = fmin(
    fn = objective_func,
    algo = tpe.suggest,
    space = params,
    max_evals = max_evals,
    trials = trials,
    rstate = np.random.RandomState(random_state)
)

# Extract best hyperparameters
new_param_dict = {}
for key,val in best_hyperparams.items():
    if key in ['lr','momentum','weight_decay','drop_rate']:
        new_param_dict[key] = val
    else:
        new_param_dict[key] = int(val)


# Save best hyperparameters
with open(param_path + 'vfl_best_params_nn_curr.pickle','wb') as handle:
    pkl.dump(new_param_dict,handle)


# Fit final model
ls_list = [new_param_dict[f"layer_size{i+1}"] for i in range(new_param_dict['num_hidden_layers'])]

nn_model = nfm.NN(
    layer_sizes = [fg_v2.scaled_data[0].shape[1],*ls_list,1],
    drop_rate = new_param_dict['drop_rate'],
    weight_decay = new_param_dict['weight_decay'],
    momentum = new_param_dict['momentum'],
    lr = new_param_dict['lr']
)

trainer_nn = nfm.Trainer(
    model = nn_model,
    data = fg_v2.scaled_data,
    opti = 'SGD'
)
trainer_nn.train(
    num_epochs = new_param_dict['epochs'],
    batch_size = new_param_dict['batch_size'],
    save_model = True,
    save_path = model_path,
    model_name = 'neuralnet_prev_curr_appdata'
)

# Predictions
yhat_train = nn_model.predict(fg_v2.scaled_data[0])
yhat_test = nn_model.predict(fg_v2.scaled_data[1])

# Evaluations
metrics_nn_prev = {
    'LOGLOSS':trainer_nn.LOGLOSS_epoch,
    'ROCAUC':trainer_nn.ROCAUC_epoch,
    'PRAUC':trainer_nn.PRAUC_epoch,
    'F1':trainer_nn.F1_epoch,
    'BRIER':trainer_nn.BRIER_epoch
}


print(f'Log-Loss on train data: {log_loss(fg_v2.data[2],yhat_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_v2.data[3],yhat_test,eps=0.000001):.4f}')
print(f"ROC-AUC on TRAIN data set: {roc_auc_score(fg_v2.scaled_data[2],yhat_train):.4f}")
print(f"ROC-AUC on TEST data set: {roc_auc_score(fg_v2.scaled_data[3],yhat_test):.4f}")
print(80*"#")
print("\t\t END OF NEURAL NETWORK FITTING for CURRENT & PREV. APPLICATIONS")
print(80*"#")



# 2.3. Random Forest
# Parameter Definition
# Fitting final model
params = {
    'n_estimators':hp.quniform('n_estimators',200,1000,25),
    'max_depth':hp.quniform('max_depth',3,8,1),
    'max_samples':hp.uniform('max_samples',0.2,0.5),
    'min_samples_split':hp.quniform('min_samples_split', 10,200,10),
    'min_impurity_decrease':hp.uniform('min_impurity_decrease',0.000001,0.0001)
}

# Hyperparameter Optimization
fmin_objective = partial(
    nfm.optimize_func_RF,
    x=fg_v2.data[0],
    y=fg_v2.data[2],
    n_splits=n_splits,
    shuffle=True,
    random_state=random_state,
    n_jobs=-1,
    verbose = 0
)

trials = Trials()

best_hyperparams = fmin(
    fn = fmin_objective,
    space = params,
    algo = tpe.suggest,
    max_evals = max_evals,
    trials = trials,
    rstate = np.random.RandomState(random_state)
)

# Extracting best hyperparameters
new_param_dict = {}

for key,val in best_hyperparams.items():
    if key in ['n_estimators','max_depth','min_samples_split']:
        new_param_dict[key] = int(val)
    else:
        new_param_dict[key] = val


# Save best hyperparameters
with open(param_path + 'vfl_best_params_rf_curr.pickle','wb') as handle:
    pkl.dump(new_param_dict,handle)


# Fitting final model
other_params = {
    'random_state':random_state,
    'criterion':'entropy',
    'max_features':'sqrt',
    'verbose':0,
    'n_jobs':-1
}

hyperopt_model = RandomForestClassifier(**{**new_param_dict,**other_params})

hyperopt_model.fit(fg_v2.data[0],fg_v2.data[2])

# Predictions
y_pred_train = hyperopt_model.predict_proba(fg_v2.data[0])[:,1] 
y_pred_test = hyperopt_model.predict_proba(fg_v2.data[1])[:,1]

# Evaluations
metrics_rf_prev = {
    'LOGLOSS':{'train':{},'test':{}},
    'ROCAUC':{'train':{},'test':{}},
    'PRAUC':{'train':{},'test':{}},
    'F1':{'train':{},'test':{}},
    'BRIER':{'train':{},'test':{}}
}

log_loss_train = log_loss(fg_v2.data[2],y_pred_train,eps=0.000001)
log_loss_test = log_loss(fg_v2.data[3],y_pred_test,eps=0.000001)
metrics_rf_prev['LOGLOSS']['train'] = log_loss_train
metrics_rf_prev['LOGLOSS']['test'] = log_loss_test

roc_auc_train = roc_auc_score(fg_v2.data[2],y_pred_train)
roc_auc_test = roc_auc_score(fg_v2.data[3],y_pred_test)
metrics_rf_prev['ROCAUC']['train'] = roc_auc_train
metrics_rf_prev['ROCAUC']['test'] = roc_auc_test

precision_train, recall_train, _ = precision_recall_curve(fg_v2.data[2],y_pred_train)
pr_auc_train = auc(recall_train,precision_train)
precision_test, recall_test, _ = precision_recall_curve(fg_v2.data[3],y_pred_test)
pr_auc_test = auc(recall_test,precision_test)
metrics_rf_prev['PRAUC']['train'] = pr_auc_train
metrics_rf_prev['PRAUC']['test'] = pr_auc_test

f1_score_train = f1_score(fg_v2.data[2],[1 if i >= 0.5 else 0 for i in y_pred_train])
f1_score_test = f1_score(fg_v2.data[3],[1 if i >= 0.5 else 0 for i in y_pred_test])
metrics_rf_prev['F1']['train'] = f1_score_train
metrics_rf_prev['F1']['test'] = f1_score_test

brier_train = brier_score_loss(fg_v2.data[2],y_pred_train)
brier_test = brier_score_loss(fg_v2.data[3],y_pred_test)
metrics_rf_prev['BRIER']['train'] = brier_train
metrics_rf_prev['BRIER']['test'] = brier_test


print(f'Log-Loss on train data: {log_loss(fg_v2.data[2],y_pred_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_v2.data[3],y_pred_test,eps=0.000001):.4f}')
print(f'ROC-AUC on train data: {roc_auc_score(fg_v2.data[2],y_pred_train):.4f}')
print(f'ROC-AUC on test data: {roc_auc_score(fg_v2.data[3],y_pred_test):.4f}')
print(80*"#")
print("\t\t END OF RANDOM FOREST FITTING for CURRENT & PREV. APPLICATIONS")
print(80*"#")



#==============================================================
#  3. Bureau Data
#==============================================================
print("\n\n\t\t 3. Bureau Data \n\n")
fg_v3 = Feature_Generator(
    data_path =  input_path,
    file_name = 'bureau_data_cleaned',
    n_jobs =  -1,
    random_state = random_state + 2,
    train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_source_1','ext_source_3'] # 'ext_source_2'
)

nfm = NonFed_Models()

# 3.1. Logistic Regression
params = {
    'lr':hp.uniform('lr',0.0001,0.01),
    'momentum': hp.uniform('momentum',0.6,0.99),
    'num_epochs': hp.quniform('num_epochs',2,5,1),
    'batch_size': hp.quniform('batch_size',32,512,8)
}

objective_func = partial(nfm.optimize_func_LogReg,data=fg_v3.scaled_data,random_state=random_state)

trials = Trials()

best_hyperparams = fmin(
    fn = objective_func,
    algo = tpe.suggest,
    space = params,
    max_evals = max_evals,
    trials = trials,
    rstate = np.random.RandomState(random_state)
)

# Extract best hyperparams
new_param_dict =  {}
for key,val in best_hyperparams.items():
    if key in ['num_epochs','batch_size']:
        new_param_dict[key] = int(val)
    else:
        new_param_dict[key] = val


# Save best hyperparameters
with open(param_path + 'vfl_best_params_lg_bur.pickle','wb') as handle:
    pkl.dump(new_param_dict,handle)


# Train Final Model
lg_model = nfm.LogReg(
    input_size = fg_v3.scaled_data[0].shape[1],momentum = new_param_dict['momentum'], lr = new_param_dict['lr']
)

trainer = nfm.Trainer(
    model = lg_model,
    data = fg_v3.scaled_data,
    opti = 'SGD'
)

trainer.train(
    num_epochs = new_param_dict['num_epochs'],
    batch_size = new_param_dict['batch_size'],
    save_model = True,
    save_path = model_path,
    model_name = 'log_reg_bureaudata'
)

# Predictions
yhat_train = lg_model.predict(fg_v3.scaled_data[0])
yhat_test = lg_model.predict(fg_v3.scaled_data[1])

# Evaluations
metrics_logreg_bur = {
    'LOGLOSS':trainer.LOGLOSS_epoch,
    'ROCAUC':trainer.ROCAUC_epoch,
    'PRAUC':trainer.PRAUC_epoch,
    'F1':trainer.F1_epoch,
    'BRIER':trainer.BRIER_epoch
}

print(f'Log-Loss on train data: {log_loss(fg_v3.data[2],yhat_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_v3.data[3],yhat_test,eps=0.000001):.4f}')
print(f"ROC-AUC on TRAIN data set: {roc_auc_score(fg_v3.scaled_data[2],yhat_train):.4f}")
print(f"ROC-AUC on TEST data set: {roc_auc_score(fg_v3.scaled_data[3],yhat_test):.4f}")
print(80*"#")
print("\t\t END OF LOGISTIC REGRESSION FITTING for BUREAU DATA")
print(80*"#")

#with open(result_path + '/Non-FL Results/Best Params Vertical/lg_best_params_bur.parquet','wb') as handle:
#    pkl.dump(best_hyperparams,handle)


# 3.2. Neural Networks
params = {
    'lr':hp.uniform('lr',0.0001,0.01),
    'momentum': hp.uniform('momentum',0.6,0.99),
    'weight_decay':hp.uniform('weight_decay',0,0.2),
    'drop_rate':hp.uniform('drop_rate',0,0.5),
    'num_hidden_layers': hp.quniform('num_hidden_layers',2,4,1),
    'layer_size1':hp.quniform('layer_size1',128,512,8),
    'layer_size2':hp.quniform('layer_size2',64,256,8),
    'layer_size3':hp.quniform('layer_size3',25,150,25),
    'layer_size4':hp.quniform('layer_size4',8,64,8),
    'epochs':hp.quniform('epochs',2,5,1),
    'batch_size':hp.quniform('batch_size',64,512,8)
}

objective_func = partial(nfm.optimize_func_NN,data=fg_v3.scaled_data,random_state=random_state,opti='SGD')

trials = Trials()

best_hyperparams = fmin(
    fn = objective_func,
    algo = tpe.suggest,
    space = params,
    max_evals = max_evals,
    trials = trials,
    rstate = np.random.RandomState(random_state)
)

# Extract best hyperparameters
new_param_dict = {}
for key,val in best_hyperparams.items():
    if key in ['lr','momentum','weight_decay','drop_rate']:
        new_param_dict[key] = val
    else:
        new_param_dict[key] = int(val)


# Save best hyperparameters
with open(param_path + 'vfl_best_params_nn_bur.pickle','wb') as handle:
    pkl.dump(new_param_dict,handle)



# Fit final model
ls_list = [new_param_dict[f"layer_size{i+1}"] for i in range(new_param_dict['num_hidden_layers'])]

nn_model = nfm.NN(
    layer_sizes = [fg_v3.scaled_data[0].shape[1],*ls_list,1],
    drop_rate = new_param_dict['drop_rate'],
    weight_decay = new_param_dict['weight_decay'],
    momentum = new_param_dict['momentum'],
    lr = new_param_dict['lr']
)

trainer_nn = nfm.Trainer(
    model = nn_model,
    data = fg_v3.scaled_data,
    opti = 'SGD'
)
trainer_nn.train(
    num_epochs = new_param_dict['epochs'],
    batch_size = new_param_dict['batch_size'],
    save_model = True,
    save_path = model_path,
    model_name = 'neuralnet_bureaudata'
)

# Predictions
yhat_train = nn_model.predict(fg_v3.scaled_data[0])
yhat_test = nn_model.predict(fg_v3.scaled_data[1])

# Evaluations
metrics_nn_bur = {
    'LOGLOSS':trainer_nn.LOGLOSS_epoch,
    'ROCAUC':trainer_nn.ROCAUC_epoch,
    'PRAUC':trainer_nn.PRAUC_epoch,
    'F1':trainer_nn.F1_epoch,
    'BRIER':trainer_nn.BRIER_epoch
}

print(f'Log-Loss on train data: {log_loss(fg_v3.data[2],yhat_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_v3.data[3],yhat_test,eps=0.000001):.4f}')
print(f"ROC-AUC on TRAIN data set: {roc_auc_score(fg_v3.scaled_data[2],yhat_train):.4f}")
print(f"ROC-AUC on TEST data set: {roc_auc_score(fg_v3.scaled_data[3],yhat_test):.4f}")
print(80*"#")
print("\t\t END OF NEURAL NETWORK FITTING for BUREAU DATA")
print(80*"#")



# 3.3. Random Forest
# Parameter Definition
# Fitting final model
params = {
    'n_estimators':hp.quniform('n_estimators',200,1000,25),
    'max_depth':hp.quniform('max_depth',3,8,1),
    'max_samples':hp.uniform('max_samples',0.2,0.5),
    'min_samples_split':hp.quniform('min_samples_split', 10,200,10),
    'min_impurity_decrease':hp.uniform('min_impurity_decrease',0.000001,0.0001)
}

# Hyperparameter Optimization
fmin_objective = partial(
    nfm.optimize_func_RF,
    x=fg_v3.data[0],
    y=fg_v3.data[2],
    n_splits=n_splits,
    shuffle=True,
    random_state=random_state,
    n_jobs=-1,
    verbose = 0
)

trials = Trials()

best_hyperparams = fmin(
    fn = fmin_objective,
    space = params,
    algo = tpe.suggest,
    max_evals = max_evals,
    trials = trials,
    rstate = np.random.RandomState(random_state)
)

# Extracting best hyperparameters
new_param_dict = {}

for key,val in best_hyperparams.items():
    if key in ['n_estimators','max_depth','min_samples_split']:
        new_param_dict[key] = int(val)
    else:
        new_param_dict[key] = val


# Save best hyperparameters
with open(param_path + 'vfl_best_params_rf_bur.pickle','wb') as handle:
    pkl.dump(new_param_dict,handle)


# Fitting final model
other_params = {
    'random_state':random_state,
    'criterion':'entropy',
    'max_features':'sqrt',
    'verbose':0,
    'n_jobs':-1
}

hyperopt_model = RandomForestClassifier(**{**new_param_dict,**other_params})

hyperopt_model.fit(fg_v3.data[0],fg_v3.data[2])

# Predictions
y_pred_train = hyperopt_model.predict_proba(fg_v3.data[0])[:,1] 
y_pred_test = hyperopt_model.predict_proba(fg_v3.data[1])[:,1]

# Evaluations
metrics_rf_bur = {
    'LOGLOSS':{'train':{},'test':{}},
    'ROCAUC':{'train':{},'test':{}},
    'PRAUC':{'train':{},'test':{}},
    'F1':{'train':{},'test':{}},
    'BRIER':{'train':{},'test':{}}
}

log_loss_train = log_loss(fg_v3.data[2],y_pred_train,eps=0.000001)
log_loss_test = log_loss(fg_v3.data[3],y_pred_test,eps=0.000001)
metrics_rf_bur['LOGLOSS']['train'] = log_loss_train
metrics_rf_bur['LOGLOSS']['test'] = log_loss_test

roc_auc_train = roc_auc_score(fg_v3.data[2],y_pred_train)
roc_auc_test = roc_auc_score(fg_v3.data[3],y_pred_test)
metrics_rf_bur['ROCAUC']['train'] = roc_auc_train
metrics_rf_bur['ROCAUC']['test'] = roc_auc_test

precision_train, recall_train, _ = precision_recall_curve(fg_v3.data[2],y_pred_train)
pr_auc_train = auc(recall_train,precision_train)
precision_test, recall_test, _ = precision_recall_curve(fg_v3.data[3],y_pred_test)
pr_auc_test = auc(recall_test,precision_test)
metrics_rf_bur['PRAUC']['train'] = pr_auc_train
metrics_rf_bur['PRAUC']['test'] = pr_auc_test

f1_score_train = f1_score(fg_v3.data[2],[1 if i >= 0.5 else 0 for i in y_pred_train])
f1_score_test = f1_score(fg_v3.data[3],[1 if i >= 0.5 else 0 for i in y_pred_test])
metrics_rf_bur['F1']['train'] = f1_score_train
metrics_rf_bur['F1']['test'] = f1_score_test

brier_train = brier_score_loss(fg_v3.data[2],y_pred_train)
brier_test = brier_score_loss(fg_v3.data[3],y_pred_test)
metrics_rf_bur['BRIER']['train'] = brier_train
metrics_rf_bur['BRIER']['test'] = brier_test


print(f'Log-Loss on train data: {log_loss(fg_v3.data[2],y_pred_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_v3.data[3],y_pred_test,eps=0.000001):.4f}')
print(f'ROC-AUC on train data: {roc_auc_score(fg_v3.data[2],y_pred_train):.4f}')
print(f'ROC-AUC on test data: {roc_auc_score(fg_v3.data[3],y_pred_test):.4f}')
print(80*"#")
print("\t\t END OF RANDOM FOREST FITTING for BUREAU DATA")
print(80*"#")




#==============================================================
#  4. Instalment Payments
#==============================================================
print("\n\n\t\t 4. Installment Payments Data \n\n")

fg_v4 = Feature_Generator(
    data_path =  input_path,
    file_name = 'installments_payments_cleaned',
    n_jobs =  -1,
    random_state = random_state + 3,
    train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_source_1','ext_source_2']
)

nfm = NonFed_Models()

# 4.1. Logistic Regression
params = {
    'lr':hp.uniform('lr',0.0001,0.01),
    'momentum': hp.uniform('momentum',0.6,0.99),
    'num_epochs': hp.quniform('num_epochs',2,5,1),
    'batch_size': hp.quniform('batch_size',32,512,8)
}

objective_func = partial(nfm.optimize_func_LogReg,data=fg_v4.scaled_data,random_state=random_state)

trials = Trials()

best_hyperparams = fmin(
    fn = objective_func,
    algo = tpe.suggest,
    space = params,
    max_evals = max_evals,
    trials = trials,
    rstate = np.random.RandomState(random_state)
)

# Extract best hyperparams
new_param_dict =  {}
for key,val in best_hyperparams.items():
    if key in ['num_epochs','batch_size']:
        new_param_dict[key] = int(val)
    else:
        new_param_dict[key] = val


# Save best hyperparameters
with open(param_path + 'vfl_best_params_lg_inst.pickle','wb') as handle:
    pkl.dump(new_param_dict,handle)



# Train Final Model
lg_model = nfm.LogReg(
    input_size = fg_v4.scaled_data[0].shape[1],momentum = new_param_dict['momentum'], lr = new_param_dict['lr']
)

trainer = nfm.Trainer(
    model = lg_model,
    data = fg_v4.scaled_data,
    opti = 'SGD'
)

trainer.train(
    num_epochs = new_param_dict['num_epochs'],
    batch_size = new_param_dict['batch_size'],
    save_model = True,
    save_path = model_path,
    model_name = 'log_reg_inst_paydata'
)

# Predictions
yhat_train = lg_model.predict(fg_v4.scaled_data[0])
yhat_test = lg_model.predict(fg_v4.scaled_data[1])

# Evaluations
metrics_logreg_inst = {
    'LOGLOSS':trainer.LOGLOSS_epoch,
    'ROCAUC':trainer.ROCAUC_epoch,
    'PRAUC':trainer.PRAUC_epoch,
    'F1':trainer.F1_epoch,
    'BRIER':trainer.BRIER_epoch
}


print(f'Log-Loss on train data: {log_loss(fg_v4.data[2],yhat_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_v4.data[3],yhat_test,eps=0.000001):.4f}')
print(f"ROC-AUC on TRAIN data set: {roc_auc_score(fg_v4.scaled_data[2],yhat_train):.4f}")
print(f"ROC-AUC on TEST data set: {roc_auc_score(fg_v4.scaled_data[3],yhat_test):.4f}")
print(80*"#")
print("\t\t END OF LOGISTIC REGRESSION FITTING for INSTALLMENT PAYMENTS")
print(80*"#")

#with open(result_path + '/Non-FL Results/Best Params Vertical/lg_best_params_inst.parquet','wb') as handle:
#    pkl.dump(best_hyperparams,handle)


# 4.2. Neural Networks
params = {
    'lr':hp.uniform('lr',0.0001,0.01),
    'momentum': hp.uniform('momentum',0.6,0.99),
    'weight_decay':hp.uniform('weight_decay',0,0.2),
    'drop_rate':hp.uniform('drop_rate',0,0.5),
    'num_hidden_layers': hp.quniform('num_hidden_layers',2,4,1),
    'layer_size1':hp.quniform('layer_size1',128,512,8),
    'layer_size2':hp.quniform('layer_size2',64,256,8),
    'layer_size3':hp.quniform('layer_size3',25,150,25),
    'layer_size4':hp.quniform('layer_size4',8,64,8),
    'epochs':hp.quniform('epochs',2,5,1),
    'batch_size':hp.quniform('batch_size',64,512,8)
}

objective_func = partial(nfm.optimize_func_NN,data=fg_v4.scaled_data,random_state=random_state,opti='SGD')

trials = Trials()

best_hyperparams = fmin(
    fn = objective_func,
    algo = tpe.suggest,
    space = params,
    max_evals = max_evals,
    trials = trials,
    rstate = np.random.RandomState(random_state)
)

# Extract best hyperparameters
new_param_dict = {}
for key,val in best_hyperparams.items():
    if key in ['lr','momentum','weight_decay','drop_rate']:
        new_param_dict[key] = val
    else:
        new_param_dict[key] = int(val)


# Save best hyperparameters
with open(param_path + 'vfl_best_params_nn_inst.pickle','wb') as handle:
    pkl.dump(new_param_dict,handle)



# Fit final model
ls_list = [new_param_dict[f"layer_size{i+1}"] for i in range(new_param_dict['num_hidden_layers'])]

nn_model = nfm.NN(
    layer_sizes = [fg_v4.scaled_data[0].shape[1],*ls_list,1],
    drop_rate = new_param_dict['drop_rate'],
    weight_decay = new_param_dict['weight_decay'],
    momentum = new_param_dict['momentum'],
    lr = new_param_dict['lr']
)

trainer_nn = nfm.Trainer(
    model = nn_model,
    data = fg_v4.scaled_data,
    opti = 'SGD'
)
trainer_nn.train(
    num_epochs = new_param_dict['epochs'],
    batch_size = new_param_dict['batch_size'],
    save_model = True,
    save_path = model_path,
    model_name = 'neuralnet_inst_paydata'
)

# Predictions
yhat_train = nn_model.predict(fg_v4.scaled_data[0])
yhat_test = nn_model.predict(fg_v4.scaled_data[1])

# Evaluations
metrics_nn_inst = {
    'LOGLOSS':trainer_nn.LOGLOSS_epoch,
    'ROCAUC':trainer_nn.ROCAUC_epoch,
    'PRAUC':trainer_nn.PRAUC_epoch,
    'F1':trainer_nn.F1_epoch,
    'BRIER':trainer_nn.BRIER_epoch
}

print(f'Log-Loss on train data: {log_loss(fg_v4.data[2],yhat_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_v4.data[3],yhat_test,eps=0.000001):.4f}')
print(f"ROC-AUC on TRAIN data set: {roc_auc_score(fg_v4.scaled_data[2],yhat_train):.4f}")
print(f"ROC-AUC on TEST data set: {roc_auc_score(fg_v4.scaled_data[3],yhat_test):.4f}")
print(80*"#")
print("\t\t END OF NEURAL NETWORK FITTING for INSTALLMENT PAYMENTS")
print(80*"#")



# 4.3. Random Forest
# Parameter Definition
# Fitting final model
params = {
    'n_estimators':hp.quniform('n_estimators',200,1000,25),
    'max_depth':hp.quniform('max_depth',3,8,1),
    'max_samples':hp.uniform('max_samples',0.2,0.5),
    'min_samples_split':hp.quniform('min_samples_split', 10,200,10),
    'min_impurity_decrease':hp.uniform('min_impurity_decrease',0.000001,0.0001)
}

# Hyperparameter Optimization
fmin_objective = partial(
    nfm.optimize_func_RF,
    x=fg_v4.data[0],
    y=fg_v4.data[2],
    n_splits=n_splits,
    shuffle=True,
    random_state=random_state,
    n_jobs=-1,
    verbose = 0
)

trials = Trials()

best_hyperparams = fmin(
    fn = fmin_objective,
    space = params,
    algo = tpe.suggest,
    max_evals = max_evals,
    trials = trials,
    rstate = np.random.RandomState(random_state)
)

# Extracting best hyperparameters
new_param_dict = {}

for key,val in best_hyperparams.items():
    if key in ['n_estimators','max_depth','min_samples_split']:
        new_param_dict[key] = int(val)
    else:
        new_param_dict[key] = val


# Save best hyperparameters
with open(param_path + 'vfl_best_params_rf_inst.pickle','wb') as handle:
    pkl.dump(new_param_dict,handle)



# Fitting final model
other_params = {
    'random_state':random_state,
    'criterion':'entropy',
    'max_features':'sqrt',
    'verbose':0,
    'n_jobs':-1
}

hyperopt_model = RandomForestClassifier(**{**new_param_dict,**other_params})

hyperopt_model.fit(fg_v4.data[0],fg_v4.data[2])

# Predictions
y_pred_train = hyperopt_model.predict_proba(fg_v4.data[0])[:,1] 
y_pred_test = hyperopt_model.predict_proba(fg_v4.data[1])[:,1]

# Evaluations
metrics_rf_inst = {
    'LOGLOSS':{'train':{},'test':{}},
    'ROCAUC':{'train':{},'test':{}},
    'PRAUC':{'train':{},'test':{}},
    'F1':{'train':{},'test':{}},
    'BRIER':{'train':{},'test':{}}
}

log_loss_train = log_loss(fg_v4.data[2],y_pred_train,eps=0.000001)
log_loss_test = log_loss(fg_v4.data[3],y_pred_test,eps=0.000001)
metrics_rf_inst['LOGLOSS']['train'] = log_loss_train
metrics_rf_inst['LOGLOSS']['test'] = log_loss_test

roc_auc_train = roc_auc_score(fg_v4.data[2],y_pred_train)
roc_auc_test = roc_auc_score(fg_v4.data[3],y_pred_test)
metrics_rf_inst['ROCAUC']['train'] = roc_auc_train
metrics_rf_inst['ROCAUC']['test'] = roc_auc_test

precision_train, recall_train, _ = precision_recall_curve(fg_v4.data[2],y_pred_train)
pr_auc_train = auc(recall_train,precision_train)
precision_test, recall_test, _ = precision_recall_curve(fg_v4.data[3],y_pred_test)
pr_auc_test = auc(recall_test,precision_test)
metrics_rf_inst['PRAUC']['train'] = pr_auc_train
metrics_rf_inst['PRAUC']['test'] = pr_auc_test

f1_score_train = f1_score(fg_v4.data[2],[1 if i >= 0.5 else 0 for i in y_pred_train])
f1_score_test = f1_score(fg_v4.data[3],[1 if i >= 0.5 else 0 for i in y_pred_test])
metrics_rf_inst['F1']['train'] = f1_score_train
metrics_rf_inst['F1']['test'] = f1_score_test

brier_train = brier_score_loss(fg_v4.data[2],y_pred_train)
brier_test = brier_score_loss(fg_v4.data[3],y_pred_test)
metrics_rf_inst['BRIER']['train'] = brier_train
metrics_rf_inst['BRIER']['test'] = brier_test


print(f'Log-Loss on train data: {log_loss(fg_v4.data[2],y_pred_train):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_v4.data[3],y_pred_test):.4f}')
print(f'ROC-AUC on train data: {roc_auc_score(fg_v4.data[2],y_pred_train):.4f}')
print(f'ROC-AUC on test data: {roc_auc_score(fg_v4.data[3],y_pred_test):.4f}')
print(80*"#")
print("\t\t END OF RANDOM FOREST FITTING for INSTALLMENT PAYMENTS")
print(80*"#")



#===================================
# Save results in structure way
#===================================
vertical_nfm_results = {
    'full_data_lg':metrics_logreg_full,
    'full_data_nn':metrics_nn_full,
    'full_data_rf':metrics_rf_full,
    
    'app_data_lg':metrics_logreg_prev,
    'app_data_nn':metrics_nn_prev,
    'app_data_rf':metrics_rf_prev,
    
    'bur_data_lg':metrics_logreg_bur,
    'bur_data_nn':metrics_nn_bur,
    'bur_data_rf':metrics_rf_bur,
    
    'inst_data_lg':metrics_logreg_inst,
    'inst_data_nn':metrics_nn_inst,
    'inst_data_rf':metrics_rf_inst
} 
   
# Save Resuts
with open(result_path + '/vertical_nfm_result_dict.pickle', 'wb') as handle:
    pkl.dump(vertical_nfm_results, handle)


"""
# Save results in structured dataframes
model_list = [key for key in vertical_nfm_results.keys()]
metric_list = [key for key in vertical_nfm_results['full_data_lg'].keys()]

# Train Set Results
result_dict = {}
for model in model_list:
    model_result_list = []
        
    if bool(re.search(r'xgb',model))==True:
        for met in metric_list:
            elem = vertical_nfm_results[model][met]['train']
            model_result_list.append(elem)
    else:
        for met in metric_list:
            elem = [val for val in vertical_nfm_results[model][met]['train'].values()][-1]
            model_result_list.append(elem)
    result_dict[model] = model_result_list

result_df_train = pd.DataFrame(result_dict)
result_df_train = result_df_train.set_index(pd.Index(metric_list))
result_df_train.to_csv(result_path + '/vert_nfm_results_train.csv')


# Test Set Results
result_dict = {}
for model in model_list:
    model_result_list = []
    if bool(re.search(r'xgb',model))==True:
        for met in metric_list:
            elem = vertical_nfm_results[model][met]['train']
            model_result_list.append(elem)
    else:
        for met in metric_list:
            elem = [val for val in vertical_nfm_results[model][met]['test'].values()][-1]
            model_result_list.append(elem)
    result_dict[model] = model_result_list

result_df_test = pd.DataFrame(result_dict)
result_df_test = result_df_test.set_index(pd.Index(metric_list))
result_df_test.to_csv(result_path + '/vert_nfm_results_test.csv')
"""