import os
import sys
import yaml
import numpy as np
import pandas as pd
import pickle as pkl
import re
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


input_path = main_path + '\\cleaned_datasets\\'
model_path =  main_path + '\\models\\'
result_path =  main_path + '\\results\\Non-FL Results\\'
param_path = main_path + '\\results\\Non-FL Results\\Best Params Horizontal\\'

# Global Parameters
#-----------------------
random_state = config['random_state']
train_size = config['htrain_size']
train_size_full = config['htrain_size']
n_splits = config['n_splits']
max_evals = config['max_evals']

#=============================================================
#  1. Data Set 1: H1 Data (Biggest Participant)
#=============================================================
print("\n\n\t\t 1. Horizontal Split: Biggest Particpant \n\n")
fg_h1 = Feature_Generator(
    data_path =  input_path,
    file_name = 'horizontal_iid_data1', # has to be changed
    n_jobs =  -1,
    train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_mean']
)

nfm_h1 = NonFed_Models()

params = {
    'lr':hp.uniform('lr',0.0001,0.01),
    'momentum': hp.uniform('momentum',0.6,0.99),
    'num_epochs': hp.quniform('num_epochs',2,5,1),
    'batch_size': hp.quniform('batch_size',32,512,8)
}

objective_func = partial(nfm_h1.optimize_func_LogReg,data=fg_h1.scaled_data)

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
with open(param_path + 'hfl_best_params_lgh1.pickle','wb') as handle:
    pkl.dump(new_param_dict,handle)

        
lg_model = nfm_h1.LogReg(
    input_size = fg_h1.scaled_data[0].shape[1],momentum = new_param_dict['momentum'], lr = new_param_dict['lr']
)

trainer = nfm_h1.Trainer(
    model = lg_model,
    data = fg_h1.scaled_data,
    opti = 'SGD'
)

trainer.train(
    num_epochs = new_param_dict['num_epochs'],
    batch_size = new_param_dict['batch_size'],
    save_model = True,
    save_path = model_path,
    model_name = 'log_reg_citydata_horizontal'
)


# Predictions
yhat_train = lg_model.predict(fg_h1.scaled_data[0])
yhat_test = lg_model.predict(fg_h1.scaled_data[1])

# Evaluation
metrics_logreg_h1 = {
    'LOGLOSS':trainer.LOGLOSS_epoch,
    'ROCAUC':trainer.ROCAUC_epoch,
    'PRAUC':trainer.PRAUC_epoch,
    'F1':trainer.F1_epoch,
    'BRIER':trainer.BRIER_epoch
}

print(f'Log-Loss on train data: {log_loss(fg_h1.data[2],yhat_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_h1.data[3],yhat_test,eps=0.000001):.4f}')
print(f"ROC-AUC on TRAIN data set: {roc_auc_score(fg_h1.scaled_data[2],yhat_train):.4f}")
print(f"ROC-AUC on TEST data set: {roc_auc_score(fg_h1.scaled_data[3],yhat_test):.4f}")
print(80*"#")
print("\t\t END OF LOGISTIC REGRESSION FITTING for H1 DATA")
print(80*"#")


# 1.2. Neural Networks
# Parameter Definition
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

objective_func = partial(nfm_h1.optimize_func_NN,data=fg_h1.scaled_data,random_state=random_state,opti='SGD')

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
with open(param_path + 'hfl_best_params_nnh1.pickle','wb') as handle:
    pkl.dump(new_param_dict,handle)


# Fit final model
ls_list = [new_param_dict[f"layer_size{i+1}"] for i in range(new_param_dict['num_hidden_layers'])]

nn_model = nfm_h1.NN(
    layer_sizes = [fg_h1.scaled_data[0].shape[1],*ls_list,1],
    drop_rate = new_param_dict['drop_rate'],
    weight_decay = new_param_dict['weight_decay'],
    momentum = new_param_dict['momentum'],
    lr = new_param_dict['lr']
)

trainer_nn = nfm_h1.Trainer(
    model = nn_model,
    data = fg_h1.scaled_data,
    opti = 'SGD'
)
trainer_nn.train(
    num_epochs = new_param_dict['epochs'],
    batch_size = new_param_dict['batch_size'],
    save_model = True,
    save_path = model_path,
    model_name = 'neuralnet_citydata_horizontal'
)

# Predictions
yhat_train = nn_model.predict(fg_h1.scaled_data[0])
yhat_test = nn_model.predict(fg_h1.scaled_data[1])

# Evaluations
metrics_nn_h1 = {
    'LOGLOSS':trainer_nn.LOGLOSS_epoch,
    'ROCAUC':trainer_nn.ROCAUC_epoch,
    'PRAUC':trainer_nn.PRAUC_epoch,
    'F1':trainer_nn.F1_epoch,
    'BRIER':trainer_nn.BRIER_epoch
}

print(f'Log-Loss on train data: {log_loss(fg_h1.data[2],yhat_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_h1.data[3],yhat_test,eps=0.000001):.4f}')
print(f"ROC-AUC on TRAIN data set: {roc_auc_score(fg_h1.scaled_data[2],yhat_train):.4f}")
print(f"ROC-AUC on TEST data set: {roc_auc_score(fg_h1.scaled_data[3],yhat_test):.4f}")
print(80*"#")
print("\t\t END OF NEURAL NETWORK FITTING for H1")
print(80*"#")



# 1.3. Random Forest
params = {
    'n_estimators':hp.quniform('n_estimators',200,1000,25),
    'max_depth':hp.quniform('max_depth',3,8,1),
    'max_samples':hp.uniform('max_samples',0.2,0.5),
    'min_samples_split':hp.quniform('min_samples_split', 10,200,10),
    'min_impurity_decrease':hp.uniform('min_impurity_decrease',0.000001,0.0001)
}

# Hyperparameter Optimization
fmin_objective = partial(
    nfm_h1.optimize_func_RF,
    x=fg_h1.data[0],
    y=fg_h1.data[2],
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
with open(param_path + 'hfl_best_params_rfh1.pickle','wb') as handle:
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

hyperopt_model.fit(fg_h1.data[0],fg_h1.data[2])

# Predictions
y_pred_train = hyperopt_model.predict_proba(fg_h1.data[0])[:,1] 
y_pred_test = hyperopt_model.predict_proba(fg_h1.data[1])[:,1]

# Evaluations
metrics_rf_h1 = {
    'LOGLOSS':{'train':{},'test':{}},
    'ROCAUC':{'train':{},'test':{}},
    'PRAUC':{'train':{},'test':{}},
    'BRIER':{'train':{},'test':{}}
}

log_loss_train = log_loss(fg_h1.data[2],y_pred_train,eps=0.000001)
log_loss_test = log_loss(fg_h1.data[3],y_pred_test,eps=0.000001)
metrics_rf_h1['LOGLOSS']['train'] = log_loss_train
metrics_rf_h1['LOGLOSS']['test'] = log_loss_test

roc_auc_train = roc_auc_score(fg_h1.data[2],y_pred_train)
roc_auc_test = roc_auc_score(fg_h1.data[3],y_pred_test)
metrics_rf_h1['ROCAUC']['train'] = roc_auc_train
metrics_rf_h1['ROCAUC']['test'] = roc_auc_test

precision_train, recall_train, _ = precision_recall_curve(fg_h1.data[2],y_pred_train)
pr_auc_train = auc(recall_train,precision_train)
precision_test, recall_test, _ = precision_recall_curve(fg_h1.data[3],y_pred_test)
pr_auc_test = auc(recall_test,precision_test)
metrics_rf_h1['PRAUC']['train'] = pr_auc_train
metrics_rf_h1['PRAUC']['test'] = pr_auc_test

brier_train = brier_score_loss(fg_h1.data[2],y_pred_train)
brier_test = brier_score_loss(fg_h1.data[3],y_pred_test)
metrics_rf_h1['BRIER']['train'] = brier_train
metrics_rf_h1['BRIER']['test'] = brier_test


# Save Random Forest Model
with open(model_path + 'rf_model_h1.pickle', 'wb') as handle:
    pkl.dump(hyperopt_model, handle)


print(f'Log-Loss on train data: {log_loss(fg_h1.data[2],y_pred_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_h1.data[3],y_pred_test,eps=0.000001):.4f}')
print(f'ROC-AUC on train data: {roc_auc_score(fg_h1.data[2],y_pred_train):.4f}')
print(f'ROC-AUC on test data: {roc_auc_score(fg_h1.data[3],y_pred_test):.4f}')
print(80*"#")
print("\t\t END OF RANDOM FOREST FITTING for H1 Data")
print(80*"#")






#=============================================================
#  2. Data Set 2: H2 Data
#=============================================================
print("\n\n\t\t 2. Horizontal Split: H2 Data \n\n")
fg_h2 = Feature_Generator(
    data_path =  input_path,
    file_name = 'horizontal_iid_data2',
    n_jobs =  -1,
    train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_mean']
)

nfm_h2 = NonFed_Models()

params = {
    'lr':hp.uniform('lr',0.0001,0.01),
    'momentum': hp.uniform('momentum',0.6,0.99),
    'num_epochs': hp.quniform('num_epochs',2,5,1),
    'batch_size': hp.quniform('batch_size',32,512,8)
}

objective_func = partial(nfm_h2.optimize_func_LogReg,data=fg_h2.scaled_data,random_state=random_state)

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
with open(param_path + 'hfl_best_params_lgh2.pickle','wb') as handle:
    pkl.dump(new_param_dict,handle)

        
lg_model = nfm_h2.LogReg(
    input_size = fg_h2.scaled_data[0].shape[1],momentum = new_param_dict['momentum'], lr = new_param_dict['lr']
)

trainer = nfm_h2.Trainer(
    model = lg_model,
    data = fg_h2.scaled_data,
    opti = 'SGD'
)

trainer.train(
    num_epochs = new_param_dict['num_epochs'],
    batch_size = new_param_dict['batch_size'],
    save_model = True,
    save_path = model_path,
    model_name = 'log_reg_suburbsdata_horizontal'
)

# Predictions
yhat_train = lg_model.predict(fg_h2.scaled_data[0])
yhat_test = lg_model.predict(fg_h2.scaled_data[1])

# Evaluations
metrics_logreg_h2 = {
    'LOGLOSS':trainer.LOGLOSS_epoch,
    'ROCAUC':trainer.ROCAUC_epoch,
    'PRAUC':trainer.PRAUC_epoch,
    'F1':trainer.F1_epoch,
    'BRIER':trainer.BRIER_epoch
}

print(f'Log-Loss on train data: {log_loss(fg_h2.data[2],yhat_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_h2.data[3],yhat_test,eps=0.000001):.4f}')
print(f"ROC-AUC on TRAIN data set: {roc_auc_score(fg_h2.scaled_data[2],yhat_train):.4f}")
print(f"ROC-AUC on TEST data set: {roc_auc_score(fg_h2.scaled_data[3],yhat_test):.4f}")
print(80*"#")
print("\t\t END OF LOGISTIC REGRESSION FITTING for H2 Data")
print(80*"#")


# 2.2. Neural Networks
# Parameter Definition
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

objective_func = partial(nfm_h2.optimize_func_NN,data=fg_h2.scaled_data,random_state=random_state,opti='SGD')

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
with open(param_path + 'hfl_best_params_nnh2.pickle','wb') as handle:
    pkl.dump(new_param_dict,handle)


# Fit final model
ls_list = [new_param_dict[f"layer_size{i+1}"] for i in range(new_param_dict['num_hidden_layers'])]

nn_model = nfm_h2.NN(
    layer_sizes = [fg_h2.scaled_data[0].shape[1],*ls_list,1],
    drop_rate = new_param_dict['drop_rate'],
    weight_decay = new_param_dict['weight_decay'],
    momentum = new_param_dict['momentum'],
    lr = new_param_dict['lr']
)

trainer_nn = nfm_h2.Trainer(
    model = nn_model,
    data = fg_h2.scaled_data,
    opti = 'SGD'
)
trainer_nn.train(
    num_epochs = new_param_dict['epochs'],
    batch_size = new_param_dict['batch_size'],
    save_model = True,
    save_path = model_path,
    model_name = 'neuralnet_suburbsdata_horizontal'
)

# Predictions
yhat_train = nn_model.predict(fg_h2.scaled_data[0])
yhat_test = nn_model.predict(fg_h2.scaled_data[1])

# Evaluations
metrics_nn_h2 = {
    'LOGLOSS':trainer_nn.LOGLOSS_epoch,
    'ROCAUC':trainer_nn.ROCAUC_epoch,
    'PRAUC':trainer_nn.PRAUC_epoch,
    'F1':trainer_nn.F1_epoch,
    'BRIER':trainer_nn.BRIER_epoch
}


print(f'Log-Loss on train data: {log_loss(fg_h2.data[2],yhat_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_h2.data[3],yhat_test,eps=0.000001):.4f}')
print(f"ROC-AUC on TRAIN data set: {roc_auc_score(fg_h2.scaled_data[2],yhat_train):.4f}")
print(f"ROC-AUC on TEST data set: {roc_auc_score(fg_h2.scaled_data[3],yhat_test):.4f}")
print(80*"#")
print("\t\t END OF NEURAL NETWORK FITTING for H2")
print(80*"#")



# 2.3. Random Forest
params = {
    'n_estimators':hp.quniform('n_estimators',200,1000,25),
    'max_depth':hp.quniform('max_depth',3,8,1),
    'max_samples':hp.uniform('max_samples',0.2,0.5),
    'min_samples_split':hp.quniform('min_samples_split', 10,200,10),
    'min_impurity_decrease':hp.uniform('min_impurity_decrease',0.000001,0.00001)
}

# Hyperparameter Optimization
fmin_objective = partial(
    nfm_h2.optimize_func_RF,
    x=fg_h2.data[0],
    y=fg_h2.data[2],
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
with open(param_path + 'hfl_best_params_rfh2.pickle','wb') as handle:
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

hyperopt_model.fit(fg_h2.data[0],fg_h2.data[2])

# Predictions
y_pred_train = hyperopt_model.predict_proba(fg_h2.data[0])[:,1] 
y_pred_test = hyperopt_model.predict_proba(fg_h2.data[1])[:,1]

# Evaluations
metrics_rf_h2 = {
    'LOGLOSS':{'train':{},'test':{}},
    'ROCAUC':{'train':{},'test':{}},
    'PRAUC':{'train':{},'test':{}},
    'BRIER':{'train':{},'test':{}}
}

log_loss_train = log_loss(fg_h2.data[2],y_pred_train,eps=0.000001)
log_loss_test = log_loss(fg_h2.data[3],y_pred_test,eps=0.000001)
metrics_rf_h2['LOGLOSS']['train'] = log_loss_train
metrics_rf_h2['LOGLOSS']['test'] = log_loss_test

roc_auc_train = roc_auc_score(fg_h2.data[2],y_pred_train)
roc_auc_test = roc_auc_score(fg_h2.data[3],y_pred_test)
metrics_rf_h2['ROCAUC']['train'] = roc_auc_train
metrics_rf_h2['ROCAUC']['test'] = roc_auc_test

precision_train, recall_train, _ = precision_recall_curve(fg_h2.data[2],y_pred_train)
pr_auc_train = auc(recall_train,precision_train)
precision_test, recall_test, _ = precision_recall_curve(fg_h2.data[3],y_pred_test)
pr_auc_test = auc(recall_test,precision_test)
metrics_rf_h2['PRAUC']['train'] = pr_auc_train
metrics_rf_h2['PRAUC']['test'] = pr_auc_test

brier_train = brier_score_loss(fg_h2.data[2],y_pred_train)
brier_test = brier_score_loss(fg_h2.data[3],y_pred_test)
metrics_rf_h2['BRIER']['train'] = brier_train
metrics_rf_h2['BRIER']['test'] = brier_test


# Save Random Forest Model
with open(model_path + 'rf_model_h2.pickle', 'wb') as handle:
    pkl.dump(hyperopt_model, handle)
    
print(f'Log-Loss on train data: {log_loss(fg_h2.data[2],y_pred_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_h2.data[3],y_pred_test,eps=0.000001):.4f}')
print(f'ROC-AUC on train data: {roc_auc_score(fg_h2.data[2],y_pred_train):.4f}')
print(f'ROC-AUC on test data: {roc_auc_score(fg_h2.data[3],y_pred_test):.4f}')
print(80*"#")
print("\t\t END OF RANDOM FOREST FITTING for H2 DATA")
print(80*"#")
   
    
    
#=============================================================
#  3. Data Set 3: H3 Data
#=============================================================
print("\n\n\t\t 3. Horizontal Split: H3 Data \n\n")
fg_h3 = Feature_Generator(
    data_path =  input_path,
    file_name = 'horizontal_iid_data3',
    n_jobs =  -1,
    train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_mean']
)

nfm_h3 = NonFed_Models()

params = {
    'lr':hp.uniform('lr',0.0001,0.01),
    'momentum': hp.uniform('momentum',0.6,0.99),
    'num_epochs': hp.quniform('num_epochs',2,5,1),
    'batch_size': hp.quniform('batch_size',32,512,8)
}

objective_func = partial(nfm_h3.optimize_func_LogReg,data=fg_h3.scaled_data,random_state=random_state)

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
with open(param_path + 'hfl_best_params_lgh3.pickle','wb') as handle:
    pkl.dump(new_param_dict,handle)


lg_model = nfm_h3.LogReg(
    input_size = fg_h3.scaled_data[0].shape[1],momentum = new_param_dict['momentum'], lr = new_param_dict['lr']
)

trainer = nfm_h3.Trainer(
    model = lg_model,
    data = fg_h3.scaled_data,
    opti = 'SGD'
)

trainer.train(
    num_epochs = new_param_dict['num_epochs'],
    batch_size = new_param_dict['batch_size'],
    save_model = True,
    save_path = model_path,
    model_name = 'log_reg_ruraldata_horizontal'
)

# Predictions
yhat_train = lg_model.predict(fg_h3.scaled_data[0])
yhat_test = lg_model.predict(fg_h3.scaled_data[1])

# Evaluations
metrics_logreg_h3 = {
    'LOGLOSS':trainer.LOGLOSS_epoch,
    'ROCAUC':trainer.ROCAUC_epoch,
    'PRAUC':trainer.PRAUC_epoch,
    'F1':trainer.F1_epoch,
    'BRIER':trainer.BRIER_epoch
}


print(f'Log-Loss on train data: {log_loss(fg_h3.data[2],yhat_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_h3.data[3],yhat_test,eps=0.000001):.4f}')
print(f"ROC-AUC on TRAIN data set: {roc_auc_score(fg_h3.scaled_data[2],yhat_train):.4f}")
print(f"ROC-AUC on TEST data set: {roc_auc_score(fg_h3.scaled_data[3],yhat_test):.4f}")
print(80*"#")
print("\t\t END OF LOGISTIC REGRESSION FITTING for H3 DATA")
print(80*"#")


# 3.2. Neural Networks
# Parameter Definition
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

objective_func = partial(nfm_h3.optimize_func_NN,data=fg_h3.scaled_data,random_state=random_state,opti='SGD')

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
with open(param_path + 'hfl_best_params_nnh3.pickle','wb') as handle:
    pkl.dump(new_param_dict,handle)


# Fit final model
ls_list = [new_param_dict[f"layer_size{i+1}"] for i in range(new_param_dict['num_hidden_layers'])]

nn_model = nfm_h3.NN(
    layer_sizes = [fg_h3.scaled_data[0].shape[1],*ls_list,1],
    drop_rate = new_param_dict['drop_rate'],
    momentum = new_param_dict['momentum'],
    weight_decay = new_param_dict['weight_decay'],
    lr = new_param_dict['lr']
)

trainer_nn = nfm_h3.Trainer(
    model = nn_model,
    data = fg_h3.scaled_data,
    opti = 'SGD'
)
trainer_nn.train(
    num_epochs = new_param_dict['epochs'],
    batch_size = new_param_dict['batch_size'],
    save_model = True,
    save_path = model_path,
    model_name = 'neuralnet_ruraldata_horizontal'
)

# Predictions
yhat_train = nn_model.predict(fg_h3.scaled_data[0])
yhat_test = nn_model.predict(fg_h3.scaled_data[1])

# Evaluations
metrics_nn_h3 = {
    'LOGLOSS':trainer_nn.LOGLOSS_epoch,
    'ROCAUC':trainer_nn.ROCAUC_epoch,
    'PRAUC':trainer_nn.PRAUC_epoch,
    'F1':trainer_nn.F1_epoch,
    'BRIER':trainer_nn.BRIER_epoch
}

print(f'Log-Loss on train data: {log_loss(fg_h3.data[2],yhat_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_h3.data[3],yhat_test,eps=0.000001):.4f}')
print(f"ROC-AUC on TRAIN data set: {roc_auc_score(fg_h3.scaled_data[2],yhat_train):.4f}")
print(f"ROC-AUC on TEST data set: {roc_auc_score(fg_h3.scaled_data[3],yhat_test):.4f}")
print(80*"#")
print("\t\t END OF NEURAL NETWORK FITTING for H3 DATA")
print(80*"#")


# 3.3. Random Forest
params = {
    'n_estimators':hp.quniform('n_estimators',200,1000,25),
    'max_depth':hp.quniform('max_depth',3,8,1),
    'max_samples':hp.uniform('max_samples',0.2,0.5),
    'min_samples_split':hp.quniform('min_samples_split', 10,200,10),
    'min_impurity_decrease':hp.uniform('min_impurity_decrease',0.000001,0.00001)
}

# Hyperparameter Optimization
fmin_objective = partial(
    nfm_h3.optimize_func_RF,
    x=fg_h3.data[0],
    y=fg_h3.data[2],
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
with open(param_path + 'hfl_best_params_rfh3.pickle','wb') as handle:
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

hyperopt_model.fit(fg_h3.data[0],fg_h3.data[2])

# Predictions
y_pred_train = hyperopt_model.predict_proba(fg_h3.data[0])[:,1] 
y_pred_test = hyperopt_model.predict_proba(fg_h3.data[1])[:,1]

# Evaluations
metrics_rf_h3 = {
    'LOGLOSS':{'train':{},'test':{}},
    'ROCAUC':{'train':{},'test':{}},
    'PRAUC':{'train':{},'test':{}},
    'BRIER':{'train':{},'test':{}}
}

log_loss_train = log_loss(fg_h3.data[2],y_pred_train,eps=0.000001)
log_loss_test = log_loss(fg_h3.data[3],y_pred_test,eps=0.000001)
metrics_rf_h3['LOGLOSS']['train'] = log_loss_train
metrics_rf_h3['LOGLOSS']['test'] = log_loss_test

roc_auc_train = roc_auc_score(fg_h3.data[2],y_pred_train)
roc_auc_test = roc_auc_score(fg_h3.data[3],y_pred_test)
metrics_rf_h3['ROCAUC']['train'] = roc_auc_train
metrics_rf_h3['ROCAUC']['test'] = roc_auc_test

precision_train, recall_train, _ = precision_recall_curve(fg_h3.data[2],y_pred_train)
pr_auc_train = auc(recall_train,precision_train)
precision_test, recall_test, _ = precision_recall_curve(fg_h3.data[3],y_pred_test)
pr_auc_test = auc(recall_test,precision_test)
metrics_rf_h3['PRAUC']['train'] = pr_auc_train
metrics_rf_h3['PRAUC']['test'] = pr_auc_test

brier_train = brier_score_loss(fg_h3.data[2],y_pred_train)
brier_test = brier_score_loss(fg_h3.data[3],y_pred_test)
metrics_rf_h3['BRIER']['train'] = brier_train
metrics_rf_h3['BRIER']['test'] = brier_test


# Save Random Forest Model
with open(model_path + 'rf_model_h3.pickle', 'wb') as handle:
    pkl.dump(hyperopt_model, handle)

print(f'Log-Loss on train data: {log_loss(fg_h3.data[2],y_pred_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_h3.data[3],y_pred_test,eps=0.000001):.4f}')
print(f'ROC-AUC on train data: {roc_auc_score(fg_h3.data[2],y_pred_train):.4f}')
print(f'ROC-AUC on test data: {roc_auc_score(fg_h3.data[3],y_pred_test):.4f}')
print(80*"#")
print("\t\t END OF RANDOM FOREST FITTING for H3 Data")
print(80*"#")



#=============================================================
#  4. Full Data (excl. global Test Data Set)
#=============================================================
print("\n\n\t\t 4. Horizontal Split: Full Data \n\n")
fg_hfull = Feature_Generator(
    data_path =  input_path,
    file_name = 'full_data_cleaned',
    n_jobs =  -1,
    train_size = train_size
    ,all_ext_sources = True
    ,vars_to_drop = ['ext_mean']
)

nfm_hfull = NonFed_Models()

params = {
    'lr':hp.uniform('lr',0.0001,0.01),
    'momentum': hp.uniform('momentum',0.6,0.99),
    'num_epochs': hp.quniform('num_epochs',2,5,1),
    'batch_size': hp.quniform('batch_size',32,512,8)
}

objective_func = partial(nfm_hfull.optimize_func_LogReg,data=fg_hfull.scaled_data,random_state=random_state)

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
with open(param_path + 'hfl_best_params_lghfull.pickle','wb') as handle:
    pkl.dump(new_param_dict,handle)

      
lg_model = nfm_hfull.LogReg(
    input_size = fg_hfull.scaled_data[0].shape[1],momentum = new_param_dict['momentum'], lr = new_param_dict['lr']
)

trainer = nfm_hfull.Trainer(
    model = lg_model,
    data = fg_hfull.scaled_data,
    opti = 'SGD'
)

trainer.train(
    num_epochs = new_param_dict['num_epochs'],
    batch_size = new_param_dict['batch_size'],
    save_model = True,
    save_path = model_path,
    model_name = 'log_reg_fulldata_horizontal'
)

# Predictions
y_pred_train = lg_model.predict(fg_hfull.scaled_data[0])
y_pred_test = lg_model.predict(fg_hfull.scaled_data[1])

# Evaluations
metrics_logreg_full = {
    'LOGLOSS':trainer.LOGLOSS_epoch,
    'LOGLOSS_batch':trainer.LOGLOSS_batch,
    'ROCAUC':trainer.ROCAUC_epoch,
    'PRAUC':trainer.PRAUC_epoch,
    'F1':trainer.F1_epoch,
    'BRIER':trainer.BRIER_epoch
}

# Evaluations (final)
metrics_lg_final = {
    'LOGLOSS':{'train':{},'test':{}},
    'ROCAUC':{'train':{},'test':{}},
    'PRAUC':{'train':{},'test':{}},
    'BRIER':{'train':{},'test':{}}
}

log_loss_train = log_loss(fg_hfull.scaled_data[2],y_pred_train,eps=0.000001)
log_loss_test = log_loss(fg_hfull.scaled_data[3],y_pred_test,eps=0.000001)
metrics_lg_final['LOGLOSS']['train'] = log_loss_train
metrics_lg_final['LOGLOSS']['test'] = log_loss_test

roc_auc_train = roc_auc_score(fg_hfull.scaled_data[2],y_pred_train)
roc_auc_test = roc_auc_score(fg_hfull.scaled_data[3],y_pred_test)
metrics_lg_final['ROCAUC']['train'] = roc_auc_train
metrics_lg_final['ROCAUC']['test'] = roc_auc_test

precision_train, recall_train, _ = precision_recall_curve(fg_hfull.scaled_data[2],y_pred_train)
pr_auc_train = auc(recall_train,precision_train)
precision_test, recall_test, _ = precision_recall_curve(fg_hfull.scaled_data[3],y_pred_test)
pr_auc_test = auc(recall_test,precision_test)
metrics_lg_final['PRAUC']['train'] = pr_auc_train
metrics_lg_final['PRAUC']['test'] = pr_auc_test

brier_train = brier_score_loss(fg_hfull.scaled_data[2],y_pred_train)
brier_test = brier_score_loss(fg_hfull.scaled_data[3],y_pred_test)
metrics_lg_final['BRIER']['train'] = brier_train
metrics_lg_final['BRIER']['test'] = brier_test

# Save final evaluation on full train and test set
with open(result_path + 'hlg_eval_full_traintest.pickle','wb') as handle:
    pkl.dump(metrics_lg_final,handle)

print(f'Log-Loss on train data: {log_loss(fg_hfull.scaled_data[2],y_pred_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_hfull.scaled_data[3],y_pred_test,eps=0.000001):.4f}')
print(f"ROC-AUC on TRAIN data set: {roc_auc_score(fg_hfull.scaled_data[2],y_pred_train):.4f}")
print(f"ROC-AUC on TEST data set: {roc_auc_score(fg_hfull.scaled_data[3],y_pred_test):.4f}")
print(80*"#")
print("\t\t END OF LOGISTIC REGRESSION FITTING for FULL DATA")
print(80*"#")


# 4.2. Neural Networks
# Parameter Definition
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

objective_func = partial(nfm_hfull.optimize_func_NN,data=fg_hfull.scaled_data,random_state=random_state,opti='SGD')

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
with open(param_path + 'hfl_best_params_nnhfull.pickle','wb') as handle:
    pkl.dump(new_param_dict,handle)


# Fit final model
ls_list = [new_param_dict[f"layer_size{i+1}"] for i in range(new_param_dict['num_hidden_layers'])]

nn_model = nfm_hfull.NN(
    layer_sizes = [fg_hfull.scaled_data[0].shape[1],*ls_list,1],
    drop_rate = new_param_dict['drop_rate'],
    weight_decay = new_param_dict['weight_decay'],
    momentum = new_param_dict['momentum'],
    lr = new_param_dict['lr']
)

trainer_nn = nfm_hfull.Trainer(
    model = nn_model,
    data = fg_hfull.scaled_data,
    opti = 'SGD'
)
trainer_nn.train(
    num_epochs = new_param_dict['epochs'],
    batch_size = new_param_dict['batch_size'],
    save_model = True,
    save_path = model_path,
    model_name = 'neuralnet_fulldata_horizontal'
)

# Predictions
y_pred_train = nn_model.predict(fg_hfull.scaled_data[0])
y_pred_test = nn_model.predict(fg_hfull.scaled_data[1])

# Evaluations
metrics_nn_full = {
    'LOGLOSS':trainer_nn.LOGLOSS_epoch,
    'LOGLOSS_batch':trainer_nn.LOGLOSS_batch,
    'ROCAUC':trainer_nn.ROCAUC_epoch,
    'PRAUC':trainer_nn.PRAUC_epoch,
    'F1':trainer_nn.F1_epoch,
    'BRIER':trainer_nn.BRIER_epoch
}


# Evaluations (final)
metrics_nn_final = {
    'LOGLOSS':{'train':{},'test':{}},
    'ROCAUC':{'train':{},'test':{}},
    'PRAUC':{'train':{},'test':{}},
    'BRIER':{'train':{},'test':{}}
}

log_loss_train = log_loss(fg_hfull.scaled_data[2],y_pred_train,eps=0.000001)
log_loss_test = log_loss(fg_hfull.scaled_data[3],y_pred_test,eps=0.000001)
metrics_nn_final['LOGLOSS']['train'] = log_loss_train
metrics_nn_final['LOGLOSS']['test'] = log_loss_test

roc_auc_train = roc_auc_score(fg_hfull.scaled_data[2],y_pred_train)
roc_auc_test = roc_auc_score(fg_hfull.scaled_data[3],y_pred_test)
metrics_nn_final['ROCAUC']['train'] = roc_auc_train
metrics_nn_final['ROCAUC']['test'] = roc_auc_test

precision_train, recall_train, _ = precision_recall_curve(fg_hfull.scaled_data[2],y_pred_train)
pr_auc_train = auc(recall_train,precision_train)
precision_test, recall_test, _ = precision_recall_curve(fg_hfull.scaled_data[3],y_pred_test)
pr_auc_test = auc(recall_test,precision_test)
metrics_nn_final['PRAUC']['train'] = pr_auc_train
metrics_nn_final['PRAUC']['test'] = pr_auc_test

brier_train = brier_score_loss(fg_hfull.scaled_data[2],y_pred_train)
brier_test = brier_score_loss(fg_hfull.scaled_data[3],y_pred_test)
metrics_nn_final['BRIER']['train'] = brier_train
metrics_nn_final['BRIER']['test'] = brier_test

# Save final evaluation on full train and test set
with open(result_path + 'hnn_eval_full_traintest.pickle','wb') as handle:
    pkl.dump(metrics_nn_final,handle)


print(f'Log-Loss on train data: {log_loss(fg_hfull.scaled_data[2],y_pred_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_hfull.scaled_data[3],y_pred_test,eps=0.000001):.4f}')
print(f"ROC-AUC on TRAIN data set: {roc_auc_score(fg_hfull.scaled_data[2],y_pred_train):.4f}")
print(f"ROC-AUC on TEST data set: {roc_auc_score(fg_hfull.scaled_data[3],y_pred_test):.4f}")
print(80*"#")
print("\t\t END OF NEURAL NETWORK FITTING for FULL DATA")
print(80*"#")



# 4.3. Random Forest
params = {
    'n_estimators':hp.quniform('n_estimators',200,1000,25),
    'max_depth':hp.quniform('max_depth',3,8,1),
    'max_samples':hp.uniform('max_samples',0.2,0.5),
    'min_samples_split':hp.quniform('min_samples_split', 10,200,10),
    'min_impurity_decrease':hp.uniform('min_impurity_decrease',0.000001,0.00001)
}

# Hyperparameter Optimization
fmin_objective = partial(
    nfm_hfull.optimize_func_RF,
    x=fg_hfull.data[0],
    y=fg_hfull.data[2],
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
with open(param_path + 'hfl_best_params_rfhfull.pickle','wb') as handle:
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

hyperopt_model.fit(fg_hfull.data[0],fg_hfull.data[2])

# Predictions
y_pred_train = hyperopt_model.predict_proba(fg_hfull.data[0])[:,1] 
y_pred_test = hyperopt_model.predict_proba(fg_hfull.data[1])[:,1]

# Evaluations
metrics_rf_full = {
    'LOGLOSS':{'train':{},'test':{}},
    'ROCAUC':{'train':{},'test':{}},
    'PRAUC':{'train':{},'test':{}},
    'BRIER':{'train':{},'test':{}}
}

log_loss_train = log_loss(fg_hfull.data[2],y_pred_train,eps=0.000001)
log_loss_test = log_loss(fg_hfull.data[3],y_pred_test,eps=0.000001)
metrics_rf_full['LOGLOSS']['train'] = log_loss_train
metrics_rf_full['LOGLOSS']['test'] = log_loss_test

roc_auc_train = roc_auc_score(fg_hfull.data[2],y_pred_train)
roc_auc_test = roc_auc_score(fg_hfull.data[3],y_pred_test)
metrics_rf_full['ROCAUC']['train'] = roc_auc_train
metrics_rf_full['ROCAUC']['test'] = roc_auc_test

precision_train, recall_train, _ = precision_recall_curve(fg_hfull.data[2],y_pred_train)
pr_auc_train = auc(recall_train,precision_train)
precision_test, recall_test, _ = precision_recall_curve(fg_hfull.data[3],y_pred_test)
pr_auc_test = auc(recall_test,precision_test)
metrics_rf_full['PRAUC']['train'] = pr_auc_train
metrics_rf_full['PRAUC']['test'] = pr_auc_test

brier_train = brier_score_loss(fg_hfull.data[2],y_pred_train)
brier_test = brier_score_loss(fg_hfull.data[3],y_pred_test)
metrics_rf_full['BRIER']['train'] = brier_train
metrics_rf_full['BRIER']['test'] = brier_test


# Save Random Forest Model
with open(model_path + 'rf_model_hfull.pickle', 'wb') as handle:
    pkl.dump(hyperopt_model, handle)

# Save final evaluation on full train and test set
with open(result_path + 'hrf_eval_full_traintest.pickle', 'wb') as handle:
    pkl.dump(metrics_rf_full, handle)

print(f'Log-Loss on train data: {log_loss(fg_hfull.data[2],y_pred_train,eps=0.000001):.4f}')
print(f'Log-Loss on test data: {log_loss(fg_hfull.data[3],y_pred_test,eps=0.000001):.4f}')
print(f'ROC-AUC on train data: {roc_auc_score(fg_hfull.data[2],y_pred_train):.4f}')
print(f'ROC-AUC on test data: {roc_auc_score(fg_hfull.data[3],y_pred_test):.4f}')
print(80*"#")
print("\t\t END OF RANDOM FOREST FITTING for FULL Data")
print(80*"#")




# Save all evaluations
horizontal_nfm_results = {
    'h1_data_lg':metrics_logreg_h1,
    'h1_data_nn':metrics_nn_h1,
    'h1_data_rf':metrics_rf_h1,
    'h2_data_lg':metrics_logreg_h2,
    'h2_data_nn':metrics_nn_h2,
    'h2_data_rf':metrics_rf_h2,
    'h3_data_lg':metrics_logreg_h3,
    'h3_data_nn':metrics_nn_h3,
    'h3_data_rf':metrics_rf_h3,
    'full_data_lg':metrics_logreg_full,
    'full_data_nn':metrics_nn_full,
    'full_data_rf':metrics_rf_full
    
} 
   
# Save Resuts
with open(result_path + 'horizontal_nfm_result_dict.pickle', 'wb') as handle:
    pkl.dump(horizontal_nfm_results, handle)


