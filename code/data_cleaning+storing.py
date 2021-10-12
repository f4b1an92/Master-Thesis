import os
import sys
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

os.path.realpath(__file__)
main_path = os.getcwd()

# Load config
with open(main_path + '\\config_general.yml','r') as f:
    config = yaml.safe_load(f)
 

sys.path.append(main_path + '\\Code\\modules')

import dataset_prep as dp
import hc_fun as hcf
import nonfed_models as nfm
import feature_gen as fg

from hyperopt import Trials, fmin, hp, tpe

# Define paths
input_path = main_path + '\\raw_data\\'
result_path = main_path + '\\cleaned_datasets\\'
model_path = main_path + '\\models\\'

data_gen = dp.DataSet_Prepper(
    input_path = input_path,
    output_path = result_path,
    models_path = model_path
)

# 1. Preparing the 'prev_app'-Data Set
arg_dict = {
    'prev_app_filename': 'previous_application',
    'app_filename':'application_train',
    'save_int_rates': True,
    'corr_thres': 0.7,
    'step_size': config['step_size'],
    'block_size': 1_000
}

data_gen.prev_app(**arg_dict)



# 2. Train an XGBRegressor to predict cnt_payment Values 
## 2.1. Random Search
"""
nfm_model = fg.Feature_Generator(
    data_path = input_path + '/cleaned_datasets',
    file_name = 'interest_rates_results',
    #param_grid =  arg_dict_rand,
    #param_grid =  arg_dict_hpt,
    #save_path = 'C:/Users/Fabian/Desktop/Studium/2.) HU Berlin/6.) SS_21/Master Thesis/Federated Learning for Credit Risk/Available Data Sets/1.) Home Credit Default Risk/models/test_models',
    n_jobs = -1,
    apr_model=True
)

arg_dict_rand = {
    'n_estimators':np.arange(50,300,25),
    'learning_rate':np.arange(0.15,0.4,0.01),
    'max_depth': np.arange(1,7),
    'subsample': np.arange(0.3,0.5,0.01),
    'gamma': np.arange(0.2,0.9,0.01),
    'min_child_weight':np.arange(1,5,1),
    #'colsample_bytree':np.arange(0.5,1,0.1),
    #'reg_lambda':np.arange(0,1,0.1),
    #'reg_alpha':np.arange(0,1,0.1)
}

nfm_model.fit_APR_model(
    param_dict = arg_dict_rand,
    random_search = True,
    train_size = 0.7,
    n_iter = 5,
    cv_folds = 3,
    random_state = 2021,
    shuffle = True,
    save_model = True,
    save_path = 'C:/Users/Fabian/Desktop/Studium/2.) HU Berlin/6.) SS_21/Master Thesis/Federated Learning for Credit Risk/Available Data Sets/1.) Home Credit Default Risk/models/test_models'
)
print(f"MSE-Score of best model after hyperparameter tuning via random search: {nfm_model.mse_score_:.4f}")
"""

## 2.2. Bayesian Optimization
nfm_data = fg.Feature_Generator(
    data_path = result_path,
    file_name = 'interest_rates_results',
    n_jobs = -1,
    apr_model=True
)

nfm_model = nfm.NonFed_Models()

arg_dict_hpt = {
    'n_estimators':hp.quniform('n_estimators',50,500,25),
    'max_depth': hp.quniform('max_depth',1,7,1),
    'learning_rate':hp.uniform('learning_rate',0.15,0.4),
    'subsample': hp.uniform('subsample',0.3,0.5),
    'gamma': hp.uniform('gamma',0.2,0.9),
    'min_child_weight':hp.quniform('min_child_weight',1,5,1),
    'colsample_bytree':hp.uniform('colsample_bytree',0.5,1),
    'reg_lambda':hp.uniform('reg_lambda',0,1),
    'reg_alpha':hp.uniform('reg_alpha',0,1),
}

nfm_model.fit_APR_model(
    data = nfm_data.data,
    param_dict = arg_dict_hpt,
    n_iter = config['max_evals'],
    random_search = False,
    train_size = config['htrain_size'],
    cv_folds = config['n_splits'],
    random_state = config['random_state'],
    shuffle = True,
    save_model = True,
    save_path = model_path
)

print(f"MSE-Score of best model after hyperparameter tuning via bayesian optimization: {nfm_model.mse_score_:.4f}")



# 3. Preparing Application Data
arg_dict = {
    'app_filename':'application_train',
    'model_filename':'hyper_opt_model',
    'corr_thres': 0.7,
    'step_size':config['step_size'],
    'block_size':1_000,
    'corr_remove':False,
    'outlier_remove':True,
    'ext_opt':False
}

data_gen.app_data(**arg_dict)



# 4. Preparing Bureau Data
arg_dict = {
    'bur_filename':'bureau',
    'bal_filename':'bureau_balance',
    'app_filename':'application_train',
    'corr_thres': 0.7
}

data_gen.bureau_data(**arg_dict)




# 5. Preparing Credit Card Balance Data
arg_dict = {
    'ccb_filename':'credit_card_balance',
    'app_filename':'application_train',
    'corr_thres': 0.7
}

data_gen.credit_card_bal_data(**arg_dict)



# 6. Preparing Installment Payments Data
arg_dict = {
    'inst_filename':'installments_payments',
    'app_filename':'application_train',
    'corr_thres': 0.7
}

data_gen.installment_data(**arg_dict)




# 7. Preparing Full Data Set
arg_dict = {
    'ccb_include':False,
    'random_state':config['random_state'],
    'train_size':config['htrain_size'],
    'corr_thres':0.7,
    'block_size':1_000,
    'step_size':config['step_size'],
    'corr_remove':False,
    'outlier_remove':True,
    'ext_opt':False,
    'cv_folds':config['n_splits'],
    'n_iter':5
}

data_gen.full_set(**arg_dict)




# 8. Preparing Horizontal Data Sets (non-iid case)
arg_dict = {
    'file_name':'full_data_cleaned',
    'size_proportions': [8,2,1],
    'def_rates': [0.019, 0.15, 0.30], # IID Case: '1.9% - 15% - 30%'; Non-IID Case: '7.83% - 7.85% - 7.90%'
    'full_data_path':result_path,
    'global_data_size': 15_000,
    'random_state':config['random_state']                    
}

data_gen.horizontal_noniid(**arg_dict)



# 9. Preparing Horizontal Data Sets (iid case)
arg_dict = {
    'file_name':'full_data_cleaned',
    'size_proportions': [8,2,1],
    'def_rates': [0.0783, 0.0785, 0.079], # IID Case: '1.9% - 15% - 30%'; Non-IID Case: '7.83% - 7.85% - 7.90%'
    'full_data_path':result_path,
    'global_data_size': 15_000,
    'random_state':config['random_state'],
    'iid':True                    
}

data_gen.horizontal_noniid(**arg_dict)
