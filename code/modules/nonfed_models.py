import numpy as np
import pandas as pd
import pickle as pkl
import time
from functools import partial

from xgboost import XGBRegressor,XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error,auc,roc_auc_score,log_loss,f1_score,brier_score_loss,precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch 
import torch.nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import TensorDataset, DataLoader


from hyperopt import Trials, fmin, hp, tpe


class NonFed_Models(object):
    """
    Super class that contains the model blue prints for the models of the non-federated (i.e. the centralized) setting,
    their trainer objects (log. reg. and neural networks only) and the optimizer functions necessary hyperparameter tuning
    via Bayesian optimization.
    
    """
    def __init__(self,n_jobs=-1):
                 self.n_jobs = n_jobs
                 
    class LogReg(torch.nn.Module):
        def __init__(self,input_size,momentum=0.9,lr=0.001,random_state=2021):
            """
            Pytorch model that implements logistic regression for the non-federated (i.e. centralized) setting.

            Parameters
            ----------
            input_size : int
                Number of features from which the model will learn (i.e. size of the input layer).
            momentum : float, optional
                Defines the momentum parameter for the SGD optimizer. Check PyTorch documentation for further details.
                The default is 0.9.
            lr : float, optional
                Defines the learning rate of the optimizer. The default is 0.001.
            random_state : int, optional
                Sets random seed. The default is 2021.

            Returns
            -------
            None.

            """
            super(NonFed_Models.LogReg,self).__init__()
            self.momentum = momentum
            self.lr = lr
            self.num_layers = 0
            torch.manual_seed(random_state)
            np.random.seed(random_state)
        
            
            self.l1 = torch.nn.Linear(input_size,1,bias=True)
            torch.nn.init.kaiming_normal_(self.l1.weight)
            
        def forward(self,x):
            x = self.l1(x)
            return x
        
        def predict(self,x):
            x = self.forward(x)
            x = torch.sigmoid(x)
            return x.detach().numpy()
        
        
    class NN(torch.nn.Module):
        def __init__(self,layer_sizes,drop_rate=0.25,weight_decay=0,momentum=0,lr=0.001,random_state=2021):
            """
            Pytorch model that implements a MLP-neural network for the non-federated (i.e. centralized) setting.

            Parameters
            ----------
            layer_sizes : list
                List that describes the sizes of the hidden layers in the network.
            drop_rate : float, optional
                Indicates how many percent of neurons should be deactivated for a given forward pass during training. 
                Is used to control overfitting but could impair the learning if not set carefully.
                The default is 0.25.
            weight_decay : float, optional
                Defines the weight_decay parameter for the ADAM optimizer. Used to control overfitting. 
                Check PyTorch documentation for further details.
                The default is 0.
            momentum : float, optional
                Defines the momentum parameter for the SGD optimizer. Check PyTorch documentation for further details.
                The default is 0.
            lr : float, optional
                Sets the learning rate for the optimizer. The default is 0.001.
            random_state : int, optional
                Sets random seed. The default is 2021.

            Returns
            -------
            None.

            """
            super(NonFed_Models.NN,self).__init__()
            self.drop_rate = drop_rate
            self.weight_decay = weight_decay
            self.momentum = momentum
            self.lr = lr
            self.layer_list = torch.nn.ModuleList([]) # torch.nn.ModuleList(self.list_1)
            #self.batch_norm_list = torch.nn.ModuleList([])
            self.num_layers = len(layer_sizes) - 2
            torch.manual_seed(random_state)
            np.random.seed(random_state)
        
            
            for i in np.arange(1,len(layer_sizes)):
                self.layer = torch.nn.Linear(layer_sizes[(i-1)],layer_sizes[i],bias=True)
                torch.nn.init.kaiming_normal_(self.layer.weight)
                self.layer_list.append(self.layer)
                #self.bn = torch.nn.BatchNorm1d(layer_sizes[i])
                #self.batch_norm_list.append(self.bn)
            
            self.dropout = torch.nn.Dropout(drop_rate)
            
            
        def forward(self,x):
            for i in np.arange((len(self.layer_list)-1)):
                x = self.layer_list[i](x)
                #x = self.batch_norm_list[i](x)
                x = F.relu(x)
                x = self.dropout(x)
            x = self.layer_list[(len(self.layer_list)-1)](x) 
            return x
        
        def predict(self,x):
            x = self.forward(x)
            x = torch.sigmoid(x)
            return x.detach().numpy()
    
    
    #==========================================
    # Trainer Subclass for PyTorch Models
    #==========================================
    class Trainer(object):
        """
        Trainer object to train log. regression and the neural network in a non-federated setting.
        
        Parameters:
        ------------
        model : PyTorch model
            Either log. regression or neural network model.
        data : torch.tensor
            Scaled data set from the Feature_Generator object.    
        opti : str, optional
            String that indicates which optimizer to use. Currently only works for SGD and ADAM.
            
        """
        def __init__(self,model,data,opti='ADAM'):
            self.model = model
            self.data = data
            # Define Optimizer
            if opti == 'SGD':
                self.opt = torch.optim.SGD(self.model.parameters(),lr=self.model.lr,momentum=self.model.momentum)
            elif opti == 'ADAM':
                self.opt = torch.optim.Adam(self.model.parameters(),lr=self.model.lr,weight_decay=self.model.weight_decay)
            else:
                raise Exception("Currently only SGD or ADAM applicable as optimizers.")
            self.LOGLOSS_batch = {}
            self.LOGLOSS_epoch = {'train':{},'test':{}}
            self.ROCAUC_epoch = {'train':{},'test':{}}
            self.PRAUC_epoch = {'train':{},'test':{}}
            self.F1_epoch = {'train':{},'test':{}}
            self.BRIER_epoch = {'train':{},'test':{}}
            
        def train(self,num_epochs,batch_size,save_model=False,save_path=None,model_name=None):
            """
            Steers the training process of the centralized models stores the results in the corresponding dictionaries.

            Parameters
            ----------
            num_epochs : int
                Number of epochs.
            batch_size : int
                Batch size to use during model training.
            save_model : bool, optional
                Indicates whether the model should be saved. The default is False.
            save_path : str, optional
                Indicates the save path on the disk. The default is None.
            model_name : str, optional
                File name for the saved model. The default is None.

            Returns
            -------
            None.

            """
            start = time.time()
            self.model.train() 
            
            ds_train = torch.utils.data.TensorDataset(self.data[0], self.data[2])
            dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last= True)
    
            # Epoch Loop
            for epoch in range(num_epochs):
                start_epoch = time.time()
                batch_loss = []
                
                # Batch Loop
                for batch_idx,(x,y) in enumerate(dl_train):
                    self.opt.zero_grad()
                    scores = self.model.forward(x.float())
                    loss = F.binary_cross_entropy_with_logits(input=scores,
                                                              target=y.reshape(-1,1),
                                                              reduction='mean')
                    
                    batch_loss.append(loss.detach().item())
                    
                    loss.backward()
                    self.opt.step()
                self.LOGLOSS_batch[f'epoch{epoch+1}'] = batch_loss
                    
                # Predictions 
                yhat_train = self.model.predict(self.data[0])
                yhat_test = self.model.predict(self.data[1])
                
                # Evaluations
                log_loss_train = log_loss(self.data[2],yhat_train,eps=0.000001)
                log_loss_test = log_loss(self.data[3],yhat_test,eps=0.000001)
                self.LOGLOSS_epoch['train'][f'epoch{epoch+1}'] = log_loss_train
                self.LOGLOSS_epoch['test'][f'epoch{epoch+1}'] = log_loss_test
                
                f1_score_train = f1_score(self.data[2],[1 if i >= 0.5 else 0 for i in yhat_train])
                f1_score_test = f1_score(self.data[3],[1 if i >= 0.5 else 0 for i in yhat_test])
                self.F1_epoch['train'][f'epoch{epoch+1}'] = f1_score_train
                self.F1_epoch['test'][f'epoch{epoch+1}'] = f1_score_test
                
                brier_train = brier_score_loss(self.data[2],yhat_train)
                brier_test = brier_score_loss(self.data[3],yhat_test)
                self.BRIER_epoch['train'][f'epoch{epoch+1}'] = brier_train 
                self.BRIER_epoch['test'][f'epoch{epoch+1}'] = brier_test
                
                precision_train, recall_train, _ = precision_recall_curve(self.data[2],yhat_train) 
                pr_auc_train = auc(recall_train,precision_train)
                precision_test, recall_test, _ = precision_recall_curve(self.data[3],yhat_test) 
                pr_auc_test = auc(recall_test,precision_test)
                self.PRAUC_epoch['train'][f'epoch{epoch+1}'] = pr_auc_train
                self.PRAUC_epoch['test'][f'epoch{epoch+1}'] = pr_auc_test
                
                roc_auc_train = roc_auc_score(self.data[2],yhat_train)
                roc_auc_test = roc_auc_score(self.data[3],yhat_test)
                self.ROCAUC_epoch['train'][f'epoch{epoch+1}'] = roc_auc_train
                self.ROCAUC_epoch['test'][f'epoch{epoch+1}'] = roc_auc_test
                
                
                end_epoch = time.time()
                print(80*'=')
                print(f"Progress: {epoch+1}/{num_epochs}")
                print(f"\t-> Avg. log-Loss on TRAIN data for epoch {epoch+1}: {log_loss_train:.4f}")
                print(f"\t-> Avg. log-Loss on TEST data for epoch {epoch+1}: {log_loss_test:.4f}")
                print(f"\t-> ROC-AUC on TRAIN data for epoch {epoch+1}: {roc_auc_train:.5f}")
                print(f"\t-> ROC-AUC on TEST data for epoch {epoch+1}: {roc_auc_test:.5f}")
                print(f"\t-> Execution time for epoch {epoch+1}: {(end_epoch - start_epoch)/60:.2f} min.")
            
            end = time.time()
            if save_model == True:
                 with open(save_path + f'{model_name}.pickle', 'wb') as handle:
                    pkl.dump(self.model, handle)
            else:
                pass
            print(80*'=')
            print(f"Total Execution of model training with {self.model.num_layers} hidden layers: {(end-start)/60:.2f} min.")
    

    #=============================
    # Optimization Functions    
    #=============================    
    def optimize_func_XGB(self,space,x,y,classifier=True,n_splits=3,shuffle=True,random_state=2021,n_jobs=-1,verbose=1):
        """
        Optimizer function used for HyperOpts implementation of Bayesian optimiztaion. Tailored for XGB and the APR model.

        Parameters
        ----------
        space : dict
            Hyperparameter space that should be tuned.
        x : pandas.DataFrame
            Features.
        y : pandas.Series
            Labels.
        classifier : bool, optional
            If False, a XGB_Regressor is tuned. Otherwise it tunes an XGB_Classifier. The default is True.
        n_splits : int, optional
            NUmber of cross-validations. The default is 3.
        shuffle : bool, optional
            Indicates if data should be shuffled after a CV-fold. The default is True.
        random_state : int, optional
            Sets random seed. The default is 2021.
        n_jobs : int, optional
            NUmber of cores to use. The default is -1.
        verbose : int, optional
            COntrols level of verbosity. The default is 1.

        Returns
        -------
        float
            Average log-loss score across all CV-folds.

        """
        np.random.seed(random_state)
        
        if classifier == True:
            model = XGBClassifier()
            objective = 'binary:logistic'
            eval_metric = 'auc'
        else:
            model = XGBRegressor()
            objective = 'reg:squarederror'
            eval_metric = 'rmse'
            
        model.set_params(
            n_estimators = int(space['n_estimators']),
            max_depth = int(space['max_depth']),
            learning_rate = space['learning_rate'],
            subsample = space['subsample'],
            gamma = space['gamma'],
            min_child_weight = space['min_child_weight'],
            colsample_bytree = space['colsample_bytree'],
            reg_lambda = space['reg_lambda'],
            reg_alpha = space['reg_alpha'],
            n_jobs = int(n_jobs),
            verbosity = int(verbose),
            use_label_encoder=False,
            eval_metric=eval_metric,
            random_state=random_state,
            objective=objective
        )
        kf_cv = KFold(n_splits=n_splits, shuffle=shuffle,random_state=random_state)
        eval_scores = []
        for idx in kf_cv.split(x,y):
            train_idx, val_idx = idx[0], idx[1]
            xtrain = x.iloc[train_idx,:]
            ytrain = y.iloc[train_idx]
            
            xval = x.iloc[val_idx,:]
            yval = y.iloc[val_idx]
            
            model.fit(xtrain,ytrain)
            if classifier == True:
                ypreds = model.predict_proba(xval)[:,1]
                scores = log_loss(yval,ypreds,eps=0.000001)
                #scores = -1 * roc_auc_score(yval,ypreds)
            else:
                ypreds = model.predict(xval)
                scores = mean_squared_error(yval,ypreds)
            eval_scores.append(scores)
        
        return np.mean(eval_scores)
    
    
    def optimize_func_RF(self,space,x,y,n_splits=3,shuffle=True,random_state=2021,n_jobs=-1,verbose=0):
        """
        Optimizer function used for HyperOpts implementation of Bayesian optimiztaion. Tailored for Random Forest.

        Parameters
        ----------
        space : dict
            Hyperparameter space that should be tuned.
        x : pandas.DataFrame
            Features.
        y : pandas.Series
            Labels.
        classifier : bool, optional
            If False, a XGB_Regressor is tuned. Otherwise it tunes an XGB_Classifier. The default is True.
        n_splits : int, optional
            NUmber of cross-validations. The default is 3.
        shuffle : bool, optional
            Indicates if data should be shuffled after a CV-fold. The default is True.
        random_state : int, optional
            Sets random seed. The default is 2021.
        n_jobs : int, optional
            NUmber of cores to use. The default is -1.
        verbose : int, optional
            COntrols level of verbosity. The default is 1.

        Returns
        -------
        float
            Average log-loss score across all CV-folds.

        """
        np.random.seed(random_state)
        
        model = RandomForestClassifier(criterion='entropy',max_features='sqrt')
            
        model.set_params(
            n_estimators = int(space['n_estimators']),
            max_depth = int(space['max_depth']),
            max_samples = space['max_samples'],
            min_samples_split = int(space['min_samples_split']),
            min_impurity_decrease = space['min_impurity_decrease'],
            n_jobs = int(n_jobs),
            verbose = int(verbose),
            random_state=random_state,
        )
        
        kf_cv = KFold(n_splits=n_splits, shuffle=shuffle,random_state=random_state)
        eval_scores = []
        for idx in kf_cv.split(x,y):
            train_idx, val_idx = idx[0], idx[1]
            xtrain = x.iloc[train_idx,:]
            ytrain = y.iloc[train_idx]
            
            xval = x.iloc[val_idx,:]
            yval = y.iloc[val_idx]
            
            model.fit(xtrain,ytrain)
            ypreds = model.predict_proba(xval)[:,1]
            scores = log_loss(yval,ypreds,eps=0.000001)
            #scores = -1 * roc_auc_score(yval,ypreds)
            eval_scores.append(scores)
        
        return np.mean(eval_scores)
    
    
    
    def optimize_func_NN(self,space,data,random_state=2021,opti='SGD'):
        """
        Optimizer function used for HyperOpts implementation of Bayesian optimiztaion. Tailored for Log. regression.

        Parameters
        ----------
        space : dict
            Hyperparameter space that should be tuned.
        data : list
            List that contains scaled features and labels.
        random_state : int, optional
            Sets random seed. The default is 2021.
        opti : str, optional
            Indicates which optimizer to use during training. CUrrently only SGD and ADAM are implemented.

        Returns
        -------
        float
            Log-loss score on test set.

        """
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        ls = [
            int(space['layer_size1']),
            int(space['layer_size2']),
            int(space['layer_size3']),
            int(space['layer_size4']),
        ]
        ls = [ls[i] for i in range(int(space['num_hidden_layers']))]
    
        model = NonFed_Models.NN(
            layer_sizes = [data[0].shape[1], *ls, 1],
            drop_rate=space['drop_rate'],
            lr=space['lr'],
            momentum=space['momentum'],
            weight_decay=space['weight_decay'],
            random_state=random_state
        )
        
        trainer = NonFed_Models.Trainer(model=model,data=data,opti=opti)
        trainer.train(num_epochs=int(space['epochs']),batch_size = int(space['batch_size']))
        yhat = model.predict(data[1])
        logloss = log_loss(data[3],yhat,eps=0.000001)
        
        return logloss
    
    
    def optimize_func_LogReg(self,space,data,random_state=2021):
        """
        Optimizer function used for HyperOpts implementation of Bayesian optimiztaion. Tailored for Log. regression.

        Parameters
        ----------
        space : dict
            Hyperparameter space that should be tuned.
        data : list
            List that contains scaled features and labels.
        random_state : int, optional
            Sets random seed. The default is 2021.
        opti : str, optional
            Indicates which optimizer to use during training. CUrrently only SGD and ADAM are implemented.

        Returns
        -------
        float
            Log-loss score on test set.

        """
        
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
        model = NonFed_Models.LogReg(
            input_size=data[0].shape[1],
            momentum=space['momentum'],
            lr=space['lr']
        )
        
        trainer = NonFed_Models.Trainer(model=model,data=data,opti='SGD')
        trainer.train(num_epochs = int(space['num_epochs']),batch_size = int(space['batch_size']))
        yhat = model.predict(data[1])
        logloss = log_loss(data[3],yhat,eps=0.000001)
        
        return logloss
        
    
    #============================
    # Wrapper Functions
    #============================
    def fit_APR_model(self,
                      data,
                      param_dict: dict,
                      random_search: bool = False,
                      train_size: float = 0.7,
                      random_state: int = 2021,
                      shuffle: bool = True,
                      n_jobs: int = -1,
                      n_iter: int = 5,
                      verbose: int = 1,
                      cv_folds: int = 3,
                      save_model: bool = True,
                      save_path: str = None
                      ):
        """
        Fits model that predicts the number of repayment periods using the credit amount and annuity amount feature  
        and some additional features that are constructed on top of these two. Training is done based on the 
        previos_application data set. Training such a model is necessary because the number of repayments (cnt_payment)
        are needed to approximate the annual percentage rate and the monthly interest rate of the credit contract, 
        which are the most important features for the final prediction task. 
        However, the cnt_payment variable is only contained in prev_applications and not in application_train which 
        requires a model that can estimate the number of repayments for the application_train set.
        

        Parameters
        ----------
        train_size : float, optional
            Indicates the size of the train set. The default is 0.7.
        random_state : int, optional
            Sets the random seed. The default is 2021.
        shuffle : bool, optional
            Indicates whether data should be shuffled after a CV-fold. The default is True.

        Returns
        -------
        Fitted and tuned APR-model as XGB-Classifier.

        """
        apr_set = data.loc[:,['sk_id_curr','sk_id_prev','cnt_payment','amt_annuity','amt_credit']]
        apr_set['app_grant_ratio'] = data['amt_credit'] / data['amt_goods_price']
        apr_set['credit_annuity_ratio'] = data['amt_credit'] / data['amt_annuity']
        
        data_apr = train_test_split(
            apr_set.loc[:,'amt_annuity':],
            apr_set.loc[:,'cnt_payment'],
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle
        )
        
        # Tuning via Random Search
        if random_search == True:
            model = RandomizedSearchCV(
                estimator = XGBRegressor(n_jobs=n_jobs),
                param_distributions = param_dict,
                n_iter = n_iter, # e.g. 5
                scoring='neg_mean_squared_error',
                verbose=verbose, # e.g. 10
                n_jobs=n_jobs, # e.g. -1
                random_state=random_state, # e.g. 2021
                cv=cv_folds # e.g. 3
            )
            
            model.fit(data_apr[0],data_apr[2])
            
            # Extracting best hyperparameters
            best_params_dict = model.best_estimator_.get_params()
            best_params_dict['random_state'] = int(random_state)
        
        # Tuning via Bayesian Optimization
        else:
            # Search for optimal hyperparameters via Bayesian Optimization
            fmin_objective = partial(self.optimize_func_XGB,
                                     x=data_apr[0],
                                     y=data_apr[2],
                                     classifier=False,
                                     n_splits=cv_folds,
                                     shuffle=shuffle,
                                     random_state=random_state,
                                     n_jobs = n_jobs,
                                     verbose = verbose)

            trials = Trials()
            
            best_hyperparams = fmin(
                fn = fmin_objective,
                space = param_dict,
                algo = tpe.suggest,
                max_evals = n_iter,
                trials = trials,
                rstate = np.random.RandomState(random_state)
            )
            
            # Extracting best hyperparameters
            best_params_dict = {}
            
            for key,val in best_hyperparams.items():
                if key == 'random_state':
                    best_params_dict[key] = int(random_state)
                elif key in ['n_estimators','max_depth']:
                    best_params_dict[key] = int(val)
                else:
                    best_params_dict[key] = val
                    
        
        # Fit Final Model with optimized hyperparameters
        self.xgb_reg = XGBRegressor(**best_params_dict)
        self.xgb_reg.fit(data_apr[0],data_apr[2])
        y_pred = self.xgb_reg.predict(data_apr[1])
        self.mse_score_ = mean_squared_error(data_apr[3],y_pred)
        
        if save_model == True:
            if random_search == True:
                with open(save_path + 'random_search_model.pickle', 'wb') as handle:
                    pkl.dump(self.xgb_reg,handle)
            else:
                with open(save_path + 'hyper_opt_model.pickle', 'wb') as handle:
                    pkl.dump(self.xgb_reg,handle)
                    
        return self
    