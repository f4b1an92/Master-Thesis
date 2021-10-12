import sys
import numpy as np
import pandas as pd
from itertools import chain
from tqdm import tqdm

import torch 
import torch.nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier

import tenseal as ts
import random
import time
from copy import deepcopy
import pickle

from sklearn.metrics import auc,roc_auc_score,log_loss,f1_score,brier_score_loss,precision_recall_curve,average_precision_score

#sys.path.append(r'C:\Users\Fabian\Desktop\Studium\2.) HU Berlin\6.) SS_21\Master Thesis\Federated Learning for Credit Risk\Code')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 300)
pd.options.display.float_format = '{:,.4f}'.format



class FedLogReg(torch.nn.Module):
    def __init__(self,input_size,momentum=0.9,lr=0.001,random_state=2021,bias=True,init='kaiming'):
        """
        Pytorch model that implements logistic regression for the federated (horizontal) setting.

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
        bias : bool, optional
            Indicates if bias should be used during the fitting process. The default is True.
        init : str, optional
            Defines the weight initialization type. The default is 'kaiming'.

        Returns
        -------
        None.

        """
        super().__init__()
        self.momentum = momentum
        self.lr = lr
        self.layer_list = torch.nn.ModuleList([]) 
        self.num_layers = 0
        self.bias = bias
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        layer = torch.nn.Linear(input_size,1,bias=self.bias)
        if init == 'kaiming':
            torch.nn.init.kaiming_normal_(layer.weight)
        else:
            torch.nn.init.uniform_(layer.weight, a=0.0, b=1.0)
        self.layer_list.append(layer)
            
    def forward(self,x):
        x = self.layer_list[0](x) # u[i]
        return x
        
    def forward_sq(self,x):
        x = self.layer_list[0](x)
        x = x**2
        return x
    
    def predict(self,x):
        x = self.forward(x)
        x = torch.sigmoid(x)
        return x.detach().numpy()



class FedNN(torch.nn.Module):
    def __init__(self,layer_sizes,drop_rate=0.25,weight_decay=0,momentum=0,lr=0.001,random_state=2021,bias=True):
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
        bias : bool, optional
            Indicates if bias should be used during the fitting process. The default is True.

        Returns
        -------
        None.

        """
        super(FedNN,self).__init__()

        self.drop_rate = drop_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr = lr
        self.layer_list = torch.nn.ModuleList([])
        self.num_layers = len(layer_sizes) - 2
        self.bias = bias
        
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        for i in np.arange(1,len(layer_sizes)):
            layer = torch.nn.Linear(layer_sizes[(i-1)],layer_sizes[i],bias=self.bias)
            torch.nn.init.kaiming_normal_(layer.weight)
            self.layer_list.append(layer)
            
        self.dropout = torch.nn.Dropout(drop_rate)
            
    def forward(self,x):
        for i in np.arange((len(self.layer_list)-1)):
            x = self.layer_list[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.layer_list[(len(self.layer_list)-1)](x) # last layer i.e. output layer
        return x
        
    def predict(self,x):
        x = self.forward(x)
        x = torch.sigmoid(x)
        return x.detach().numpy()



class FLP_Trainer_WOA(object):
    def __init__(self,models,data,opti='ADAM',verbose=1):
        """
        Trainer class for the centralized (WOA - without average which hints at the FedAvg algorithm in federated learning) models that are fitted as reference 
        during the experiments for horizontal federated learning setting.

        Parameters
        ----------
        model : PyTorch model
            Either log. regression or neural network model.
        data : torch.tensor
            Scaled data set from the Feature_Generator object.    
        opti : str, optional
            String that indicates which optimizer to use. Currently only works for SGD and ADAM.
        verbose : TYPE, optional
            Controls verbosity. If set to 1, intermediate results after each batch are printed to the console. 
            The default is 1.

        
        """
        self.models = models
        self.data = data
        self.opts = {}
        self.verbose = verbose
        
        # Define Optimizers
        if opti == 'SGD':
            for key in self.models.keys():
                self.opts[key] = torch.optim.SGD(self.models[key].parameters(),lr=self.models[key].lr,momentum=self.models[key].momentum)
        elif opti == 'ADAM':
            for key in self.models.keys():
                self.opts[key] = torch.optim.Adam(self.models[key].parameters(),lr=self.models[key].lr,weight_decay=self.models[key].weight_decay)
        else:
            raise Exception("Currently only SGD or ADAM applicable as optimizers.")
        
        self.LOGLOSS_batch = {key:{} for key in self.models.keys()}
        self.LOGLOSS_epoch = {key:{'train':{},'test':{}} for key in self.models.keys()}
        self.ROCAUC_epoch = {key:{'train':{},'test':{}} for key in self.models.keys()}
        self.PRAUC_epoch = {key:{'train':{},'test':{}} for key in self.models.keys()}
        self.BRIER_epoch = {key:{'train':{},'test':{}} for key in self.models.keys()}
        
        
    def train(self,num_epochs,batch_size):
        """
        Steers the training process of the centralized models and stores the results in the corresponding dictionaries.

        Parameters
        ----------
        num_epochs : int
            Number of epochs.
        batch_size : int
            Batch size to use during model training.

        Returns
        -------
        Fitted model and training results in the form of dictionaries (see above)

        """
        start = time.time()
        for key in self.models.keys():
            self.models[key].train() # set models into training mode
        
        ## DESIGN TRANING PROCESS FIRST OUTSIDE A LOOP -> THEN PUT THAT INTO A LOOP
        ds_dict = {}
        dl_dict = {}
        for key in self.models.keys():
            ds_dict[key] = torch.utils.data.TensorDataset(self.data[key]['fed_xtrain_data'], self.data[key]['fed_ytrain_data'])
            dl_dict[key] = torch.utils.data.DataLoader(ds_dict[key], batch_size=batch_size, shuffle=True, drop_last= True)

        # Epoch Loop
        for epoch in range(num_epochs):
            start_epoch = time.time()
            batch_loss = {key:[] for key in self.models.keys()}
            
            # MODEL LOOP
            yhat_train_dict = {}
            yhat_test_dict = {}
            log_loss_train_dict = {}
            log_loss_test_dict = {}
            roc_auc_train_dict = {}
            roc_auc_test_dict = {}    
            pr_auc_train_dict = {}
            pr_auc_test_dict = {}    
            brier_train_dict = {}
            brier_test_dict = {}    
                
            for key in self.models.keys():
                # BATCH LOOP 
                for batch_idx,(x,y) in enumerate(dl_dict[key]):
                    self.opts[key].zero_grad()
                    scores = self.models[key].forward(x.float())
                    loss = F.binary_cross_entropy_with_logits(input=scores,
                                                              target=y.reshape(-1,1),
                                                              reduction='mean')

                    batch_loss[key].append(loss.detach().item())

                    loss.backward()
                    self.opts[key].step()
                    # END BATCH LOOP
            
                # ADD LISTS OF BATCH LOSSES TO EPOCH LIST
                self.LOGLOSS_batch[key][f'epoch{epoch+1}'] = batch_loss[key]
                #self.LOGLOSS_batch[key][f'epoch{epoch+1}'] = batch_loss[key]
                #self.LOGLOSS_batch[key][f'epoch{epoch+1}'] = batch_loss[key]

                # MAKE PREDICTIONS
                yhat_train_dict[key] = self.models[key].predict(self.data[key]['fed_xtrain_data'])
                yhat_test_dict[key] = self.models[key].predict(self.data[key]['fed_xtest_data'])

                # EVALUATIONS
                ## 1. Log Loss
                log_loss_train_dict[key] = log_loss(self.data[key]['fed_ytrain_data'],yhat_train_dict[key],eps=0.000001)
                log_loss_test_dict[key] = log_loss(self.data[key]['fed_ytest_data'],yhat_test_dict[key],eps=0.000001)
                
                self.LOGLOSS_epoch[key]['train'][f'epoch{epoch+1}'] = log_loss_train_dict[key]
                self.LOGLOSS_epoch[key]['test'][f'epoch{epoch+1}'] = log_loss_test_dict[key]
                
                ## 2. ROC-AUC 
                roc_auc_train_dict[key] = roc_auc_score(self.data[key]['fed_ytrain_data'],yhat_train_dict[key])
                roc_auc_test_dict[key] = roc_auc_score(self.data[key]['fed_ytest_data'],yhat_test_dict[key])
                
                self.ROCAUC_epoch[key]['train'][f'epoch{epoch+1}'] = roc_auc_train_dict[key]
                self.ROCAUC_epoch[key]['test'][f'epoch{epoch+1}'] = roc_auc_test_dict[key]
                
                # 3. PR-AUC
                precision_train, recall_train, _ = precision_recall_curve(self.data[key]['fed_ytrain_data'],yhat_train_dict[key]) 
                precision_test, recall_test, _ = precision_recall_curve(self.data[key]['fed_ytest_data'],yhat_test_dict[key]) 
                
                pr_auc_train_dict[key] = auc(recall_train,precision_train)
                pr_auc_test_dict[key] = auc(recall_test,precision_test)
                
                self.PRAUC_epoch[key]['train'][f'epoch{epoch+1}'] = pr_auc_train_dict[key]
                self.PRAUC_epoch[key]['test'][f'epoch{epoch+1}'] = pr_auc_test_dict[key]
                
                # 4. BRIER-Score
                brier_train_dict[key] = brier_score_loss(self.data[key]['fed_ytrain_data'],yhat_train_dict[key])
                brier_test_dict[key] = brier_score_loss(self.data[key]['fed_ytest_data'],yhat_test_dict[key])
                
                self.BRIER_epoch[key]['train'][f'epoch{epoch+1}'] = brier_train_dict[key]
                self.BRIER_epoch[key]['test'][f'epoch{epoch+1}'] = brier_test_dict[key]
                
                # END MODEL LOOP
                
            end_epoch = time.time()
            if self.verbose == 1:
                print(80*'=')
                print(f"Progress: {epoch+1}/{num_epochs}")
                print(f"\t Avg. LOG-LOSS on epoch {epoch+1}:")
                print("\t\t-> TRAIN: " + ", ".join(str(round(val,4)) for val in log_loss_train_dict.values()))
                print("\t\t-> TEST : " + ", ".join(str(round(val,4)) for val in log_loss_test_dict.values()))
                print(f"\t ROC-AUC on epoch {epoch+1}:")
                print("\t\t-> TRAIN: " + ", ".join(str(round(val,4)) for val in roc_auc_train_dict.values()))
                print("\t\t-> TEST : " + ", ".join(str(round(val,4)) for val in roc_auc_test_dict.values()))
                print(f"\t Execution time for epoch {epoch+1}: {(end_epoch - start_epoch)/60:.2f} min.")
            else:
                pass
            # END EPOCH LOOP
        end = time.time()
        print(f'Total Training Time: {(end-start)/60:.2f} min.')
        
        
class FLC_Trainer_CKKS(object):
    def __init__(self,models,data,poly_mod_degree = 16384, coeff_mod_bit_sizes = [60,60,60,60],global_scale=2 ** 60,opti='ADAM',verbose=1):
        """
        Trainer class for the horizontal federated models that are fitted using homomorphic encryption (CKKS-scheme) for privacy protection. 
        
        Parameters
        ----------
        models : dict
            Contains the models per federatin client.
        data : list
            List of scaled data sets from the Feature_Generator object.    
        poly_mod_degree : int, optional
            Polynomial modulo degree. The default is 16384.
        coeff_mod_bit_sizes : TYPE, optional
            Bit sizes for modulo coefficients. The default is [60,60,60,60].
        global_scale : TYPE, optional
            Controls accuracy of the encrpyted parameters. The default is 2 ** 60 (and the maximum).
        opti : str, optional
            String that indicates which optimizer to use. Currently only works for SGD and ADAM.
        verbose : TYPE, optional
            Controls verbosity. If set to 1, intermediate results after each batch are printed to the console. 
            The default is 1.

        Returns
        -------
        None.

        """
        self.models = models
        self.data = data
        self.opts = {}
        self.verbose = verbose
        self.ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
        self.ctx_eval.global_scale = global_scale
        self.ctx_eval.generate_galois_keys()
        self.secret_key = self.ctx_eval.secret_key()
        self.ctx_eval.make_context_public()
        
        # Define Optimizers
        if opti == 'SGD':
            for key in self.models.keys():
                self.opts[key] = torch.optim.SGD(self.models[key].parameters(),lr=self.models[key].lr,momentum=self.models[key].momentum)
        elif opti == 'ADAM':
            for key in self.models.keys():
                self.opts[key] = torch.optim.Adam(self.models[key].parameters(),lr=self.models[key].lr,weight_decay=self.models[key].weight_decay)
        else:
            raise Exception("Currently only SGD or ADAM applicable as optimizers.")
        
        self.LOGLOSS_batch = {key:{} for key in self.models.keys()}
        self.LOGLOSS_epoch = {key:{'train':{},'test':{}} for key in self.models.keys()}
        self.LOGLOSS_epoch_agg = {key:{'train':{},'test':{}} for key in self.models.keys()}
        self.ROCAUC_epoch = {key:{'train':{},'test':{}} for key in self.models.keys()}
        self.ROCAUC_epoch_agg = {key:{'train':{},'test':{}} for key in self.models.keys()}
        self.PRAUC_epoch = {key:{'train':{},'test':{}} for key in self.models.keys()}
        self.PRAUC_epoch_agg = {key:{'train':{},'test':{}} for key in self.models.keys()}
        self.BRIER_epoch = {key:{'train':{},'test':{}} for key in self.models.keys()}
        self.BRIER_epoch_agg = {key:{'train':{},'test':{}} for key in self.models.keys()}
        
        
    def train(self,num_epochs,batch_size):
        """
        Steers the training process of the centralized models and stores the results in the corresponding dictionaries.

        Parameters
        ----------
        num_epochs : int
            Number of epochs.
        batch_size : int
            Batch size to use during model training.

        Returns
        -------
        Fitted model and training results in the form of dictionaries (see above).

        """
        start = time.time()
        for key in self.models.keys():
            self.models[key].train() # set models into training mode
        
        # Epoch Loop        
        for epoch in range(num_epochs):
            start_epoch = time.time()
            batch_loss = {key:[] for key in self.models.keys()}
            fed_params_dict = {}
            for mod in self.models.keys():
                fed_params_dict[mod] = {f"l{lay+1}":{} for lay,_ in enumerate(self.models[mod].layer_list)}

            # MODEL LOOP
            ds_dict = {}
            dl_dict = {}
            yhat_train_dict = {}
            yhat_test_dict = {}
            log_loss_train_dict = {}
            log_loss_test_dict = {}
            roc_auc_train_dict = {}
            roc_auc_test_dict = {}    
                
            for key in self.models.keys():
                
                ds_dict[key] = torch.utils.data.TensorDataset(self.data[key]['fed_xtrain_data'], self.data[key]['fed_ytrain_data'])
                dl_dict[key] = torch.utils.data.DataLoader(ds_dict[key], batch_size=batch_size, shuffle=True, drop_last= True)

                # BATCH LOOP 
                for batch_idx,(x,y) in enumerate(dl_dict[key]):
                    self.opts[key].zero_grad()
                    scores = self.models[key].forward(x.float())
                    loss = F.binary_cross_entropy_with_logits(input=scores,
                                                              target=y.reshape(-1,1),
                                                              reduction='mean')

                    batch_loss[key].append(loss.detach().item())

                    loss.backward()
                    self.opts[key].step()
                    # END BATCH LOOP
            
                # ADD LISTS OF BATCH LOSSES TO EPOCH LIST
                self.LOGLOSS_batch[key][f'epoch{epoch+1}'] = batch_loss[key]
                
                # MAKE PREDICTIONS
                yhat_train_dict[key] = self.models[key].predict(self.data[key]['fed_xtrain_data'])
                yhat_test_dict[key] = self.models[key].predict(self.data[key]['fed_xtest_data'])

                # EVALUATIONS
                ## 1. Log Loss
                log_loss_train_dict[key] = log_loss(self.data[key]['fed_ytrain_data'],yhat_train_dict[key],eps=0.000001)
                log_loss_test_dict[key] = log_loss(self.data[key]['fed_ytest_data'],yhat_test_dict[key],eps=0.000001)
                
                self.LOGLOSS_epoch[key]['train'][f'epoch{epoch+1}'] = log_loss_train_dict[key]
                self.LOGLOSS_epoch[key]['test'][f'epoch{epoch+1}'] = log_loss_test_dict[key]
                
                ## 2. ROC-AUC 
                roc_auc_train_dict[key] = roc_auc_score(self.data[key]['fed_ytrain_data'],yhat_train_dict[key])
                roc_auc_test_dict[key] = roc_auc_score(self.data[key]['fed_ytest_data'],yhat_test_dict[key])
                
                self.ROCAUC_epoch[key]['train'][f'epoch{epoch+1}'] = roc_auc_train_dict[key]
                self.ROCAUC_epoch[key]['test'][f'epoch{epoch+1}'] = roc_auc_test_dict[key]

                # Encrpyt Model Weights and Biases
                with torch.no_grad():
                    for mod in fed_params_dict.keys():
                        for idx,lay in enumerate(fed_params_dict[mod].keys()):
                            # Encryption of model weights
                            num_chunks = int(np.ceil(self.models[mod].layer_list[idx].weight.flatten().shape[0] / 10_000))
                            weight_chunks = np.array_split(self.models[mod].layer_list[idx].weight.flatten().detach().numpy(),num_chunks)        
                            weight_list_crypt = [ts.ckks_vector(self.ctx_eval,weight_chunks[j]) for j in range(num_chunks)] 
                            fed_params_dict[mod][lay]['weights'] = weight_list_crypt #ts.ckks_vector(self.ctx_eval,self.models[mod].layer_list[idx].weight.flatten().tolist())
                            # Encryption of model biases
                            fed_params_dict[mod][lay]['bias'] = ts.ckks_vector(self.ctx_eval,self.models[mod].layer_list[idx].bias.tolist())
                # END MODEL LOOP
            
            # AGGREGATOR
            data_sizes = {k:self.data[k]['fed_xtrain_data'].shape[0] for k in self.data.keys()}
            full_data_size = sum([v for v in data_sizes.values()])
            avg_weights = {k : v/full_data_size for k,v in data_sizes.items()}
            
            # Encrypt weights for averaging
            avg_weights_crypt = {}
            for mod in avg_weights.keys():
                avg_weights_crypt[mod] = {f"l{lay+1}":{} for lay,_ in enumerate(self.models[mod].layer_list)}

            for mod in avg_weights_crypt.keys():
                for idx,lay in enumerate(avg_weights_crypt[mod].keys()):
                    # Encryption of averaging-weights for model weights
                    num_chunks = int(np.ceil(self.models[mod].layer_list[idx].weight.flatten().shape[0] / 10_000))
                    weight_chunks = np.array_split(self.models[mod].layer_list[idx].weight.flatten().detach().numpy(),num_chunks)        
                    avg_list_crypt = [ts.ckks_vector(self.ctx_eval,np.repeat(avg_weights[mod],len(weight_chunks[j]))) for j in range(num_chunks)]         
                    avg_weights_crypt[mod][lay]['weights'] = avg_list_crypt
                    # Encryption of averaging-weights for model biases
                    avg_weights_crypt[mod][lay]['bias'] = ts.ckks_vector(self.ctx_eval,np.repeat(avg_weights[mod],self.models[mod].layer_list[idx].bias.flatten().shape[0])) 
            
            
            # at AGGREGATOR: Aggregate Model Weights and Biases
            model_keys = [key for key in self.models.keys()]
            new_weights_crypt = {key:[] for key in fed_params_dict[model_keys[0]].keys()}
            new_bias_crypt = {key:[] for key in fed_params_dict[model_keys[0]].keys()}
            num_chunks_layer = {}
            with torch.no_grad():
                for mod in fed_params_dict.keys():
                    for lay in fed_params_dict[mod].keys():
                        # Weight Averaging
                        num_chunks_layer[lay] = len(fed_params_dict[mod][lay]['weights'])
                        mul_weight_list_crypt = [fed_params_dict[mod][lay]['weights'][j] * avg_weights_crypt[mod][lay]['weights'][j] for j in range(num_chunks_layer[lay])]
                        new_weights_crypt[lay].append(mul_weight_list_crypt)
                        # Bias Averaging
                        new_bias_crypt[lay].append(fed_params_dict[mod][lay]['bias'] * avg_weights_crypt[mod][lay]['bias'])

                updates_weights = {} # to be send to the federation participants by the aggregator
                updates_bias = [] # # to be send to the federation participants by the aggregator
                for lay in new_weights_crypt.keys():
                    num_chunks = num_chunks_layer[lay]
                    chunk_agg_list = []
                    for nc in range(num_chunks):
                        chunks = [new_weights_crypt[lay][mod][nc] for mod in range(len(new_weights_crypt[lay]))]
                        chunk_sum = sum(chunks)
                        chunk_agg_list.append(chunk_sum)
                    updates_weights[lay] = chunk_agg_list
                    updates_bias.append(sum(new_bias_crypt[lay]))
                    
            # at FEDERATION PARTICIPANTS: Update their local model parameters            
            with torch.no_grad():
                updates = []
                for idx,lay in enumerate(updates_weights.values()):
                    lay_weight_decrypt = list(chain.from_iterable([chunk.decrypt(self.secret_key) for chunk in lay]))
                    updates.append(lay_weight_decrypt)
                    lay_bias_decrypt = updates_bias[idx].decrypt(self.secret_key)
                    updates.append(lay_bias_decrypt)
                    
                for mod in self.models.keys():
                    for i,p in enumerate(self.models[mod].parameters()):
                        if len(p.shape) == 2:
                            shape0 = p.shape[0]
                            shape1 = p.shape[1]
                            p.copy_(torch.tensor(updates[i]).reshape(shape0,shape1))
                        else:
                            p.copy_(torch.tensor(updates[i]))
            
            
            # EVALUATION OF AGGREGATED MODEL ON ALL 3 DATA SETS
            yhat_train_dict_agg = {}
            yhat_test_dict_agg = {}
            log_loss_train_dict_agg = {}
            log_loss_test_dict_agg = {}
            roc_auc_train_dict_agg = {}
            roc_auc_test_dict_agg = {}
            prauc_train_dict_agg = {}
            prauc_test_dict_agg = {}
            brier_train_dict_agg = {}
            brier_test_dict_agg = {}
            
            for key in self.models.keys(): 
                # MAKE PREDICTIONS
                yhat_train_dict_agg[key] = self.models[key].predict(self.data[key]['fed_xtrain_data'])
                yhat_test_dict_agg[key] = self.models[key].predict(self.data[key]['fed_xtest_data'])

                # EVALUATIONS
                ## 1. Log Loss
                log_loss_train_dict_agg[key] = log_loss(self.data[key]['fed_ytrain_data'],yhat_train_dict_agg[key],eps=0.000001)
                log_loss_test_dict_agg[key] = log_loss(self.data[key]['fed_ytest_data'],yhat_test_dict_agg[key],eps=0.000001)
                
                self.LOGLOSS_epoch_agg[key]['train'][f'epoch{epoch+1}'] = log_loss_train_dict_agg[key]
                self.LOGLOSS_epoch_agg[key]['test'][f'epoch{epoch+1}'] = log_loss_test_dict_agg[key]
                
                ## 2. ROC-AUC 
                roc_auc_train_dict_agg[key] = roc_auc_score(self.data[key]['fed_ytrain_data'],yhat_train_dict_agg[key])
                roc_auc_test_dict_agg[key] = roc_auc_score(self.data[key]['fed_ytest_data'],yhat_test_dict_agg[key])
                
                self.ROCAUC_epoch_agg[key]['train'][f'epoch{epoch+1}'] = roc_auc_train_dict_agg[key]
                self.ROCAUC_epoch_agg[key]['test'][f'epoch{epoch+1}'] = roc_auc_test_dict_agg[key]
                
                # 3. PR-AUC
                precision_train, recall_train, _ = precision_recall_curve(self.data[key]['fed_ytrain_data'],yhat_train_dict_agg[key]) 
                precision_test, recall_test, _ = precision_recall_curve(self.data[key]['fed_ytest_data'],yhat_test_dict_agg[key]) 
                
                prauc_train_dict_agg[key] = auc(recall_train,precision_train)
                prauc_test_dict_agg[key] = auc(recall_test,precision_test)
                
                self.PRAUC_epoch_agg[key]['train'][f'epoch{epoch+1}'] = prauc_train_dict_agg[key]
                self.PRAUC_epoch_agg[key]['test'][f'epoch{epoch+1}'] = prauc_test_dict_agg[key]
                
                # 4. Brier Score
                brier_train_dict_agg[key] = brier_score_loss(self.data[key]['fed_ytrain_data'],yhat_train_dict_agg[key])
                brier_test_dict_agg[key] = brier_score_loss(self.data[key]['fed_ytest_data'],yhat_test_dict_agg[key])
                
                self.BRIER_epoch_agg[key]['train'][f'epoch{epoch+1}'] = brier_train_dict_agg[key]
                self.BRIER_epoch_agg[key]['test'][f'epoch{epoch+1}'] = brier_test_dict_agg[key]
                
                
            end_epoch = time.time()
            if self.verbose == 1:
                print(80*'=')
                print(f"Progress: {epoch+1}/{num_epochs}")
                print(f"\t Avg. LOG-LOSS on epoch {epoch+1}:")
                print("\t\t-> TRAIN: " + ", ".join(str(round(val,4)) for val in log_loss_train_dict_agg.values()))
                print("\t\t-> TEST : " + ", ".join(str(round(val,4)) for val in log_loss_test_dict_agg.values()))
                print(f"\t ROC-AUC on epoch {epoch+1}:")
                print("\t\t-> TRAIN: " + ", ".join(str(round(val,4)) for val in roc_auc_train_dict_agg.values()))
                print("\t\t-> TEST : " + ", ".join(str(round(val,4)) for val in roc_auc_test_dict_agg.values()))
                print(f"\t Execution time for epoch {epoch+1}: {(end_epoch - start_epoch)/60:.2f} min.")
            else:
                pass
            # END EPOCH LOOP
        end = time.time()
        print(f'Total Training Time: {(end-start)/60:.2f} min.')

class HFL_FedForest():
    def __init__(self,data,params=None,param_path=None,param_file_names=None,random_state=2021):
        """
        Model class for the horizontal federated forest. 

        Parameters
        ----------
        data : list
            List of scaled data sets from the Feature_Generator object.
        params : dict, optional
            Dictionary containing all hyperparameters that need to be set for the regular random forest (sklearn implementation). 
            The default is None.
        param_path : str, optional
            Path to the pickle-file that contains the tuned hyperparameters. The default is None.
        param_file_names : TYPE, optional
            Name of the pickle-file that contains the tuned hyperparameters. The default is None.
        random_state : TYPE, optional
            Sets random seed. The default is 2021.

        Returns
        -------
        None.

        """
        self.data = data
        self.random_state = random_state
        self.models = {}
        
        # Create context object for CKKS encryption 
        self.ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, 16384, -1, [60,60,60,60])
        self.ctx_eval.global_scale = 2**60
        self.ctx_eval.generate_galois_keys()
        self.secret_key = self.ctx_eval.secret_key()
        self.ctx_eval.make_context_public()
        
        for i in range(len(data)):
            self.models[f'm{i+1}'] = RandomForestClassifier(**params[i],random_state=self.random_state)
        self.models['mglobal'] = None
            
        self.LOGLOSS = {key:{'train':None,'test':None} for key in self.models.keys()}
        self.ROCAUC = {key:{'train':None,'test':None} for key in self.models.keys()}
        self.PRAUC = {key:{'train':None,'test':None} for key in self.models.keys()}
        self.BRIER = {key:{'train':None,'test':None} for key in self.models.keys()}
        
    def train(self):
        """
        Trains the horizontal federated forest using only using the class attributes that were specified upon initialization of the model class.

        Returns
        -------
        Fitted model and training results in the form of dictionaries (see above)

        """
        # @participants: fit local models
        for key in tqdm(self.data.keys()):
            self.models[key].fit(self.data[key]['fed_xtrain_data'], self.data[key]['fed_ytrain_data'])
        
        # @participants: after local model fitting, extract cutoffs of every decision tree in the local forest, encrypt those cutoffs and 
        #                replace them with NaN in the model copy that will be sent to the coordinator.
        models_copy = deepcopy([self.models[key] for key in self.data.keys()])
        forest_thr_list_crypt = []
        for i in range(len(models_copy)): #hfl_ff.models.keys():
            dt_thr_crypt_list = []
            for dt in tqdm(models_copy[i].estimators_):
                dt_thrs = deepcopy(dt.tree_.threshold)
                dt.tree_.threshold[:] = np.nan
                
                dt_thrs_crypt = ts.ckks_vector(self.ctx_eval,dt_thrs)
                dt_thr_crypt_list.append(dt_thrs_crypt)
            forest_thr_list_crypt.append(dt_thr_crypt_list) # per participant: send dt_thr_crypt_list to coordinator (creates forest_thr_list_crypt)


        # @ coordinator: shuffle local forest structures and combine to one global forest. 
        random.seed(self.random_state)
        random.shuffle(models_copy)
        fedforest = deepcopy(models_copy[0])
        
        for i in range(1,len(models_copy)):
            fedforest.estimators_ += models_copy[i].estimators_
            fedforest.n_estimators += models_copy[i].n_estimators
        
        # Also shuffle the corresponding tree thresholds (forest_thr_list_crypt) -> have to have the same order as the local forest when they were combined.
        random.seed(self.random_state)
        random.shuffle(forest_thr_list_crypt)
        forest_thrs_crypt = list(chain(*forest_thr_list_crypt))
        # send forest_thr_list_crypt and fedforest over to each participant.
        print("Start Participant Role")
        # @participants: decrypt cutoffs in forest_thr_list_crypt and set the cutoffs in the global model in accordance with the recently decrypted cutoffs
        # Important: elements in forest_thr_list_crypt and the models in models_copy have to be in the same order
        for idx,thr in enumerate(forest_thrs_crypt):
            fedforest.estimators_[idx].tree_.threshold[:] = thr.decrypt(self.secret_key)
        
        self.models['mglobal'] = fedforest
        
        
        
    def evaluate(self,data_test=None):
        if data_test is None:
            for key in tqdm(self.data.keys()):
                yhat_train = self.models[key].predict_proba(self.data[key]['fed_xtrain_data'])[:,1]
                yhat_train_global = self.models['mglobal'].predict_proba(self.data[key]['fed_xtrain_data'])[:,1]
                yhat_test = self.models[key].predict_proba(self.data[key]['fed_xtest_data'])[:,1]
                yhat_test_global = self.models['mglobal'].predict_proba(self.data[key]['fed_xtest_data'])[:,1]
                
                # LOG-LOSS
                self.LOGLOSS[key]['train'] = log_loss(self.data[key]['fed_ytrain_data'],yhat_train,eps=0.000001)
                self.LOGLOSS[key]['test'] = log_loss(self.data[key]['fed_ytest_data'],yhat_test,eps=0.000001)
                self.LOGLOSS[key]['train_fl'] = log_loss(self.data[key]['fed_ytrain_data'],yhat_train_global,eps=0.000001)
                self.LOGLOSS[key]['test_fl'] = log_loss(self.data[key]['fed_ytest_data'],yhat_test_global,eps=0.000001)
                
                # ROC-AUC
                self.ROCAUC[key]['train'] = roc_auc_score(self.data[key]['fed_ytrain_data'] , yhat_train)
                self.ROCAUC[key]['test'] = roc_auc_score(self.data[key]['fed_ytest_data'] , yhat_test)
                self.ROCAUC[key]['train_fl'] = roc_auc_score(self.data[key]['fed_ytrain_data'] , yhat_train_global)
                self.ROCAUC[key]['test_fl'] = roc_auc_score(self.data[key]['fed_ytest_data'] , yhat_test_global)
                
                
                # PR-AUC
                precision_train, recall_train, _ = precision_recall_curve(self.data[key]['fed_ytrain_data'],yhat_train) 
                pr_auc_train = auc(recall_train,precision_train)
                precision_train_global, recall_train_global, _ = precision_recall_curve(self.data[key]['fed_ytrain_data'],yhat_train_global) 
                pr_auc_train_global = auc(recall_train_global,precision_train_global)
                
                precision_test, recall_test, _ = precision_recall_curve(self.data[key]['fed_ytest_data'],yhat_test) 
                pr_auc_test = auc(recall_test,precision_test)
                precision_test_global, recall_test_global, _ = precision_recall_curve(self.data[key]['fed_ytest_data'],yhat_test_global) 
                pr_auc_test_global = auc(recall_test_global,precision_test_global)
                
                self.PRAUC[key]['train'] = pr_auc_train
                self.PRAUC[key]['test'] = pr_auc_test
                self.PRAUC[key]['train_fl'] = pr_auc_train_global
                self.PRAUC[key]['test_fl'] = pr_auc_test_global
                
                # BRIER-Score
                self.BRIER[key]['train'] = brier_score_loss(self.data[key]['fed_ytrain_data'],yhat_train)
                self.BRIER[key]['test'] = brier_score_loss(self.data[key]['fed_ytest_data'],yhat_test)
                self.BRIER[key]['train_fl'] = brier_score_loss(self.data[key]['fed_ytrain_data'],yhat_train_global)
                self.BRIER[key]['test_fl'] = brier_score_loss(self.data[key]['fed_ytest_data'],yhat_test_global)
                
                
        else:
            for key in self.data.keys():
                yhat = self.models[key].predict_proba(data_test['fed_xtest_data'])[:,1]
                yhat_global = self.models['mglobal'].predict_proba(data_test['fed_xtest_data'])[:,1]
                
                # LOG-LOSS
                self.LOGLOSS[key]['oos'] = log_loss(data_test['fed_ytest_data'],yhat,eps=0.000001)
                self.LOGLOSS[key]['oos_fl'] = log_loss(data_test['fed_ytest_data'],yhat_global,eps=0.000001)
                
                # ROC-AUC
                self.ROCAUC[key]['oos'] = roc_auc_score(data_test['fed_ytest_data'] , yhat)
                self.ROCAUC[key]['oos_fl'] = roc_auc_score(data_test['fed_ytest_data'] , yhat_global)
                
                # PR-AUC
                precision, recall, _ = precision_recall_curve(data_test['fed_ytest_data'],yhat) 
                precision_global, recall_global, _ = precision_recall_curve(data_test['fed_ytest_data'],yhat_global) 
                
                pr_auc = auc(recall,precision)
                pr_auc_global = auc(recall_global,precision_global)
                
                self.PRAUC[key]['oos'] = pr_auc
                self.PRAUC[key]['oos_fl'] = pr_auc_global
                
                # BRIER-Score
                self.BRIER[key]['oos'] = brier_score_loss(data_test['fed_ytest_data'],yhat)
                self.BRIER[key]['oos_fl'] = brier_score_loss(data_test['fed_ytest_data'],yhat_global)
    
    
                
