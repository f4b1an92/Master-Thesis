import sys
import numpy as np
import pandas as pd
from itertools import chain
from itertools import product
from functools import partial
from functools import reduce


import pickle
import yaml
import torch 
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim
from torch.utils.data import TensorDataset, DataLoader

import tenseal as ts
import random
import time
from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.metrics import auc,roc_auc_score,log_loss,f1_score,brier_score_loss,precision_recall_curve
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

sys.path.append(r'C:\Users\Fabian\Desktop\Studium\2.) HU Berlin\6.) SS_21\Master Thesis\Federated Learning for Credit Risk\Code')

from feature_gen import *
from hfed_models import *


#================================
# Model Definition
#================================

# Vertical Log. Regression
#==========================
class FedLogReg(torch.nn.Module):
    def __init__(self,input_size,momentum=0.9,lr=0.001,random_state=2021,bias=True,init='kaiming'):
        """
        Pytorch model that implements logistic regression for the federated (vertical) setting.

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
    
    
# SplitNN
#=====================    
class SplitNN(torch.nn.Module):
    def __init__(self,layer_sizes,lr=0.001,momentum=0,weight_decay=0,drop_rate=0.25,init='kaiming',bias=True,random_state=2021):
        """
        Pytorch model that implements a MLP-neural network for the non-federated (i.e. centralized) setting.

        Parameters
        ----------
        layer_sizes : list
            List that describes the sizes of the hidden layers in the network.
        lr : float, optional
            Sets the learning rate for the optimizer. The default is 0.001.
        momentum : float, optional
            Defines the momentum parameter for the SGD optimizer. Check PyTorch documentation for further details.
            The default is 0.
        weight_decay : float, optional
            Defines the weight_decay parameter for the ADAM optimizer. Used to control overfitting. 
            Check PyTorch documentation for further details.
            The default is 0.
        drop_rate : float, optional
            Indicates how many percent of neurons should be deactivated for a given forward pass during training. 
            Is used to control overfitting but could impair the learning if not set carefully.
            The default is 0.25.
        init : str, optional
            Defines how initialize the model weights. The default is 'kaiming'.
        bias : bool, optional
            Indicates if bias should be used during the fitting process. The default is True.
        random_state : int, optional
            Sets random seed. The default is 2021.
        
        Returns
        -------
        None.

        """
        super(SplitNN,self).__init__()
        self.layer_sizes = layer_sizes
        self.layer_list = torch.nn.ModuleList([])
        self.lr = lr
        self.momentum = momentum 
        self.weight_decay = weight_decay
        self.dropout = torch.nn.Dropout(drop_rate)
        self.init = init
        self.bias = bias
        self.random_state = 2021
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Network Structure
        for i in np.arange(1,len(self.layer_sizes)):
            self.layer_list.append(torch.nn.Linear(self.layer_sizes[i-1],self.layer_sizes[i],bias=self.bias))
            if self.init == 'kaiming':
                torch.nn.init.kaiming_normal_(self.layer_list[i-1].weight)
            elif self.init == 'uniform':
                torch.nn.init.uniform_(self.layer_list[i-1].weight)
            else:
                raise Exception("The argument 'init' must either be set to 'kaiming' or 'uniform'.")
                
        
    def forward(self,x):
        self.input = []
        self.output = []
        
        for layer in self.layer_list:
            # detach from prev. history
            x = Variable(x.data,requires_grad=True)
            self.input.append(x)
            # compute output
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)
            # add output to list of outputs
            self.output.append(x)
        return x
    
    def forward_last_model(self,x):
        self.input = []
        self.output = []
        
        for layer in self.layer_list:
            x = Variable(x.data,requires_grad=True)
            self.input.append(x)
            x = layer(x)
            self.output.append(x)
            return x
        
    def predict(self,x):
        logits = self.forward_last_model(x)
        logits = Variable(logits.data,requires_grad=True)
        
        self.input.append(logits)
        preds = torch.sigmoid(logits)
        self.output.append(preds)
        return preds
    
    
    def evaluate(self,x,y,red='none'):
        logits = self.forward_last_model(x)
        logits = Variable(logits.data,requires_grad=True)
        
        self.input.append(logits)
        preds = torch.sigmoid(logits)
        self.output.append(preds)
        
        preds = Variable(preds.data,requires_grad=True)
        self.input.append(preds)
        crit = torch.nn.BCELoss(reduction=red)
        loss = crit(preds.flatten(), y.flatten()).reshape(-1,1)
        self.output.append(loss)
        return loss    
       
    
    def backward(self, g):
        for i, output in reversed(list(enumerate(self.output))):
            #print(i)
            if i == (len(self.output) - 1):
                # for node that receives gradients from downstream network (i.e. from the coordinator)
                output.backward(g,retain_graph=True)
            else:
                output.backward(self.input[i+1].grad.data,retain_graph=True)
                
                
                
# Federated Tree
#=====================
class Fed_DecisionTree():
    def __init__(
            self,
            num_parties,
            target_values,
            id_col_name,
            max_depth,
            min_samples_split=10,
            min_impurity_decrease=0.0001,
            crit='entropy',
            encryption_context = None,
            secret_key = None,
            random_state=2021,
        ):
        """
        Implementaion of a decision tree classifier that is tailored for vertical federated learning. For data splitting it uses
        the sklearn implementation of the regular CART decision tree classifier. 

        Parameters
        ----------
        num_parties : int
            NUmber of federation clients.
        target_values : list
            List that contains unique target values.
        id_col_name : TYPE
            Name of the ID column in the data set.
        max_depth : TYPE
            Maximum depth a decision tree is allowed to grow. Supposed to control overfitting
        min_samples_split : TYPE, optional
            Minimum number of samples in a node to make another split. The default is 10.
        min_impurity_decrease : TYPE, optional
            Minimum requirement a new split. The default is 0.0001.
        crit : TYPE, optional
            Criterium to measure node impurity. The default is 'entropy'.
        encryption_context : TYPE, optional
            TenSeal context object for encryption and decryption. The default is None.
        secret_key : TYPE, optional
            Secret key of a TenSeal context object. The default is None.
        random_state : TYPE, optional
            Sets random seed. The default is 2021.
         
        Returns
        -------
        None.

        """
        self.sub_tree_list = [{} for i in range(num_parties)]
        self.leaf_dict = {}
        self.target_values = target_values
        self.id_col_name = id_col_name
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.crit = crit
        self.random_state = random_state
        self.cnt = 0
        self.depth = 0 
        
        # Encryption Scheme (CKKS)
        if (encryption_context is None) & (secret_key is None):
            self.ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, 16_384, -1, [60,60,60,60])
            self.ctx_eval.global_scale = 2 ** 60
            self.ctx_eval.generate_galois_keys()
            self.secret_key = self.ctx_eval.secret_key()
            self.ctx_eval.make_context_public()
        else:
            self.ctx_eval = encryption_context
            self.secret_key = secret_key

        
    def fit(self,X,y,depth=0,tree_path='o'):
        self.cnt += 1
        
        # 1. Check stopping criterias
        # 1.1 pure node -> terminate
        if len(np.unique(y)) == 1:
            self.leaf_dict[f'n{self.cnt}'] = {'tree_path':tree_path,'value':int(self._mode(y)), 'p_dist':self._probabilities(y)}
            return self._final_node(y,depth=depth,tree_path=tree_path)
        
        # 1.2. max_depth
        elif depth >= self.max_depth:
            self.leaf_dict[f'n{self.cnt}'] = {'tree_path':tree_path,'value':int(self._mode(y)), 'p_dist':self._probabilities(y)}
            return self._final_node(y,depth=depth,tree_path=tree_path)
        
        # 1.3. min_samples_split
        elif X[0].shape[0] < self.min_samples_split: # it is sufficient to only check X[0] since all data sets in X have the sample ids
            self.leaf_dict[f'n{self.cnt}'] = {'tree_path':tree_path,'value':int(self._mode(y)), 'p_dist':self._probabilities(y)}
            return self._final_node(y,depth=depth,tree_path=tree_path)
        
        else:
            # @ all participants: 
            clf_list = []
            split_info_dict = {}
            col_mask_list = []
            for i in range(len(X)):
                # Fit Decision Tree
                participant_result_list = []
                col_mask_list.append(X[i].columns != self.id_col_name)
                clf = DecisionTreeClassifier(random_state=self.random_state + self.cnt,max_depth=1,criterion=self.crit)
                clf.fit(X[i].loc[:,col_mask_list[i]],y)
                clf_list.append(clf)
                
                # Measure IG per Participant & prepare the info for the coordinator (incl. weights & impurity measures)
                if len(clf.tree_.n_node_samples) == 1: # Sklearn's decision tree was unable to find a split given this feature subset
                    w1 = 0.5
                    w2 = 0.5
                    impurity_parent = clf.tree_.impurity[0]
                    impurity_left = clf.tree_.impurity[0]
                    impurity_right = clf.tree_.impurity[0]
                    ig = 0
                    participant_result_list.extend([impurity_parent,impurity_left,impurity_right,w1,w2,ig])
                    
                    # Also provide the split feature ID (local ID not global ID)
                    participant_result_list.append(1) 
                   
                else:
                    w1 = clf.tree_.n_node_samples[1]/clf.tree_.n_node_samples[0]
                    w2 = clf.tree_.n_node_samples[2]/clf.tree_.n_node_samples[0]
                    impurity_parent = clf.tree_.impurity[0]
                    impurity_left = clf.tree_.impurity[1]
                    impurity_right = clf.tree_.impurity[2] 
                    ig = impurity_parent - w1 * impurity_left - w2 * impurity_right
                    participant_result_list.extend([impurity_parent,impurity_left,impurity_right,w1,w2,ig])
                    
                    # Also provide the split feature ID (local ID not global ID)
                    participant_result_list.append(clf.tree_.feature[0]+1) # +1 because id-column is dropped during decision tree fitting
                    
                # Encrypt the participant result list
                crypt_list = ts.ckks_vector(self.ctx_eval,participant_result_list)
                split_info_dict[f'p{i}'] = crypt_list
                
            # @ Coordinator:
            # 1. Decrypt splitting info provided by the federation participants
            impurity_parent_list = [val.decrypt(self.secret_key)[0] for val in split_info_dict.values()]
            ig_list = [val.decrypt(self.secret_key)[5] for val in split_info_dict.values()]
            feature_id_list = [int(round(val.decrypt(self.secret_key)[-1],0)) for val in split_info_dict.values()]
            
            # 2. Determine DT which brought the highest IG 
            best_dt_idx = np.argmax(ig_list)
            
            # 3. Check another stopping criteria: Minimum Impurity Decrease
            if ig_list[best_dt_idx] <= self.min_impurity_decrease:
                self.leaf_dict[f'n{self.cnt}'] = {'tree_path':tree_path,'value':int(self._mode(y)), 'p_dist':self._probabilities(y)}
                return self._final_node(y,depth=depth,tree_path=tree_path)
            
            #print(feature_id_list)
            #print(best_dt_idx)
            #print(clf_list[best_dt_idx].tree_.threshold)
            # @ Participant which provided the best split (notified by coordinator):
            # Generate indices for further use down the tree & send to coordinator w/o encryption (ID info will be provided by coordinator to other participants)            
            bool_left = X[best_dt_idx].iloc[:,feature_id_list[best_dt_idx]] <= clf_list[best_dt_idx].tree_.threshold[0]
            bool_right = X[best_dt_idx].iloc[:,feature_id_list[best_dt_idx]] >  clf_list[best_dt_idx].tree_.threshold[0]
            
            leftids = X[best_dt_idx].loc[bool_left,self.id_col_name].values 
            rightids = X[best_dt_idx].loc[bool_right,self.id_col_name].values 
            
            # @ Coordinator: Add relevant infos to self.global_tree_dict 
            global_tree_dict = { 
                'node':int(self.cnt),
                'tree_path':tree_path,
                'leaf':False,
                'tree_depth':int(depth),
                #'split_log': self._split_log(y), # needs to be dropped in final version -> exclusive to federation pariticpant with best split at this node
                'impurity':float(impurity_parent_list[best_dt_idx]),
                'p_id': int(best_dt_idx), # participant id
                'f_id': int(feature_id_list[best_dt_idx]), # feature id
                #'cutoff': float(clf_list[best_dt_idx].tree_.threshold[0]), # needs to be dropped in final version -> exclusive to federation pariticpant with best split at this node
                'left_ids':leftids, # are transmitted by participant with the best split
                'right_ids':rightids, # are transmitted by participant with the best split
                
            }
            
            # @ Participant which provided the best split: Save split info for the current node in the tree
            sub_tree_dict_inp = {
                'tree_path':tree_path,
                'leaf':False,
                'tree_depth':int(depth),
                'impurity':float(impurity_parent_list[best_dt_idx]),
                'f_id': int(feature_id_list[best_dt_idx]), # feature id
                'cutoff': float(clf_list[best_dt_idx].tree_.threshold[0]),
                #'left_ids':leftids,
                #'right_ids':rightids,
                'split_log': self._split_log(y) # reflects the actual split at that node as if the tree was grown fully at this participant (works b/c split data set is passed recursively with each .fit-call)
                # 'value' & 'p_dist' don't need to be stored at the participant since those are provided by the coordinator anyway
            }
            self.sub_tree_list[best_dt_idx][f'n{self.cnt}'] = sub_tree_dict_inp # self.sub_tree_list is only a vessel to save subtrees in one place, in reality this object wouldn't exist since the subtrees are stored locally
            
            
            # Grow the tree further (recursively) -> this is now done again by every participant i.e. start of a new round
            X_left = [X[i].loc[X[i][self.id_col_name].isin(leftids).values,:] for i in range(len(X))] 
            X_right = [X[i].loc[X[i][self.id_col_name].isin(rightids).values,:] for i in range(len(X))] 
            y_left = y.loc[y.index.isin(leftids)]
            y_right = y.loc[y.index.isin(rightids)]
            
            global_tree_dict['left'] = self.fit(X_left,y_left, depth = depth + 1,tree_path = tree_path + 'l')
            global_tree_dict['right'] = self.fit(X_right,y_right, depth = depth + 1, tree_path = tree_path + 'r')
            
            self.fitted_tree = global_tree_dict
            
            return global_tree_dict
        
    
    def predict(self,X,pred_type='prob',drop_idx=True):
        """
        Collaborative prediction round across all federation clients for the fed. decision tree classifier.

        Parameters
        ----------
        X : list
            List containing the features.
        pred_type : TYPE, optional
            Indicates whether probabilities or labels should be predicted. The default is 'prob'.
        drop_idx : TYPE, optional
            Clients don't save indices locally. The default is True.

        final_pred : numpy.array
            numpy array containg the prediction results.

        """
        pred_idx_dict = {f'p{k}':{} for k in range(len(self.sub_tree_list))}
        for part in range(len(self.sub_tree_list)): 
            for leaf in self.leaf_dict.keys():
                id_list_for_leaves = []
                # START LOOP 3
                for key in self.sub_tree_list[part].keys():
                    if self.sub_tree_list[part][key]['tree_path'] in self.leaf_dict[leaf]['tree_path']: # here: key == n14
                        split_mask = X[part].iloc[:,self.sub_tree_list[part][key]['f_id']] <= self.sub_tree_list[part][key]['cutoff']
                                    
                        lids = X[part].loc[split_mask,self.id_col_name].values
                        rids = X[part].loc[~split_mask,self.id_col_name].values
                    
                        if self.sub_tree_list[part][key]['tree_path'] + 'l' in self.leaf_dict[leaf]['tree_path']:
                            id_list_for_leaves.append(lids)
                        else:
                            id_list_for_leaves.append(rids) 
                # END LOOP 3
                
                # Get intersection of all subtree splits possible at the respective participant to form the ID set at the corresponding leaf node 
                if len(id_list_for_leaves) == 0:
                    pred_idx_dict[f'p{part}'][leaf] = X[part].loc[:,self.id_col_name].values
                elif len(id_list_for_leaves) == 1:
                    pred_idx_dict[f'p{part}'][leaf] = id_list_for_leaves # change 0 to part
                elif len(id_list_for_leaves) == 2:
                    pred_idx_dict[f'p{part}'][leaf] = np.intersect1d(*id_list_for_leaves) # change 0 to part
                else:
                    pred_idx_dict[f'p{part}'][leaf] = reduce(np.intersect1d, (id_list_for_leaves)) # change 0 to part
            # END LOOP 2
        # END LOOP 1
        
        # @ Coordinator: Get the intersection of all ID sets at every leaf node that were provided by each participant
        key_list = [key for key in pred_idx_dict.keys()]
        valkey_list = [key for key in pred_idx_dict[key_list[0]].keys()] # yields the keys of the nested dict (i.e. the node IDs)
        
        final_pred_list = []
        if pred_type == 'prob':
            pred_key = 'p_dist'
        elif pred_type == 'vote':
            pred_key = 'value'
        else:
            raise Exception("For the argument 'pred_type' only 'vote' or 'prob' are allowed. \n'Vote' translates into class predictions by majority vote & 'prob' into class probabilities.")
        for inner_key in valkey_list:
            final_leaf_node_ids = [pred_idx_dict[outer_key][inner_key] for outer_key in pred_idx_dict.keys()]
            idx_arr = reduce(np.intersect1d, (final_leaf_node_ids))
            idx_arr = idx_arr.reshape(-1,1)
            pred_arr = np.broadcast_to(self.leaf_dict[inner_key][pred_key],shape=(idx_arr.shape[0],len(self.target_values)))
            pred_arr = np.concatenate((idx_arr,pred_arr),axis=1)
            final_pred_list.append(pred_arr)
        
        final_pred = np.concatenate((final_pred_list),axis=0)
        if drop_idx == True:
            final_pred = final_pred[final_pred[:,0].argsort(),1:]
        else:
            final_pred = final_pred[final_pred[:,0].argsort()]
            
        return final_pred  

        
    def _final_node(self,y,depth,tree_path):
        """
        Create leaf node. 
        """
        leaf_node = {
            "node":int(self.cnt),
            "leaf":True,
            "tree_path":tree_path,
            "tree_depth":int(depth),
            "entropy": float(round(self._entropy(y), 4)),
            "split_log": self._split_log(y),
            "value": int(self._mode(y)),
            "p_dist":self._probabilities(y)
        }
        return leaf_node

    
    def _split_log(self, y):
        counts = [sum(y.values == self.target_values[i]) for i in range(len(self.target_values))]
        split_counts_list = [f"{self.target_values[i]}: {counts[i]}" for i in range(len(counts))]
        return " | ".join(split_counts_list)

    
    def _probabilities(self,y):
         """
         Class probabilities. 
         """
         freqs = [sum(y.values == self.target_values[i])/len(y) for i in range(len(self.target_values))]
         return freqs
    
    @staticmethod
    def _entropy(y):
        _,counts = np.unique(y,return_counts=True)
        p = counts / np.array(y).shape[0]
        e = sum([-i * np.log2(i) for i in p])
        return e
         
    @staticmethod    
    def _mode(y):
        """
        Majority vote with tie breaker.
        """
        values, counts = np.unique(y, return_counts=True)
        return int(np.random.choice(values[counts == np.max(counts)]))
        
    
    
# Federated Forest
#=====================
class FederatedForest():
    def __init__(
            self,
            num_parties,
            target_values,
            id_col_name,
            n_estimators,
            max_depth,
            max_features = 'sqrt',
            max_samples = 0.2,
            min_samples_split=10,
            min_impurity_decrease=0.0001,
            crit='entropy',
            random_state=2021
        ):
        """
        Implementaion of the vertical federated forest that uses the Fed_DecisionTree-class to construct 
        decison trees collectively inside the federation.

        Parameters
        ----------
        num_parties : int
            Number of federation clients.
        target_values : list
            Unique values of the target variable.
        id_col_name : str
            Name of the ID column.
        n_estimators : int
            Number of trees to grow for the forest.
        max_depth : int
            Maximum depth of the federated decision trees.
        max_features : str or float, optional
            Maximum number of features to be used for tree growining after sampling them randomly. 
            The default is 'sqrt' (see sklearn docu for decision tree).
        max_samples : float, optional
            Sample size to be used for finding the best split criteria. The default is 0.2.
        min_samples_split : int, optional
            Minimum number of samples in a node to make another split. The default is 10.
        min_impurity_decrease : float, optional
            Minimum requirement a new split. The default is 0.0001.
        crit : str, optional
            Criterium to measure node impurity. The default is 'entropy'.
        random_state : int, optional
            Sets random seed. The default is 2021.
        
        Returns
        -------
        None.

        """
        self.num_parties = num_parties
        self.sub_tree_list = [{} for i in range(self.num_parties)]
        self.leaf_dict = {}
        self.target_values = target_values
        self.id_col_name = id_col_name
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_samples = max_samples
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.crit = crit
        self.random_state = random_state
        
        # Encryption Scheme (CKKS)
        self.ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, 16_384, -1, [60,60,60,60])
        self.ctx_eval.global_scale = 2 ** 60
        self.ctx_eval.generate_galois_keys()
        self.secret_key = self.ctx_eval.secret_key()
        self.ctx_eval.make_context_public()
        
        # Storing of Global Forest Information (i.e. feature selections per fitted tree & the fitted trees themselves)
        self.feature_idx_memory = {} 
        self.forest_dict = {}
        self.tree_structure_dict = {}
        
    def fit_forest(self,X,y):
        """
        Fits a federated forest using the Fed_DecisionTree class to build the decision trees. 

        Parameters
        ----------
        X : list 
            Contains the features for every participant of the federation. Should be a list of numpy.arrays or a list of pandas.DataFrames.
        y : array or pandas.Series
            Contains the training labels. Should be of type numpy.array or pandas.Series.

        Returns
        -------
        A fitted FederatedForest-model

        """
        np.random.seed(self.random_state)
        
        for j in tqdm(range(self.n_estimators)):
            # 1. @ Coordinator: Create a bootstrap sample and select certain share of features for each participant
            # 1.1. Random draw of features
            if self.max_features == 'sqrt':
                t = [round(np.sqrt(d.shape[1])) for d in X]
            elif type(self.max_features) == float:
                t = [round(d.shape[1] * self.max_features) for d in X]
            else:
                raise Exception("The argument 'max_features' has to be either a float between 0 and 1 or 'sqrt'.")
                
            feat_len_list = [d.shape[1] for d in X]
    
            feature_idx = [np.random.choice(np.arange(1,feat_len_list[i]),t[i],replace=False) for i in range(len(t))]
            self.feature_idx_memory[f't{j}'] = feature_idx # store the used feature indices for predictions -> use self. here
    
    
            # 1.2. Bootstrapping of samples
            len_ds = np.min([d.shape[0] for d in X])
            bootstrap_idx = np.random.randint(0,len_ds,int(len_ds * self.max_samples))
    
            # 1.3. Send the bootstrap indices and the feature indices over to the participants
            #      who will filter their local data sets accordingly -> no encryption necessary at this step 
            X_tr = [d.iloc[bootstrap_idx,[0,*feature_idx[i]]] for i,d in enumerate(X)]
            y_tr = y.iloc[bootstrap_idx]
    
            # 2. @ Coordinator & Participants: Instantiate Decision Tree
            arg_dict_fdt = {
                'num_parties':self.num_parties,
                'target_values':self.target_values,
                'id_col_name':self.id_col_name,
                'max_depth':self.max_depth,
                'min_samples_split':self.min_samples_split,
                'min_impurity_decrease':self.min_impurity_decrease,
                'crit':self.crit,
                'encryption_context':self.ctx_eval,
                'secret_key':self.secret_key,
                'random_state':self.random_state + j
            }
            fed_dt = Fed_DecisionTree(**arg_dict_fdt)
            
            # Fit a federated decision tree and save it in the model dict
            # Also save the corresponding tree structure separtely
            self.tree_structure_dict[f't{j}'] = fed_dt.fit(X_tr,y_tr,depth=0,tree_path='o')
            self.forest_dict[f't{j}'] = fed_dt
            
        
    def predict(self,X,pred_type='prob',drop_idx=True):
        tree_predictions_list = []
        for key in tqdm(self.forest_dict.keys()):
            # select features for the different participants
            X_pred = [X[i].iloc[:,[0,*self.feature_idx_memory[key][i]]] for i in range(len(X))]
            
            tree_pred = self.forest_dict[key].predict(
                X=X_pred,
                pred_type=pred_type,
                drop_idx=drop_idx
            )
            tree_predictions_list.append(tree_pred)
            final_pred = np.mean(tree_predictions_list,axis=0)
        return final_pred         

        



#================================
# Trainers
#================================  
# Vertical Logisitic Regression:  Yang et al. 2019 approach
class VFLC_Trainer_Yang(object):
    def __init__(self,
                 data,
                 models, 
                 poly_mod_degree = 16384, 
                 coeff_mod_bit_sizes = [60,60,60,60],
                 global_scale=2 ** 60,
                 verbose=1,
                 random_state = 2021
                 ):
        """
        Implementation of the vertical log. regression following Yang et al. 2019 -> https://arxiv.org/pdf/1911.09824

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        models : TYPE
            DESCRIPTION.
        poly_mod_degree : TYPE, optional
            DESCRIPTION. The default is 16384.
        coeff_mod_bit_sizes : TYPE, optional
            DESCRIPTION. The default is [60,60,60,60].
        global_scale : TYPE, optional
            DESCRIPTION. The default is 2 ** 60.
        verbose : TYPE, optional
            DESCRIPTION. The default is 1.
        random_state : TYPE, optional
            DESCRIPTION. The default is 2021.

        Returns
        -------
        None.

        """
        self.data = data
        self.models = models
        self.verbose = verbose
        self.random_state = random_state
        self.batch_times = []
        self.poly_mod_degree = poly_mod_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.global_scale = global_scale
        self.lr_dict = {key:self.models[key].lr for key in self.models.keys()}
        self.momentum_dict = {key:self.models[key].momentum for key in self.models.keys()}
        self.change_memory = {key:{'weights':torch.tensor(np.zeros(self.models[key].layer_list[0].weight.shape[1])),'bias':torch.tensor(0) } for key in self.models.keys()}
        self.opts = {}
        
        # Dictionaries for monitoring progress
        self.LOGLOSS_train = {}
        self.LOGLOSS_test = {}
        self.ROCAUC_train = {}
        self.ROCAUC_test = {}
        self.PRAUC_train = {}
        self.PRAUC_test = {}
        self.BRIER_train = {}
        self.BRIER_test = {}
        

    def train(self,num_epochs,batch_size,progress_freq=5,compute_loss=True,encryption=True,opti='custom',pred_approach='host'):
        """
        Fitting function that steers the model training process and stores the results in the corresponding dictionaries. 

        Parameters
        ----------
        num_epochs : int
            Number of epochs.
        batch_size : int
            Sizes of batches to split of data into.
        progress_freq : int, optional
            Indicates after how many batches intermediate model results should be printed to the console in order
            to monitor the training process. The default is 5.
        encryption : bool, optional
            Boolean that indicates if encrytion methods should be used. The default is True.
        opti : str, optional
            Indicates which optimizer to use. Currently only a custom SGD is implemented. The default is 'custom'.
        
        Returns
        -------
        Fitted model and training results in the form of dictionaries (see above)

        """
        # Start of Federation Protocol:
        # 1. @ host: create context object with public and secret key & "send" public key to participants
        ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, self.poly_mod_degree, -1, self.coeff_mod_bit_sizes)
        ctx_eval.global_scale = self.global_scale
        ctx_eval.generate_galois_keys()
        secret_key = ctx_eval.secret_key()
        ctx_eval.make_context_public()
        
        # Define Optimizers (only for non-encrypted pytorch approach, which was used for benchmarking the results)
        if opti == 'pytorch':
            for key in self.models.keys():
                self.opts[key] = torch.optim.SGD(self.models[key].parameters(),lr=self.models[key].lr,momentum=self.models[key].momentum)
        
        # used for selecting the label holder (has to be first element in models dictionary)
        key_list = [key for key in self.models.keys()]
        
        # set models into training mode
        for key in self.models.keys():
            self.models[key].train() 
        
        # START EPOCH LOOP
        for epoch in range(num_epochs):
            # Dictionaries for performance logging during training
            ROCAUC_train_epoch_dict = {'fed_pred':{},'host_pred':{}} 
            ROCAUC_test_epoch_dict = {'fed_pred':{},'host_pred':{}} 
            LOGLOSS_train_epoch_dict = {'fed_pred':{},'host_pred':{}}
            LOGLOSS_test_epoch_dict = {'fed_pred':{},'host_pred':{}}
            PRAUC_train_epoch_dict = {'fed_pred':{},'host_pred':{}}
            PRAUC_test_epoch_dict = {'fed_pred':{},'host_pred':{}}
            BRIER_train_epoch_dict = {'fed_pred':{},'host_pred':{}}
            BRIER_test_epoch_dict = {'fed_pred':{},'host_pred':{}}
            
            # Create batches via indices (shuffled once per new epoch)
            new_idx = shuffle(range(self.data[key_list[0]]['fed_xtrain_data'].shape[0]), 
                              random_state = self.random_state + epoch)
            num_splits = int(len(new_idx)/batch_size)
            new_idx = new_idx[:batch_size * num_splits]
            batches = np.array_split(new_idx,num_splits)
            
            # START BATCH LOOP
            for bat_idx,bat in enumerate(batches):  
                start_batch = time.time()
                
                if opti == 'pytorch':
                    for key in self.models.keys():
                        self.opts[key].zero_grad()
                        
                # 2. At every participant: generate u
                u_dict = {}
    
                for key in self.models.keys():
                    # Forward Passes
                    u = self.models[key].forward(self.data[key]['fed_xtrain_data'][bat,:])
                    # Encryption of forward passes from every federation participant
                    if encryption == True:
                        u = ts.ckks_vector(ctx_eval,u.flatten().tolist())
                    u_dict[key] = u
                    
                # 3. @Host: Decrypt u's and sum them up to create y_hat,prediction error & loss
                if encryption == True:
                    logits = sum([torch.tensor(u.decrypt(secret_key)) for u in u_dict.values()]).reshape(-1,1)
                else:
                    logits = sum([u for u in u_dict.values()])
                yhat = torch.sigmoid(logits).flatten()
                
                y = self.data[key_list[0]]['fed_ytrain_data'][bat]
                pred_error =  y - yhat
                
                if encryption == True:
                    pred_error_crypt = ts.ckks_vector(ctx_eval,pred_error.flatten().tolist())
                
                if opti == 'pytorch':
                    loss = F.binary_cross_entropy_with_logits(input=logits,
                                                          target=y.reshape(-1,1),
                                                          reduction='mean')
                    loss.backward()
                
                # 4. Compute Gradients at A,B & C & send encrypted gradients to A for decryption (here: through storing in same dictionary)
                grad_bias = (-1/batch_size) * pred_error.flatten().sum().detach().numpy() # created by host & sent to guests
                
                grad_crypt_dict = {}
                noise_dict = {}
                for key in self.models.keys():
                    if encryption == True:
                        if len(self.data[key]) > 2: # i.e. host with labels
                            grad_crypt_dict[key] = (-1/batch_size) * pred_error.matmul(self.data[key]['fed_xtrain_data'][bat])
                        else: # i.e. guests (data provider w/o labels)
                            # Compute gradients on encrypted prediction error for guests
                            grad_crypt_tmp = pred_error_crypt.matmul(self.data[key]['fed_xtrain_data'][bat])
                            # Add noise via uniform dist. to encrypted gradients of guests to disguise them for the host after decryption
                            max_w = max(self.models[key].layer_list[0].weight[0]).detach()
                            min_w = min(self.models[key].layer_list[0].weight[0]).detach()
                            noise_dict[key] = np.random.uniform(min_w,max_w,self.data[key]['fed_xtrain_data'].shape[1])
                            grad_crypt_dict[key] = grad_crypt_tmp + noise_dict[key]
                        
                # 5. + 6. @host: decrypt gradients for B & C and send them back for denoising @guests (for convenience, they are stored here with the gradients of A in the same dict.)
                grad_decrypt_dict = {}
                for key in self.models.keys():
                    if encryption == True:
                        if len(self.data[key]) > 2: # i.e. host with labels
                            grad_decrypt_dict[key] = grad_crypt_dict[key]
                        else: # i.e. guests (data provider w/o labels)
                            # the following line incorporates two consecutive steps at different parties: Decryption of uest gradients at host + denoising of those gradients which is done at the guests after host sends the decrypted gradients over
                            grad_decrypt_dict[key] = (-1/batch_size) * (torch.tensor(grad_crypt_dict[key].decrypt(secret_key)) - torch.tensor(noise_dict[key]))
                    else:
                        grad_decrypt_dict[key] = (-1/batch_size) * pred_error.matmul(self.data[key]['fed_xtrain_data'][bat])

                # 7. @each participant: update parameters
                if opti == 'pytorch': # for reference
                    for key in self.models.keys():
                        self.opts[key].step()
                elif opti == 'custom':
                    with torch.no_grad():
                        for key in self.models.keys():
                            eta = self.lr_dict[key]
                            momentum = self.momentum_dict[key]
                            for i,p in enumerate(self.models[key].parameters()):
                                if len(p.shape) == 2:
                                    update_weight = momentum * self.change_memory[key]['weights'] + grad_decrypt_dict[key]
                                    p[0].add_(update_weight,alpha=-eta)
                                    self.change_memory[key]['weights'] = update_weight
                                else:
                                    update_bias = momentum * self.change_memory[key]['bias'] + grad_bias
                                    p[0].add_(update_bias,alpha=-eta)
                                    self.change_memory[key]['bias'] = update_bias
                                    
                else:
                    raise ValueError("Either use 'pytorch' or 'custom' as argument.")
                                
                # 8. Evaluate Model(s): 
                # 8.1. Predictions
                ## 8.1.1. Predictions using the federation
                logit_list_train = [self.models[key].forward(self.data[key]['fed_xtrain_data']) for key in self.models.keys()]
                logit_sum_train = sum(logit_list_train)
                yhat_train_fed = torch.sigmoid(logit_sum_train).flatten().detach().numpy()
                
                logit_list_test = [self.models[key].forward(self.data[key]['fed_xtest_data']) for key in self.models.keys()]
                logit_sum_test = sum(logit_list_test)
                yhat_test_fed = torch.sigmoid(logit_sum_test).flatten().detach().numpy()
                
                ## 8.1.2. Predictions using only the host (i.e. label owner) after training on federation
                yhat_train_host = self.models[key_list[0]].predict(self.data[key_list[0]]['fed_xtrain_data'])
                yhat_test_host = self.models[key_list[0]].predict(self.data[key_list[0]]['fed_xtest_data'])
                
                
                # 8.2. Log-Loss
                ## 8.2.1. Log-Loss for predictions based on "federation data"
                log_loss_test_fed = log_loss(self.data[key_list[0]]['fed_ytest_data'],yhat_test_fed,eps=0.000001)
                log_loss_train_fed = log_loss(self.data[key_list[0]]['fed_ytrain_data'],yhat_train_fed,eps=0.000001)
                LOGLOSS_test_epoch_dict['fed_pred'][f"Batch_{bat_idx+1}"] = log_loss_train_fed
                LOGLOSS_train_epoch_dict['fed_pred'][f"Batch_{bat_idx+1}"] = log_loss_test_fed
                
                ## 8.2.2. Log-Loss for predictions based on "host data" only
                log_loss_test_host = log_loss(self.data[key_list[0]]['fed_ytest_data'],yhat_test_host,eps=0.000001)
                log_loss_train_host = log_loss(self.data[key_list[0]]['fed_ytrain_data'],yhat_train_host,eps=0.000001)
                LOGLOSS_test_epoch_dict['host_pred'][f"Batch_{bat_idx+1}"] = log_loss_train_host
                LOGLOSS_train_epoch_dict['host_pred'][f"Batch_{bat_idx+1}"] = log_loss_test_host
                
                
                # 8.3. ROC-AUC
                ## 8.3.1. ROC-AUC for predictions based on "federation data" 
                ROCAUC_test_epoch_dict['fed_pred'][f"Batch_{bat_idx+1}"] = roc_auc_score(self.data[key_list[0]]['fed_ytest_data'] , yhat_test_fed)
                ROCAUC_train_epoch_dict['fed_pred'][f"Batch_{bat_idx+1}"] = roc_auc_score(self.data[key_list[0]]['fed_ytrain_data'] , yhat_train_fed)
                
                ## 8.3.2. ROC-AUC for predictions based on "host data" only
                ROCAUC_test_epoch_dict['host_pred'][f"Batch_{bat_idx+1}"] = roc_auc_score(self.data[key_list[0]]['fed_ytest_data'] , yhat_test_host)
                ROCAUC_train_epoch_dict['host_pred'][f"Batch_{bat_idx+1}"] = roc_auc_score(self.data[key_list[0]]['fed_ytrain_data'] , yhat_train_host)
                    
                  
                # 8.4. PR-AUC
                ## 8.4.1. PR-AUC for predictions based on "federation data"
                precision_test_fed, recall_test_fed, _ = precision_recall_curve(self.data[key_list[0]]['fed_ytest_data'],yhat_test_fed) 
                pr_auc_test_fed = auc(recall_test_fed,precision_test_fed)
                precision_train_fed, recall_train_fed, _ = precision_recall_curve(self.data[key_list[0]]['fed_ytrain_data'],yhat_train_fed) 
                pr_auc_train_fed = auc(recall_train_fed,precision_train_fed)
                PRAUC_test_epoch_dict['fed_pred'][f"Batch_{bat_idx+1}"] = pr_auc_test_fed
                PRAUC_train_epoch_dict['fed_pred'][f"Batch_{bat_idx+1}"] = pr_auc_train_fed
                
                ## 8.4.2. PR-AUC for predictions based on "host data" only
                precision_test_host, recall_test_host, _ = precision_recall_curve(self.data[key_list[0]]['fed_ytest_data'],yhat_test_host) 
                pr_auc_test_host = auc(recall_test_host,precision_test_host)
                precision_train_host, recall_train_host, _ = precision_recall_curve(self.data[key_list[0]]['fed_ytrain_data'],yhat_train_host) 
                pr_auc_train_host = auc(recall_train_host,precision_train_host)
                PRAUC_test_epoch_dict['host_pred'][f"Batch_{bat_idx+1}"] = pr_auc_test_host
                PRAUC_train_epoch_dict['host_pred'][f"Batch_{bat_idx+1}"] = pr_auc_train_host
                
                # 8.5. Brier Score
                ## 8.5.1. PR-AUC for predictions based on "federation data"
                BRIER_test_epoch_dict['fed_pred'][f"Batch_{bat_idx+1}"] = brier_score_loss(self.data[key_list[0]]['fed_ytest_data'],yhat_test_fed)
                BRIER_train_epoch_dict['fed_pred'][f"Batch_{bat_idx+1}"] = brier_score_loss(self.data[key_list[0]]['fed_ytrain_data'],yhat_train_fed)
                
                ## 8.5.2. PR-AUC for predictions based on "host data" only
                BRIER_test_epoch_dict['host_pred'][f"Batch_{bat_idx+1}"] = brier_score_loss(self.data[key_list[0]]['fed_ytest_data'],yhat_test_host)
                BRIER_train_epoch_dict['host_pred'][f"Batch_{bat_idx+1}"] = brier_score_loss(self.data[key_list[0]]['fed_ytrain_data'],yhat_train_host)
                
                end_batch = time.time()
                self.batch_times.append(end_batch-start_batch)
                # Monitor Progress
                if self.verbose == 1:
                    if (bat_idx + 1) % progress_freq == 0: 
                        print(65*'=')
                        print(f"Progress: {bat_idx+1}/{len(batches)} batches in {epoch+1}/{num_epochs} epochs")
                        print(f"\t ROC-AUC on batch {bat_idx+1}:")
                        print("\t\t-> TRAIN FED: " + str(round(ROCAUC_train_epoch_dict['fed_pred'][f'Batch_{bat_idx+1}'],4)))
                        print("\t\t-> TEST FED: " + str(round(ROCAUC_test_epoch_dict['fed_pred'][f'Batch_{bat_idx+1}'],4)))
                        print("\t\t" + 55*"-")
                        print("\t\t-> TRAIN HOST: " + str(round(ROCAUC_train_epoch_dict['host_pred'][f'Batch_{bat_idx+1}'],4)))
                        print("\t\t-> TEST HOST: " + str(round(ROCAUC_test_epoch_dict['host_pred'][f'Batch_{bat_idx+1}'],4)))
                        if len(self.batch_times) > progress_freq:
                            print(f"\t Execution time for the last 5 batches: {sum(self.batch_times[-5:])/60:.2f} min.")
                        print(f"\t Total execution time so far: {sum(self.batch_times)/60:.2f} min.")
                    else:
                        pass
                else:
                    pass
                # END BATCH LOOP
                
            # Save ROC-AUC development across the epoch 
            self.LOGLOSS_train[f"Epoch_{epoch+1}"] = LOGLOSS_train_epoch_dict
            self.LOGLOSS_test[f"Epoch_{epoch+1}"] = LOGLOSS_test_epoch_dict
            self.ROCAUC_train[f"Epoch_{epoch+1}"] = ROCAUC_train_epoch_dict
            self.ROCAUC_test[f"Epoch_{epoch+1}"] = ROCAUC_test_epoch_dict
            self.PRAUC_train[f"Epoch_{epoch+1}"] = PRAUC_train_epoch_dict
            self.PRAUC_test[f"Epoch_{epoch+1}"] = PRAUC_test_epoch_dict
            self.BRIER_train[f"Epoch_{epoch+1}"] = BRIER_train_epoch_dict
            self.BRIER_test[f"Epoch_{epoch+1}"] = BRIER_test_epoch_dict
            # END EPOCH LOOP




class SplitNN_Trainer(object):
    def __init__(self,models,data,poly_mod_degree = 32768, coeff_mod_bit_sizes = [60,60,60,60], global_scale=2 ** 60, opti='ADAM', verbose=1,random_state=2021):
        """
        Trianer class for the SplitNN as described in http://arxiv.org/pdf/1812.00564v1 and https://doi.org/10.1016/j.jnca.2018.05.003

        Parameters
        ----------
        models : dict
            dictionary that carries the models of all three clients separately.
        data : dict
            dictionary that carries the data of all three clients separately.
        poly_mod_degree : int, optional
            Polynomial modulo degree. The default is 16384.
        coeff_mod_bit_sizes : TYPE, optional
            Bit sizes for modulo coefficients. The default is [60,60,60,60].
        global_scale : TYPE, optional
            Controls accuracy of the encrpyted parameters. The default is 2 ** 60 (and the maximum).
        opti : str, optional
            String that indicates which optimizer to use. Currently only works for SGD and ADAM. The default is 'ADAM'.
        verbose : TYPE, optional
            Controls verbosity. If set to 1, intermediate results after each batch are printed to the console. 
            The default is 1.
        random_state : int, optional
            Sets random seed. The default is 2021.

        
        Returns
        -------
        Fitted model and training results in the form of dictionaries (see above)
        
        """
        self.models = models
        self.data = data
        self.opts = {}
        self.batch_times = []
        self.verbose = verbose
        self.poly_mod_degree = poly_mod_degree 
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.global_scale = global_scale
        self.random_state = random_state
        
        # Define Encryption context
        self.ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, self.poly_mod_degree, -1, self.coeff_mod_bit_sizes)
        self.ctx_eval.global_scale = self.global_scale
        self.ctx_eval.generate_galois_keys()
        self.secret_key = self.ctx_eval.secret_key()
        self.ctx_eval.make_context_public()

        # used for selecting the label holder (has to be first element in data dictionary)
        self.model_key_list = [key for key in self.models.keys()]
        self.data_key_list = [key for key in self.data.keys()]
        
        # Define Optimizers
        if opti == 'SGD':
            for key in self.models.keys():
                self.opts[key] = torch.optim.SGD(self.models[key].parameters(),lr=self.models[key].lr,momentum=self.models[key].momentum)
        elif opti == 'ADAM':
            for key in self.models.keys():
                self.opts[key] = torch.optim.Adam(self.models[key].parameters(),lr=self.models[key].lr,weight_decay=self.models[key].weight_decay)
        else:
            raise Exception("Currently only SGD or ADAM applicable as optimizers.")
        
        # Dictionaries for monitoring progress
        self.LOGLOSS_train = {}
        self.LOGLOSS_test = {}
        self.ROCAUC_train = {}
        self.ROCAUC_test = {}
        self.PRAUC_train = {}
        self.PRAUC_test = {}
        self.BRIER_train = {}
        self.BRIER_test = {}
    
    
    def ensemble_forward(self,bat_ids=None,encryption=False,train_set = True):  #,context=None,skey=None
        dp_output_dict = {}
                
        for key in self.data.keys():
            if train_set == True:
                if bat_ids is None:
                    dp_output = self.models[key].forward(self.data[key]['fed_xtrain_data'])
                    bat_ids = np.arange(0,self.data[key]['fed_xtrain_data'].shape[0])
                else:
                    dp_output = self.models[key].forward(self.data[key]['fed_xtrain_data'][bat_ids,:])
            else:
                if bat_ids is None:
                    dp_output = self.models[key].forward(self.data[key]['fed_xtest_data'])
                    bat_ids = np.arange(0,self.data[key]['fed_xtest_data'].shape[0])
                else:
                    dp_output = self.models[key].forward(self.data[key]['fed_xtest_data'][bat_ids,:])
                   
            if encryption == True:
                num_chunks = int(np.ceil(dp_output.flatten().shape[0] / 5_000))
                dp_chunks = np.array_split(dp_output.flatten().detach().numpy(),num_chunks)        
                dp_list_crypt = [ts.ckks_vector(self.ctx_eval,dp_chunks[j]) for j in range(num_chunks)] 
                dp_output_dict[key] = dp_list_crypt    
                
            else:
                dp_output_dict[key] = dp_output
        
        
        ## 2. Further processing @ coordinator
        if encryption == True:
            # Decrypt score chunks and piece them togehter to get the inputs of the three data providers
            tmp_list = []
            for val in dp_output_dict.values():
                decrypt_list = [l.decrypt(self.secret_key) for l in val]
                decrypt_list = list(chain(*decrypt_list))
                tmp_list.append(torch.tensor(decrypt_list).reshape(len(bat_ids),-1))
            
            # Now stack the actual inputs of the data providers together and feed the resulting tensor into the coordinator model
            coordinator_input =  torch.cat(tmp_list,dim=1)
            coord_output = self.models[self.model_key_list[-2]].forward(coordinator_input)
            
            # Again, break the outputs into chunks in order to be able to encrypt them. Then, send them over to the label holder.
            num_chunks = int(np.ceil(coord_output.flatten().shape[0] / 5_000))
            coord_chunks = np.array_split(coord_output.flatten().detach().numpy(),num_chunks)        
            coord_output = [ts.ckks_vector(self.ctx_eval,coord_chunks[j]) for j in range(num_chunks)] 
        
        else:
            tmp_list = [val for val in dp_output_dict.values()]
            coordinator_input =  torch.cat(tmp_list,dim=1)
            coord_output = self.models[self.model_key_list[-2]].forward(coordinator_input)
        
        return coord_output
        
    
    def train(self,num_epochs,batch_size,encryption=True,red='mean',progress_freq=1,stop_criteria=None):
        """
        Fitting function that steers the model training process and stores the results in the corresponding dictionaries. 

        Parameters
        ----------
        num_epochs : int
            Number of epochs.
        batch_size : int
            Sizes of batches to split of data into.
        encryption : bool, optional
            Boolean that indicates if encrytion methods should be used. The default is True.
        red : str, optional
            Reduction method for computing the loss during training. Options are 'mean' or 'sum'. Default is 'mean'. 
        progress_freq : int, optional
            Indicates after how many batches intermediate model results should be printed to the console in order
            to monitor the training process. The default is 5.
        stop_criteria : float, optional
            Ends the training if the loss fell below a predefined level. The default is None.
        
        Returns
        -------
        Fitted model and training results in the form of dictionaries (see above)

        """
        # Dictionaries for performance logging during training
        ROCAUC_train_epoch_dict = {} 
        ROCAUC_test_epoch_dict = {}
        LOGLOSS_train_epoch_dict = {}
        LOGLOSS_test_epoch_dict = {}
        PRAUC_train_epoch_dict = {}
        PRAUC_test_epoch_dict = {}
        BRIER_train_epoch_dict = {}
        BRIER_test_epoch_dict = {}
        
        
        # set models into training mode
        for key in self.models.keys():
            self.models[key].train() 
        
        # Epoch Loop        
        for epoch in range(num_epochs):
            
            # Create batches via indices (shuffled once per new epoch)
            new_idx = shuffle(range(self.data[self.data_key_list[0]]['fed_xtrain_data'].shape[0]), 
                              random_state = self.random_state + epoch)
            num_splits = int(len(new_idx)/batch_size)
            new_idx = new_idx[:batch_size * num_splits]
            batches = np.array_split(new_idx,num_splits)

            # BATCH LOOP 
            for bat_idx,bat in enumerate(batches):
                start_batch = time.time()
                for key in self.models.keys():
                    self.opts[key].zero_grad()
                    
                # Forward pass
                ## 1. do forward pass until you get to label holder    
                if encryption == True:
                    coord_output = self.ensemble_forward(bat_ids = bat,encryption=True,train_set=True) #context=ctx_eval,skey=secret_key
                    
                    decrypt_list = [l.decrypt(self.secret_key) for l in coord_output]
                    decrypt_list = list(chain(*decrypt_list))
                    coord_output = torch.tensor(decrypt_list).reshape(batch_size,-1)
                    
                else:
                    coord_output = self.ensemble_forward(bat_ids = bat,encryption=False,train_set=True) #,context=None,skey=None
                ## 2. Coordinator sends coord_output over to labelholder for evaluation & initiation of backward pass
                loss  = self.models[self.model_key_list[-1]].evaluate(coord_output,self.data[self.data_key_list[0]]['fed_ytrain_data'][bat],red=red)
                
                # Stopping criteria
                if stop_criteria is None:
                    pass
                else:
                    if loss < stop_criteria:
                        break
                
                # Backward pass
                ## 1. @ Labelholder (last section of the network)
                self.models[self.model_key_list[-1]].backward(torch.tensor(1.0).reshape(-1,1))
                grad_input_coord = self.models[self.model_key_list[-1]].input[0].grad.data # potentially add encrytion here 
                self.opts[self.model_key_list[-1]].step()
                
                ## 2. @ Coordinator
                self.models[self.model_key_list[-2]].backward(grad_input_coord)
                grad_input_dp = self.models[self.model_key_list[-2]].input[0].grad.data
                self.opts[self.model_key_list[-2]].step()
                
                # Split gradient for data providers & send corresponding (decrypted) gradients to each of them
                grad_input_dp_dict = {}
                cnt = 0
                for key in self.data.keys():
                    lim = self.models[key].layer_list[-1].weight.shape[0]
                    grad_input_dp_sub = grad_input_dp[:,cnt:cnt+lim]
                    if encryption == True:
                        num_chunks = int(np.ceil(grad_input_dp_sub.flatten().shape[0] / 5_000))
                        grad_chunks = np.array_split(grad_input_dp_sub.flatten().detach().numpy(),num_chunks)        
                        grad_list_crypt = [ts.ckks_vector(self.ctx_eval,grad_chunks[j]) for j in range(num_chunks)] 
                        grad_input_dp_dict[key] = grad_list_crypt
                        
                    else:
                        grad_input_dp_dict[key] = grad_input_dp_sub
                    
                    cnt += lim
                    
                ## 3. @ Data Providers (incl. labelholder)
                for k,v in grad_input_dp_dict.items():
                    if encryption == True:
                        decrypt_list = [l.decrypt(self.secret_key) for l in v]
                        decrypt_list = list(chain(*decrypt_list))
                        grad_dp = torch.tensor(decrypt_list).reshape(batch_size,-1)
                        
                    else:
                        grad_dp = v
                    self.models[k].backward(grad_dp)
                    self.opts[k].step()
                    
                    
                # Evaluations
                ## 1. Forward pass
                coord_output_train = self.ensemble_forward(bat_ids=None,encryption=False,train_set=True) 
                coord_output_test = self.ensemble_forward(bat_ids=None,encryption=False,train_set=False) 

                yhat_test = self.models[self.model_key_list[-1]].predict(coord_output_test).detach().numpy()                
                yhat_train = self.models[self.model_key_list[-1]].predict(coord_output_train).detach().numpy()
                                
                
                ## 2. Log-Loss
                log_loss_test = log_loss(self.data[self.model_key_list[0]]['fed_ytest_data'],yhat_test,eps=0.000001)
                log_loss_train = log_loss(self.data[self.model_key_list[0]]['fed_ytrain_data'],yhat_train,eps=0.000001)
                LOGLOSS_test_epoch_dict[f"Batch_{bat_idx+1}"] = log_loss_train
                LOGLOSS_train_epoch_dict[f"Batch_{bat_idx+1}"] = log_loss_test
                
                
                ## 3. ROC-AUC
                ROCAUC_test_epoch_dict[f"Batch_{bat_idx+1}"] = roc_auc_score(self.data[self.model_key_list[0]]['fed_ytest_data'] , yhat_test)
                ROCAUC_train_epoch_dict[f"Batch_{bat_idx+1}"] = roc_auc_score(self.data[self.model_key_list[0]]['fed_ytrain_data'] , yhat_train)
                
                
                ## 4. PR-AUC
                precision_test, recall_test, _ = precision_recall_curve(self.data[self.model_key_list[0]]['fed_ytest_data'],yhat_test) 
                pr_auc_test = auc(recall_test,precision_test)
                precision_train, recall_train, _ = precision_recall_curve(self.data[self.model_key_list[0]]['fed_ytrain_data'],yhat_train) 
                pr_auc_train = auc(recall_train,precision_train)
                PRAUC_test_epoch_dict[f"Batch_{bat_idx+1}"] = pr_auc_test
                PRAUC_train_epoch_dict[f"Batch_{bat_idx+1}"] = pr_auc_train
                
                ## 5. BRIER Score
                BRIER_test_epoch_dict[f"Batch_{bat_idx+1}"] = brier_score_loss(self.data[self.model_key_list[0]]['fed_ytest_data'],yhat_test)
                BRIER_train_epoch_dict[f"Batch_{bat_idx+1}"] = brier_score_loss(self.data[self.model_key_list[0]]['fed_ytrain_data'],yhat_train)
                
                
                end_batch = time.time()
                self.batch_times.append(end_batch-start_batch)
                # Monitor Progress
                if self.verbose == 1:
                    if (bat_idx + 1) % progress_freq == 0: 
                        print(65*'=')
                        print(f"Progress: {bat_idx+1}/{len(batches)} batches in {epoch+1}/{num_epochs} epochs")
                        print(f"\t Log-Loss on batch {bat_idx+1}:")
                        print("\t\t-> TRAIN: " + str(round(LOGLOSS_train_epoch_dict[f'Batch_{bat_idx+1}'],4)))
                        print("\t\t-> TEST: " + str(round(LOGLOSS_test_epoch_dict[f'Batch_{bat_idx+1}'],4)))
                        print("\t" + 60*"-")
                        print(f"\t ROC-AUC on batch {bat_idx+1}:")
                        print("\t\t-> TRAIN: " + str(round(ROCAUC_train_epoch_dict[f'Batch_{bat_idx+1}'],4)))
                        print("\t\t-> TEST: " + str(round(ROCAUC_test_epoch_dict[f'Batch_{bat_idx+1}'],4)))
                        if len(self.batch_times) > progress_freq:
                            print(f"\t Execution time for the last 5 batches: {sum(self.batch_times[-5:])/60:.2f} min.")
                        print(f"\t Total execution time so far: {sum(self.batch_times)/60:.2f} min.")
                    else:
                        pass
                else:
                    pass
                # END BATCH LOOP
                
        # Save ROC-AUC development across the epoch 
        self.LOGLOSS_train[f"Epoch_{epoch+1}"] = LOGLOSS_train_epoch_dict
        self.LOGLOSS_test[f"Epoch_{epoch+1}"] = LOGLOSS_test_epoch_dict
        self.ROCAUC_train[f"Epoch_{epoch+1}"] = ROCAUC_train_epoch_dict
        self.ROCAUC_test[f"Epoch_{epoch+1}"] = ROCAUC_test_epoch_dict
        self.PRAUC_train[f"Epoch_{epoch+1}"] = PRAUC_train_epoch_dict
        self.PRAUC_test[f"Epoch_{epoch+1}"] = PRAUC_test_epoch_dict
        self.BRIER_train[f"Epoch_{epoch+1}"] = BRIER_train_epoch_dict
        self.BRIER_test[f"Epoch_{epoch+1}"] = BRIER_test_epoch_dict
        # END EPOCH LOOP
