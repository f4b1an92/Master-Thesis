import numpy as np
import pandas as pd
import pickle as pkl
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch 


class Feature_Generator(object):
    """
    Class that loads preprocessed data and prepares it for the respective experiment scenario (horizontal vs. vertical).
    This includes unifying sample IDs across data sets (i.e. find intersection), scaling, train-test splitting and allocating
    the data set splits to the clients. Data sets are saved in the Feature_Generator object in accordance with their 
    respective federation client (client 1 - "m1", client 2 - "m2", client 3 - "m3") to simplify pointing to the correct 
    data sets in the experiments. The Feature_Generator object stores scaled (log. reg. & neural networks) and regular data 
    sets (for random forest).

    Parameters
    ----------
    data_path : str
        String that indicates from where to load the data sets.
    file_name : str
        File name of the data set that should be loaded. Not possible to load raw data. 
        Outputs from "data_cleaning+storing.py" are required here.
    n_jobs : int, optional
        Indicates the number of cores to use. The default is -1.
    random_state : int, optional
        Sets random seed. The default is 2021.
    train_size : float, optional
        Size of the training set given in relative terms. The default is 0.7.
    all_ext_sources : bool, optional
        Boolean that indicates whether all external source variables should be joined to the data. 
        The default is False.
    vars_to_drop : list, optional
        List that indicates which variables to drop before creating the final model input data set. 
        The default is None.
    apr_model : bool, optional
        Indicates if data should be prepared for training the APR model. If True, 
        most preprocessing steps like scaling and creating the train test split are skipped 
        and only code of block 1 is executed (see comment in line 73).  
        The default is False.
    drop_target : bool, optional
        Indicates whether target variable should be dropped. Only needed for vertical federated 
        learning where only one client holds the labels. 
        The default is False.
    
    
    Returns
    -------
    None.
    
    """
    def __init__(self,
                 data_path: str,
                 file_name: list,
                 n_jobs: int = -1,
                 random_state: int = 2021,
                 train_size: float = 0.7,
                 all_ext_sources: bool = False,
                 vars_to_drop: list = None,
                 apr_model: bool = False,
                 drop_target: bool = False,
                 vertical_lg_host: bool = False
                 ):
        
        self.data_path = data_path
        self.file_name = file_name
        self.vars_to_drop = vars_to_drop
        self.all_ext_sources = all_ext_sources
        self.n_jobs = n_jobs
        self.mse_score_ = np.nan
        self.random_state = random_state
        self.train_size = train_size
        self.apr_model = apr_model
        self.drop_target = drop_target
        self.scaler = StandardScaler()
        self.vertical_lg_host = vertical_lg_host
        
        # Block 1
        # Unscaled Data
        if type(self.file_name) == str:
            self.data = pd.read_parquet(self.data_path + self.file_name + '.parquet')
            self.data.columns = [col.lower() for col in self.data.columns]
        
        elif (type(self.file_name)==list) & (len(self.file_name) == 1):
            self.data = pd.read_parquet(self.data_path + self.file_name[0] + '.parquet')
            self.data.columns = [col.lower() for col in self.data.columns]
        
        elif (type(self.file_name) == list) & (len(self.file_name) > 1):
            dlist = [pd.read_parquet(self.data_path + i + '.parquet') for i in self.file_name]
            self.data = dlist[0]
            for i in np.arange(1,len(dlist)):
                self.data = self.data.merge(dlist[i],how='right')
                self.data.columns = [col.lower() for col in self.data.columns]
                self.data = self.data.loc[~self.data['target'].isnull(),:]
                
        else:
            raise TypeError("Input 'file_name' must either be string or list.")
        
        # Block 2
        # Checks if data for apr_model is needed. If so, scaling and other data processing steps necessary for the regular models are skipped.
        if self.apr_model == False: 
            # Join Target Variable
            if self.all_ext_sources == True:
                data_ext = pd.read_parquet(self.data_path + 'ext_sources.parquet' )
                self.data = self.data.merge(data_ext,how='left')    
            else:
                data_ext = pd.read_parquet(self.data_path + 'ext_sources.parquet' )
                data_ext = data_ext.loc[:,['sk_id_curr','target']]
                self.data = self.data.merge(data_ext,how='left')
            
            del data_ext
            
            # Reduce to IDs that are present in all data sets (save the remaining in another df)
            data_full = pd.read_parquet(self.data_path + 'full_data_cleaned.parquet')
            self.data_comp = self.data.loc[~self.data['sk_id_curr'].isin(np.unique(data_full['sk_id_curr'])),:].reset_index(drop=True)
            self.data = self.data.loc[self.data['sk_id_curr'].isin(np.unique(data_full['sk_id_curr'])),:].reset_index(drop=True)
            del data_full
            
            # In case there were variables specified to be dropped
            if self.vars_to_drop is None:
                pass
            elif 'target' in self.vars_to_drop:
                raise Exception("Variable 'target' must not be in 'vars_to_drop'. If you want to drop the target because of vertical FL, do so using the respective argument (i.e. 'drop_target')")
            else:
                self.data = self.data.drop(labels=self.vars_to_drop,axis=1)
            
            # Set target variable to -1 and 1 if vertical log. regression is conducted (host carries labels) 
            if self.vertical_lg_host == True:
                self.data['target'] = self.data['target'].apply(lambda x: -1 if x == 0 else x)
            else:
                pass
            
            # Scaled Data
            self.diff_list = [col for col in self.data.columns if col not in ['sk_id_curr','target']]
            self.scale_list = [col for col in self.diff_list if len(np.unique(self.data[col])) > 2]
            self.unscale_list = [col for col in self.data.columns if col not in self.scale_list]
            
            self.scaled_data = self.scaler.fit_transform(self.data.loc[:,self.scale_list])
            self.scaled_data = pd.DataFrame(self.scaled_data,columns=self.scale_list)
            self.scaled_data = pd.concat([self.data.loc[:,self.unscale_list],self.scaled_data],axis=1)
            self.scaled_data = self.scaled_data.sort_index(axis=1)
            
            if self.data_comp.shape[0] == 0:
                self.scaled_data_comp = self.data_comp
            else:
                self.scaled_data_comp = self.scaler.transform(self.data_comp.loc[:,self.scale_list])
                self.scaled_data_comp = pd.DataFrame(self.scaled_data_comp,columns=self.scale_list)
                self.scaled_data_comp = pd.concat([self.data_comp.loc[:,self.unscale_list],self.scaled_data_comp],axis=1)
                self.scaled_data_comp = self.scaled_data_comp.sort_index(axis=1)
            
            
            # Create Train-Test Splits
            self.data = train_test_split(
                    self.data.loc[:,self.diff_list],
                    self.data.loc[:,'target'],
                    train_size=self.train_size,
                    random_state=self.random_state,
                    shuffle=True
            )
            if self.drop_target == True:
                self.data = self.data[0], self.data[1]
                self.data_comp = [self.data_comp.loc[:,self.diff_list]]
            else:
                self.data_comp = self.data_comp.loc[:,self.diff_list], self.data_comp.loc[:,'target']
                
                
            self.scaled_data = train_test_split(
                    self.scaled_data.loc[:,self.diff_list],
                    self.scaled_data.loc[:,'target'],
                    train_size=self.train_size,
                    random_state=self.random_state,
                    shuffle=True
            )
            if self.drop_target == True:
                self.scaled_data = self.scaled_data[0], self.scaled_data[1]
                self.scaled_data_comp = [self.scaled_data_comp.loc[:,self.diff_list]]
            else:
                self.scaled_data_comp = self.scaled_data_comp.loc[:,self.diff_list], self.scaled_data_comp.loc[:,'target']
            
            self.scaled_data = [torch.tensor(i.values,dtype=torch.float32) for i in self.scaled_data]
            self.scaled_data_comp = [torch.tensor(i.values,dtype=torch.float32) for i in self.scaled_data_comp]
        else:
            pass


        
