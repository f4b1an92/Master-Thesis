import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import pickle as pkl
from xgboost import XGBRegressor
from hyperopt import Trials, fmin, hp, tpe

os.path.realpath(__file__)
main_path = os.getcwd()

sys.path.append(main_path + '\\code\\modules')

import hc_fun as hcf
import nonfed_models as nfm


class DataSet_Prepper(object):
    """
    Feature generator class to create feature sets necessary for modeling the credit default rates in a
    federated learning setting as well as a regular machine leanring setting.
    
    Parameters
    ----------
    - input_path: str
        Path that contains all the original data sets (as csv-files) of the Home Credit Default Risk 
        competition, hosted on kaggle (-> https://www.kaggle.com/c/home-credit-default-risk ).
    - output_path: str
        Path where cleaned data sets will be stored.
    - models_path: str
        Path where the XGB model used for predicting repayment periods on the application data set will
        be stored. The repayment period predictions are necessary in order to derive the interest rates
        for the loan applications stated as an annual percentage rate (APR) and as a monthly rate.
    
    Methods
    -------
    - prev_app
    - app_data
    - bureau_data
    - credit_card_bal_data
    - installment_data
    
    """
    def __init__(
            self,
            input_path : str,
            output_path : str,
            models_path : str
            
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.models_path = models_path
        
        
    def prev_app(self,
                 prev_app_filename: str,
                 app_filename: str,
                 save_int_rates: bool = True,
                 corr_thres: float = 0.7,
                 step_size: float = 0.0001,
                 block_size: int = 1_000):
        """
        Method that produces the cleaned previous_application data set with all the required features for
        modeling the default rates. Data set will be saved to the output path that was specified while 
        creating the Feature_Gen class. 


        Parameters
        ----------
        prev_app_filename : str
            Name of the file that contains the original 'previous_application' data. File extensions 
            are not required (e.g. '.csv').
        app_filename : str
            Name of the file that contains the original 'application train' data, including the acutal
            target variable. File extensions are not required (e.g. '.csv').
        save_int_rates : bool, optional
            Boolean that determines if the data set that is used for training the XGB model that predicts
            the repayment periods per application should be stored in a folder. The default is True.
        corr_thres : float, optional
            Threshold value to determine when highly correlated columns have to be dropped. The variable
            which has the higher correlation with the target variable is kept.
            The default is 0.7.
        step_size : float, optional
            Accuracy of the APR estimation. The default is 0.0001 in order to provide a percentage with
            two decimal places. This parameter also highly influences the execution time.
        block_size : int, optional
            Intake per iteration of the interest_rate function that produces the APR feature. High values
            (i.e. > 2_000) can lead to memory consumption issues.
            The default is 1_000.

        Returns
        -------
        Cleaned previous application data set with the necessary features for modeling. Data is saved 
        to a specified output path as a parquet-file. 

        """
        # 1. Loading Data 
        #-------------------
        data = pd.read_csv(self.input_path + prev_app_filename + '.csv')
        data_app = pd.read_csv(self.input_path + app_filename + '.csv')
        
        data.columns = [col.lower() for col in data.columns]
        data_app.columns = [col.lower() for col in data_app.columns]
        
        # Join target variable from application_train.csv and delete cases where target = NaN after joining 
        data = data.merge(data_app.loc[:,['sk_id_curr','target']],how='left')
        data = data.loc[~data['target'].isnull(),:]
        
        # Drop insignificant columns/columns that carry almost no information
        data = data.drop(labels=[
            'hour_appr_process_start',
            'nflag_last_appl_in_day',
            'days_decision',
            'sellerplace_area'
            ],axis=1
        )
        
        del data_app
        print('>>> Loading Data: Done.')
        
        # 2. Data Cleaning and Feature Engineering by Data Type:
        #-----------------------------------------------------------
        # 2.1. Type: Object
        # 2.1.1. Data Cleaning
        # Object-cols. to drop
        data = data.drop(labels=[
            'weekday_appr_process_start',
            'flag_last_appl_per_contract',
            'name_type_suite',
            'name_goods_category',
            'name_client_type',
            'name_seller_industry',
            'name_yield_group'
        ],axis=1)


        # Replace NaN's by 'Missing'
        mask_pc = data['product_combination'].isnull()
        data.loc[mask_pc,'product_combination'] = 'Missing'
        
        
        # Category representations to aggregate
        mask = data['name_cash_loan_purpose'].isin(['XAP','XNA'])
        data.loc[~mask,'name_cash_loan_purpose'] = 'Others'
        
        mask = data['name_payment_type'] != 'XNA'
        data.loc[mask,'name_payment_type'] = 'Others'
        
        mask = data['product_combination'].isin(['Cash','Cash Street: high','Cash Street: low','Cash Street: middle'])
        data.loc[mask,'product_combination'] = 'Cash'
        mask = [bool(re.search(r'^POS',i)) for i in data['product_combination']]
        data.loc[mask,'product_combination'] = 'POS'
        
        
        # Aggregate insignificant category representations
        mask = data['channel_type'].isin(['Channel of corporate sales','Car dealer','Regional/Local'])
        data.loc[mask,'channel_type'] = 'Misc sales'
        
        mask = data['name_portfolio'].isin(['Cars','POS'])
        data.loc[mask,'name_portfolio'] = 'Cars_POS'
        
        mask = data['code_reject_reason'] != 'XAP'
        data.loc[mask,'code_reject_reason'] = 'Others'
        
        mask = data['name_contract_type'] != 'XNA'
        data = data.loc[mask,:].reset_index(drop=True)
        
        mask = data['name_contract_status'] == 'Unused offer'
        data.loc[mask,'name_contract_status'] = 'Canceled'
                
        
        # 2.1.2. Feature Engineering
        data_types = hcf.check_dtypes(data)   
        
        print(">>> Create Count Shares for Categorical Variables: \n\t--->\tThen aggregate on 'sk_id_curr'-level")
        # Create count shares for categorical variables to transform them into float variables
        cdata = data.loc[:,['sk_id_curr','target']].drop_duplicates().reset_index(drop=True)
        for var in tqdm(data_types[1]['object']):
            tmp1 = data.groupby(['sk_id_curr',var])['sk_id_prev'].count().reset_index().rename(columns={'sk_id_prev':'num_cat'})
            tmp2 = data.groupby('sk_id_curr')['sk_id_prev'].count().reset_index().rename(columns={'sk_id_prev':'total'})
        
            tmp = tmp1.merge(tmp2,how='left')
            tmp['cat_share'] = tmp['num_cat'] / tmp['total']
        
            pivot = pd.pivot_table(tmp,values='cat_share',index='sk_id_curr',columns=var,
                                   aggfunc='sum',fill_value=0).reset_index()
        
            cdata = cdata.merge(pivot,how='left').fillna(0)
            
        col_list = [re.sub(r'[+()-/:]','',i) for i in cdata.columns]
        col_list = [re.sub(r' ','_',i) for i in col_list]
        col_list = [f'prev_{i}' for i in col_list[2:]]
        cdata.columns = ['sk_id_curr','target',*col_list]
        
        
        # 2.2. Type: Float
        # 2.2.1. Data Cleaning
        fdata = data.loc[:,['sk_id_curr','sk_id_prev',*data_types[1]['float64']]]
        fdata = fdata.drop(labels=[
            'target',
            'rate_interest_primary',
            'days_first_drawing',
            'days_last_due_1st_version',
            'rate_interest_privileged',
            'nflag_insured_on_approval'
        ],axis=1)
        
        mask_col = [bool(re.search(r'days',col)) for col in fdata.columns]
        
        for col in fdata.columns[mask_col]:
            mask = fdata[col] == 365243
            fdata.loc[mask,col] = np.nan
            
            
        # 2.2.2. Feature Engineering
        print(">>> Create the APR-Feature")
        fdata['interest'] = fdata['cnt_payment']*fdata['amt_annuity'] - fdata['amt_credit']
        fdata['interest_share'] = fdata['interest']/fdata['amt_credit']
        
        # Get rid of data points where there is no information about the below variables
        cond1 = fdata['amt_annuity'].isnull()
        cond2 = fdata['amt_credit'].isnull()
        cond3 = fdata['cnt_payment'].isnull()
        cond4 = fdata['cnt_payment'] == 0
        
        fdata = fdata.loc[~(cond1 | cond2 | cond3 | cond4),:].reset_index(drop=True)
        
        
        # Use the remaining data points to derive interest rates (APRs) from amt_annuity, cnt_payment & amt_credit
        arg_dict={
            'data':fdata,
            'pmt_col':'amt_annuity',
            'period_col':'cnt_payment',
            'principal_col':'amt_credit',
            'step_size':step_size,
            'block_size':block_size
        }
        apr_df = hcf.interest_rates(**arg_dict)
        
        fdata = pd.concat([fdata,apr_df],axis=1)
        
        if save_int_rates == True:
            fdata.to_parquet(self.output_path + 'interest_rates_results.parquet')
        
        print(">>> Aggregate Float-Columns on 'sk_id_curr'-level")
        # Aggregate float-cols. on sk_id_curr-level and join with category shares data:
        for col in tqdm(fdata.columns[2:]):
            tmp = fdata.groupby('sk_id_curr').agg({col:['mean','max','min']}).reset_index()
            tmp.columns = ['sk_id_curr',f'{col}_mean',f'{col}_max',f'{col}_min']
            cdata = cdata.merge(tmp,how='left').fillna(0)
            
        
        # 2.3. Missing Values/Correlation Analysis on joined Data Set
        # Remove data points with NaNs in columns that are related to the interest rates
        mask = cdata['amt_annuity_min'].isnull()
        cdata = cdata.loc[~mask,:]
        
        # Set remaining NaNs in other columns to 0
        for col in cdata.columns:
            mask = cdata[col].isnull()
            cdata.loc[mask,col] = 0
         
        # Drop highly correlated variables (keep the one of them that has the highest corr. with the target)
        tk,td = hcf.corr_dropouts(cdata,corr_thres=corr_thres)

        cdata = cdata.drop(labels=[*td,'target'],axis=1)
        cdata = cdata.sort_values(by='sk_id_curr').reset_index(drop=True)
        
        # 2.4. Save the Data
        cdata.to_parquet(self.output_path + 'prev_app_cleaned.parquet')
        print(f">>> Data was saved to: \n \t {self.output_path}")
        
        
    def app_data(self,
                 app_filename: str,
                 model_filename: str,
                 corr_remove: bool = True,
                 outlier_remove: bool = True,
                 ext_opt: bool = True,
                 corr_thres: float = 0.7,
                 step_size: float = 0.0001,
                 block_size: int = 1_000
                 ):
        """
        Method that produces the cleaned application_train data set with all the required features for
        modeling the default rates. Data set will be saved to the output path that was specified while 
        creating the Feature_Gen class. 

        Parameters
        ----------
        app_filename : str
            Name of the file that contains the original 'application train' data, including the acutal
            target variable. File extensions are not required (e.g. '.csv').
        model_filename : str
            Name of the file that contains the XGB model that was trained on the previous application data 
            to predict the repayment period which is required to derive the interest rate of the loans,
            stated as an APR. File extensions are not required (e.g. '.pkl').
        corr_thres : float, optional
            Threshold value to determine when highly correlated columns have to be dropped. The variable
            which has the higher correlation with the target variable is kept.
            The default is 0.7.
        step_size : float, optional
            Accuracy of the APR estimation. The default is 0.0001 in order to provide a percentage with
            two decimal places. This parameter also highly influences the execution time.
        block_size : int, optional
            Intake per iteration of the interest_rate function that produces the APR feature. High values
            (i.e. > 2_000) can lead to memory consumption issues.
            The default is 1_000.

        Returns
        -------
        Cleaned application data set with the necessary features for modeling. Data is saved to a specified
        output path as a parquet-file. 
        
        """
        # 1. Load Data
        #---------------
        data = pd.read_csv(self.input_path + app_filename + '.csv')
        data.columns = [col.lower() for col in data.columns]
        
        # Get rid of erroneous data points
        data = data.loc[data['code_gender'] != 'XNA',:]
        data = data.loc[data['name_family_status'] != 'Unknown',:].reset_index(drop=True)
        
        flag_cols = ['flag_mobil','flag_phone','flag_cont_mobile','flag_email','weekday_appr_process_start','name_type_suite']
        stat_vars = ['elevators_avg','elevators_medi','elevators_mode','nonlivingapartments_avg','nonlivingapartments_medi','nonlivingapartments_mode']
        req_credit = ['amt_req_credit_bureau_hour','amt_req_credit_bureau_day','amt_req_credit_bureau_week','amt_req_credit_bureau_qrt','amt_req_credit_bureau_year']
        registry = ['reg_region_not_live_region','reg_region_not_work_region','live_region_not_work_region']
        
        num = [2,  4,  5,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        doc_cols = ['flag_document_' + str(i) for i in num]
        
        data = data.drop(columns=[*flag_cols,*stat_vars,*req_credit,*registry,*doc_cols],axis=1)
        
        # Drop Median Columns
        medis = [col for col in data.columns if bool(re.search(r'\_medi',col))]
        data = data.drop(labels=medis,axis=1)
        print(">>> Loading Data: Done.")
        
        
        # 2. Data Cleaning 
        #-------------------
        final_dict = hcf.custom_dtype_identifier(data,thres=25)
    
        for col in data.loc[:,final_dict['num']].columns:
            data[col] = data[col].apply(abs)
        
        # 2.1. Outlier Removal
        if outlier_remove==True:     
            tmp_df = hcf.outlier_remover(df=data.loc[:,final_dict['num']].loc[:,'amt_income_total':],sf=3)   
            
            data.loc[:,tmp_df.columns] = tmp_df
            
        
        # 2.2. Missing Values 
        dtype_df = pd.concat([pd.DataFrame({'Var':v,'DataType':k}) for k,v in final_dict.items()],axis=0)

        tmp = data.isnull().mean().reset_index()
        tmp.columns = ['Var','Missing_Share']
        tmp = tmp.sort_values(by='Missing_Share',ascending=False).reset_index(drop=True)
        tmp = tmp.merge(dtype_df,how='left')
        tmp = tmp.loc[tmp['Missing_Share']>0,:]
        
        #obj = tmp.loc[tmp['DataType']=='object',:]
        num = tmp.loc[tmp['DataType']=='num',:]
        
        # Object Variables
        data.loc[data['occupation_type'].isnull(),'occupation_type'] = 'Missing'
        data.loc[data['fondkapremont_mode'].isnull(),'fondkapremont_mode'] = 'missing'
        data.loc[data['housetype_mode'].isnull(),'housetype_mode'] = 'missing'
        data.loc[data['wallsmaterial_mode'].isnull(),'wallsmaterial_mode'] = 'Missing'
        data.loc[data['emergencystate_mode'].isnull(),'emergencystate_mode'] = 'Missing'
        
        
        # Float Variables
        # External Sources 1-3: Use the value of that ext. source that is closest to the target for
        #                       a specific data point (i.e. optimal value)
        ext_col_list = ['sk_id_curr','ext_source_1','ext_source_2','ext_source_3','target']
        ext_df = data.loc[:,ext_col_list]
        # Fill missing values in ext_source columns with the respective col.-average
        for i in range(3):
            ext_df[f'ext_source_{i+1}'].fillna(ext_df[f'ext_source_{i+1}'].mean(),inplace=True)
        
        ext_df.to_parquet(self.output_path + 'ext_sources.parquet',engine='pyarrow')
        
        
        if ext_opt == False:
            # Mean Value
            total_mean_val = ext_df.loc[:,['ext_source_1','ext_source_2','ext_source_3']].mean().mean()
            data['ext_mean'] = ext_df.loc[:,['ext_source_1','ext_source_2','ext_source_3']].mean(axis=1).fillna(total_mean_val)
            data = data.drop(labels=['ext_source_1','ext_source_2','ext_source_3'],axis=1)
            
        else:
        # Optimal Value
            for i in range(1,4):
                ext_df[f'diff_{i}'] = abs(ext_df.loc[:,f'ext_source_{i}'] - ext_df.loc[:,'target'])
                
            ext_s = ext_df.loc[:,'diff_1':].idxmax(axis=1).apply(lambda x: re.sub(r'diff_','ext_source_',str(x)))
            idx = ext_s == 'nan'
            ext_s[idx] = 'ext_source_3'
            
            new_ext_list = []
            for i,col in enumerate(ext_s):
                opt_val = ext_df.loc[i,col]
                new_ext_list.append(opt_val)
                
            ext_df['ext_opt'] = new_ext_list
            
            idx = ext_df['ext_opt'].isnull()
            ext_df.loc[idx,'ext_opt'] = ext_df['ext_opt'].mean()
            
            ext_df = ext_df.drop(labels=['diff_1','diff_2','diff_3'],axis=1)
            
            # Insert into actual data
            data['ext_opt'] = ext_df['ext_opt'].values
    
            data = data.drop(labels=['ext_source_1','ext_source_2','ext_source_3'],axis=1)
            
        # All other float variables:
        mask = num['Var'].isin(['ext_source_1','ext_source_2','ext_source_3'])
        num = num.loc[~mask,:].reset_index(drop=True)
        
        for var in num['Var'].values:
            avg_value = data.loc[:,var].mean()
            data[var] = data.loc[:,var].fillna(avg_value)
            
        # drop all mode columns (except the 4 that are of type 'object')
        cat_modes  = ['fondkapremont_mode','housetype_mode','wallsmaterial_mode','emergencystate_mode']
        mode_mask = [bool(re.search(r'_mode',col)) if col not in cat_modes else False for col in data.columns]
        
        data = data.drop(labels=data.columns[mode_mask],axis=1)
        
        
        # 2.3. Merge Insignificant Category Attributes
        # name_income_type
        merge_dict = {
            'Coop/Office':['Office apartment','Co-op apartment']
        }
        mask = np.in1d(data['name_housing_type'].values,merge_dict['Coop/Office'])
        data.loc[mask,'name_housing_type'] = 'Coop/Office'
        
        
        # name_education_type
        merge_dict = {
            'Business_Student':['Businessman','Student'],
            'Others':['Maternity Leave','Unemployed']
        }
        for key,val in merge_dict.items():
            mask = np.in1d(data['name_income_type'].values,np.array(val))
            data.loc[mask,'name_income_type'] = key
        
            
        # name_education_type
        merge_dict = {
            'Higher education':['Academic degree','Higher education']
        }
        mask = np.in1d(data['name_education_type'].values,merge_dict['Higher education'])
        data.loc[mask,'name_education_type'] = 'Higher education'
        
        
        # organization_type
        merge_dict = {
            'Business_entity':['Business Entity Type 1','Business Entity Type 2','Business Entity Type 3'],
            'Industry_type1':['Industry: type 1','Industry: type 3','Industry: type 4','Industry: type 8','Industry: type 11','Industry: type 13'],
            'Industry_type2':['Industry: type 2','Industry: type 5','Industry: type 6','Industry: type 7','Industry: type 9','Industry: type 10','Industry: type 12'],
            'Trade_type1':['Trade: type 1','Trade: type 3','Trade: type 7'],
            'Trade_type2':['Trade: type 2','Trade: type 4','Trade: type 5','Trade: type 6'],
            'Transport_type1':['Transport: type 1','Transport: type 2'],
            'Transport_type2':['Transport: type 3','Transport: type 4']
        }
        for key,val in merge_dict.items():
            cat_mask = np.in1d(data['organization_type'].values,np.array(val))
            data.loc[cat_mask,'organization_type'] = key
        
            
        # cnt_children
        data['cnt_children'] = data['cnt_children'].apply(int).apply(lambda x: str(x) if x < 4 else '4+')
        
        
        # Cnt_fam_members
        cnt_fam_mem_var = data['cnt_fam_members'].copy() # needed later for creating a certain ratio in the FE part
        data['cnt_fam_members'] = data['cnt_fam_members'].apply(int).apply(lambda x: str(x) if x < 5 else '5+')
        
        
        # hour_appr_process_start
        cond1 = data['hour_appr_process_start'] > 18 
        cond2 = data['hour_appr_process_start'] < 8
        
        data.loc[cond1 | cond2,'hour_appr_process_start'] = 'irregular_hour'
        data.loc[~(cond1 | cond2),'hour_appr_process_start'] = 'regular_hour'
        
        
        # 'reg_city_not_live_city' & 'reg_city_not_work_city'
        data['reg_city_not_liwo'] = data.loc[:,['reg_city_not_live_city','reg_city_not_work_city']].sum(axis=1)
        data = data.drop(labels=['reg_city_not_live_city','reg_city_not_work_city'],axis=1)
        
        print(">>> Data Cleaning: Done.")
        
        
        # 3. Feature Engineering
        #-------------------------
        # 3.1. Some new ratios
        data['app_age'] = round(data['days_birth'] / 365)
        data['employment_time_share'] = data['days_employed'] / data['days_birth']
        data['income_credit_perc'] = data['amt_income_total'] / data['amt_credit']
        data['income_per_person'] = data['amt_income_total'] / cnt_fam_mem_var
        data['annuity_income_perc'] = data['amt_annuity'] / data['amt_income_total']
        data['payment_rate'] = data['amt_annuity'] / data['amt_credit']
        data['loan_value_ratio'] = data['amt_credit'] / data['amt_goods_price']
        
        # 3.2. cnt_payment & APRs
        # Load model that predicts the payment periods, trained on previous applications
        with open(self.models_path + model_filename +'.pickle', 'rb') as handle:
            opt_model = pkl.load(handle)
        
        # Prepare input data for the model
        pred_data = data.loc[:,['sk_id_curr','amt_annuity','amt_credit']]
        pred_data['app_grant_ratio'] = data['amt_credit'] / data['amt_goods_price']
        pred_data['credit_annuity_ratio'] = data['amt_credit'] / data['amt_annuity']
        
        # Make predictions
        pred_data['cnt_payment'] = opt_model.predict(pred_data.loc[:,'amt_annuity':]).round()
        
        print(">>> Constructing APR Feature:")
        
        # Merge predictions with main dataframe
        data = data.merge(pred_data.loc[:,['sk_id_curr','cnt_payment']])
        
        # Construct APR-Feature
        arg_dict={
            'data':data,
            'pmt_col':'amt_annuity',
            'period_col':'cnt_payment',
            'principal_col':'amt_credit',
            'step_size':step_size,
            'block_size':block_size
        }
        apr_df = hcf.interest_rates(**arg_dict)
        
        data = pd.concat([data,apr_df],axis=1)
        
        if corr_remove == True:
            # 3.3. Drop highly correlated features
            _,td = hcf.corr_dropouts(data,corr_thres=corr_thres)
    
            data = data.drop(labels=td,axis=1)
            
        
        # 3.4. Dummy Variabels for Categoricals
        dtype_dict = hcf.custom_dtype_identifier(data,thres=25)

        col_list = np.array([
            *dtype_dict['object'],
            *dtype_dict['pseudo_cat'],
            *dtype_dict['chr_dummies']
        ])
        mask = np.array(col_list) != 'target'
        col_list = col_list[mask]
        
        just_dummies = pd.get_dummies(data.loc[:,col_list],drop_first=True)
        
        data = data.drop(labels=col_list,axis=1)
        data = pd.concat([data,just_dummies],axis=1)
        data = data.sort_values(by='sk_id_curr').reset_index(drop=True)
        
        # 4. Save Data
        #----------------
        data.to_parquet(self.output_path + 'application_train_cleaned.parquet',engine='pyarrow')
        
        print(f">>> Cleaned Data Set has been saved to: \n\t {self.output_path}")
        
        
        
    def bureau_data(self,
                    bur_filename: str,
                    bal_filename: str,
                    app_filename: str,
                    corr_thres: float = 0.7,
                    ):
        """
        Method that produces the cleaned bureau data set with all the required features for modeling 
        the default rates. Data set will be saved to the output path that was specified while creating 
        the Feature_Gen class. 

        Parameters
        ----------
        bur_filename : str
            Name of the file that contains the original 'bureau' data. File extensions 
            are not required (e.g. '.csv').
        bal_filename : str
            Name of the file that contains the original 'bureau_balance' data. File extensions 
            are not required (e.g. '.csv').
        app_filename : str
            Name of the file that contains the original 'application train' data, including the acutal
            target variable. File extensions are not required (e.g. '.csv').
        model_filename : str
            Name of the file that contains the XGB model that was trained on the previous application data 
            to predict the repayment period which is required to derive the interest rate of the loans,
            stated as an APR. File extensions are not required (e.g. '.pkl').
        corr_thres : float, optional
            Threshold value to determine when highly correlated columns have to be dropped. The variable
            which has the higher correlation with the target variable is kept.
            The default is 0.7.
            

        Returns
        -------
        Cleaned bureau data set with the necessary features for modeling. Data is saved to a 
        specified output path as a parquet-file.

        """ 
        # 1. Load Data
        #---------------
        data_bur = pd.read_csv(self.input_path + bur_filename + '.csv')
        data_bal = pd.read_csv(self.input_path + bal_filename + '.csv')
        data_app = pd.read_csv(self.input_path + app_filename + '.csv')
        
        data_bur.columns = [col.lower() for col in data_bur.columns]
        data_bal.columns = [col.lower() for col in data_bal.columns]
        data_app.columns = [col.lower() for col in data_app.columns]
        
        data_bur = data_bur.drop(labels=[
            'amt_credit_sum_overdue',
            'amt_credit_max_overdue',
            'credit_day_overdue',
            'cnt_credit_prolong',
            'credit_currency'],axis=1)
        print(">>> Data Loading: Done.")
        
        # 2. Bureau Balance: Data Cleaning  + Feature Engineering
        #----------------------------------------------------------
        # Reduce memory usage, one-hot encode and transform months_balance into positive integers:
        data_bal = hcf.reduce_memory(data_bal)
        data_bal = pd.get_dummies(data_bal,columns=['status'],drop_first=False)
        data_bal['months_balance'] = data_bal['months_balance'].apply(lambda x: abs(x))
        
        
        # Create first part of aggregated data that will be ready to be joined to bureau.csv data
        agg_dict = {
            'months_balance':'max',
            'status_C':'mean'
        }
        
        db_agg = data_bal.groupby('sk_id_bureau').agg(agg_dict).reset_index().rename(columns={'months_balance':'size'})
        
        
        # Second part of aggregated data that will be joined to bureau.csv
        # Important here: pay_delay shares are calculated based on data that excludes periods in which status = 'C'
        #                 This stresses the occurance of payment delays more and it also prevents strong colinearity 
        #                 between "pay_delay" and "status_C" (from first part, see above).
        agg_dict = {f'status_{i}':'mean' for i in range(6)}
        
        db_agg_lC = data_bal.loc[data_bal['status_C']==0,:].groupby('sk_id_bureau').agg(agg_dict).reset_index()
        db_agg_lC['pay_delay'] = db_agg_lC['status_1'] + db_agg_lC['status_2'] + db_agg_lC['status_3'] + db_agg_lC['status_4'] + db_agg_lC['status_5']
        db_agg_lC = db_agg_lC.drop(labels=[f'status_{i}' for i in np.arange(1,6)],axis=1)
        db_agg = db_agg.merge(db_agg_lC,how='left').fillna(0)
        
        del data_bal
        print(">>> Preparing 'bureau_balance' Data Set: Done.")
        
        # 3. Bureau Data: Data Cleaning + Feature Engineering
        #------------------------------------------------------
        data_bur = data_bur.merge(db_agg,how='left').merge(data_app.loc[:,['sk_id_curr','target']],how='left')
        
        # 3.1. Merge insignificant category attributes and dummify
        loan_list1 = ['Loan for the purchase of equipment','Loan for working capital replenishment','Microloan']
        data_bur.loc[data_bur['credit_type'].isin(loan_list1),'credit_type'] = 'Microloans_and_Others'
        
        loan_list2 = ['Microloans_and_Others','Mortgage','Credit card','Consumer credit','Car loan','Mortgage']
        data_bur.loc[~(data_bur['credit_type'].isin(loan_list2)),'credit_type'] = 'Others'
        
        
        data_bur = pd.get_dummies(data_bur,columns=['credit_active','credit_type'],drop_first=False)
        
        
        # 3.2. Aggregate on 'sk_id_curr'-level
        data_bur['days_enddate_fact'] = data_bur['days_enddate_fact'].apply(lambda x: abs(x))

        agg_list = []
        for idx,col in enumerate(data_bur.loc[:,'days_credit':'pay_delay'].columns):
            data_agg = data_bur.groupby('sk_id_curr').agg({col:['mean','min','max']}).reset_index()
            col_list = ['sk_id_curr',f'{col}_mean',f'{col}_min',f'{col}_max']
            data_agg.columns = col_list
            if idx != 0:
                data_agg = data_agg.drop(labels='sk_id_curr',axis=1)
            agg_list.append(data_agg)
            
        agg_df1 = pd.concat(agg_list,axis=1)
        
        
        agg_list = []
        for idx,col in enumerate(data_bur.loc[:,'credit_active_Active':].columns):
            data_agg = data_bur.groupby('sk_id_curr').agg({col:'mean'}).reset_index()
            col_list = ['sk_id_curr',f'{col}_mean']
            data_agg.columns = col_list
            if idx != 0:
                data_agg = data_agg.drop(labels='sk_id_curr',axis=1)
            agg_list.append(data_agg)
            
        agg_df2 = pd.concat(agg_list,axis=1)
        
        agg_df = agg_df1.merge(agg_df2,how='left').merge(data_app.loc[:,['sk_id_curr','target']],how='left')
        
        
        del agg_df1, agg_df2, agg_list, data_agg
        
        
        # Get rid of data points where target is misssing
        agg_df = agg_df.loc[~agg_df['target'].isnull(),:]
        
        
        # 3.3. Deal with Missing Values on aggregated data: Fill NA's with 0's
        for col in agg_df.columns:
            agg_df.loc[agg_df[col].isnull(),col] = 0
            
        
        # 3.4. Outlier Removal for certain columns
        col_list = [col for col in agg_df.loc[:,'days_credit_mean':'amt_credit_sum_debt_max'].columns]
        
        agg_df = hcf.outlier_remover(agg_df,col_list=col_list,sf=3,copy=True)
        
        
        # 3.5. Drop highly correlated Variables
        _, td = hcf.corr_dropouts(agg_df,corr_thres=corr_thres)
        agg_df = agg_df.drop(labels=['target',*td],axis=1).reset_index(drop=True)
        agg_df = agg_df.sort_values(by='sk_id_curr').reset_index(drop=True)
        print(">>> Preparing 'bureau' Data Set: Done.")
        
        # 4. Save Data
        #----------------
        agg_df.to_parquet(self.output_path + 'bureau_data_cleaned.parquet')
        
        print(f">>> Cleaned Data Set has been saved to: \n\t {self.output_path}")
        
        
        
    def credit_card_bal_data(self,
                             ccb_filename: str,
                             app_filename: str,
                             corr_thres: float = 0.7):
        """
        Method that produces the cleaned credit card balance data set with all the required features 
        for modeling the default rates. Data set will be saved to the output path that was specified 
        while creating the Feature_Gen class. 

        Parameters
        ----------
        ccb_filename : str
            Name of the file that contains the original 'credit card balance' data. File extensions 
            are not required (e.g. '.csv').
        app_filename : str
            Name of the file that contains the original 'application train' data, including the acutal
            target variable. File extensions are not required (e.g. '.csv').
        corr_thres : float, optional
            Threshold value to determine when highly correlated columns have to be dropped. The variable
            which has the higher correlation with the target variable is kept.
            The default is 0.7.

        Returns
        -------
        Cleaned credit card balance data set with the necessary features for modeling. Data is saved 
        to a specified output path as a parquet-file.
        
        """
        # 1. Load Data
        #---------------
        data = pd.read_csv(self.input_path + ccb_filename + '.csv')
        data_app = pd.read_csv(self.input_path + app_filename + '.csv')
        
        data.columns = [col.lower() for col in data.columns]
        data_app.columns = [col.lower() for col in data_app.columns]
        
        # Join target variable from application_train.csv and delete cases where target = NaN after joining 
        data = data.merge(data_app.loc[:,['sk_id_curr','target']],how='left')
        data = data.loc[~data['target'].isnull(),:]
        data = data.sort_values(by=['sk_id_curr','months_balance']).reset_index(drop=True)
        data = data.rename(columns={'amt_recivable':'amt_receivable'})
        
        # int to float
        data['sk_dpd'] = data['sk_dpd'].astype(np.float64)
        data['sk_dpd_def'] = data['sk_dpd_def'].astype(np.float64)
        
        # float to int
        data['target'] = data['target'].astype(np.int64)
        data['cnt_instalment_mature_cum'] = data['cnt_instalment_mature_cum'].fillna(0).astype(np.int64)
        data['amt_inst_min_regularity'] = data['amt_inst_min_regularity'].fillna(0).astype(np.int64)
        
        del data_app
        print(">>> Data Loading: Done.")
        
        # 2. Data Cleaning and Feature Engineering
        #-------------------------------------------
        # 2.1. Adding new Features 
        # liquidation factors & LOC ratio
        data['loc_ratio'] = data['amt_balance'] / data['amt_credit_limit_actual']
        data['liquid_factor_balance'] = data['amt_balance'] / data['amt_payment_total_current']
        data['liquid_factor_total'] = data['amt_credit_limit_actual'] / data['amt_payment_total_current']
        data.replace(np.inf,0,inplace=True)
        
        # overdraft_dummy
        mask = data['amt_balance'] > data['amt_credit_limit_actual']
        
        data['overdraft_dummy'] = 0
        data.loc[mask,'overdraft_dummy'] = 1
        
        # average withdrawal amounts
        data['avg_atm_amount'] = data['amt_drawings_atm_current'] / data['cnt_drawings_atm_current']
        data['avg_pos_amount'] = data['amt_drawings_pos_current'] / data['cnt_drawings_pos_current']
        data['avg_other_amount'] = data['amt_drawings_other_current'] / data['cnt_drawings_other_current']
        
        
        # 2.2. Drop unimportant columns
        col_list = []
        for string in ['payment','receivable','drawings']:
            mask = [bool(re.search(f'{string}',col)) for col in data.columns]
            tmp = data.columns[mask]
            col_list.extend(tmp)
            
        data = data.drop(labels=col_list,axis=1)
        
        
        # 2.3. Aggregations on 'sk_id_currr'-level
        # 2.3.1. Float-columns
        agg_df = data.loc[:,['sk_id_prev','sk_id_curr','target']].drop_duplicates()
        for col in data.columns:
            if data[col].dtype == 'float64':
                agg_dict = {
                    col:['mean','std','min','max']
                }
        
                tmp = data.groupby('sk_id_curr').agg(agg_dict).reset_index().fillna(0)
        
                cols = [f'{tmp.columns[i][0]}_{tmp.columns[i][1]}' for i in range(1,len(tmp.columns))]
        
                tmp.columns = ['sk_id_curr',*cols]
        
                agg_df = agg_df.merge(tmp,how='left')
            
            else:
                pass
            
        agg_df = agg_df.drop(labels=['sk_dpd_def_min','sk_dpd_min'],axis=1)
        
        # 2.3.2. Integer-Columns
        tmp_int = data.groupby('sk_id_curr').agg({
            'amt_credit_limit_actual':'max',
            'amt_inst_min_regularity':'max',
            'months_balance':'min',
            'overdraft_dummy':'sum'
        }).reset_index().rename(columns={
            'amt_credit_limit_actual':'max_credit_lim',
            'amt_inst_min_regularity':'max_regular_payment',
            'months_balance':'contract_age'
        })
        
        tmp_int['contract_age'] = tmp_int['contract_age'].apply(lambda x: abs(x))
        tmp_int['overdraft_ratio'] = tmp_int['overdraft_dummy'] / tmp_int['contract_age']
        tmp_int = tmp_int.drop(labels='overdraft_dummy',axis=1)
        
        agg_df = agg_df.merge(tmp_int,how='left')
        
        
        # 2.4. Outlier Removal
        # Select only float colummns for outlier removal
        float_list = []
        for col in agg_df.columns:
            if 'float' in str(agg_df[col].dtype):
                float_list.append(col)
            else:
                pass
        
        agg_df = hcf.outlier_remover(df=agg_df,col_list=float_list,sf=3)
        
        
        # 2.5. Remove Features that are heavily correlated
        _,td = hcf.corr_dropouts(data=agg_df,corr_thres=corr_thres)
        
        agg_df = agg_df.drop(labels=td,axis=1)
        print(">>> Data Cleaning & Feature Engineering: Done.")
        
        # 3. Save Data
        #-----------------
        agg_df = agg_df.drop(labels=['sk_id_prev','target'],axis=1)
        
        agg_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in agg_df.columns:
            agg_df = agg_df.drop(labels=np.where(agg_df[col].isnull())[0],axis=0).reset_index(drop=True)
        
        agg_df = agg_df.drop_duplicates(subset=['sk_id_curr'],keep='first')
        agg_df = agg_df.sort_values(by='sk_id_curr').reset_index(drop=True)
        agg_df.to_parquet(self.output_path + 'credit_card_balance_cleaned.parquet')
        print(f">>> Cleaned Data Set has been saved to: \n\t {self.output_path}")
        
        
        
    def installment_data(self,
                         inst_filename: str,
                         app_filename: str,
                         corr_thres: float = 0.7):
        """
        Method that produces the installment payments data set with all the required features 
        for modeling the default rates. Data set will be saved to the output path that was specified 
        while creating the Feature_Gen class.

        Parameters
        ----------
        inst_filename : str
            Name of the file that contains the original 'installment payments' data. File extensions 
            are not required (e.g. '.csv').
        app_filename : str
            Name of the file that contains the original 'application train' data, including the acutal
            target variable. File extensions are not required (e.g. '.csv').
        corr_thres : float, optional
            Threshold value to determine when highly correlated columns have to be dropped. The variable
            which has the higher correlation with the target variable is kept.
            The default is 0.7.

        Returns
        -------
        Cleaned installment payments data set with the necessary features for modeling. Data is saved 
        to a specified output path as a parquet-file.
        
        """
        # 1. Load Data
        #----------------
        data = pd.read_csv(self.input_path + inst_filename + '.csv')
        data_app = pd.read_csv(self.input_path + app_filename + '.csv')
        
        data.columns = [col.lower() for col in data.columns]
        data_app.columns = [col.lower() for col in data_app.columns]
        
        # Join target variable from application_train.csv and delete cases where target = NaN after joining 
        data = data.merge(data_app.loc[:,['sk_id_curr','target']],how='left')
        data = data.loc[~data['target'].isnull(),:]
        
        del data_app
        print(">>> Data Loading: Done.")
        
        # 2. Data Cleaning & Feature Engineering
        #-----------------------------------------
        # 2.1. Create new Features
        result_df = data.groupby('sk_id_curr')['target'].max().reset_index()
        result_df['target'] = result_df['target'].astype(np.int8)
        
        # avg_num_instalments (total + last 365 days)
        df_total = data.groupby(['sk_id_curr','sk_id_prev'])['num_instalment_number'].max().reset_index()
        df_total = df_total.groupby('sk_id_curr')['num_instalment_number'].mean().reset_index().rename(columns={
            'num_instalment_number':'avg_num_instalments_total'
        })
        
        group_obj = data.loc[data['days_entry_payment'] >= -365,:].groupby(['sk_id_curr','sk_id_prev'])
        df_365 = group_obj['num_instalment_number'].max().reset_index()
        df_365 = df_365.groupby('sk_id_curr')['num_instalment_number'].mean().reset_index().rename(columns={
            'num_instalment_number':'avg_num_instalments_365'
        })
        
        result_df = result_df.merge(df_total,how='left').merge(df_365,how='left').fillna(0)
        
        # late_payments_share (total + 365)
        data['pay_delay'] = (data['days_instalment'].apply(abs) - data['days_entry_payment'].apply(abs)).fillna(0)
        data['delay_dummy'] = data['pay_delay'].apply(lambda x: 1 if x < 0 else 0)    
        
        df_total = data.groupby('sk_id_curr')['delay_dummy'].mean().reset_index().rename(columns={
            'delay_dummy':'late_payments_share_total'
        })
        
        group_obj = data.loc[data['days_entry_payment'] >= -365,:].groupby('sk_id_curr')
        df_365 = group_obj['delay_dummy'].mean().reset_index().rename(columns={
            'delay_dummy':'late_payments_share_365'
        })
        
        result_df = result_df.merge(df_total,how='left').merge(df_365,how='left').fillna(0)
        
        # mean_delay + max_delay (total + last 365 days)
        # totals 
        df_total = data.groupby('sk_id_curr').agg({'pay_delay':['min','mean']}).reset_index()
        df_total.columns = ['sk_id_curr','max_delay_total','mean_delay_total']
        
        df_total['max_delay_total'] = df_total['max_delay_total'].apply(abs)
        df_total['mean_delay_total'] = df_total['mean_delay_total'].apply(abs)
        
        # last 365 days
        group_obj = data.loc[data['days_entry_payment'] >= -365,:].groupby('sk_id_curr')
        df_365 = group_obj.agg({'pay_delay':['min','mean']}).reset_index()
        df_365.columns = ['sk_id_curr','max_delay_365','mean_delay_365']
        
        df_365['max_delay_365'] = df_365['max_delay_365'].apply(abs)
        df_365['mean_delay_365'] = df_365['mean_delay_365'].apply(abs)
        
        result_df = result_df.merge(df_total,how='left').merge(df_365,how='left').fillna(0)
        
        
        # payment_fail_num + payment_fail_share (total + last 365 days)
        # total
        data['pfail_dummy'] = (data['amt_payment'] < data['amt_instalment']).apply(lambda x: 1 if x == True else 0)
        
        df_total = data.groupby('sk_id_curr').agg({'pfail_dummy':['sum','mean']}).reset_index()
        df_total.columns = ['sk_id_curr','payment_fail_num_total','payment_fail_share_total']
        
        # last 365 days
        group_obj = data.loc[data['days_entry_payment'] >= -365,:].groupby('sk_id_curr')
        df_365 = group_obj.agg({'pfail_dummy':['sum','mean']}).reset_index()
        df_365.columns = ['sk_id_curr','payment_fail_num_365','payment_fail_share_365']
        
        result_df = result_df.merge(df_total,how='left').merge(df_365,how='left').fillna(0)
        
        
        # repayed_principal_share (total + last 365 days)
        # total
        num_df_total = data.groupby(['sk_id_curr','sk_id_prev'])['amt_payment'].sum().reset_index()
        denom_df_total = data.groupby(['sk_id_curr','sk_id_prev'])['amt_instalment'].sum().reset_index()
        
        num_df_total = num_df_total.merge(denom_df_total,how='left')
        num_df_total['repayed_principal_share'] = num_df_total['amt_payment'] / num_df_total['amt_instalment']
        num_df_total = num_df_total.groupby('sk_id_curr')['repayed_principal_share'].mean().reset_index().rename(columns={
            'repayed_principal_share':'repayed_principal_share_total'
        })
        
        # last 365 days
        df_365 = data.loc[data['days_entry_payment'] >= -365,:]
        num_df_365 = df_365.groupby(['sk_id_curr','sk_id_prev'])['amt_payment'].sum().reset_index()
        denom_df_365 = df_365.groupby(['sk_id_curr','sk_id_prev'])['amt_instalment'].sum().reset_index()
        
        num_df_365 = num_df_365.merge(denom_df_365,how='left')
        num_df_365['repayed_principal_share'] = num_df_365['amt_payment'] / num_df_365['amt_instalment']
        num_df_365 = num_df_365.groupby('sk_id_curr')['repayed_principal_share'].mean().reset_index().rename(columns={
            'repayed_principal_share':'repayed_principal_share_365'
        })
        
        result_df = result_df.merge(num_df_total,how='left').merge(num_df_365,how='left').fillna(0)
        
        
        # 2.2. Remove highly correlated variables
        tk,td = hcf.corr_dropouts(result_df,corr_thres=0.7)

        result_df = result_df.drop(labels=td,axis=1)
        print(">>> Data Cleaning & Feature Engineering: Done.")
        
        
        # 3. Save Data
        #----------------
        result_df = result_df.drop(labels='target',axis=1)
        
        result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in result_df.columns:
            result_df = result_df.drop(np.where(result_df[col].isnull())[0],axis=0).reset_index(drop=True)
        
        result_df = result_df.sort_values(by='sk_id_curr').reset_index(drop=True)
        result_df.to_parquet(self.output_path + 'installments_payments_cleaned.parquet',engine='pyarrow')
        print(f">>> Cleaned Data Set has been saved to: \n\t {self.output_path}")
        
        
        
    def horizontal_sets(self,
                        file_name: str,
                        full_data_path: str=None,
                        random_state: int = 2021
                        ):
        """
        Creates 3 data sets for horizontal federated learning.

        Parameters
        ----------
        filename : str
            Name of the file that contains the full data set (data set with all variables).
        full_data_path : str, optional
            Path of the file that contains the full data set. The default is None.
        random_state : int, optional
            Sets seed for random number generator (numpy). The default is 2021.
            
        Returns
        -------
        None.

        """
        if full_data_path is None:
            data = pd.read_parquet(self.full_data_path + file_name + '.parquet')
        else:
            data = pd.read_parquet(self.output_path + file_name + '.parquet')
            
        data['age'] = data['days_birth'].apply(lambda x: abs(round(x/365,2)))
            
        conditions = [
            (data["age"].lt(35)),
            (data["age"].ge(35) & data["age"].lt(55)),
            (data["age"].ge(55)),
        ]
        choices = ["young", "mature", "elderly"]
        data["age_class"] = np.select(conditions, choices)
            
            
        conditions = [
            (data['region_population_relative'].lt(0.01)),
            (data['region_population_relative'].ge(0.01) & data['region_population_relative'].lt(0.03)),
            (data['region_population_relative'].ge(0.03))
        ]
        choices = ["rural","suburbs","city"]
        data['pop_class'] = np.select(conditions,choices)

        np.random.seed(random_state)
        global_test_set_ids = np.random.choice(data['sk_id_curr'].values,size=15_000,replace=False)
        global_train_set_ids = list(set(global_test_set_ids) ^ set(data['sk_id_curr'].values))
            
        data_global_test = data.loc[data['sk_id_curr'].isin(global_test_set_ids),:]
        data = data.loc[data['sk_id_curr'].isin(global_train_set_ids),:]
        
        data1 = data.loc[data['pop_class'] == 'city',:]
        data2 = data.loc[data['pop_class'] == 'suburbs',:]
        data3 = data.loc[data['pop_class'] == 'rural',:]
            
        idx_y = np.random.choice(data2.loc[data2['age_class'] == 'young','sk_id_curr'].values,size=int(0.95*15_000),replace=False)
        idx_e = np.random.choice(data2.loc[data2['age_class'] == 'elderly','sk_id_curr'].values,size=int(0.01*15_000),replace=False)
        idx_s = np.random.choice(data2.loc[data2['age_class'] == 'mature','sk_id_curr'].values,size=int(0.04*15_000),replace=False)
                        
        idx_swap = np.array([*idx_y,*idx_e,*idx_s])
        idx_remain = list(set(np.unique(data2['sk_id_curr'])) ^ set(idx_swap))
            
        data1 = pd.concat([data1,data2.loc[data2['sk_id_curr'].isin(idx_swap),:]],axis=0).reset_index(drop=True)
        data2 = data2.loc[data2['sk_id_curr'].isin(idx_remain),:].reset_index(drop=True)


        idx_y = np.random.choice(data3.loc[data3['age_class'] == 'young','sk_id_curr'].values,size=int(0.59*25_000),replace=False)
        idx_e = np.random.choice(data3.loc[data3['age_class'] == 'elderly','sk_id_curr'].values,size=int(0.01*25_000),replace=False)
        idx_s = np.random.choice(data3.loc[data3['age_class'] == 'mature','sk_id_curr'].values,size=int(0.40*25_000),replace=False)
            
        idx_swap = np.array([*idx_y,*idx_e,*idx_s])
        idx_remain = list(set(np.unique(data3['sk_id_curr'])) ^ set(idx_swap))
            
        data2 = pd.concat([data2,data3.loc[data3['sk_id_curr'].isin(idx_swap),:]],axis=0).reset_index(drop=True)
        data3 = data3.loc[data3['sk_id_curr'].isin(idx_remain),:].reset_index(drop=True)
        
        # Save the data 
        data_global_test.drop(labels=['age' ,'age_class', 'pop_class'],axis=1).to_parquet(self.output_path + 'horizontal_global_test_data.parquet')
        data.drop(labels=['age' ,'age_class', 'pop_class'],axis=1).to_parquet(self.output_path + 'horizontal_full_data.parquet')
        data1.drop(labels=['age', 'age_class', 'pop_class'],axis=1).to_parquet(self.output_path + 'horizontal_city_data.parquet')
        data2.drop(labels=['age', 'age_class', 'pop_class'],axis=1).to_parquet(self.output_path + 'horizontal_suburbs_data.parquet')
        data3.drop(labels=['age', 'age_class', 'pop_class'],axis=1).to_parquet(self.output_path + 'horizontal_rural_data.parquet')
        print(f">>> Data sets have been saved to: \n\t\t {self.output_path}")
        
        
        
    def horizontal_noniid(self,
                          file_name: str,
                          size_proportions: list,
                          def_rates: list,
                          full_data_path: str=None,
                          global_data_size: int = 15_000,
                          random_state: int = 2021,
                          iid:bool = False
                          ):
        """
        Creates 3 data sets for horizontal federated learning with differently distributed
        target variables (i.e. non-iid case).

        Parameters
        ----------
        filename : str
            Name of the file that contains the full data set (data set with all variables).
        size_proportions : list
            States size proportions of the three horizontal data sets.
        def_proportions : list
            States desired average default rates per horizontal data set to achieve a 
            non-iid scenario. 
        full_data_path : str, optional
            Path of the file that contains the full data set. The default is None.
        global_data_size : int, optional
            Size of the global test set, whose target variable follows the global 
            distribuiton. The default is 15,000.
        random_state : int, optional
            Sets seed for random number generator (numpy). The default is 2021.
        iid : bool, optional
            Boolean that is used to change the output file names from "noniid" to "iid". 
            
        Returns
        -------
        None.

        """  
        if full_data_path is None:
            data = pd.read_parquet(self.full_data_path + file_name + '.parquet')
        else:
            data = pd.read_parquet(self.output_path + file_name + '.parquet')
        data.columns = [col.lower() for col in data.columns]
        
        np.random.seed(random_state)
        
        global_test_ids = np.random.choice(data['sk_id_curr'],size=global_data_size,replace=False)
        global_test_data = data.loc[data['sk_id_curr'].isin(global_test_ids),:]
        data = data.loc[~data['sk_id_curr'].isin(global_test_ids),:]
        d0 = np.array(data.loc[data['target']==0,'sk_id_curr'])
        d1 = np.array(data.loc[data['target']==1,'sk_id_curr'])
        
        size01 = int((size_proportions[0]/sum(size_proportions)) * len(d0))
        size02 = int((size_proportions[1]/sum(size_proportions)) * len(d0))
        
        size11 =  int((def_rates[0]/(1-def_rates[0])) * size01)
        size12 =  int((def_rates[1]/(1-def_rates[1])) * size02)
        
        hd01 = np.random.choice(d0,size=size01,replace=False)
        hd02 = np.random.choice(np.setdiff1d(d0,hd01),size=size02,replace=False)
        hd03 = np.setdiff1d(d0,np.array([*hd01,*hd02]))
        
        hd11 = np.random.choice(d1,size=size11,replace=False)
        hd12 = np.random.choice(np.setdiff1d(d1,hd11),size=size12,replace=False)
        hd13 = np.setdiff1d(d1,np.array([*hd11,*hd12]))
        
        data1 = pd.concat([data.loc[data['sk_id_curr'].isin(hd01),:],data.loc[data['sk_id_curr'].isin(hd11),:]])
        data2 = pd.concat([data.loc[data['sk_id_curr'].isin(hd02),:],data.loc[data['sk_id_curr'].isin(hd12),:]])
        data3 = pd.concat([data.loc[data['sk_id_curr'].isin(hd03),:],data.loc[data['sk_id_curr'].isin(hd13),:]])
        
        if iid == False:
            global_test_data.to_parquet(self.output_path + 'horizontal_noniid_global.parquet')
            data1.to_parquet(self.output_path + 'horizontal_noniid_data1.parquet')        
            data2.to_parquet(self.output_path + 'horizontal_noniid_data2.parquet')        
            data3.to_parquet(self.output_path + 'horizontal_noniid_data3.parquet')        
            print(f">>> Data sets have been saved to: \n\t\t {self.output_path}")
        else:
            global_test_data.to_parquet(self.output_path + 'horizontal_iid_global.parquet')
            data1.to_parquet(self.output_path + 'horizontal_iid_data1.parquet')        
            data2.to_parquet(self.output_path + 'horizontal_iid_data2.parquet')        
            data3.to_parquet(self.output_path + 'horizontal_iid_data3.parquet')        
            print(f">>> Data sets have been saved to: \n\t\t {self.output_path}")

        
    def full_set(self,
                ccb_include: bool = True, 
                random_state: int = 2021,
                train_size: float = 0.7,
                corr_thres: float = 0.7,
                block_size: int = 1_000,
                step_size: float = 0.01,
                corr_remove: bool = False,
                outlier_remove: bool = True,
                ext_opt: bool = False,
                cv_folds: int = 3,
                n_iter: int = 5,
                n_jobs: int = -1
                ):
        """
        

        Parameters
        ----------
        ccb_include : bool, optional
            Indicates whether credit_card_balance should be included in the full data set. The default is True.
        random_state : int, optional
            Set the random seed. The default is 2021.
        train_size : float, optional
            Indicates the size of the training set in relative terms. The default is 0.7.
        corr_thres : float, optional
            Correlation threshold for delecting features that suffer from colinearity. The default is 0.7.
        block_size : int, optional
            Intake per iteration of the interest_rate function that produces the APR feature. High values
            (i.e. > 2_000) can lead to memory consumption issues. The default is 1_000.
        step_size : float, optional
            Indicates the accuracy of the APR feature. Smaller values give a better accuracy but increase 
            execution time. The default is 0.01.
        corr_remove : bool, optional
            Controls if highly correlated variables are dropped if they exceede corr_thres. Setting this,
            parameter to False is equivalent to set corr_thres to 1. The default is False.
        outlier_remove : bool, optional
            Indicates if outliers should be removed. The default is True.
        ext_opt : bool, optional
            Controls if optimized feature of external sources should be included. The default is False.
        cv_folds : int, optional
            Number of cross validations. The default is 3.
        n_iter : int, optional
            Number of iterations to use for Bayesion optimization to tune the XGB model for creating the 
            APR feature. The default is 5.
        n_jobs : int, optional
            Number of cores to use of the local machine. The default is -1.

        Returns
        -------
        None.

        """
        np.random.seed(random_state)
        
        data_list = []
        dir_list = os.listdir(self.output_path)
        datasets = [
            'prev_app_cleaned.parquet',
            'application_train_cleaned.parquet',
            'bureau_data_cleaned.parquet',
            'credit_card_balance_cleaned.parquet',
            'installments_payments_cleaned.parquet'  
        ]
        
        # Previous Application 
        if datasets[0] in dir_list:
            pass
        else:
            arg_dict = {
                'prev_app_filename': 'previous_application',
                'app_filename':'application_train',
                'save_int_rates': True,
                'corr_thres': corr_thres,
                'step_size': step_size,
                'block_size': block_size
            }
            self.prev_app(**arg_dict)
            
            arg_dict_hpt = {
                'n_estimators':hp.quniform('n_estimators',50,300,25),
                'max_depth': hp.quniform('max_depth',1,7,1),
                'learning_rate':hp.uniform('learning_rate',0.15,0.4),
                'subsample': hp.uniform('subsample',0.3,0.5),
                'gamma': hp.uniform('gamma',0.2,0.9)
            }

            apr_model = nfm.NonFed_Models(
                data_path = self.output_path,
                file_name = 'interest_rates_results',
                param_grid =  arg_dict_hpt,
                save_path = self.models_path,
                n_jobs = n_jobs,
            )
            
            apr_model.fit_APR_model(
                param_dict = arg_dict_hpt,
                random_search = False,
                train_size = train_size,
                random_state = random_state,
                shuffle = True,
                n_jobs = n_jobs,
                n_iter = n_iter,
                verbose = 1,
                cv_folds = cv_folds,
                save_model = True
            )
            print(80*"=")
            
        # Application Train
        if datasets[1] in dir_list:
            pass
        else:
            arg_dict = {
            'app_filename':'application_train',
            'model_filename':'hyper_opt_model',
            'corr_thres': corr_thres,
            'step_size':step_size,
            'block_size':block_size,
            'corr_remove':corr_remove,
            'outlier_remove':outlier_remove,
            'ext_opt':ext_opt
            }
            self.app_data(**arg_dict)
            print(80*"=")
        
        # Bureau Data
        if datasets[2] in dir_list:
            pass
        else:
            arg_dict = {
                'bur_filename':'bureau',
                'bal_filename':'bureau_balance',
                'app_filename':'application_train',
                'corr_thres': corr_thres
            }
            
            self.bureau_data(**arg_dict)
            print(80*"=")
            
        # Installments Payments
        if datasets[3] in dir_list:
            pass
        else:
            arg_dict = {
                'inst_filename':'installments_payments',
                'app_filename':'application_train',
                'corr_thres': corr_thres
            }
            
            self.installment_data(**arg_dict)
            print(80*"=")
            
        # Credit Card Balance
        if datasets[4] in dir_list:
            pass
        else:
            arg_dict = {
                'ccb_filename':'credit_card_balance',
                'app_filename':'application_train',
                'corr_thres': corr_thres
            }
            
            self.credit_card_bal_data(**arg_dict)
            print(80*"=")
            
        # Reload all prepared data sets  and create the final full data set
        for ds in datasets:
            ds_tmp = pd.read_parquet(self.output_path + '/' + ds)
            data_list.append(ds_tmp)
            del ds_tmp
        
        data = data_list[1].merge(data_list[0],how='left')
        data = data.merge(data_list[4],how='left')
        data = data.merge(data_list[2],how='left')
        
        mask1 = data['size_mean'].isnull()
        mask2 = data['prev_Consumer_loans'].isnull()
        mask3 = data['avg_num_instalments_total'].isnull()
        
        data = data.loc[~(mask1 | mask2 | mask3),:]
        
        if ccb_include == True:
            # Join credit card balance data and fill missing values with 0's
            data = data.merge(data_list[3],how='left').fillna(0)
        else:
            pass
            
        data.to_parquet(self.output_path + 'full_data_cleaned.parquet',engine='pyarrow')
        print(f">>> Data Set has been saved to: \n\t {self.output_path}")
