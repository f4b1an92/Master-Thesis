import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm import tqdm

def aggregator(df,levels):
    tmp_df = df.copy()
    if type(levels) == str:
        tmp_df.loc[:,levels] = tmp_df.loc[:,levels].apply(str)
    else:
        for lev in levels:
            tmp_df.loc[:,lev] = tmp_df.loc[:,lev].apply(str)
    tmp = tmp_df.groupby(levels)
    tmp = tmp.agg({'sk_id_curr':'count','target':'mean'}).reset_index().rename(columns={'sk_id_curr':'count','target':'target_share'})
    tmp['target_share'] = tmp['target_share'].round(4)
    tmp['count_share'] = (tmp['count'] / tmp['count'].sum()).round(4)
    if type(levels) == list: 
        tmp = tmp.loc[:,[*levels,'count','count_share','target_share']]
    else:
        tmp = tmp.loc[:,[levels,'count','count_share','target_share']]
    return tmp


def outlier_thresholds(data,var,sf=1.5):
    """
    Helper function for outlier remover.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe for whoch outlier thesholds should be determined.
    var : str
        Indicates the column name for which the outlier threshold should be determined.
    sf : float, optional
        Scaling factor that determines when a data point is considered as an outlier according to 
        the formula d < sf * IQR | d > sf * IQR. The default is 1.5.

    Returns
    -------
    lower_thres : float
        Lower threshold for outlier detection.
    upper_thres : float
        Lower threshold for outlier detection.

    """
    scale_factor = sf
    quart1 = data.loc[~data[var].isnull(),var].quantile(0.25)
    quart3 = data.loc[~data[var].isnull(),var].quantile(0.75) 
    iqr = abs(quart3 - quart1)
    
    if (data[var].min() < 0) & (data[var].max() <= 0):
        lower_bound = quart1 - scale_factor * iqr
        upper_bound = quart3 + scale_factor * iqr
    elif(data[var].min() < 0) & (data[var].max() > 0):
        lower_bound = quart1 - scale_factor * iqr
        upper_bound = quart3 + scale_factor * iqr 
    else:
        lower_bound = max(quart1 - scale_factor * iqr,0)
        upper_bound = quart3 + scale_factor * iqr 
    
    lower_thres = data.loc[data[var] > lower_bound, var].min()
    upper_thres = data.loc[data[var] < upper_bound, var].max()
    
    return lower_thres, upper_thres



def outlier_remover(df,col_list=None,sf=1.5,copy=True):
    """
    Replaces outliers with the upper (lower) boundary in accordance with a scaling factor that can be set manually. 

    Parameters
    ----------
    df : pandas.DataFrame
        Data sets for which outlier should be removed.
    col_list : list, optional
        Selection of features in the data for which outlier should be removed. The default is None.
    sf : float, optional
        Scaling factor that determines when a data point is considered as an outlier according to 
        the formula d < sf * IQR | d > sf * IQR. The default is 1.5.
    copy : bool, optional
        Indicates if a copy of the data set should be created in memory. The default is True.

    Returns
    -------
    df : pandas.DataFrame
        Input data for which outlier have been replaced by upper (lower) boundaries.

    """
    if copy == True:
        df = df.copy(deep=True)
    if col_list is None:
        col_list = df.columns
    
    for col in col_list:
        lower_thr,upper_thr = outlier_thresholds(data=df,var=col,sf=sf)

        df[col].apply(lambda x: upper_thr if x > upper_thr else x)
        df[col].apply(lambda x: lower_thr if x < lower_thr else x)
        
    return df


def box_plot(data,col_list,dim1,dim2,fsize1,fsize2,lower_thr=False):
    """
    Creates box plot for numeric and pseudo-numeric features.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe.
    col_list : TYPE
        List of features for which the box-plots should be created.
    dim1 : int
        How many plots should the rows contain.
    dim2 : int
        How many plots should the columns contain.
    fsize1 : int
        Indicates height of the plot.
    fsize2 : int
        Indicates width of the plot.
    lower_thr : bool, optional
        Threshold for truncation to be used. The default is False.

    Returns
    -------
    None.

    """
    fig,ax = plt.subplots(dim1,dim2,figsize=(fsize1,fsize2),squeeze=False)

    cnt = 0
    #for idx,col in enumerate(col_list):
    for i in range(dim1):
        for j in range(dim2):
            if (cnt == len(col_list)) & (cnt % 2 == 1) :
                pass
            else:
                col = col_list[cnt]
                lower,upper = outlier_thresholds(data=data,var=col,sf=1.5)

                tmp = data.loc[data[col]<=upper,:].copy()
                if lower_thr==True:
                    tmp = tmp.loc[tmp[col]>=lower,:].copy()
                    
                if tmp.loc[:,col].max()/1_000 >= 100:
                    cond = True
                    tmp.loc[:,col] = tmp.loc[:,col]/1_000
                else:
                    cond = False
                sns.boxplot(data=tmp,y='target',x=col,ax=ax[i,j],orient="h")
                if cond==True:
                    ax[i,j].set_xlabel(f'{col} in Thousands')
                cnt += 1
    plt.show()


def check_dtypes(data):
    """
    Checks how many features per data type there are in the input data set.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    col_types_counter : dict
        Counts per data type of the features contained in the input data set.
    col_types : dict
        Dictionary that maps column names to their data types which are stored as string.

    """
    col_types_counter = {}
    col_types = {}
    
    for col in data.columns:
        col_type = str(data.loc[:,col].dtype)
        if col_type in col_types_counter.keys():
            col_types_counter[col_type] += 1
            col_types[col_type].append(col)
        else:
            col_types_counter[col_type] = 1
            col_types[col_type] = [col]

    return (col_types_counter,col_types)


def reduce_memory(df):
    """
    Reduces memory consumption of the data sets by compressing the features to the "smallest"
    data type possible given their set of unique values.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe that should be compressed.

    Returns
    -------
    df : pandas.DataFrame
        Returns the compressed version of the input dataframe.

    """
    df = df.copy()
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != 'object':
            min_val = df[col].min()
            max_val = df[col].max()
            if 'int' in str(df[col].dtype):
                if (min_val > np.iinfo(np.int8).min) & (max_val < np.iinfo(np.int8).max):
                    df[col] = df[col].astype(np.int8)
                elif (min_val > np.iinfo(np.int16).min) & (max_val < np.iinfo(np.int16).max):
                    df[col] = df[col].astype(np.int16)
                elif (min_val > np.iinfo(np.int32).min) & (max_val < np.iinfo(np.int32).max):
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            elif 'float' in str(df[col].dtype):
                if (min_val > np.finfo(np.float16).min) & (max_val < np.finfo(np.float16).max):
                    df[col] = df[col].astype(np.float16)
                elif (min_val > np.finfo(np.float32).min) & (max_val < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
            else:        
                df[col] = df[col].apply(str)
        else:
            df[col] = df[col].astype('category')
    return df



def custom_dtype_identifier(data,thres=25):
    final_dict = {}
    temp_dict = {}
    abc_dummy = []

    for col in data.columns:
        data_type = str(data[col].dtype)
        data_type = re.sub(r'[0-9]+','',data_type)
        if data_type == 'object':
            if len(np.unique(data.loc[:,col].apply(str))) == 2:
                abc_dummy.append(col)
            elif data_type not in final_dict.keys():
                final_dict[data_type] = []
                final_dict[data_type].append(col)
            else:
                final_dict[data_type].append(col)
        elif (data_type == 'int') | (data_type == 'float'):
            temp_dict[col] = len(np.unique(data[col]))
        elif data_type not in final_dict.keys():
            final_dict[data_type] = []
            final_dict[data_type].append(col)
        else:
            final_dict[data_type].append(col)

    pseudo_cat = []
    num_vars = []
    for key,val in temp_dict.items():
        if val <= thres:
            pseudo_cat.append(key)
        else:
            num_vars.append(key)

    final_dict['num'] = num_vars
    final_dict['pseudo_cat'] = pseudo_cat
    final_dict['chr_dummies'] = abc_dummy

    return final_dict


def corr_dropouts(data,corr_thres=0.7):
    """
    Determines variables to be dropped because of too high correlation. It has a dependency with the
    custom_dtype_identifier which will look for numeric variables in order to create a correlation-matrix
    from only those columns. The correlation threshold is controlled via corr_thres (default = 0.7). 

    Parameters
    ----------
    data : pandas.DataFrame
        A dataframe containing all the columns (not only numeric cols).
    corr_thres : float, optional
        Threshold that determines when the correlation is too high to be tolerated. Variables with 
        correlation above this threshold have to be dropped. The default is 0.7.

    Returns
    -------
    to_keep : numpy.array
        Column names that are kept.
    to_drop : numpy.array
        Column names that have to dropped.

    """
    dtype_dict = custom_dtype_identifier(data,thres=25)
    num_cols = dtype_dict['num']
    pcat_cols = dtype_dict['pseudo_cat']
    corr_df = data.loc[:,[*pcat_cols,*num_cols]].corr()

    to_keep = []
    to_drop = []

    for col in corr_df.columns:
        corr_list = []
        corr_cols_idx = np.where(abs(corr_df[col])>corr_thres)[0]
        corr_arr = corr_df[col].reset_index().iloc[corr_cols_idx,0].values

        for i in range(len(corr_arr)):
            corr_list.append(abs(corr_df.loc[corr_arr[i],'target']))

        max_corr = max(corr_list)
        max_idx = np.where(corr_list == max_corr)[0]

        to_keep = np.append(to_keep,corr_arr[max_idx])
        to_drop = np.append(to_drop,np.delete(corr_arr, max_idx))

    to_keep = np.unique(to_keep)
    to_drop = np.unique(to_drop)

    return to_keep,to_drop




def interest_rates(data,pmt_col,period_col,principal_col,step_size=0.0001,block_size=1_000):
    iters = int(data.shape[0] / block_size) + 1
    result_list = []
    len_inter = len(np.arange(step_size,1,step_size))

    for i in tqdm(range(iters)):
        if i+1 == iters:
            subset_df = data.iloc[i*block_size:,:]
        else:
            subset_df = data.iloc[i*block_size:(i+1)*block_size,:]

        r_mat = np.arange(step_size,1,step_size)
        r_mat = np.tile(r_mat,subset_df.shape[0]).reshape(subset_df.shape[0],-1)
        c_mat = np.repeat(np.array(subset_df[pmt_col]),len_inter).reshape(-1,len_inter)
        n = np.repeat(np.array(subset_df[period_col]),len_inter).reshape(-1,len_inter)
        B = np.ones(r_mat.shape) - np.power(np.power(1+r_mat,n),-1)

        PV_mat = c_mat * np.power(r_mat,-1) * B
        diffs = abs(PV_mat - np.array(subset_df[principal_col]).reshape(-1,1))

        result_df = pd.DataFrame({'monthly_rate':(np.argmin(diffs,axis=1) + 1) / (1/step_size)})
        result_df['apr'] = result_df['monthly_rate'] * 12
        result_list.append(result_df)

    output_df = pd.concat(result_list,axis=0).reset_index(drop=True)
    return(output_df)




def progress_plot(data,fsize=(16,10),bsizes=[700,1400],colors=['red','blue','orange','green'],save=False,save_path=None,file_name=None):
    """
    Plots the learning progress of the federated logistic regression or fed. neural networks (depending on input data)
    in terms of log-loss per epoch iteration. The plot is split by IID-ness and client.

    Parameters
    ----------
    data : pandas.DataFrame
        .
    fsize : tuple, optional
        Plot size. The default is (16,10).
    bsizes : TYPE, optional
        Batch sizes that were used to train the models. It will use the list values to filter for the 
        corresponding batch sizes in the input data. The default is [700,1400].
    colors : list, optional
        Colors for the lines in the plot. Length should match the product of len(batch_size) * 2. 
        The default is ['red','blue','orange','green'].
    save : bool, optional
        Indicates if plots should be save to disk. The default is False.
    save_path : str, optional
        Save path on local disk where to save the plots. The default is None.
    file_name : TYPE, optional
        File name for the plot. The default is None.

    Returns
    -------
    Progress plots for the federated logistic regression or the federated neural networks for 
    horizontally partitioned data (depending on the input data).

    """
    colors = np.array(colors)
    idx = int(len(colors)/2)
    
    xticks = []
    for i in np.unique(data['m2']['iid']['index']):
        if i % 10 == 0:
            xticks.append(i)
            
    # lg iid
    tmp1 = data['m2']['iid']
    tmp1 = tmp1.loc[tmp1['Batch Size'].isin(bsizes),:]
    tmp2 = data['m2']['noniid']
    tmp2 = tmp2.loc[tmp2['Batch Size'].isin(bsizes),:]

    tmp3 = data['m3']['iid']
    tmp3 = tmp3.loc[tmp3['Batch Size'].isin(bsizes),:]
    tmp4 = data['m3']['noniid']
    tmp4 = tmp4.loc[tmp4['Batch Size'].isin(bsizes),:]

    sns.set_style('dark')
    fig,ax = plt.subplots(2,2,figsize=fsize,sharey='row')
    sns.set_style('white')
    sns.lineplot(
        data=tmp1,x='index',y='loss',hue='Batch Size',style='Model Type',ax=ax[0,0],palette = list(colors[:idx])
    )
    ax[0,0].set_title(r"$\bf{IID}$",fontsize=16)
    ax[0,0].set_ylabel(r"$\bf{Client 2}$" + "\n Log-Loss",fontsize=14)
    ax[0,0].tick_params(axis='y',labelsize=13)
    ax[0,0].set_xlabel(None)
    ax[0,0].set_xticks(xticks)
    ax[0,0].set_xticklabels(xticks,fontsize=13)
    ax[0,0].legend(loc='upper right',prop={'size': 14},ncol=2)

    sns.lineplot(
        data=tmp2,x='index',y='loss',hue='Batch Size',style='Model Type',ax=ax[0,1],palette = list(colors[:idx])
    )
    ax[0,1].set_title(r"$\bf{Non-IID}$",fontsize=16)
    ax[0,1].set_xlabel(None)
    ax[0,1].set_xticks(xticks)
    ax[0,1].set_xticklabels(xticks,fontsize=13)
    ax[0,1].legend(loc='upper right',prop={'size': 14},ncol=2)

    sns.lineplot(
        data=tmp3,x='index',y='loss',hue='Batch Size',style='Model Type',ax=ax[1,0],palette = list(colors[idx:])
    )
    ax[1,0].set_ylabel(r"$\bf{Client 3}$" + "\n Log-Loss",fontsize=14)
    ax[1,0].tick_params(axis='y',labelsize=13)
    ax[1,0].set_xlabel("Number of Batches",fontsize=16)
    ax[1,0].tick_params(axis='x',labelsize=13)
    ax[1,0].legend(loc='upper right',prop={'size': 14},ncol=2)

    sns.lineplot(
        data=tmp4,x='index',y='loss',hue='Batch Size',style='Model Type',ax=ax[1,1],palette = list(colors[idx:])
    )
    ax[1,1].set_xlabel("Number of Batches",fontsize=16)
    ax[1,1].tick_params(axis='x',labelsize=13)
    ax[1,1].legend(loc='upper right',prop={'size': 14},ncol=2)

    plt.tight_layout()
    if save == True:
        plt.savefig(save_path + file_name + '.png')
 
    plt.show()