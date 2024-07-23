import pandas as pd
import mne as mne
import os 
import time
import numpy as np
import matplotlib.pyplot as plt
import joblib
import constants
from IPython.utils import io
import time
import sys
import yasa
from scipy.signal import welch

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold

#Import my modules
import format_eeg_data
import constants
import eeg_stat_ts
import run_expts

#TS Fresh Parameter Settings
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_extraction import extract_features

#Define script parameters _______________________________
# data_types = ['REM', 'N1', 'N2', 'N3', 'Wake'][int(sys.argv[1])] 

TS_Fresh_setting, connectivity_setting = 'Minimal' , 'beta'

settings = [] 

for data_type in  ['N1','REM', 'N2', 'N3', 'Wake']:
    for expt_num in [1,2,3,4]:
        #Define Classifiers , they must be defined after num_splits but before clf_dicts
        DT_dict = {'DT' :  DecisionTreeClassifier(random_state = 5) }
        RF_dict = {'RF' : RandomForestClassifier(random_state = 5) }
        Ada_dict = {'Ada' : AdaBoostClassifier(random_state = 5) }
        Svc_dict = {'SVC' : SVC()}
        
        for clf_dict in [DT_dict , RF_dict , Ada_dict , Svc_dict]:
            # for clf_dict in [RF_dict]:
                settings.append((data_type, expt_num , clf_dict))
            

data_type, expt_num , clf_dict = settings[int(sys.argv[1])]

# Print commencing message 
model = list(clf_dict.keys())[0]
save_name = model + '_' + data_type + '_' + TS_Fresh_setting + '_' + connectivity_setting + '_' + 'expt' + str(expt_num) + '_results_df.pkl'

print('commencing for ... ' + save_name)

#Adding extra code to check if the results already exist____________ 
print('Checking if results already exists..') 
folder = 'non_tuned_models_and_results/dual_region_'
model = list(clf_dict.keys())[0]
save_name = model + '_' + data_type + '_' + TS_Fresh_setting + '_' + connectivity_setting + '_' + 'expt' + str(expt_num) + '_results_df.pkl'
 
save_path = folder + save_name

if os.path.exists(save_path):
    print('Path already exists!')
    run_expt = False
else:
    print('Path does not exist yet')
    run_expt = True
    

t1 = time.time()

if run_expt:
    # Load the features 
    load_path = '/user/home/ko20929/work/RBD_using_custom_package/Data/freq_6_second_files/'
    load_path_bpw = '/user/home/ko20929/work/RBD_using_custom_package/Data/freq_6_second_files/'
    load_path_max_freqs = '/user/home/ko20929/work/RBD_using_custom_package/Execute New Experiments/Baseline_Extensions/Gen_New_Features/generated_feats/'
    
    loaded_data = {}
    
    X_y_groups = {}
        
    X_bpw = pd.read_hdf(load_path_bpw + data_type +  'six_second_freq_df.h5', key='df', mode='r')
    y = pd.read_hdf(load_path + data_type +  '_y.h5', key='df', mode='r') 
    groups = pd.read_hdf(load_path + data_type +  '_groups.h5', key='df', mode='r')
    X_bpw, y , groups = X_bpw.reset_index(drop = True) , y.reset_index(drop = True) , groups.reset_index(drop = True)
    
    X_max_freqs = pd.read_hdf(load_path_max_freqs + data_type +  'six_second_max_freq_stats_df.h5', key='df', mode='r')
    X_max_freqs = X_max_freqs.reset_index(drop = True)
    
    X = pd.concat([X_bpw , X_max_freqs], axis = 1)
    
    #Transform the X into TS_Fresh Features___
    # 1. Convert to TS_Fresh format Dataframe 
    ts_fresh_df = format_eeg_data.convert_sktime_df_to_ts_fresh_format(X, ts_cols = list(X.columns))
    
    # 2. Extract TS_Fresh Features from the dataframe
    if TS_Fresh_setting == 'Minimal':
        settings = MinimalFCParameters()
    
    extracted_ts_fresh_df = extract_features(ts_fresh_df, column_id = 'id' , column_sort = 'time', default_fc_parameters=settings)
    
    # 3. Asign extract_ts_fresh_df to the variable X
    X = extracted_ts_fresh_df.copy()
    print(len(X.columns))
    #Drop columns where all values are NA 
    X = X.dropna(axis = 1)
    print(len(X.columns))
    
    #Drop columns where all values are the same
    # Find columns where all values are the same
    same_value_columns = X.columns[X.nunique() == 1]
    # Drop columns with the same values
    X = X.drop(columns=same_value_columns)
    print(len(X.columns))
    
    #___________load the appropriate connectivity features don't concatenate to non connectivity features until after dictionary generation
    
    connectivity_folder = '/user/home/ko20929/work/RBD_using_custom_package/Blue_pebble/Connectivity/'
    X_connectivity = pd.read_hdf(connectivity_folder + data_type+ '_pli__df.h5')
    
    X_connectivity = X_connectivity[[col for col in X_connectivity.columns if connectivity_setting in col]]
    
    #Generate a region to features dictionary - this will enable us to run expts regionally as before
    regional_features_dict = {}
    region_channel_dict = constants.region_to_channel_dict
    regions = list(region_channel_dict.keys())
    for region in regions:
        region_features = [col for col in X.columns if region + '_' in col]
        if len(region_features) > 0 : 
            regional_features_dict[region] = region_features + list(X_connectivity.columns)

    #Added line - Generate the combined regions dictionary 
    regions = list(regional_features_dict.keys())
    combined_regions_features_dict = {}
    
    for i, region_1 in enumerate(regions):
        for region_2 in regions[i+1:]:
            new_key = region_1 + '_' + region_2
            combined_regions_features_dict[new_key] = regional_features_dict[region_1] + regional_features_dict[region_2] + list(X_connectivity.columns)
        
    #Now concatenate the regional frequency features dataframe with the connectivity dataframe
    X = X.reset_index(drop=True)
    X_connectivity = X_connectivity.reset_index(drop=True)
    X = pd.concat([X , X_connectivity], axis = 1)
    
    #Run the experiment with no hyperparameter tuning
    X_expt , y_expt , groups_expt, expt_info = run_expts.generate_expt_x_y_groups(X,y,groups, expt_num)
    results_df, clfs_dict = run_expts.run_mv_tsc(X_expt,y_expt,groups_expt,clf_dict, return_df = True , subset_names_and_cols = combined_regions_features_dict, random_states = [1,2], return_clfs = True)
    
    #Save the results dataframe 
    folder = 'non_tuned_models_and_results/dual_region_'
    model = list(clf_dict.keys())[0]
    save_name = model + '_' + data_type + '_' + TS_Fresh_setting + '_' + connectivity_setting + '_' + 'expt' + str(expt_num) + '_results_df.pkl'
    joblib.dump(results_df , folder + save_name)

    clfs_save_name =  model + '_' + data_type + '_' + TS_Fresh_setting + '_' + connectivity_setting + '_' + 'expt' + str(expt_num) + '_clfs_dict.pkl'
    joblib.dump(clfs_dict , folder + clfs_save_name)
    
    t2 = time.time()
    print(save_name)
    print(t2-t1)