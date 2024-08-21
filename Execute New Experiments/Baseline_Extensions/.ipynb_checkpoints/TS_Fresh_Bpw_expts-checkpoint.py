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

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score

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

# Set display options to show all rows and columns
pd.set_option('display.max_rows', 50)  # Show rows
pd.set_option('display.max_columns', 160)  # Show columns

data_type = ['Wake','N1', 'N2', 'N3', 'REM'][int(sys.argv[1])]
TS_Fresh_setting = ['Minimal' , 'Efficient'][int(sys.argv[2])]

print(data_type)

load_path = '/user/home/ko20929/work/RBD_using_custom_package/Execute New Experiments/Baseline_Extensions/Gen_New_Features/generated_feats/'

#Change load path to the band power time series folder
load_path = '/user/home/ko20929/work/RBD_using_custom_package/Data/freq_6_second_files/'


loaded_data = {}

X_y_groups = {}
    
X = pd.read_hdf(load_path + data_type +  'six_second_freq_df.h5', key='df', mode='r')
y = pd.read_hdf(load_path + data_type +  '_y.h5', key='df', mode='r') 
groups = pd.read_hdf(load_path + data_type +  '_groups.h5', key='df', mode='r')
X, y , groups = X.reset_index(drop = True) , y.reset_index(drop = True) , groups.reset_index(drop = True)

#Transform the X into TS_Fresh Features___
# 1. Convert to TS_Fresh format Dataframe 
ts_fresh_df = format_eeg_data.convert_sktime_df_to_ts_fresh_format(X, ts_cols = list(X.columns))

# 2. Extract TS_Fresh Features from the dataframe
if TS_Fresh_setting == 'Minimal':
    settings = MinimalFCParameters()
elif TS_Fresh_setting == 'Efficient':
    settings = EfficientFCParameters()
else:
    raise Exception('No TS Fresh Parameter Setting Set!!')
    

extracted_ts_fresh_df = extract_features(ts_fresh_df, column_id = 'id' , column_sort = 'time',  default_fc_parameters=settings)

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

#Generate the three dictionaries per data type _________________________________________________________________________________
#Generate regional features dict per data type 
#Generate a region to features dictionary - this will enable us to run expts regionally as before
regional_features_dict = {}
region_channel_dict = constants.region_to_channel_dict
regions = list(region_channel_dict.keys())
for region in regions:
    region_features = [col for col in X.columns if region + '_' in col]
    if len(region_features) > 0 : 
        regional_features_dict[region] = region_features

# 2. #Create the combined regions dictionary
regions = list(regional_features_dict.keys())
combined_regions_features_dict = {}
    
for i, region_1 in enumerate(regions):
    for region_2 in regions[i+1:]:
        new_key = region_1 + '_' + region_2
        combined_regions_features_dict[new_key] = regional_features_dict[region_1] + regional_features_dict[region_2]

#3. Use all of the features
all_data_dict = {'All_regions' : list(X.columns) , 'All_regions_2' : list(X.columns)}
    
X_y_groups['X'] = X
X_y_groups['y'] = y
X_y_groups['groups'] = groups

X_y_groups['regions_dict'] = regional_features_dict
X_y_groups['combined_regions_dict'] = combined_regions_features_dict
X_y_groups['all_feats_dict'] = all_data_dict

loaded_data[data_type] = X_y_groups

#Run all experiments

t1 = time.time()

all_expt_results = {}

for expt_num in [1,2,3,4] :
    t3 = time.time()
    
    expt_results = {}
    
    X_y_groups = loaded_data[data_type]
    
    X = X_y_groups['X']
    y = X_y_groups['y'] 
    groups = X_y_groups['groups'] 
        
    #1.Generate expt specific X,y,groups
    X_expt , y_expt , groups_expt, expt_info = run_expts.generate_expt_x_y_groups(X,y,groups, expt_num )

    results_df_regional = run_expts.run_mv_tsc(X_expt,y_expt,groups_expt,  {'RF' : RandomForestClassifier(random_state = 5) , 'DT' : DecisionTreeClassifier() , 'Ada_B' : AdaBoostClassifier(random_state = 5)} , return_df = True , subset_names_and_cols = X_y_groups['regions_dict'], random_states = [1,2] )
    print('regional done...')
    results_df_regions_combined = run_expts.run_mv_tsc(X_expt,y_expt,groups_expt,  {'RF' : RandomForestClassifier(random_state = 5) , 'DT' : DecisionTreeClassifier() , 'Ada_B' : AdaBoostClassifier(random_state = 5)} , return_df = True , subset_names_and_cols = X_y_groups['combined_regions_dict'] , random_states = [1,2] )
    print('regions combined done...')
    results_df_all_feats = run_expts.run_mv_tsc(X_expt,y_expt,groups_expt,  {'RF' : RandomForestClassifier(random_state = 5) , 'DT' : DecisionTreeClassifier() , 'Ada_B' : AdaBoostClassifier(random_state = 5)} , return_df = True , subset_names_and_cols = X_y_groups['all_feats_dict'], random_states = [1,2] )
    
    expt_results[data_type] = {'regional' : results_df_regional , 'regions_combined' : results_df_regions_combined , 'all_feats' : results_df_all_feats}

    all_expt_results[expt_num] = expt_results
    
    #Saving Results 
    #Save Name --> data_type + TS_Fresh_efficient ...
    
    joblib.dump(all_expt_results, data_type + 'BPW_TS_Fresh_' + TS_Fresh_setting + '-BPW_feats_results.pkl')
    t4 = time.time()
    loop_time = t4-t3
    print(str(loop_time))

t2 = time.time()

total_time = str(t2-t1)

print('total time taken was ' + total_time)