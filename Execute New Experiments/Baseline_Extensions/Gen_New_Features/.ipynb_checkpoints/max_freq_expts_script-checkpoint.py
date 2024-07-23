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

#Import my modules
import format_eeg_data
import constants
import eeg_stat_ts
import run_expts

#This script is created form the Experiment_2_Max_Freq_static_features notebook with minor mods made
print('Script has commenced....')

core_path = '/user/home/ko20929/work/RBD_using_custom_package/Execute New Experiments/Baseline_Extensions/Gen_New_Features/generated_feats/'
data_types = ['Wake','N1', 'N2', 'N3', 'REM']

loaded_data = {}

for data_type in data_types:
    X_y_groups = {}
    
    load_path = core_path + data_type
    X = pd.read_hdf(load_path + 'six_second_max_freq_stats_df.h5', key='df', mode='r')
    y = pd.read_hdf(load_path + '_y.h5', key='df', mode='r') 
    groups = pd.read_hdf(load_path + '_groups.h5', key='df', mode='r')  
    
    # Replace time sereis data with mean of the data ______________________________________________________________________________________________________

    # Defining a function to replace time series of values with their mean
    def function(x):
        return x.values.mean()
        
    #Construct the static features
    static_features_df = X.apply(np.vectorize(function))
    X = static_features_df.copy()

    X_y_groups['X'] = X
    X_y_groups['y'] = y
    X_y_groups['groups'] = groups
    
    loaded_data[data_type] = X_y_groups

# Generate the required dictionaries ________________________________________________________________________________________________
# The dictionaries are the same for all data types because features are the same for all data types

# 1. #Generate region to features dictionary to enable experiments to be run regionally
regional_features_dict = {}
region_channel_dict = constants.region_to_channel_dict
regions = list(region_channel_dict.keys())
for region in regions:
    region_features = [col for col in X.columns if '_' + region in col]
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



# Run the experiments , 1 to 4 ______________________________________________________________________________________________________________
t1 = time.time()

all_expt_results = {}

for expt_num in [1,2,3,4] :
    expt_results = {}
    for data_type in data_types:
        X_y_groups = loaded_data[data_type]
        
        X = X_y_groups['X']
        y = X_y_groups['y'] 
        groups = X_y_groups['groups'] 
            
        #1.Generate expt specific X,y,groups
        X_expt , y_expt , groups_expt, expt_info = run_expts.generate_expt_x_y_groups(X,y,groups, expt_num )
    
        results_df_regional = run_expts.run_mv_tsc(X_expt,y_expt,groups_expt,  {'RF' : RandomForestClassifier(random_state = 5) , 'DT' : DecisionTreeClassifier() , 'Ada_B' : AdaBoostClassifier(random_state = 5)} , return_df = True , subset_names_and_cols = regional_features_dict, random_states = [1,2] )
        results_df_regions_combined = run_expts.run_mv_tsc(X_expt,y_expt,groups_expt,  {'RF' : RandomForestClassifier(random_state = 5) , 'DT' : DecisionTreeClassifier() , 'Ada_B' : AdaBoostClassifier(random_state = 5)} , return_df = True , subset_names_and_cols = combined_regions_features_dict, random_states = [1,2] )
        results_df_all_feats = run_expts.run_mv_tsc(X_expt,y_expt,groups_expt,  {'RF' : RandomForestClassifier(random_state = 5) , 'DT' : DecisionTreeClassifier() , 'Ada_B' : AdaBoostClassifier(random_state = 5)} , return_df = True , subset_names_and_cols = all_data_dict, random_states = [1,2] )
        
        expt_results[data_type] = {'regional' : results_df_regional , 'regions_combined' : results_df_regions_combined , 'all_feats' : results_df_all_feats}
    
    all_expt_results[expt_num] = expt_results

t2 = time.time()

print(t2-t1)

# Save the results _______________________________________________________________________________________________________________________________________
#Saving Results 
joblib.dump(all_expt_results, 'static_frequency_feats_results_2.pkl')
