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

#Define Parameter Grids for CV ____________________

#Lets trim the grids based on what the findings were showing about the best classifier?
RF_param_grid = { 
    'n_estimators': [1,2,3, 10, 40,100,250],
    'max_features': [None, 'sqrt'],
    'max_depth' : [2,3,5,8, None],
    'criterion' :['gini',  'entropy'],
    'min_samples_split' : [2,3,4,5]}

DT_params =  {
    'min_samples_leaf': [1, 2, 3 , 5 ,10],
    'max_depth': [1, 2, 3, 5, None],
    'criterion': ["gini", "entropy"],
    'max_features': [None, 'sqrt']}

Ada_grid =  { 
    'n_estimators': [2, 3, 5, 10, 20, 40, 50, 100],
    'learning_rate': [0.01,0.05,  0.1, 0.2, 0.4, 1.0 , 2.0, 10.0]
    }


SVC_grid = {'C': [0.1, 1, 10, 100, 1000], 
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['linear', 'rbf']} 

t1 = time.time()

settings = [] 
TS_Fresh_setting = 'no_setting'

for data_type in  ['REM', 'N1', 'N2', 'N3', 'Wake']:
    for expt_num in [1,2]:
        for num_splits in [4]:
            #Define Classifiers , they must be defined after num_splits but before clf_dicts
            DT_dict = {'DT' : GridSearchCV( DecisionTreeClassifier(), DT_params , refit = True, verbose = 1, cv = GroupKFold(n_splits = num_splits) ) }
            # RF_dict = {'RF' : GridSearchCV( RandomForestClassifier(n_jobs = -1), RF_param_grid , refit = True, verbose = 1, cv = GroupKFold(n_splits = num_splits)  ) }
            Ada_dict = {'Ada' : GridSearchCV( AdaBoostClassifier(), Ada_grid , refit = True, verbose = 1, cv = GroupKFold(n_splits = num_splits) ) }
            # Svc_dict = {'SVC' : GridSearchCV( SVC(), SVC_grid , refit = True, verbose = 1, cv = GroupKFold(n_splits = num_splits) ) }
            
            # for clf_dict in [DT_dict , Ada_dict , Svc_dict]:
            # for clf_dict in [RF_dict]:
            for clf_dict in [DT_dict , Ada_dict ]:
                settings.append( (data_type, expt_num , num_splits , clf_dict,TS_Fresh_setting))


data_type, expt_num , num_splits , clf_dict, TS_Fresh_setting = settings[int(sys.argv[1])]

folder = '/user/home/ko20929/work/RBD_using_custom_package/Execute New Experiments/Baseline_Extensions/Fitting Hyperparam Test/tuned_region_only_models_and_results/baseline_'
model = list(clf_dict.keys())[0]
save_name = model + '_' + data_type + '_' + TS_Fresh_setting + '_' + 'expt' + str(expt_num) + '_' +  str(num_splits) + '_fold_results_df.pkl'
print('commencing for ... ' + save_name)


#Change load path to the band power time series folder
load_path = '/user/home/ko20929/work/RBD_using_custom_package/Data/freq_6_second_files/'
load_path_bpw = '/user/home/ko20929/work/RBD_using_custom_package/Data/freq_6_second_files/'
load_path_max_freqs = '/user/home/ko20929/work/RBD_using_custom_package/Execute New Experiments/Baseline_Extensions/Gen_New_Features/generated_feats/'

X_bpw = pd.read_hdf(load_path_bpw + data_type +  'six_second_freq_df.h5', key='df', mode='r')
y = pd.read_hdf(load_path + data_type +  '_y.h5', key='df', mode='r') 
groups = pd.read_hdf(load_path + data_type +  '_groups.h5', key='df', mode='r')
X_bpw, y , groups = X_bpw.reset_index(drop = True) , y.reset_index(drop = True) , groups.reset_index(drop = True)

X_max_freqs = pd.read_hdf(load_path_max_freqs + data_type +  'six_second_max_freq_stats_df.h5', key='df', mode='r')
X_max_freqs = X_max_freqs.reset_index(drop = True)

X = pd.concat([X_bpw , X_max_freqs], axis = 1)

#Transform the X into mean of time series wihin dataframe ___
def function(x):
    return x.values.mean()
    
#Construct the static features
static_features_df = X.apply(np.vectorize(function))
X = static_features_df.copy()

#Drop columns where all values are the same
# Find columns where all values are the same
same_value_columns = X.columns[X.nunique() == 1]
# Drop columns with the same values
X = X.drop(columns=same_value_columns)
print(len(X.columns))


#Generate a region to features dictionary - this will enable us to run expts regionally as before
regional_features_dict = {}
region_channel_dict = constants.region_to_channel_dict
regions = list(region_channel_dict.keys())

for region in regions:
    region_features = [col for col in X.columns if '_' + region in col]
    regional_features_dict[region] = region_features 

#Run the experiment with hyperparameter tuning
X_expt , y_expt , groups_expt, expt_info = run_expts.generate_expt_x_y_groups(X,y,groups, expt_num)
results_df, clfs_dict = run_expts.run_mv_tsc(X_expt,y_expt,groups_expt,clf_dict, return_df = True , subset_names_and_cols = regional_features_dict, random_states = [1,2], groups_for_fit = True, best_params = True, return_clfs = True)

#Save the results dataframe 
#Commenting this out for now for overfitting the RF 
folder = '/user/home/ko20929/work/RBD_using_custom_package/Execute New Experiments/Baseline_Extensions/Fitting Hyperparam Test/tuned_region_only_models_and_results/baseline_'
model = list(clf_dict.keys())[0]
save_name = model + '_' + data_type + '_' + TS_Fresh_setting + '_' + 'expt' + str(expt_num) + '_' +  str(num_splits) + '_fold_results_df.pkl'

joblib.dump(results_df , folder + save_name)

clfs_save_name = model + '_' + data_type + '_' + TS_Fresh_setting  + '_' + 'expt' + str(expt_num) + '_' +  str(num_splits) + '_clfs_dict.pkl'
joblib.dump(clfs_dict , folder + clfs_save_name)

t2 = time.time()
print(save_name)
print(t2-t1)