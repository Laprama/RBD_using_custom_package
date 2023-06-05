from IPython.utils import io
import pandas as pd
import os
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import seaborn as sns
import time
import joblib
from os.path import exists
import shutil
import sys
import time

from sklearn.model_selection import train_test_split
#From my EEG package 
import run_expts

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# defining parameter range for grid search SVC 
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['linear', 'rbf']} 

classifier_list  = [{'Random_Forest': RandomForestClassifier()},
 {'SVC':GridSearchCV(SVC(), param_grid, refit = True, verbose = 1)},
 {'Decision_Tree': DecisionTreeClassifier()} ]   



#Directory containing all main experiment folders
core_path = '/export/sphere/ebirah/ko20929/RBD_files/notebooks/frequency_analysis/constructed_data/band_power_time_series/expt_dfs/'

folders = ['N1_full_expts', 'N2_full_expts', 'N3_full_expts', 'REM_full_expts', 'EC_full_expts']

t1 = time.time()

for clf in classifier_list:
    clf_name = list(clf.keys())[0]
    print(clf_name)

    for folder in folders:
        load_path = core_path + folder + '/'

        X = pd.read_hdf(load_path + 'X.h5', key='df', mode='r')
        y = pd.read_hdf(load_path + 'y_full.h5', key='df', mode='r') 
        groups = pd.read_hdf(load_path + 'groups.h5', key='df', mode='r')  

        # Defining a function to replace time series of values with their mean
        def function(x):
            return x.values.mean()

        #Construct the static features
        static_features_df = X.apply(np.vectorize(function))
        X = static_features_df.copy()

        for num in [1,2,3,4]:
            #1.Generate expt specific X,y,groups
            X_expt , y_expt , groups_expt, expt_info = run_expts.generate_expt_x_y_groups(X,y,groups,num)

            save_path = 'Results/Results_4/' + clf_name + 'expt_type_' + str(num) + '_' + folder[:3] 

            run_expts.run_mv_tsc(X_expt ,y_expt , groups_expt, clf, save_path, random_states = [1 , 2])
            
            t2 = time.time()
            
            print(t2 - t1)