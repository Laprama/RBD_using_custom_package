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

#Import models from sktime
from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.classification.interval_based import DrCIF
from sktime.classification.kernel_based import RocketClassifier

#Directory containing all main experiment folders
core_path = '/export/sphere/ebirah/ko20929/RBD_files/notebooks/frequency_analysis/constructed_data/band_power_time_series/expt_dfs/'

folders = ['N1_customised_40_expt_files', 'N2_120_expt_files_v2', 'N3_82_expt_files', 'REM_19_expt_files', 'EC_8_expts',
           'N1_full_expts', 'N2_full_expts', 'N3_full_expts', 'REM_full_expts', 'EC_full_expts']


t1 = time.time()

classifier_list  = [{'Rocket': RocketClassifier(use_multivariate='yes')},
 {'C_I_F': CanonicalIntervalForest()},
 {'Dr_C_I_F': DrCIF()} ]   

for clf in classifier_list:
    for folder in folders:
        load_path = core_path + folder + '/'

        X = pd.read_hdf(load_path + 'X.h5', key='df', mode='r')
        y = pd.read_hdf(load_path + 'y_full.h5', key='df', mode='r') 
        groups = pd.read_hdf(load_path + 'groups.h5', key='df', mode='r')  

        for num in [1,2,3,4]:
            #1.Generate expt specific X,y,groups
            X_expt , y_expt , groups_expt, expt_info = run_expts.generate_expt_x_y_groups(X,y,groups,num)
            
            clf_name = list(clf.keys() ) [0]

            save_path = 'Results/Results_2/' + clf_name + 'expt_type_' + str(num) + '_' + folder 
            
            print('commencing first experiment ....')
            
            #2.Run the experiments per model and save the results per fold in a dataframe at save_path
            results_df = run_expts.run_mv_tsc(X_expt ,y_expt , groups_expt, clf, save_path = save_path , random_states = [1,2])

            t2 = time.time()

            print(t2 - t1)