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

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

#Import my modules
import format_eeg_data
import constants
import eeg_stat_ts
import run_expts


from mne_connectivity import spectral_connectivity_epochs
from mne.datasets import sample
import seaborn as sns

from scipy.signal import welch
import yasa
import constants
import numpy as np

data_type = ['REM', 'N1', 'N2', 'N3', 'Wake'][int(sys.argv[1])] 
# method = [ 'wpli2_debiased' , 'ppc', 'imcoh' ][int(sys.argv[2])]

method = ['coh', 'plv', 'ciplv', 'pli', 'dpli', 'wpli'][int(sys.argv[2])]

print('commencing for ' + data_type + '...')
t1 = time.time()

#Define power band names
power_bands = power_bands = {'delta' : (0.5,4) , 'theta' : (4,8) , 'alpha' : (8,12) , 'sigma' : (12,16) , 'beta' : (16,30) , 'gamma' : (30,40), 'all' : (0.5,40) }
power_band_names = [val for val in power_bands.keys()]

#Load the data
folder = '/user/home/ko20929/work/RBD_using_custom_package/Blue_pebble/Connectivity/'
X = pd.read_hdf(folder + data_type + '_' + method + '_' + '_df.h5')
X = X.reset_index(drop = True)
y = pd.read_hdf('/user/home/ko20929/work/RBD_using_custom_package/Blue_pebble/C_22_data/Full_dfs/' + data_type + '_y.h5')
groups = pd.read_hdf('/user/home/ko20929/work/RBD_using_custom_package/Blue_pebble/C_22_data/Full_dfs/' + data_type + '_groups.h5')

#Define subsets of features to be used in Experiments
subsets_dict = {}
for p_band in power_band_names:
    subsets_dict[p_band] = [col for col in X.columns if p_band in col]
subsets_dict['all_features'] = [col for col in X.columns]

save_folder = '/user/home/ko20929/work/RBD_using_custom_package/Execute New Experiments/Connectivity/Connectivity_Results/Experiment_set_2/'

for expt_num in [1,2,3,4]:
    clfs_dict =  {'RF' : RandomForestClassifier(), 'DT' : DecisionTreeClassifier() , 'Ada_B' : AdaBoostClassifier() , 'SVM' : SVC()} 
    save_path = save_folder + data_type + method + '_expt_' + str(expt_num) + '_'
    X_expt , y_expt , groups_expt, expt_info = run_expts.generate_expt_x_y_groups(X,y,groups, expt_num)
    results_df = run_expts.run_mv_tsc(X_expt,y_expt,groups_expt, clfs_dict , save_path = save_path, return_df = True , subset_names_and_cols = subsets_dict, random_states = [1,2] )

t2 = time.time()
time_taken = str( (t2-t1)/ 60 )
print(data_type + ' took ' + time_taken + ' mins')