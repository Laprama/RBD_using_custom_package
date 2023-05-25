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

#Import my modules
import format_eeg_data
import constants
import eeg_stat_ts
import custom_ts_length
import run_expts

#Ignore warnings for now 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#Models
from sktime.classification.kernel_based import RocketClassifier


data_type_num = int(sys.argv[1])
dtypes =  ['Wake', 'N1', 'N2', 'N3','REM']
data_type = dtypes[data_type_num]

folder_num = int(sys.argv[2]) #short or long expts? 0 short , 1 long 

expt_num = int(sys.argv[3]) #which expt to run? 1,2,3,4 --> Going with 1 and 2 to begin with 

#1.Load the correct data for the expt, informed by type of processed data being uesed and data type (X, y , groups)
#These file paths need to be correct for bp because this will be executed on bp
# bp_folder = /user/home/ko20929/RBD_using_custom_package/Data
core_folder = '/user/home/ko20929/RBD_using_custom_package/Data/'
load_folders = ['freq_6_second_files_proc_short/' , 'freq_6_second_files_proc_long/' ]
load_folder = load_folders[folder_num]


X = pd.read_hdf(core_folder + load_folder + data_type + '_X.h5' , key = 'df', mode = 'r')
y = pd.read_hdf(core_folder + load_folder + data_type + '_y.h5' , key = 'df', mode = 'r')
groups = pd.read_hdf(core_folder + load_folder + data_type + '_groups.h5' , key = 'df', mode = 'r')

#2.Configure for the correct experiment type [1,2,3,4]
X_expt ,y_expt , groups_expt, expt_info = run_expts.generate_expt_x_y_groups(X , y , groups , expt_type = expt_num)


#3.Use the correct function to run the experiment, give it the correct random seeds and correct save path for the results
save_path = '/user/home/ko20929/RBD_using_custom_package/Data/Execute New Experiments/Results/Results_3/'
save_path = save_path + 'expt_' + str(expt_num) + '_' + data_type + '_' 
clf = {'ROCKET' : RocketClassifier(use_multivariate='yes') }

results_df = run_expts.run_mv_tsc(X_expt ,y_expt , groups_expt, clf, save_path = save_path , random_states = [1,2])