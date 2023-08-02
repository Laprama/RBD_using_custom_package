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
import mne

from sklearn.model_selection import train_test_split
#From my EEG package 
import run_expts
import format_eeg_data
import constants
import eeg_stat_ts
import custom_ts_length

from sktime.transformations.panel.catch22 import Catch22

feature_list = ['DN_HistogramMode_5', 'DN_HistogramMode_10', 'SB_BinaryStats_diff_longstretch0', 'DN_OutlierInclude_p_001_mdrmd', 'DN_OutlierInclude_n_001_mdrmd', 
 'CO_f1ecac', 'CO_FirstMin_ac', 'SP_Summaries_welch_rect_area_5_1', 'SP_Summaries_welch_rect_centroid', 'FC_LocalSimple_mean3_stderr', 'CO_trev_1_num', 
 'CO_HistogramAMI_even_2_5', 'IN_AutoMutualInfoStats_40_gaussian_fmmi', 'MD_hrv_classic_pnn40', 'SB_BinaryStats_mean_longstretch1', 'SB_MotifThree_quantile_hh',
 'FC_LocalSimple_mean1_tauresrat', 'CO_Embed2_Dist_tau_d_expfit_meandiff', 'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1', 'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1', 
 'SB_TransitionMatrix_3ac_sumdiagcov', 'PD_PeriodicityWang_th0_01' , 'StandardDeviation' , 'Mean']

transformer = Catch22(features = feature_list , catch24 = True)

try:
    path_num = int(sys.argv[1]) #This will be input to the script
except:
    raise ValueError ('Error with path number!!!')

data_type = str(sys.argv[2])

if data_type not in ['REM', 'N1', 'N2', 'N3', 'Wake']:
    raise ValueError('Data type incorrect, it was: ' + str(data_type) )

channels = constants.channel_list

core_path = '/user/home/ko20929/work/RBD_using_custom_package/Blue_pebble/'

try:
    paths_dict = joblib.load(core_path + data_type + '_paths.pkl')
except: 
    raise ValueError ('Error with loading path dictionary')
       
#Then assign the lists to the appropriate variables
selected_paths = paths_dict['selected_paths']
s_class_list = paths_dict['s_class_list']
s_night_list = paths_dict['s_night_list']
s_sleep_type = paths_dict['s_sleep_type']
s_p_id = paths_dict['s_p_id']

#Now everything that was done for multiple paths is done for the one selected path (everything in parallel)
#You really only need the path --> as oll supplementary info goes into groups , class_list and y 

selected_path = selected_paths[path_num]

#2.Load corresponding data into dataframe, df 
data_epo = mne.read_epochs(selected_path)
data = data_epo._data * 1e6  # convert signal from V to uV
df_full = data_epo.to_data_frame()
df = df_full[channels].copy()

#3.Load all of the data into a single dataframe with each cell containing a time series
ts_row_list = []
row = {}
for col in df.columns:
    row[col] = df[col]
ts_row_list.append(row)

#Create dataframe from that single row (previously was dataframe from multiple rows)____________________________________
#All of the main pieces of data to save 
eeg_data_df = pd.DataFrame.from_records(ts_row_list)
#Trims down to 45 minutes worth of data
eeg_data_df = custom_ts_length.customise_df_ts_length(eeg_data_df,691200 , impute = False ) 


#4. Transform the dataframe _______________________________________________________________________________________
t1 = time.time()

if str(sys.argv[3]) == 'test':
    transformed_df = transformer.fit_transform(eeg_data_df.iloc[:,:1])

else:
    transformed_df = transformer.fit_transform(eeg_data_df)
t2 = time.time()

print(t2-t1)

# Save the transformed_df 
transformed_df.to_hdf('/user/home/ko20929/work/RBD_using_custom_package/Blue_pebble/C_22_data/' + str(path_num) + data_type + '_c_22_features.h5', key = 'df', mode = 'w')