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

#Let me see as many results as I want to see
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

from sktime.classification.feature_based import Catch22Classifier
from sklearn.ensemble import RandomForestClassifier
from sktime.datasets import load_unit_test
from sktime.transformations.panel.catch22 import Catch22

feature_list = ['DN_HistogramMode_5', 'DN_HistogramMode_10', 'SB_BinaryStats_diff_longstretch0', 'DN_OutlierInclude_p_001_mdrmd', 'DN_OutlierInclude_n_001_mdrmd', 
 'CO_f1ecac', 'CO_FirstMin_ac', 'SP_Summaries_welch_rect_area_5_1', 'SP_Summaries_welch_rect_centroid', 'FC_LocalSimple_mean3_stderr', 'CO_trev_1_num', 
 'CO_HistogramAMI_even_2_5', 'IN_AutoMutualInfoStats_40_gaussian_fmmi', 'MD_hrv_classic_pnn40', 'SB_BinaryStats_mean_longstretch1', 'SB_MotifThree_quantile_hh',
 'FC_LocalSimple_mean1_tauresrat', 'CO_Embed2_Dist_tau_d_expfit_meandiff', 'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1', 'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1', 
 'SB_TransitionMatrix_3ac_sumdiagcov', 'PD_PeriodicityWang_th0_01' , 'StandardDeviation' , 'Mean']

transformer = Catch22(features = feature_list , catch24 = True )

t1 = time.time()

for data_type in ['REM']:

    print('Generating Catch22 features for ' + data_type + ' data')
    
    transformer = Catch22(features = feature_list , catch24 = True )
    channels = constants.channel_list
    paths , class_list, sleep_night_list , sleep_type_list , participant_id_list = constants.generate_paths_and_info()

    #1. select the appropriate paths and supplementary information - store in lists
    selected_paths , s_class_list , s_night_list , s_sleep_type , s_p_id = [], [], [], [], []


    for path , class_name, night , p_id in zip(paths, class_list, sleep_night_list, participant_id_list ):
        if data_type in path:
            selected_paths.append(path) 
            s_class_list.append(class_name)
            s_night_list.append(night)
            s_sleep_type.append(data_type)
            s_p_id.append(p_id)

    #2. Load corresponding data into dataframes , store in dataframe list
    df_list = []
    error_paths = []
    with io.capture_output() as captured:
        for path in selected_paths:
            try:
                data_epo = mne.read_epochs(path)
                data = data_epo._data * 1e6  # convert signal from V to uV
                df_full = data_epo.to_data_frame()
                df = df_full[channels].copy()
                df_list.append(df)
            except:
                #error with loading data
                error_paths.append(path)
                
    #Remove paths with errors from lists 
    for path in error_paths:
        path_index = selected_paths.index(path)
        #pop that index from all lists
        selected_paths.pop(path_index) 
        s_class_list.pop(path_index)
        s_night_list.pop(path_index)
        s_sleep_type.pop(path_index)
        s_p_id.pop(path_index)

            
    #Now we have the 57 channel EEG data in df's in df_list and corresponding supplementary information in the lists 
    #Selected_paths , s_class_list , s_night_list , s_sleep_type , s_p_id

    #3. Load all of the data into a single dataframe with each cell containing a time series 
    ts_row_list = []

    for df in df_list:
        row = {}
        for col in df.columns:
            row[col] = df[col]
        ts_row_list.append(row)
        
    #All of the main pieces of data to save 
    eeg_data_df = pd.DataFrame.from_records(ts_row_list)
    groups = pd.Series(s_p_id)
    class_list = pd.Series(s_class_list)
    y = class_list.map({'HC': 0 , 'PD' : 1 , 'PD+RBD' : 2 , 'RBD' : 3})

t2 = time.time()
print('Time taken to load data and convert to dataframe format: ' + str(t2-t1) + ' seconds')

t1 = time.time()
eeg_data_subset = eeg_data_df.iloc[:, :5].copy()
length_aligned_sub_df = custom_ts_length.customise_df_ts_length(eeg_data_subset,691200 ) 
t2 = time.time()

print('Time taken to customise length of time series: ' + str(t2-t1) + ' seconds')

channel_sample_combos = [(1, 1), (1, 4), (1, 8), (2, 1), (2, 4), (2, 8), (4, 1),
                         (4, 4), (4, 8) , (1,12) , (2,12) , (4,12)]
times = []
for channel , sample in channel_sample_combos:
    print('Channels: ' + str(channel) + ' Samples: ' + str(sample))
    t1 = time.time()
    transformed_df = transformer.fit_transform(length_aligned_sub_df.iloc[:sample, :channel])
    t2 = time.time()
    times.append(t2-t1)
    print('Time taken to transform data: ' + str(t2-t1) + ' seconds')

#write times to text file
with open('Catch22_times.txt', 'w') as f:
    for item in times:
        f.write("%s\n" % item)
        