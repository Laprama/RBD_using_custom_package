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

# num = int(sys.argv[1])

# data_type = ['Wake', 'N1', 'N2', 'N3','REM'][num] #User inputs which data type should be used

for data_type in ['REM', 'N1', 'N2', 'N3', 'Wake']:
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

    #4. Transform the dataframe _______________________________________________________________________________________
    transformed_df = transformer.fit_transform(eeg_data_df)

    #Generate the rename mapping dictionary to rename the transformed dataframe using more appropriate names____________________
    transformed_names = transformed_df.columns
    channel_names = eeg_data_df.columns
    new_names = [channel + '_' + feature for channel in channel_names for feature in feature_list] #This is hard to follow but it is correct

    rename_mapping_dict = {}
    for old_name, new_name in zip(transformed_names,new_names):
        rename_mapping_dict[old_name] = new_name
        
    final_transformed_df = transformed_df.rename(rename_mapping_dict, axis=1)


    #5. Save everything in the appropriate place ---->  final_transformed_df, groups , y
    folder = 'Catch_22_features/'
    final_transformed_df.to_hdf(folder + data_type + '_c_22_feautures.h5' , key = 'df', mode = 'w')
    groups.to_hdf(folder + data_type + '_groups.h5' , key = 'df', mode = 'w')
    y.to_hdf(folder + data_type + '_y.h5' , key = 'df', mode = 'w')
