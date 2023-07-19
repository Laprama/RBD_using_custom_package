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

t0 = time.time()

for data_type in ['REM', 'N1', 'N2', 'N3', 'Wake']:
    t1 = time.time()

    channels = constants.channel_list
    paths , class_list, sleep_night_list , sleep_type_list , participant_id_list = constants.generate_paths_and_info(blue_pebble = True)

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
        
    # All of the main pieces of data to save 
    eeg_data_df = pd.DataFrame.from_records(ts_row_list)
    groups = pd.Series(s_p_id)
    class_list = pd.Series(s_class_list)
    y = class_list.map({'HC': 0 , 'PD' : 1 , 'PD+RBD' : 2 , 'RBD' : 3})

    # Save these main pieces of data
    folder = 'eeg_data/'
    eeg_data_df.to_hdf(folder + data_type + '_c_22_feautures.h5' , key = 'df', mode = 'w')
    groups.to_hdf(folder + data_type + '_c_22_groups.h5' , key = 's', mode = 'w')
    class_list.to_hdf(folder + data_type + '_c_22_class_list.h5' , key = 's', mode = 'w')
    y.to_hdf(folder + data_type + '_c_22_y.h5' , key = 's', mode = 'w')

    t2 = time.time()
    time_taken = t2 - t1

    print('Time taken ' + data_type + '  :' + str(time_taken) + ' seconds')

t3 = time.time()

print('Total time taken: ' + str(t3-t0) + ' seconds')
