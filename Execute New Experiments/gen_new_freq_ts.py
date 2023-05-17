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

#Ignore warnings for now 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


channels = constants.channel_list





#1. select the appropriate paths and supplementary information - store in lists

# for data_type in ['Wake', 'N1', 'N2', 'N3','REM']

for data_type in ['N2', 'N3','REM', 'Wake', 'N1']:   
    
    t1 = time.time()
    
    paths , class_list, sleep_night_list , sleep_type_list , participant_id_list = constants.generate_paths_and_info()
    
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

    #Convert each dataframe of raw chanel EEG data into a single row of TS data , with bpw statistics calculated per region
    # Store each row in ts_row_list
    ts_row_list = []

    for df in df_list:
        #1.Generate the window indices 
        window_indices = eeg_stat_ts.gen_window_indices(6, df , samp_freq = 256)
        #2. Calculate bpw vals per window
        bpw_per_win_df = eeg_stat_ts.gen_statistic_per_window(df , window_indices , stat = 'bpw')
        #3. Convert bpw per window per channel into bpw per window per region
        regional_df = eeg_stat_ts.convert_chan_stats_to_region(bpw_per_win_df, constants.channel_list , constants.region_to_channel_dict)
        #4. Convert into a single row of a new dataframe where each cell is a series
        new_row = eeg_stat_ts.dataframe_to_ts_row(regional_df, list(regional_df.columns[:-1]) )
        ts_row_list.append(new_row)
        
    # Save everything in the appropriate place ---->  final_transformed_df, groups , y
    folder = 'new_freq_ts_2/'
    
    ts_df = pd.DataFrame.from_records(ts_row_list)
    groups = pd.Series(s_p_id)
    s_class_list = pd.Series(s_class_list)
    y = s_class_list.map({'HC': 0 , 'PD' : 1 , 'PD+RBD' : 2 , 'RBD' : 3})

    ts_df.to_hdf(folder + data_type + 'six_second_freq_df.h5' , key = 'df', mode = 'w')
    groups.to_hdf(folder + data_type + '_groups.h5' , key = 'df', mode = 'w')
    y.to_hdf(folder + data_type + '_y.h5' , key = 'df', mode = 'w')
    print('Done for ' + data_type + ' !....')
    
    t2 = time.time()
    
    print(t2 - t1)

print(paths_with_errors)