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


t0 = time.time()

#Lets trouble shoot with N1  as it only takes 82 seconds to create that dataframe
# for data_type in ['REM', 'N1', 'N2', 'N3', 'Wake']:

for data_type in ['Wake', 'N2', 'REM', 'N1', 'N3', ]:
    
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
       
    #Save the list of data paths to a .pkl file 
    paths_dict = {}
    paths_dict['selected_paths'] = selected_paths
    paths_dict['s_class_list'] = s_class_list
    paths_dict['s_night_list'] = s_night_list
    paths_dict['s_sleep_type'] = s_sleep_type
    paths_dict['s_p_id'] = s_p_id

    joblib.dump(paths_dict, data_type + '_paths.pkl')



    #Saving the data is taken out of the loop for investigation purposes
    t2 = time.time()
    time_taken = t2 - t1

    print('Time taken ' + data_type + '  :' + str(time_taken) + ' seconds')

t3 = time.time()

print('Total time taken: ' + str(t3-t0) + ' seconds')
