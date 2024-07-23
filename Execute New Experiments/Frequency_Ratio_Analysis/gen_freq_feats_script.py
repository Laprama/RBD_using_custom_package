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

d_num = int( sys.argv[1] )
data_type = ['N2', 'N3','REM', 'Wake', 'N1'][d_num]

channels = constants.channel_list

# 1. generate all path names and class list(s) etc. 
folder = '/user/home/ko20929/work/RBD_using_custom_package/Blue_pebble/'
paths = joblib.load(folder + data_type + '_paths.pkl') # keys : ['selected_paths', 's_class_list', 's_night_list', 's_sleep_type', 's_p_id']

# 2. Load corresponding data into dataframes, store in dataframe list
df_list = []

with io.capture_output() as captured:
    for path in paths['selected_paths']:
        data_epo = mne.read_epochs(path)
        data = data_epo._data * 1e6  # convert signal from V to uV
        df_full = data_epo.to_data_frame()
        df = df_full[channels].copy()
        df_list.append(df)

ts_row_list = []

for df in df_list:
    #1. Calculate bpw values
    bpw_df = eeg_stat_ts.gen_band_power_vals_and_freq_ratios(df, list(df.columns), win_s = 8)    
    bpw_df['window_no.'] = 0
    #2. Convert bpw per channel into bpw per region
    regional_df = eeg_stat_ts.convert_chan_stats_to_region(bpw_df, constants.channel_list , constants.region_to_channel_dict)
    #4. Convert into a single row of a new dataframe where each cell is a series
    new_row = eeg_stat_ts.dataframe_to_ts_row(regional_df, list(regional_df.columns[:-1]), time_steps = False )
    ts_row_list.append(new_row)
    
ts_df = pd.DataFrame.from_records(ts_row_list)
joblib.dump(ts_df, data_type + '_frequency_data.pkl')
