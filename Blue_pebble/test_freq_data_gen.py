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

#Let's time 1 dataframe transformation
t1 = time.time()

channels = constants.channel_list
paths_with_errors = []
paths , class_list, sleep_night_list , sleep_type_list , participant_id_list = constants.generate_paths_and_info(blue_pebble = True)


df_list = []

path = paths[0]

with io.capture_output() as captured:
    for path in paths[:20]:
        try:
            data_epo = mne.read_epochs(path)
            data = data_epo._data * 1e6  # convert signal from V to uV
            df_full = data_epo.to_data_frame()
            df = df_full[channels].copy()
            df_list.append(df)
        
        except:
            pass
print('done loading the data .....')
        
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
    
ts_df = pd.DataFrame.from_records(ts_row_list)    
    
t2 = time.time()

print('Time taken was .........')
print(t2-t1)

print('We did this for 20 paths mate(y)')