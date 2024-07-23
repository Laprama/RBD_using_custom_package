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

from mne_connectivity import spectral_connectivity_epochs
from mne.datasets import sample
import seaborn as sns

from scipy.signal import welch
import yasa
import constants
import numpy as np

data_type = ['Wake', 'N1', 'N2','N3', 'REM'][int(sys.argv[1])]

folder = '/user/home/ko20929/work/RBD_using_custom_package/Blue_pebble/'
paths = joblib.load(folder + data_type + '_paths.pkl')

method = 'pli'

connectivity_df_list = []

t1 = time.time()
#Because I want std of distribution each power band needs to be processed seperately
power_bands_dict = {'delta' : (0.5,4) , 'theta' : (4,8) , 'alpha' : (8,12) , 'sigma' : (12,16) , 'beta' : (16,30) , 'gamma' : (30,40), 'all' : (0.5,40) }

    
for path in paths['selected_paths']:
    #For each path the power band features (mean and std) are calculated seperately then concatenated
    power_band_dfs = []
    for power_band in power_bands_dict.keys():
        fmin = power_bands_dict[power_band][0]
        fmax = power_bands_dict[power_band][1]
        
        
        channels = constants.channel_list
        channel_names = constants.channel_list
        
        data_epo = mne.read_epochs(path)
        df_full = data_epo.to_data_frame()
        
        
        #generate coherence data across electrodes
        con_pli =  spectral_connectivity_epochs(data_epo , method=method , sfreq=256,fmin=fmin, fmax=fmax, faverage=False)
        
        connectivity_data = con_pli.get_data('dense')
        channel_data = connectivity_data
        
        # Create an empty DataFrame for the means
        df = pd.DataFrame(index=channel_names, columns=channel_names)
        
        # Fill the DataFrame with connectivity values
        for i in range(len(channel_names)):
            for j in range(len(channel_names)):
                channel_1 = channel_names[i]
                channel_2 = channel_names[j]
                connectivity_value = channel_data[i, j]
                df.loc[channel_1, channel_2] = connectivity_value.mean()
                df.loc[channel_2, channel_1] = connectivity_value.mean()
                
        df_mean = df.apply(pd.to_numeric)
        
        
        
        # Create an empty DataFrame for the stds
        df = pd.DataFrame(index=channel_names, columns=channel_names)
        
        # Fill the DataFrame with connectivity values
        for i in range(len(channel_names)):
            for j in range(len(channel_names)):
                channel_1 = channel_names[i]
                channel_2 = channel_names[j]
                connectivity_value = channel_data[i, j]
                df.loc[channel_1, channel_2] = connectivity_value.std()
                df.loc[channel_2, channel_1] = connectivity_value.std()
                
        df_std = df.apply(pd.to_numeric)
        
        new_df_row = {}
        
        for df , df_type in zip([df_mean , df_std] , ['mean' , 'std' ] ): 

            #4. Go through channel vs channel matrix dfs and transform into features for single row in df
            for i, channel in enumerate(channels):
                for channel_2 in channels[i+1:]:
                    val = df.loc[channel, channel_2]
                    new_df_row[power_band + '_' + df_type + '_' + channel + '_' + channel_2] = [val]
                    
        new_df = pd.DataFrame.from_dict(new_df_row, orient = 'columns')
        
        power_band_dfs.append(new_df)
    
    sample_df = pd.concat(power_band_dfs, axis = 1)
    connectivity_df_list.append(sample_df)

save_folder = '/user/home/ko20929/work/RBD_using_custom_package/Blue_pebble/Connectivity/test/'
connectivity_df = pd.concat(connectivity_df_list)

connectivity_df.to_hdf(save_folder + data_type + '_' + method + '_mean_and_std_' + '_df.h5' , key = 'df' , mode = 'w')

t2 = time.time()
print(str(t2-t1))