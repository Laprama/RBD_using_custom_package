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

#Wake connectivity already generated, N2 to be done in a seperate script 
data_type = ['REM', 'N1', 'N3'][int(sys.argv[1])]
paths = joblib.load(data_type + '_paths.pkl')

print('paths loaded ...')

connectivity_df_list = []

t1 = time.time()

for path in paths['selected_paths']:
    #1. Load the data 
    channels = constants.channel_list
    
    
    print(path)
    data_epo = mne.read_epochs(path)
    df_full = data_epo.to_data_frame()
    
    # 2. Define the power bands
    
    power_bands = {'delta' : (0.5,4) , 'theta' : (4,8) , 'alpha' : (8,12) , 'sigma' : (12,16) , 'beta' : (16,30) , 'gamma' : (30,40) }
    
    fmin = [float(val[0]) for val in power_bands.values()]
    fmax = [float(val[1]) for val in power_bands.values()]
    
    #generate coherence data across electrodes
    con_pli =  spectral_connectivity_epochs(data_epo , method='coh' , sfreq=256,fmin=fmin, fmax=fmax, faverage=True)
    
    #3. Generate all of the power band dataframes for that sample and add to power_band_coherence_dfs dictionary
    power_band_coherence_dfs = {}
    for power_band_index, power_band in enumerate(list(power_bands.keys())):
        
        channel_names = channels
        connectivity_data = con_pli.get_data('dense')[:, : , power_band_index]
        
        channel_data = connectivity_data
        
        # Create an empty DataFrame
        df = pd.DataFrame(index=channel_names, columns=channel_names)
        
        # Fill the DataFrame with connectivity values
        for i in range(len(channel_names)):
            for j in range(len(channel_names)):
                channel_1 = channel_names[i]
                channel_2 = channel_names[j]
                connectivity_value = channel_data[i, j]
                df.loc[channel_1, channel_2] = connectivity_value
                df.loc[channel_2, channel_1] = connectivity_value
        
        df = df.apply(pd.to_numeric)
        
        power_band_coherence_dfs[power_band] = df
    
    new_df_row = {}
    
    #4. Go through all of the power_bands and add data as a column for a new dataframe with power_band + channel_1 + channel_2 as feature(s) 
    for power_band in list(power_bands.keys()):
        print(power_band)
        df = power_band_coherence_dfs[power_band]
        for i, channel in enumerate(channels):
            for channel_2 in channels[i+1:]:
                val = df.loc[channel, channel_2]
                new_df_row[power_band + '_' + channel + '_' + channel_2] = [val]
    
    new_df = pd.DataFrame.from_dict(new_df_row, orient = 'columns')

    connectivity_df_list.append(new_df)

folder = 'Connectivity/'
connectivity_df = pd.concat(connectivity_df_list)
connectivity_df.to_hdf(folder + data_type + '_coherence_df.h5' , key = 'df' , mode = 'w')

t2 = time.time()
print(str(t2-t1))
