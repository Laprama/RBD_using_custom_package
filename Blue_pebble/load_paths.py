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

t1 = time.time()

for data_type in ['Wake', 'N2', 'REM', 'N1', 'N3', ]:

    loaded_dict = joblib.load( data_type + '_paths.pkl')
    selected_paths = loaded_dict['selected_paths']
    s_class_list = loaded_dict['s_class_list']
    s_night_list = loaded_dict['s_night_list']
    s_sleep_type = loaded_dict['s_sleep_type'] 
    s_p_id = loaded_dict['s_p_id']

    print(data_type)
    print(len(selected_paths))

t2 = time.time()

print('Total time taken waz ' + str(t2-t2) + ' seconds')

