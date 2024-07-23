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

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score

#Import my modules
import format_eeg_data
import constants
import eeg_stat_ts
import run_expts
import custom_ts_length

#TS Fresh Parameter Settings
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_extraction import extract_features

# Set display options to show all rows and columns
pd.set_option('display.max_rows', 50)  # Show rows
pd.set_option('display.max_columns', 160)  # Show columns

input_num = int(sys.argv[1]) #Determines the data type via -> ['Wake', 'N1', 'N2','N3', 'REM'][input_num]

t1 = time.time()

save_folder = '/user/home/ko20929/work/RBD_using_custom_package/Execute New Experiments/TS_Fresh_Raw_EEG/Data/'

#generate all path names and class list(s) etc. 
channels = constants.channel_list
paths , class_list, sleep_night_list , sleep_type_list , participant_id_list = constants.generate_paths_and_info(blue_pebble = True)

folder = '/user/home/ko20929/work/RBD_using_custom_package/Blue_pebble/'
data_type = ['Wake', 'N1', 'N2','N3', 'REM'][input_num]
paths = joblib.load(folder + data_type + '_paths.pkl')

#Load each raw dataframe and convert into a row for the larger dataframe 
overall_dfs_rows = []

for path in paths['selected_paths']:
    data_epo = mne.read_epochs(path)
    df_full = data_epo.to_data_frame()

    #Convert the dataframe into a row
    new_row = {}
    for col in df_full.columns: #iterate through feature columns   
        series = df_full[col] #This gives you the series that you want to append to the new dataframe
        new_row[str(col)] = series
    
    overall_dfs_rows.append(new_row)

df_all_samples = pd.DataFrame(overall_dfs_rows)

#Choose how many minutes of the Raw EEG data you actually want to use
mins = 10
new_length = 256*60*mins #convert minutes to the number of data points
df_snipped = custom_ts_length.customise_df_ts_length(df_all_samples, new_length, impute = False)
df_snipped = df_snipped.drop(columns = ['time', 'condition', 'epoch'] )

# convert into TS_Fresh format Dataframe so that TS Fresh statistics can be calculated
ts_fresh_df = format_eeg_data.convert_sktime_df_to_ts_fresh_format(df_snipped, ts_cols = list(df_snipped.columns))

setting = EfficientFCParameters()

extracted_ts_fresh_df = extract_features(ts_fresh_df, column_id = 'id' , column_sort = 'time',  default_fc_parameters=setting, n_jobs = 8)

save_name = 'TS_Fresh_Stats_Efficient_10_min_' + data_type + '.pkl'
save_name = save_folder + save_name

joblib.dump(extracted_ts_fresh_df, save_name)

t2 = time.time()

time_taken = t2 - t1

print(time_taken)