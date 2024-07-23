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

## Use decision tree as a rough way for splitting based on that feature
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn import tree

#Import my modules
import format_eeg_data
import constants
import eeg_stat_ts
import run_expts

from constants import regions

# 1 Data type is defined by script input ______________________________________________________
d_num = int( sys.argv[1] )
data_type = ['N2', 'N3','REM', 'Wake', 'N1'][d_num]
expt_num = int(sys.argv[2])


for region in regions:
    #2. Load the data based on data type ___________________________________________________________
    df = joblib.load(os.path.join(os.path.abspath('..'), data_type + '_psd_normalised_data.pkl') )
    
    single_region_df = df[[col for col in df.columns if col.endswith('_' + region)]]
    frequency_vals  = np.arange(0.5,40.125, 0.125)
    single_region_df.columns = frequency_vals
    
    folder = '/user/home/ko20929/work/RBD_using_custom_package/Blue_pebble/'
    paths = joblib.load(folder + data_type + '_paths.pkl')
    
    groups = pd.Series(paths['s_p_id'])
    s_class_list = pd.Series(paths['s_class_list'])
    y = s_class_list.map({'HC': 0 , 'PD' : 1 , 'PD+RBD' : 2 , 'RBD' : 3})
    
    #3 Generate the frequency band search space ____________________________________________________
    width_slide_list = [(0.5,0.5)] 
    for window_width in range(1, 15 ):
        width_slide_list.append((window_width, 0.5) )
    
    #4 Calculate all of the features ________________________________________________________________
    feature_dfs = []
    
    for window_width_hz, window_slide_hz in width_slide_list:
        description = 'window width : ' + str(window_width_hz) + ' .  window stride : ' + str(window_slide_hz)
        
        #1.Set Window width in Hz and Window Slide in Hz
        # window_width_hz = 3
        window_len = (window_width_hz/0.125)+1
        assert window_len%1 == 0
        window_len = int(window_len)
        
        # window_slide_hz = 0.5 
        window_slide_len = window_slide_hz/0.125 
        assert window_slide_len%1 == 0
        window_slide_len = int(window_slide_len)
        
        
        #2.Calculate band values and store in dataframe calculated_df
        
        # Start of the window is the middle_freq value minus window_width_hz/2 
        # End of the window is the middle freq value plus window_width_hz/2
        window_len
        
        middle_freq = []
        final_cols = []
        
        i = 0
        while i < len(frequency_vals) - window_len:
            middle_freq.append( frequency_vals[i:i+window_len].mean() )
            band_vals = single_region_df.iloc[:,i:i+window_len].mean(axis = 1) #For every row calculate the mean for the appropriate elements
            
            final_cols.append(band_vals)
            
            # scaled_psd.append( psd_values[i:i+factor].mean() )
            i+= window_slide_len
        
        calculated_df = pd.DataFrame(final_cols).T
        calculated_df.columns = middle_freq
        
        calculated_df.columns = [ str(col) + '_width_' + str(window_width_hz) for col in calculated_df.columns]
        
        
        #3.Calculate Information Gain Based on Features
        #Change to binary HC vs PD / PD+RBD ---> Generate expt specific X,y,groups 
        X_expt , y_expt , groups_expt, expt_info = run_expts.generate_expt_x_y_groups(calculated_df,y,groups,expt_num)
    
        feature_dfs.append(X_expt)
            
    #5 Concatenate all feature_dfs into a single dataframe_________________________________________________________________
    X_expt_concatenated = pd.concat(feature_dfs, axis=1)
    
    #Edit from here downwards - 
    #1 Load the appropriate scores and column combos files
    f_name =  'Results/' + data_type + '_' + region +  '_expt_' + str(expt_num) + '_scores.pkl'
    scores = joblib.load(f_name)
    
    f_name =  'Results/' + data_type + '_' + region + '_expt_' + str(expt_num) + '_col_combos.pkl'
    col_combos = joblib.load(f_name)

    #2. Determine the top n column combos
    n = 1500
    top_n_indices = list(np.argsort(np.array(scores))[-n:])
    top_n_scores = np.array(scores)[top_n_indices]
    
    top_n_col_combos = np.array(col_combos)[top_n_indices]
    top_n_col_combos = [list(val) for val in top_n_col_combos]

    #3 Generate the reduced cols list 
    reduced_slide_list = [ (0.5, 0.5), (1, 0.5), (2, 0.5), (3, 0.5), (4, 0.5), (6, 0.5), (8, 0.5), (10, 1), (12, 1), (14, 1)]
    reduced_cols_list = []
    for window_width_hz, window_slide_hz in reduced_slide_list:
        description = 'window width : ' + str(window_width_hz) + ' .  window stride : ' + str(window_slide_hz)
            
        #1.Set Window width in Hz and Window Slide in Hz
        # window_width_hz = 3
        window_len = (window_width_hz/0.125)+1
        assert window_len%1 == 0
        window_len = int(window_len)
        
        # window_slide_hz = 0.5 
        window_slide_len = window_slide_hz/0.125 
        assert window_slide_len%1 == 0
        window_slide_len = int(window_slide_len)
    
        #2. Calculate band values and store in dataframe calculated_df
        middle_freq = []
    
        i = 0
    
        while i < len(frequency_vals) - window_len:
            middle_freq.append( frequency_vals[i:i+window_len].mean() )
            # scaled_psd.append( psd_values[i:i+factor].mean() )
            i+= window_slide_len
    
        cols_list = [str(col) + '_width_' + str(window_width_hz) for col in middle_freq]
        reduced_cols_list += cols_list

    #4. Create a new set of combos consisting of three cols
    three_col_combos_list = []
    for two_combo in top_n_col_combos:
        for col in reduced_cols_list:
            three_col = two_combo + [col]
            three_col_combos_list.append(three_col)
    
    scores = []
    for col_combo in three_col_combos_list: 
        X_selected = X_expt_concatenated[ col_combo ]
        # clf = RidgeClassifier().fit(X_selected, y_expt)
        clf = LinearDiscriminantAnalysis().fit(X_selected, y_expt)
        score = accuracy_score(y_expt,  clf.predict(X_selected))
        scores.append(score)
    
    f_name =  'Results/Three_Search/' + data_type + '_' + region +  '_expt_' + str(expt_num) + '_scores.pkl'
    joblib.dump(scores,f_name)
    
    f_name =  'Results/Three_Search/' + data_type + '_' + region + '_expt_' + str(expt_num) + '_col_combos.pkl'
    joblib.dump(three_col_combos_list,f_name)