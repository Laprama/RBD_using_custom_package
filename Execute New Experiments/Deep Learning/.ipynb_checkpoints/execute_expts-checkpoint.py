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
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold

from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian

#Import my modules
import format_eeg_data
import constants
import eeg_stat_ts

#Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#Example command to execute the script : 
# python execute_expts.py Wake [52,20] 0.75 Prefrontal 200 100  80 100
#                         type seeds  overlap Region  g_std hop segment_length epochs
#Script Inputs : 
data_type = sys.argv[1]

seeds = sys.argv[2] #This is a user input list
seeds = seeds.strip('[]').split(',')
seeds = [int(num) for num in seeds]

overlap = float(sys.argv[3])

region = sys.argv[4]

g_std = int(sys.argv[5])  #200 User input
window_width = g_std*2 #could be changed to a user input

hop = int(sys.argv[6]) #Hop in samples is user input

segment_length = int(sys.argv[7]) #80 #User input (I need to edit the code to have this as a user input)

epochs = int(sys.argv[8]) #User input of epochs

# Set the folder to save the data 
save_folder = 'Results/Folder_1/'

start_time = time.time()

print('script started...')

# 1.Load the data __________________________________________________________________________________
df_list = joblib.load(data_type + '_normalised_dataframes.pkl')
# 1. generate all path names and class list(s) etc. 
folder = '/user/home/ko20929/work/RBD_using_custom_package/Blue_pebble/'
paths = joblib.load(folder + data_type + '_paths.pkl') # keys : ['selected_paths', 's_class_list', 's_night_list', 's_sleep_type', 's_p_id']

class_label_dict = {'HC': 0 , 'PD' : 1 , 'PD+RBD' : 2 , 'RBD' : 3} #Dictionary used to label the classes for reference
y = np.array([class_label_dict[c_name] for c_name in paths['s_class_list'] ] )
groups = paths['s_p_id']

wake_dfs_binary = []
y_binary = []
groups_binary = []

for df , class_label , group in zip(df_list, y, groups):
    if class_label in [0,1]:
        wake_dfs_binary.append(df)
        y_binary.append(class_label)
        groups_binary.append(group)

y_binary = np.array(y_binary)

#2.Generate the Spectrograms _________________________________________________________________________
regions = constants.regions
region_channel_dict = constants.region_to_channel_dict
channel_list = constants.channel_list
region_channel_dict[region]


#For each dataframe generate a spectrogram
regional_spectrograms = []

for df in wake_dfs_binary:
    
    channel_spectrograms = []
    for channel in region_channel_dict[region]:
        
        #for each channel generate a spectrogram, then take the mean of the spectrograms to get the regional spectrogram
           
        eeg_data = df[channel].values 

        # g_std is standard deviation for Gaussian window in samples
        w = gaussian(window_width, std=g_std, sym=True)  # symmetric Gaussian window of total width 'window_width' samples
        mfft = max(256 , g_std*2)
        
        #Perform STFT 
        SFT = ShortTimeFFT(w, hop=hop, fs=256, mfft=mfft, scale_to='magnitude')   
        Sx = SFT.stft(eeg_data)  
        Sx_abs = abs(Sx)
        
        #I can obtain the corresponding frequency values for this spectrogram with 
        frequency_vals = SFT.f
        
        # I want to snip the frequency values to the range I'm interested in 0-40 Hz
        def find_first_above(array, threshold):
            for index, value in enumerate(array):
                if value > threshold:
                    return index
            return -1  # Return -1 if no such value is found
        
        ind = find_first_above(frequency_vals, 40)
        
        frequency_vals = frequency_vals[:ind+1]
        
        #To update the time values I need to input the number of vals in the original sequency 
        num_seq_vals = len(eeg_data)
        time_vals = SFT.t(num_seq_vals)
        
        # snip the spectrogram array to values that are of interest 
        Sx_abs = Sx_abs[:ind+1 , :]

        #Generate the power from the magnitude
        Sx_db = 10*np.log10( np.square(Sx_abs))
        
        channel_spectrograms.append(Sx_db)
        # [Sx_db , time_vals , frequency_vals]
    
    stacked_arrays = np.stack( channel_spectrograms)
    mean_spectrogram = np.mean(stacked_arrays, axis = 0)
    regional_spectrograms.append(mean_spectrogram)
    
input_height = len(frequency_vals)

#3. Generate Spectrogram Slices _____________________________________________________________________________________
#segment_length is user input and overlap is user input at the start of the script

spectrogram_slices = []
y_slice_labels = []
y_slice_groups = []

for spectrogram, label, group in zip(regional_spectrograms, y_binary, groups_binary):
    
    num_segments = int( np.floor(spectrogram.shape[1]/segment_length) )
    new_specs = []
    
    for i in np.arange(0,num_segments, 1 - overlap):
        if i > num_segments - 1 :
            # if signal is 4 full minutes, I don't want it to try and take a window from 3.5 minutes to 4.5 minutes
            # will be an incomplete slice causing errors downstream, needs to stop 
            break
          
        start_index = int( np.floor(i*segment_length) )
        end_index = start_index + segment_length
        new_spec = spectrogram[: , start_index : end_index]
        
        spectrogram_slices.append(new_spec)
        y_slice_labels.append(label)
        y_slice_groups.append(group)
        
plt.hist(np.array(y_slice_labels) )

#4. Create the CNN _________________________________________________________________________________

def out_dim(x):
    '''
    Function to calculate output dimensions for neural network to make it easy for me to define NN for changing window sizes. 
    Action on width and height dim sizes are currently equivalent
    '''
    x = x-3 #convolutional filter of size 4 applied 
    x = int( np.floor(x/4) ) # effect of maxpooling 4 x 4 
    x = x-3 #convolutional filter of size 4 applied
    x = int( np.floor(x/4) ) # effect of maxpooling 4 x 4 
    return x


class AdaptiveConvolutionalNetwork(nn.Module):
    def __init__(self, input_width = 110 , input_height = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,4,1)
        self.bn1 = nn.BatchNorm2d(6)  # Batch norm after first conv layer
        self.conv2 = nn.Conv2d(6, 16, 4, 1)
        self.bn2 = nn.BatchNorm2d(16)  # Batch norm after second conv layer

        self.out_h = out_dim(input_height) 
        self.out_w = out_dim(input_width)
        
        self.fc1 = nn.Linear(16*self.out_h*self.out_w, int(np.floor( 16*self.out_h*self.out_w / 2) ))
        self.bn3 = nn.BatchNorm1d(int(np.floor( 16*self.out_h*self.out_w / 2) ))

        
        self.fc2 = nn.Linear(int(np.floor( 16*self.out_h*self.out_w / 2) ), 50)
        # self.fc3 = nn.Linear(80,10)
        # self.fc4 = nn.Linear(10,1)
        self.fc3 = nn.Linear(50,1)
        
    def forward(self, X):
        
        X = F.relu(self.bn1(self.conv1(X)))
        X = nn.Dropout2d(p=0.2)(X) # Dropout after first conv layer
        
        X = F.max_pool2d(X,4,4)
        X = F.relu(self.bn2(self.conv2(X)))
        X = nn.Dropout2d(p=0.2)(X) # Dropout after second conv layer

        X = F.max_pool2d(X,4,4)

        
        X = X.view(-1, 16*self.out_h*self.out_w)
        X = F.relu(self.bn3(self.fc1(X) ) )
        X = nn.Dropout(p=0.15)(X) # Dropout after first FC layer
        
        X = F.relu(self.fc2(X))
        X = nn.Dropout(p=0.15)(X) # Dropout after first FC layer
        
        # X = F.relu(self.fc3(X) )
        X = self.fc3(X)
        # X = self.fc4(X) 
                
        # return F.log_softmax(X, dim=1 )
        return X

#5. Do the train test splits _________________________________________________________________

# Train and Validation splits only ----> NO TEST
# spectrogram_slices, y_slice_labels and y_slice_groups to work with

train_val_dict = {}

for value in ['train' , 'val']:
    train_val_dict[value] = {}

X = np.stack(spectrogram_slices)
y = np.array(y_slice_labels)
groups = np.array( [int(group) for group in y_slice_groups] )

gkf = GroupKFold(n_splits = 4) 
fold = 0

for train_index, val_index   in gkf.split(X, y, groups*1):
    fold += 1
    
    X_train, y_train, groups_train  = X[train_index], y[train_index] , groups[train_index]
    X_val, y_val, groups_val =  X[val_index], y[val_index] , groups[val_index]   
    
    train_val_dict['train'][fold] = X_train, y_train, groups_train
    train_val_dict['val'][fold]   = X_val, y_val, groups_val
    
    total_len = len(X) 
    val_percent = 100*(len(X_val) / total_len)
    train_percent = 100*(len(X_train) / total_len)
   
    print('fold ' + str(fold) ) 
    print( str(train_percent)[:3] + ' | '  + str(val_percent)[:3] + ' |' )

    # testing that the splits are as expected
    print( np.unique(groups_train) )
    print( np.unique(groups_val) )
    
    print('__________________________________________________________________________')

#Output from this section of code is X_train, y_train, groups_train AND X_test, y_test, groups_test 

#6.Fit the model and draw the graph + save it _________________________________________________________________


rows = len(seeds) # Make the figure the right size 

fig = plt.figure()
fig = plt.figure(figsize=(24,4*rows),dpi=100)

# k is for subplots within the overall figure 

k = 1

for fold in [1,2,3,4]:
    print(fold)
    X_train, y_train, groups_train = train_val_dict['train'][fold]
    X_val, y_val, groups_val = train_val_dict['val'][fold]  
    
    # Creating train and test data loaders
    input_width = segment_length
    train_data = [ (torch.from_numpy(spectrogram).float().view(1,input_height,input_width), val) for spectrogram, val in zip(X_train, y_train) ] 
    train_loader = DataLoader(train_data, batch_size=30, shuffle=True)
    
    val_data = [ (torch.from_numpy(spectrogram).float().view(1,input_height,input_width), val) for spectrogram, val in zip(X_val, y_val) ] 
    val_loader = DataLoader(val_data , batch_size=30, shuffle=False)
    
   
    
    
    
    # for seed in [2,5,15,50]:
    for seed in seeds:
        # set all seeds 
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(device) - to check that device is actually cuda
        
        model = AdaptiveConvolutionalNetwork(input_width = input_width , input_height = input_height)
        model.to(device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001 )

        train_losses = []
        val_losses = []
        test_losses = []
        
        train_correct = []
        val_correct = []
        test_correct = []
        
        for i in range(epochs):
            
            trn_corr = 0
            val_corr = 0
            tst_corr = 0
             
            
            trn_loss = 0
            val_loss = 0
            tst_loss = 0
            
            model.train()
            # Run the training batches
            for b, (X_train_batch, y_train_batch) in enumerate(train_loader):
                b+=1
        
                #Move train data to the GPU
                X_train_batch = X_train_batch.to(device)
                y_train_batch = y_train_batch.to(device)
                
                # Apply the model
                y_pred = model(X_train_batch)  # we don't flatten X-train here
                loss = criterion(y_pred, y_train_batch.unsqueeze(1).float())
         
                # Tally the number of correct predictions
                predicted = torch.round(F.sigmoid(y_pred.detach() ) )
                predicted = predicted.reshape(y_train_batch.shape)
                
                batch_corr = (predicted == y_train_batch).sum()
                trn_corr += batch_corr
                trn_loss += loss
                
                # Update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            train_losses.append(trn_loss)
            train_correct.append(trn_corr)
        
            # Run the validation batches
            # Some of the variables in this loop have the same name as the variables in the above loop... be aware of that plz!
            model.eval()
            with torch.no_grad():
                for b, (X_val_batch, y_val_batch) in enumerate(val_loader):
                    b+=1
                    
                    #Move train data to the GPU
                    X_val_batch = X_val_batch.to(device)
                    y_val_batch = y_val_batch.to(device)
        
                    # Apply the model
                    y_val = model(X_val_batch)
        
                    # Tally the number of correct predictions
                    predicted = torch.round(F.sigmoid(y_val.detach() ) )
                    predicted = predicted.reshape(y_val_batch.shape)
                    
                    batch_corr = (predicted == y_val_batch).sum()
                    val_corr += batch_corr
        
                    
                    loss = criterion(y_val, y_val_batch.unsqueeze(1).float())
                    val_loss += loss 
                   
            val_losses.append(val_loss)
            val_correct.append(val_corr)
        
            
           
        
        # Plot the outcome from the loop
        
        ax = fig.add_subplot(rows,4,k)
        k+=1
        plt.title('fold ' + str(fold), fontsize = 10)
        plt.plot([(val.cpu() / len(X_train) ) for val in train_correct], label='training set accuracy')
        plt.plot([(val.cpu()/len(X_val) ) for val in val_correct], label='validation set accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epochs') 
        plt.grid()
    
    
    plt.tight_layout()


plt.legend()   

#Add text at the bottom of the figure
# fig.text(0.5, 0, 'This is a caption at the bottom of the figure | Model : ' + str(model) , va='bottom')
fig.text(0.5, 0, f'\nDuration: {time.time() - start_time:.0f} seconds' , ha='center', va='bottom')

plt.tight_layout(pad = 2.0)

save_name = data_type + '_' + region + '_seeds_' + str(seeds) + '_' + 'overlap_' + str(overlap) + '_' + 'window_gstd_' + str(g_std) + '_hop_' + str(hop) + '_segement_length' + '_' + str(time_vals[segment_length])[:5] + '_secs'

plt.savefig(save_folder + save_name +'.png')

# Create a text file and write to it_________________________

file_name = save_folder + save_name + ".txt"
with open(file_name, 'w') as file:
    file.write(save_name)
    file.write('\n')
    file.write(str(model) )

print('Done')
