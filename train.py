import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

'''
Note:
This df_final.csv file (we will generate from this script) 
can be created from merging 5 files given named as
'Neuro_Detect_train_{1 to 5}.parquet'

This code is for reference of one of the many methods that can be used to 
create merge files from different parquet files of eeg and spectrogram data
and combining it to given train.csv file
'''

# Defining paths for base datafiles for eeg and spectrogram data
path_train_eeg = 'path of all eeg files'
path_train_spectrograms = 'path of all spectrogram files'

# Getting datafile of training
df_train = 'path of train file named train.csv'


# Creating data from provided parquet files for eeg and spectrogram.
df_eeg_data = []
df_spectrogram_data = []
for sample in range(df_train.shape[0]):
	df_train_sample = df_train.iloc[sample, :]

	eeg_file = pd.read_parquet(f"{path_train_eeg}{df_train_sample['eeg_id']}.parquet")
	# 200 samples were taken for every second in eeg files
	df_train_eeg_seconds = int(df_train_sample['eeg_label_offset_seconds'] * 200)
	df_eeg_data.append(list(eeg_file.iloc[df_train_eeg_seconds, :]))

	spectrogram_file = pd.read_parquet(f"{path_train_spectrograms}{df_train_sample['spectrogram_id']}.parquet")
	# we need to find a row with time = (spectrogram_label_offset_seconds + 1) but the row number is (spectrogram_label_offset_seconds/2)
	df_train_spectrogram_seconds = int(df_train_sample['spectrogram_label_offset_seconds'] / 2)
	df_spectrogram_data.append(list(spectrogram_file.iloc[df_train_spectrogram_seconds, :]))


# merging and creating final file
cols_eeg = pd.read_parquet(f'{path_train_eeg}/eeg_1000913311.parquet.parquet')
df_eeg_all_data = pd.DataFrame(df_eeg_data,columns = cols_eeg)

cols_spectrograms = pd.read_parquet(f'{path_train_spectrograms}/spectrogram_1000086677.parquet')
df_spectrograms_all_data = pd.DataFrame(df_spectrogram_data,columns = cols_spectrograms)

df_final = df_train.merge(df_eeg_all_data,left_index=True,right_index=True)
df_final = df_final.merge(df_spectrograms_all_data,left_index=True,right_index=True)

df_final.to_parquet('/kaggle/working/final.parquet')
