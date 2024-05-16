#%%
import numpy as np 
import pandas as pd 
import tqdm
import skimage.io
import matplotlib.pyplot as plt

# Define a random seed for shuffling the rows to partition into train, validation, and test sets
seed = 387962 #Generated from numpy.random.randint(0, 1E6, 1)

# Load and shuffle the data
data = pd.read_csv('../data/promoter_sequences.csv')
data = data.sample(frac=1, random_state=seed)

# Define the fractions for the train, validation, and test sets
train_frac = 0.8
dev_frac = 0.1
test_frac = 0.1

train_data = data.iloc[:int(train_frac*len(data))]
dev_data = data.iloc[int(train_frac*len(data)):int((train_frac+dev_frac)*len(data))]
test_data = data.iloc[int((train_frac+dev_frac)*len(data)):]

# Define a base encoder
encoder = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

# Iterate over the data sets
for d, f in zip([train_data, dev_data, test_data], ['train', 'dev', 'test']):
    for i in tqdm.tqdm(range(len(d))):
        seq = d.iloc[i]['sequence_5-3']
        encoded_seq = np.array([encoder[base] for base in seq])
        promoter_id = d.iloc[i]['promoter_id']
        promoter_name = d.iloc[i]['promoter_name']

        # Generate a one-hot encoding of the promoter sequence
        onehot = np.zeros((4, len(encoded_seq)))
        for j, base in enumerate(encoded_seq):
            onehot[base, j] = 1
        onehot = onehot.astype('uint8')
        skimage.io.imsave(f'../data/images/{f}/{promoter_id}-{promoter_name}.bmp', onehot,
                          check_contrast=False)
