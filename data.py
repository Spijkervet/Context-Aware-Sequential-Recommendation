import os
import re
import numpy as np
import torch.utils.data as data

class AmazonDataset(data.Dataset):

    def __init__(self, data, seq_length):

        self._seq_length = seq_length
        self._data = list([item for sublist in data.values() for item in sublist])

        self._chars = sorted(list(set(self._data)))
        self._data_size, self._vocab_size = len(self._data), len(self._chars)

        print("Initialize dataset with {} characters, {} unique.".format(
            self._data_size, self._vocab_size))
        self._char_to_ix = { ch:i for i,ch in enumerate(self._chars) }
        self._ix_to_char = { i:ch for i,ch in enumerate(self._chars) }
        self._offset = 0
        
        offset = np.random.randint(0, len(self._data)-self._seq_length-2)

    def __getitem__(self, item):
        offset = np.random.randint(0, len(self._data)-self._seq_length-2)
        inputs =  [self._char_to_ix[ch] for ch in self._data[offset:offset+self._seq_length]]
        targets = [self._char_to_ix[ch] for ch in self._data[offset+1:offset+self._seq_length+1]]
        return inputs, targets

    def convert_to_product_id(self, char_ix):
        return ''.join(self._ix_to_char[ix] for ix in char_ix)

    def __len__(self):
        return self._data_size

    @property
    def vocab_size(self):
        return self._vocab_size