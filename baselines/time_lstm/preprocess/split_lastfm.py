# -*- coding: utf-8 -*-
#! /usr/bin/env python
import os
import numpy as np
import math

BASE_DIR = 'data'
DATA_SOURCE = 'music'

files = {
    'user_item_record': 'user-item.lst',
    'user_item_delta_time_record': 'user-item-delta-time.lst',
    'user_item_accumulate_time_record': 'user-item-accumulate-time.lst'
}

def load_lines(fileName):
    print("Loading file:", fileName)
    with open(os.path.join(BASE_DIR, DATA_SOURCE, fileName)) as f:
        lineList = f.readlines()
    return np.array(lineList)

loaded_data = {i: load_lines(files[i]) for i in files}

print("Permutate")
perms = np.random.permutation(loaded_data['user_item_record'].shape[0])
loaded_data = {i: loaded_data[i][perms] for i in loaded_data }

train_percentage = 0.8
print("Split")
train_size = int(loaded_data['user_item_record'].shape[0] * train_percentage)
train_set = {i: loaded_data[i][:train_size] for i in loaded_data }
test_set = {i: loaded_data[i][train_size:] for i in loaded_data }

for name in files:
    print("Converting file '" + name + "' ")
    training_file = open(os.path.join(BASE_DIR, DATA_SOURCE, 'tr_' + files[name]), 'w')
    test_file = open(os.path.join(BASE_DIR, DATA_SOURCE, 'te_' + files[name]), 'w')

    training_file.write("".join(train_set[name].tolist()))
    test_file.write("".join(test_set[name].tolist()))

    training_file.close()
    test_file.close()