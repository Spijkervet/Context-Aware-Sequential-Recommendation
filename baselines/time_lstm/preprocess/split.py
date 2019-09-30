# -*- coding: utf-8 -*-
#! /usr/bin/env python
import os
import numpy as np
import math

BASE_DIR = 'data'
DATA_SOURCE = 'amazon'

training_size = 0.8

files = {
    'user_item_record': 'user-item.lst',
    'user_item_delta_time_record': 'user-item-delta-time.lst',
    'user_item_accumulate_time_record': 'user-item-accumulate-time.lst'
}

def load_lines(fileName):
    with open(os.path.join(BASE_DIR, DATA_SOURCE, fileName)) as f:
        lineList = f.readlines()
    return np.array(lineList)


def write_lines(fileName, prefix, lines):
    out = open(os.path.join(BASE_DIR, DATA_SOURCE, prefix+fileName), 'w')
    for line in np.nditer(lines):
        out.write(str(line))

    out.close()


loaded_data = {i: load_lines(files[i]) for i in files}

permutation = np.random.permutation(loaded_data['user_item_record'].shape[0])
shuffled_data = {i: loaded_data[i][permutation] for i in loaded_data}

abs_training_size = math.floor(loaded_data['user_item_record'].shape[0] * 0.8)

for dset in shuffled_data:
    write_lines(files[dset], 'tr_', shuffled_data[dset][:abs_training_size])
    write_lines(files[dset], 'te_', shuffled_data[dset][abs_training_size:])
