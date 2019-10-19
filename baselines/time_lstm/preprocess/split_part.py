# -*- coding: utf-8 -*-
#! /usr/bin/env python
import os
import numpy as np
import math

BASE_DIR = 'data'
DATA_SOURCE = 'movielens'
AMOUNT_OF_TEST_LINES = 2000
files = {
    'user_item_record': 'user-item.lst',
    'user_item_delta_time_record': 'user-item-delta-time.lst',
    'user_item_accumulate_time_record': 'user-item-accumulate-time.lst'
}

def load_lines(fileName):
    with open(os.path.join(BASE_DIR, DATA_SOURCE, fileName)) as f:
        lineList = f.readlines()
    return np.array(lineList)

loaded_data = {i: load_lines(files[i]) for i in files}

perms = np.random.permutation(loaded_data['user_item_record'].shape[0])
loaded_data = {i: loaded_data[i][perms] for i in loaded_data }


for name in files:
    loaded_lines = loaded_data[name]
    print("Converting file '"+ name + "' with "+str(len(loaded_lines))+" lines")
    training_file = open(os.path.join(BASE_DIR, DATA_SOURCE, 'tr_'+ files[name]), 'w')
    test_file = open(os.path.join(BASE_DIR, DATA_SOURCE, 'te_' + files[name]), 'w')

    for i, line in enumerate(loaded_lines):
        splitted_line = line.split()
        training = ' '.join(splitted_line[:-2]) + "\n"
        test = ' '.join(splitted_line) + "\n"

        training_file.write(training)
        if i < AMOUNT_OF_TEST_LINES:
            test_file.write(test)

    training_file.close()
    test_file.close()
