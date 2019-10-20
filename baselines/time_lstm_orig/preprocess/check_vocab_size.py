import os
import numpy as np
import math

BASE_DIR = 'data'
DATA_SOURCE = 'beauty'

files = {
    'user_item_record': 'te_user-item.lst',
    'user_item_delta_time_record': 'user-item-delta-time.lst',
    'user_item_accumulate_time_record': 'user-item-accumulate-time.lst'
}

def load_lines(fileName):
    with open(os.path.join(BASE_DIR, DATA_SOURCE, fileName)) as f:
        lineList = f.readlines()
    return np.array(lineList)

lines = load_lines(files['user_item_record'])

count = {}

for line in lines:
    user, history = line.split(',')
    history_split = history.split()

    for item in history_split:
        count[item] = 1

print("Items: "+str(len(count)))