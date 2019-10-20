from __future__ import print_function
import pandas as pd
import pickle
import os

BASE_DIR = 'data'
DATA_SOURCE = 'beauty'

inputFile = os.path.join('../../../data/', 'Beauty.txt')

user_item_record = os.path.join(BASE_DIR, DATA_SOURCE, 'user-item.lst')
user_item_delta_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'user-item-delta-time.lst')
user_item_accumulate_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'user-item-accumulate-time.lst')
index2item_path = os.path.join(BASE_DIR, DATA_SOURCE, 'index2item')
item2index_path = os.path.join(BASE_DIR, DATA_SOURCE, 'item2index')

def generate_data():
    out_ui = open(user_item_record, 'w')
    out_uidt = open(user_item_delta_time_record, 'w')
    out_uiat = open(user_item_accumulate_time_record, 'w')

    print("Start")

    data = pd.read_csv(inputFile, sep=' ',
                      error_bad_lines=False,
                      header=None,
                      names=['user', 'item', 'rating', 'timestamp'])

    count = 0
    user_group = data.groupby(['user'])
    # short sequence comes first
    for userid, length in user_group.size().sort_values().iteritems():
        if count % 10 == 0:
            print("=====count %d======" % count)
        count += 1
        print('%s %d' % (userid, length))
        # oldest data comes first
        user_data = user_group.get_group(userid).sort_values(by='timestamp')
        item_seq = user_data['item']
        time_seq = user_data['timestamp']
        # filter the null data.
        item_seq = item_seq[item_seq.notnull()]
        time_seq = time_seq[time_seq.notnull()]
        # calculate the difference between adjacent items. -1 means using t[i] = t[i] - t[i+1]
        delta_time = pd.to_datetime(time_seq, unit='s').diff(-1).astype('timedelta64[s]') * -1

        # map music to index
        delta_time = delta_time.tolist()
        delta_time[-1] = 0
        time_accumulate = [0]
        for delta in delta_time[:-1]:
            next_time = time_accumulate[-1] + delta
            time_accumulate.append(next_time)
        out_ui.write(str(userid) + ',')
        out_ui.write(' '.join(str(x) for x in item_seq) + '\n')
        out_uidt.write(str(userid) + ',')
        out_uidt.write(' '.join(str(x) for x in delta_time) + '\n')
        out_uiat.write(str(userid) + ',')
        out_uiat.write(' '.join(str(x) for x in time_accumulate) + '\n')

    out_ui.close()
    out_uidt.close()
    out_uiat.close()

if __name__ == '__main__':
    generate_data()
