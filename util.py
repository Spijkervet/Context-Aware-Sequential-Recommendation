import sys
import copy
import random
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone, timedelta

class TimeStamp():
    
    def __init__(self, timestamp):
        self.day = timestamp.strftime("%-d")
        self.hour = timestamp.strftime("%-H")
        self.date = timestamp.strftime("%c")

class UserItems():

    # TODO: Days ago relative to first date!
    def __init__(self, item, timestamp):
        self.item = item
        self.timestamp = datetime.fromtimestamp(timestamp).astimezone(timezone.utc)
        self.ts = TimeStamp(self.timestamp)
        self.day = self.ts.day

def data_partition(fpath):
    '''
    Temporarily taken from https://github.com/kang205/SASRec/blob/master/util.py
    '''

    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(fpath, 'r')
    for line in f:
        u, i, t = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        t = int(t)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)

        to_add = UserItems(i, t)
        User[u].append(to_add)

    # Create delta_time
    for user in User:
        most_recent_timestamp = User[user][-1].timestamp
        for u in User[user]:
            delta_timestamp = most_recent_timestamp - u.timestamp
            delta_timestamp = delta_timestamp.days # TODO: Add this as an argument
            if delta_timestamp > 31: # TODO: Add this as an argument
                delta_timestamp = 31

            u.delta_time = delta_timestamp
            # if u.delta_time != 0 and u.delta_time != 31:
            #     print(u.delta_time)
            # print(most_recent_timestamp, u.timestamp, delta_timestamp)

    # Partition data into three parts: train, valid, test.
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

def evaluate(model, dataset, args, sess):

    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        timeseq = np.zeros([args.maxlen], dtype=np.int32)

        idx = args.maxlen - 1
        seq[idx] = valid[u][0].item
        timeseq[idx] = valid[u][0].delta_time

        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i.item
            timeseq[idx] = i.delta_time
            idx -= 1
            if idx == -1: break
        rated = set([i.item for i in train[u]])
        rated.add(0)
        item_idx = [test[u][0].item]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], [timeseq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(list(range(1, usernum + 1)), 10000)
    else:
        users = list(range(1, usernum + 1))
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        timeseq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i.item
            timeseq[idx] = i.delta_time
            idx -= 1
            if idx == -1: break

        rated = set([i.item for i in train[u]])
        rated.add(0)
        item_idx = [valid[u][0].item]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [seq], [timeseq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
