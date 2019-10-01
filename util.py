import sys
import copy
import random
import numpy as np
import math
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

def get_delta_time(ts, bin_in_hours=48, max_bins=200, log_scale=False, min_ts=None, max_ts=None):

    '''
    Determines bin for an individual time delta.

    Arguments
    ---------

    ts : float
        Time delta for which bin should be calculated
    bin_in_hours : int
        Bin size in hours (only used if log scale not applied).
    max_bins : int
        Maximum number of bins.
    log_scale : bool
        Whether bin should be inferred from a log scale (where each bin is of equal log-length).
    min_ts : float
        Minimum timedelta in dataset.
    max_ts : float
        Maximum timedelta (defines the boundary of the right-most bin). Everything beyond max_ts
        will be grouped in last bin.
    '''

    if log_scale:

        # determine the extents of the log scale
        min_ts_log = np.log(min_ts)
        max_ts_log = np.log(max_ts)

        # log-transform the current timedelta
        ts_log = np.log(ts)

        # determine the size of each bin
        bin_size = (max_ts_log - min_ts_log) / max_bins

        # determine in which bin the current delta should live
        delta_timestamp = math.floor(ts_log / bin_size)
    else:

        # determine in which bin the current delta should live
        delta_timestamp = math.floor(ts.total_seconds()//3600 / bin_in_hours)

    # if the bin is larger than the maximum number of bins, bring it back to the last bin
    if delta_timestamp > max_bins:
        delta_timestamp = max_bins

    return delta_timestamp

def get_delta_range(User, max_percentile=90):

    '''
    Function to determine the maximum and minimum time deltas present in the data.

    Arguments
    ---------
    User : User object
        Contains all sequences for all users.

    max_percentile : int
        If maximum timedelta should be taken at a particular percentile in the dataset.
        Eventually, all observations beyond this percentile will then be mapped to
        the last bin.

    Returns
    -------

    min_timedelta : float
        Minimum time difference observed between current and previous
        interactions in a sequence. (Likely to be 0.0 in most datasets.)

    max_timedelta : float
        Maximum time difference observed between current and previous
        interactions in a sequence, at the specified max percentile.
    '''

    all_timedeltas = []

    for user in User:

        ts = User[user][-1].timestamp

        for u in User[user]:

            # get time difference with last-known observations
            delta_ts = ts - u.timestamp

            all_timedeltas.append(delta_ts)

    all_timedeltas = np.array(all_timedeltas)

    max_timedelta = np.percentile(all_timedeltas, 90)
    min_timedelta = np.amin(all_timedeltas)

    return min_timedelta, max_timedelta


def data_partition(args, fpath):
    '''
    Temporarily taken from https://github.com/kang205/SASRec/blob/master/util.py
    '''

    log_scale = args.log_scale

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

    # if positional embedding is calculated on the basis of a log scale get min and max timediff
    if log_scale:
        min_timedelta, max_timedelta = get_delta_range(User)


    # Create delta_time
    for user in User:
        most_recent_timestamp = User[user][-1].timestamp
        for u in User[user]:
            delta_timestamp = most_recent_timestamp - u.timestamp
            # delta_timestamp = delta_timestamp.days # TODO: Add this as an argument
            if log_scale:
                u.delta_time = get_delta_time(delta_timestamp, min_ts=min_timedelta, max_ts=max_timedelta)
            else:
                u.delta_time = get_delta_time(delta_timestamp, bin_in_hours=48, max_bins=200)
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
