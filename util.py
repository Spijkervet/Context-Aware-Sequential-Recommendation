import sys
import copy
import random
import numpy as np
import math
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime, timezone, timedelta

dayDict = {
	"Monday":1,
	"Tuesday":2,
	"Wednesday":3,
	"Thursday":4,
	"Friday":5,
	"Saturday":6,
	"Sunday":7
}

class TimeStamp():

    def __init__(self, timestamp):
        self.day = dayDict[timestamp.strftime("%A")]
        self.hour = int(timestamp.strftime("%H")) + 1
        # self.date = timestamp.strftime("%c")


class UserItems():

    # TODO: Days ago relative to first date!
    def __init__(self, item, rating, timestamp):
        self.item = item
        self.rating = rating
        self.timestamp_raw = timestamp
        self.timestamp = datetime.fromtimestamp(
            timestamp).astimezone(timezone.utc)
            
        self.ts = TimeStamp(self.timestamp)
        self.day = self.ts.day


def get_bin_size(min_ts, max_ts, max_bins):
    # determine the extents of the log scale
    # log(0) is undefined (-inf), add eps:
    # if min_ts == 0:
    #     min_ts += np.finfo(float).eps
    # if max_ts == 0:
    #     max_ts += np.finfo(float).eps

    min_ts_log = np.log(min_ts)
    max_ts_log = np.log(max_ts)

    # determine the size of each bin
    bin_size = (max_ts_log - min_ts_log) / max_bins
    return bin_size


def get_timedelta_bin(ts, bin_in_hours=48, max_bins=200, log_scale=False, min_ts=None, max_ts=None):
    '''
    Determines bin for an individual time delta.

    Arguments
    ---------

    ts : float
        Time delta in seconds for which bin should be calculated
    bin_in_hours : int
        Bin size in hours (only used if log scale not applied).
    max_bins : int
        Maximum number of bins.
    log_scale : bool
        Whether bin should be inferred from a log scale (where each bin is of equal log-length).
    min_ts : float
        Minimum hours in dataset.
    max_ts : float
        Maximum hours (defines the boundary of the right-most bin). Everything beyond max_ts
        will be grouped in last bin.
    '''

    if log_scale:
        # NOTE: (?) Add 1, since log(1) = 0
        min_ts += 1
        max_ts += 1
        ts += 1

        # determine in which bin the current delta should live
        bin_size = get_bin_size(min_ts, max_ts, max_bins)

        # log-transform the current timedelta, add eps if zero
        # if ts == 0:
        #     ts += np.finfo(float).eps

        ts_log = np.log(ts)
        time_bin = math.floor(ts_log / bin_size)

    else:

        # determine in which bin the current delta should live
        time_bin = math.floor(ts // 3600 / bin_in_hours)

    # if the bin is larger than the maximum number of bins, bring it back to the last bin
    if time_bin > max_bins:
        time_bin = max_bins

    return time_bin


def get_delta_range(User, max_percentile=90):
    '''
    Function to determine the maximum and minimum time deltas present in the data in seconds.

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
        Minimum time difference in seconds observed between current and previous
        interactions in a sequence. (Likely to be 0.0 in most datasets.)

    max_timedelta : float
        Maximum time difference in seconds observed between current and previous
        interactions in a sequence, at the specified max percentile.
    '''

    all_timedeltas = []
    for _, sequences in User.items():
        ts = sequences[-1].timestamp
        for u in sequences:
            # get time difference with last-known observations
            delta_ts = (ts - u.timestamp).total_seconds()
            all_timedeltas.append(delta_ts)

    all_timedeltas = np.array(all_timedeltas)
    max_timedelta = np.percentile(all_timedeltas, 90)
    min_timedelta = np.amin(all_timedeltas)
    return min_timedelta, max_timedelta


def get_users(fpath):
    usernum = 0
    itemnum = 0
    ratingnum = 0
    f = open(fpath, 'r')
    User = defaultdict(list)
    for line in f:
        u, i, r, t = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        r = float(r)
        t = int(t)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        ratingnum = max(r, ratingnum)

        to_add = UserItems(i, r, t)
        User[u].append(to_add)
    f.close()
    return User, usernum, itemnum, ratingnum


def add_time_bin(User, log_scale, bin_in_hours=48, max_bins=200):
    # if positional embedding is calculated on the basis of a log scale get min and max timediff
    if log_scale:
        min_timedelta, max_timedelta = get_delta_range(User)

    for _, sequences in User.items():
        most_recent_timestamp = sequences[-1].timestamp
        for s in sequences:
            time_delta = (most_recent_timestamp - s.timestamp).total_seconds()

            if log_scale:
                s.time_bin = get_timedelta_bin(time_delta, max_bins=max_bins,
                                               log_scale=True, min_ts=min_timedelta, max_ts=max_timedelta)
            else:
                s.time_bin = get_timedelta_bin(time_delta, bin_in_hours=bin_in_hours, max_bins=max_bins,
                                               log_scale=False)
    return User


def data_partition(fpath, log_scale=False):
    '''
    Temporarily taken from https://github.com/kang205/SASRec/blob/master/util.py
    '''

    user_train = {}
    user_valid = {}
    user_test = {}
    User, usernum, itemnum, ratingnum = get_users(fpath)

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
    return [user_train, user_valid, user_test, usernum, itemnum, ratingnum]


def evaluate(model, dataset, args, sess):

    [train, valid, test, usernum, itemnum, ratingnum] = copy.deepcopy(dataset)
    
    min_timedelta, max_timedelta = get_delta_range(train)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in tqdm(users):
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        orig_seq = [0] * args.maxlen
        timeseq = np.zeros([args.maxlen], dtype=np.int32)
        hours_seq = np.zeros([args.maxlen], dtype=np.int32)
        days_seq = np.zeros([args.maxlen], dtype=np.int32)

        idx = args.maxlen - 1
        seq[idx] = valid[u][0].item
        orig_seq[idx] = valid[u][0]
        hours_seq[idx] = valid[u][0].ts.hour
        days_seq[idx] = valid[u][0].ts.day

        valid_to_test_delta = (
            test[u][0].timestamp - valid[u][0].timestamp).total_seconds()
        timeseq[idx] = get_timedelta_bin(
            valid_to_test_delta, bin_in_hours=args.bin_in_hours, max_bins=args.max_bins, log_scale=False)
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i.item
            orig_seq[idx] = i
            hours_seq[idx] = i.ts.hour
            days_seq[idx] = i.ts.day
            idx -= 1
            if idx == -1:
                break

        # Get the timestamps between the most recent timestamp and the item's timestamp, and calculate its bin
        most_recent_timestamp = orig_seq[-1].timestamp
        for idx, s in enumerate(orig_seq):
            if s != 0:
                time_delta = (most_recent_timestamp -
                            s.timestamp).total_seconds()

                if args.log_scale:
                    timeseq[idx] = get_timedelta_bin(time_delta, bin_in_hours=args.bin_in_hours, max_bins=args.max_bins,
                                                   log_scale=True, min_ts=min_timedelta, max_ts=max_timedelta)
                else:
                    timeseq[idx] = get_timedelta_bin(time_delta, bin_in_hours=args.bin_in_hours, max_bins=args.max_bins,
                                                    log_scale=False)
            else:
                timeseq[idx] = 0

        rated = set([i.item for i in train[u]])
        rated.add(0)
        item_idx = [test[u][0].item]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        
        if args.test_model:

            if not args.test_seq_len:
                raise Exception('test_seq_len is not provided')
             
            max_seq_len = args.maxlen
            test_seq_len = args.test_seq_len

            # If test sequence length is larger than max, then set it equal to (sanity check)
            if test_seq_len > max_seq_len:
                test_seq_len = max_seq_len

            seq[:-test_seq_len] = 0
            timeseq[:-test_seq_len] = 0
            hours_seq[:-test_seq_len] = 0
            days_seq[:-test_seq_len] = 0

        predictions = -model.predict(sess, [u], [seq], item_idx, timeseq=[timeseq], hours_seq=[hours_seq], days_seq=[days_seq])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, sess):
    [train, valid, _, usernum, itemnum, ratingnum] = copy.deepcopy(dataset)

    min_timedelta, max_timedelta = get_delta_range(train)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum > 10000:
        users = random.sample(list(range(1, usernum + 1)), 10000)
    else:
        users = list(range(1, usernum + 1))
    for u in tqdm(users):
        if len(train[u]) < 1 or len(valid[u]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        hours_seq = np.zeros([args.maxlen], dtype=np.int32)
        days_seq = np.zeros([args.maxlen], dtype=np.int32)
        orig_seq = [0] * args.maxlen
        # for test data, the most recent item is always in the 0 bin
        timeseq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i.item
            orig_seq[idx] = i
            hours_seq[idx] = i.ts.hour
            days_seq[idx] = i.ts.day
            idx -= 1
            if idx == -1:
                break

        most_recent_timestamp = orig_seq[-1].timestamp
        for idx, s in enumerate(orig_seq):
            if s != 0:
                time_delta = (most_recent_timestamp -
                              s.timestamp).total_seconds()

                if args.log_scale:
                    timeseq[idx] = get_timedelta_bin(time_delta, bin_in_hours=args.bin_in_hours, max_bins=args.max_bins,
                                                   log_scale=True, min_ts=min_timedelta, max_ts=max_timedelta)
                else:
                    timeseq[idx] = get_timedelta_bin(time_delta, bin_in_hours=args.bin_in_hours, max_bins=args.max_bins,
                                                log_scale=False)
            else:
                timeseq[idx] = 0

        rated = set([i.item for i in train[u]])
        rated.add(0)
        item_idx = [valid[u][0].item]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        if args.test_model:

            if not args.test_seq_len:
                raise Exception('test_seq_len is not provided')
             
            max_seq_len = args.maxlen
            test_seq_len = args.test_seq_len

            # If test sequence length is larger than max, then set it equal to (sanity check)
            if test_seq_len > max_seq_len:
                test_seq_len = max_seq_len

            seq[:-test_seq_len] = 0
            timeseq[:-test_seq_len] = 0
            hours_seq[:-test_seq_len] = 0
            days_seq[:-test_seq_len] = 0



        predictions = -model.predict(sess, [u], [seq], item_idx, timeseq=[timeseq], hours_seq=[hours_seq], days_seq=[days_seq])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user
