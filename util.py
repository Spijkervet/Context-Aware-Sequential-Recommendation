import sys
import copy
import random
import numpy as np
from collections import defaultdict


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
        User[u].append(i)

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