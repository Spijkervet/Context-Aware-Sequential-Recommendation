import gzip
import json
import os
from tqdm import tqdm
import datetime
from collections import defaultdict

import logging

class DataReader():

    """
    SAMPLE
    {
        "reviewerID": "A2SUAM1J3GNN3B",
        "asin": "0000013714",
        "reviewerName": "J. McDonald",
        "helpful": [2, 3],
        "reviewText": "I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!",
        "overall": 5.0,
        "summary": "Heavenly Highway Hymns",
        "unixReviewTime": 1252800000,
        "reviewTime": "09 13, 2009"
    }
    """

    def __init__(self, path, dataset_fp, type, limit=None):
        self.path = path
        self.limit = limit
        self.dataset_fp = dataset_fp
        self.type = type
        self.logger = logging.getLogger('ir2')

    def parse_amazon(self):
        g = gzip.open(self.path, 'rb')
        for i, l in enumerate(g):
            if self.limit and i > self.limit:
                break
            yield eval(l)

    def preprocess(self):
        assert type(self.type) == str
        if self.type == 'amazon':
            self.preprocess_amazon()
        elif self.type == 'movielens':
            self.preprocess_movielens()

    def parse_movielens(self):
        f = open(self.path, 'r')
        for l in f:
            yield l.rstrip()

    def preprocess_movielens(self):
        f = open(self.dataset_fp, 'w')
        for l in tqdm(self.parse_movielens(), total=1000000):
            user, item, rating, timestamp = l.split('::')
            f.write('{} {} {}\n'.format(user, item, timestamp))

    def preprocess_amazon(self):
        countU = defaultdict(lambda: 0)
        countP = defaultdict(lambda: 0)
        total = 8898041 if not self.limit else self.limit

        logging.info('Reading and processing {}'.format(self.path))
        f = open(self.dataset_fp, 'w')
        for l in tqdm(self.parse_amazon(), total=total):
            f.write(" ".join([l['reviewerID'], l['asin'], str(l['overall']), str(l['unixReviewTime'])]) + ' \n')
            asin = l['asin']
            rev = l['reviewerID']
            time = l['unixReviewTime']
            countU[rev] += 1
            countP[asin] += 1
        f.close()

        logging.info('Creating user map dictionary')
        usermap = dict()
        usernum = 0
        itemmap = dict()
        itemnum = 0
        User = dict()
        for l in tqdm(self.parse(), total=total):
            asin = l['asin']
            rev = l['reviewerID']
            time = l['unixReviewTime']

            # Minimum of 5:
            if countU[rev] < 5 or countP[asin] < 5:
                continue

            if rev in usermap:
                userid = usermap[rev]
            else:
                usernum += 1
                userid = usernum
                usermap[rev] = userid
                User[userid] = []
            if asin in itemmap:
                itemid = itemmap[asin]
            else:
                itemnum += 1
                itemid = itemnum
                itemmap[asin] = itemid
            User[userid].append([time, itemid])

        logging.info('Sorting reviews for every user on time')
        # sort reviews in User according to time
        for userid in User.keys():
            User[userid].sort(key=lambda x: x[0])
        
        f = open(self.dataset_fp, 'w')
        for user in tqdm(User.keys()):
            for i in User[user]:
                f.write('%d %d %d\n' % (user, i[1], i[0]))
        f.close()

        # product map
        logging.info('Writing product item map')
        d = os.path.dirname(self.dataset_fp)
        bn = os.path.basename(self.dataset_fp)
        metadata_fp = os.path.join(d, bn + '_product_map.txt')
        with open(metadata_fp, 'w') as f:
            for k, v in tqdm(itemmap.items()):
                f.write('{} {}\n'.format(k, v))