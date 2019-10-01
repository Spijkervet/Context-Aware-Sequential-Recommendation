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

    def __init__(self, path, dataset_fp, type, limit=None, maxlen=None):
        self.path = path
        self.limit = limit
        self.dataset_fp = dataset_fp
        self.type = type
        self.logger = logging.getLogger('ir2')
        self.maxlen = maxlen

    def preprocess(self):
        assert type(self.type) == str
        if self.type == 'amazon':
            self.preprocess_amazon()
        elif self.type == 'movielens':
            self.preprocess_movielens()
        elif self.type == 'amazon_ratings':
            self.preprocess_amazon_ratings()

    def parse_amazon(self):
        g = gzip.open(self.path, 'rb')
        for i, l in enumerate(g):
            if self.limit and i > self.limit:
                break
            yield eval(l)

    def parse_movielens(self):
        f = open(self.path, 'r')
        for i, l in enumerate(f):
            if self.limit and i > self.limit:
                break
            yield l.rstrip()

    def preprocess_movielens(self):
        logging.info('Reading and processing {}'.format(self.path))

        countU = defaultdict(lambda: 0)
        countP = defaultdict(lambda: 0)
        total = 100000 if not self.limit else self.limit

        delim = '::'
        f = open(self.dataset_fp, 'w')
        for l in tqdm(self.parse_movielens(), total=total):
            user, item, rating, timestamp = l.split(delim)
            f.write('{} {} {}\n'.format(user, item, timestamp))
            asin = int(item) 
            rev = int(user)
            time = int(timestamp)
            countU[rev] += 1
            countP[asin] += 1
        f.close()

        logging.info('Creating user map dictionary')
        usermap = dict()
        usernum = 0
        itemmap = dict()
        itemnum = 0
        User = dict()
        for l in tqdm(self.parse_movielens(), total=total):
            rev, asin, rating, time = l.split(delim)
            rev = int(rev)
            asin = int(asin)
            time = int(time)

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
            User[userid].append([itemid, time])

        logging.info('Sorting reviews for every user on time')
        # sort reviews in User according to time
        for userid in User.keys():
            User[userid].sort(key=lambda x: x[1])
            
        # Original data writer
        f = open(self.dataset_fp, 'w')
        for user in User.keys():
            for i in User[user]:
                f.write('%d %d %d\n' % (user, i[0], i[1]))
        f.close()

        # f = open(self.dataset_fp, 'w')
        # for user, v in User.items():
        #     # NOTE: self.maxlen - 3, since we want to hold one out for validation and for testing and have self.maxlen as training set!!!
        #     to_write = v[-self.maxlen-3:] if self.maxlen else v
        #     for tw in to_write:
        #         f.write('%d %d %d\n' % (user, tw[0], tw[1]))
        # f.close()

        # Test maxlen
        # ts = open(self.dataset_fp, 'r')
        # users = defaultdict(list)
        # for l in ts:
        #     user, item, ts = l.split(' ')
        #     users[user].append(item)
        
        # for k, v in users.items():
        #     if len(v) > self.maxlen + 3:
        #         exit('Maxlen exceeded in {}'.format(k, v))


        # tsv metadata file (index/label)
        logging.info('Writing tsv metadata file (index/label)')
        d = os.path.dirname(self.dataset_fp)
        bn = os.path.basename(self.dataset_fp)

        movies_dict = {}
        movies_labels_path = os.path.join(os.path.dirname(self.path), 'movies.dat')
        with open(movies_labels_path, 'r', encoding='ISO-8859-1') as f:
            for l in f:
                key, movie, genre, = tuple(l.rstrip().split(delim))
                movies_dict[int(key)] = [movie, genre]

        metadata_fp = os.path.join(d, bn + '_metadata.tsv')
        genre_fp = open(os.path.join(d, bn + '_metadata_genres.tsv'), 'w')
        with open(metadata_fp, 'w') as f:
            for k, v in tqdm(itemmap.items()):
                # asin=k, index=v
                f.write('{} {}\n'.format(v, movies_dict[k][0])) # movie
                genre_fp.write('{} {}\n'.format(v, movies_dict[k][1])) #genre

    def preprocess_amazon_ratings(self):
        countU = defaultdict(lambda: 0)
        countP = defaultdict(lambda: 0)
        total = 8898041 if not self.limit else self.limit

        logging.info('Reading and processing {}'.format(self.path))
        f = open(self.dataset_fp, 'w')
        for l in tqdm(self.parse_movielens(), total=total):
            user, item, rating, timestamp = l.split(',')
            f.write('{} {} {}\n'.format(user, item, timestamp))

            asin = item
            rev = user
            countU[rev] += 1
            countP[asin] += 1
        f.close()

        logging.info('Creating user map dictionary')
        usermap = dict()
        usernum = 0
        itemmap = dict()
        itemnum = 0
        User = dict()

        for l in tqdm(self.parse_movielens(), total=total):
            user, item, rating, timestamp = l.split(',')
            asin = item
            rev = user
            time = timestamp

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
                f.write('%s %s %s\n' % (user, i[1], i[0]))
        f.close()

        # product map
        logging.info('Writing product item map')
        d = os.path.dirname(self.dataset_fp)
        bn = os.path.basename(self.dataset_fp)
        metadata_fp = os.path.join(d, bn[:-4] + '_product_map.txt')
        with open(metadata_fp, 'w') as f:
            for k, v in tqdm(itemmap.items()):
                f.write('{} {}\n'.format(k, v))

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
        for l in tqdm(self.parse_amazon(), total=total):
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