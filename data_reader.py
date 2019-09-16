import pandas as pd
import gzip
import json
from tqdm import tqdm
from data import Data

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

    def __init__(self, path, limit=None):
        self.path = path
        self.limit = limit
        # self.data = Data(self.convert_to_df())

    def parse(self):
        g = gzip.open(self.path, 'rb')
        for i, l in enumerate(g):
            if self.limit and i >= self.limit:
                break
            yield eval(l)

    def convert_to_df(self):
        i = 0
        df = {}
        for d in self.parse():
            df[i] = d
            i += 1
            if self.limit and i >= self.limit:
                break
        return pd.DataFrame.from_dict(df, orient='index')

    def preprocess(self, path):
        df = pd.DataFrame()
        d = {}
        i = 0
        for l in tqdm(self.parse(), total=self.limit):
            reviewer_id = l['reviewerID']
            tmp_pid = l['asin']
            review_time = l['unixReviewTime']
            if tmp_pid not in d:
                d[tmp_pid] = i
                i += 1
            
            product_id = d[tmp_pid]

            df = df.append(pd.Series({'reviewer_id': reviewer_id, \
                'product_id': product_id, \
                'orig_product_id': tmp_pid, \
                'review_time': review_time}), ignore_index=True)

        df = df.groupby('reviewer_id').apply(lambda x: x.sort_values(['review_time'], ascending=True)).reset_index(drop=True)
        df = df.astype({'reviewer_id': str, 'product_id': int, 'orig_product_id': str, 'review_time': int})
        df.to_csv(path)