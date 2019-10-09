import unittest

from preprocess import main as preprocess
import util


class CAST(unittest.TestCase):

    def test_preprocess_data(self):
        """
        Test whether the MovieLens-1m dataset is read properly
        """
        preprocess('data/ml-1m/ratings.dat',
                          'data/ml-1m.txt', 'movielens', limit=None, maxlen=200)
        with open('data/ml-1m.txt', 'r') as f:
            for l in f:
                user, item, timestamp = tuple(map(int, l.rstrip().split(' ')))
                self.assertEqual(user, 1)
                self.assertEqual(item, 32)
                self.assertEqual(timestamp, 978300019)
                break

    def test_data_partition(self):
        dataset_path = 'data/ml-1m.txt'

        dataset = util.data_partition(dataset_path, log_scale=False)
        [train, _, _, usernum, itemnum] = dataset

        cc = sum([len(v) for v in train.values()])
        avg_seq_len = cc / len(train)

        self.assertEqual(round(avg_seq_len, 1), 163.5)
        self.assertEqual(usernum, 6040)
        self.assertEqual(itemnum, 3416)

    def test_delta_time(self):
        """
        Test the calcuation of addition of the time bins in the user item sequence
        """

        dataset_path = 'data/ml-1m.txt'
        User, usernum, itemnum = util.get_users(dataset_path)
        self.assertEqual(usernum, 6040)
        self.assertEqual(itemnum, 3416)

        # Test log scale
        min_timedelta, max_timedelta = util.get_delta_range(
            User, max_percentile=90)
        self.assertEqual(min_timedelta, 0)
        self.assertEqual(max_timedelta, 58896613)
        
        # Test delta_time
        User = util.add_time_bin(User, log_scale=True)
        self.assertEqual(User[1][0].time_bin, 147)

        User = util.add_time_bin(User, log_scale=False)
        self.assertEqual(User[1][0].time_bin, 3)

if __name__ == '__main__':
    unittest.main()