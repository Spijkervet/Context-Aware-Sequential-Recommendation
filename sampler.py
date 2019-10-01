import numpy as np
from multiprocessing import Process, Queue


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    # pick a product id that is NOT in the set of unique product ids of this sequence
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        # Get a random user_id, make sure it has more than x interactions (which we already checked?):
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: 
            user = np.random.randint(1, usernum + 1)

        # Create sequence / pos / negative with zero padding :maxlen:
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1].item
        timeseq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        
        # print('sequence', user_train[user])
        # print('get most recent product in sequence', nxt)
        # Sequence shape has maxlen zero padding
        assert seq.shape[0] == maxlen

        # Get unique product ids in sequence
        ts = set([i.item for i in user_train[user]])
        # NOTE: Reverse sequence (ascending -> descending), except for the last interaction
        for i in reversed(user_train[user][:-1]):
            # print('idx', idx, 'i', i, 'nxt', nxt)
            seq[idx] = i.item
            timeseq[idx] = i.delta_time # NOTE: CONTEXT SCOPE IS CHANGED HERE
            pos[idx] = nxt
            if nxt != 0: # TODO: What does nxt != 0 mean?
                # print('nxt', nxt)
                # Pick a random product id between 1 and :itemnum: NOT in the set of unique product ids of this sequence
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i.item # nxt becomes i
            idx -= 1
            if idx == -1: break

        # print('user_id', user)
        # print('sequence', seq)
        # print('positive examples (incl. recent)', pos)
        # print('negative examples', neg)
        return (user, seq, pos, neg, timeseq)

    np.random.seed(SEED)
    # TODO: I have no idea what this while loop does?
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())
        result_queue.put(zip(*one_batch))

class WarpSampler(object):
    """
    (???)
    To avoid heavy computation on all user-item pairs, we
    followed the strategy in [14], [48]. For each user u, we
    randomly sample 100 negative items, and rank these items
    with the ground-truth item.
    """

    """
    JANNE:
    This class implements a parallel (multiprocessing, it's confusing) batch loader, given user interaction data,
    the number of users and number of  items, it builds batches of size :batch_size:.
    The batches consist of four tuples of size :batch_size: rows with :maxlen: columns, which are zero-padded vectors.
    These vectors are: seq, pos and neg. The userid is also returned.
    The seq vector has :maxlen: items, filled at the end with the most recent product ids (except the most recent).
    The pos vector has :maxlen: items, filled at the end with the most recent product ids (including the most recent, excluding oldest).
    The neg vector has :maxlen: items, filled at the end with 'negative' samples, i.e. randomly drawn product ids that do not exist in the current interaction data.
    """
    def __init__(self, args, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []

        if args.seed:
            seed = args.seed
        else:
            seed = np.random.randint(2e9)
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      seed
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
