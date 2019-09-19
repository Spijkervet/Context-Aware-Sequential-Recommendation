import sys
import os
import argparse
from data_reader import DataReader

import logging
from util import data_partition

from torch.utils import data


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler("{0}/{1}.log".format('.', 'output')),
        logging.StreamHandler()
])

if __name__ == '__main__':
    logger = logging.getLogger('ir2')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dataset')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--limit', default=None, type=int)
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=50, type=int)

    args = parser.parse_args()

    # Check if dataset exists
    if not os.path.exists(args.dataset):
        logger.info('Pre-process the data first using the --preprocess flag')
        sys.exit()

    # Check if training directory structure exists
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
  
    # Start the data reader and read the dataset
    dr = DataReader(args.raw_dataset, args.dataset, limit=args.limit)
    if args.preprocess:
        dr.preprocess()

    # Partition data
    dataset = data_partition(args.dataset)
    [train, valid, test, usernum, itemnum] = dataset
    num_batch = len(train) / args.batch_size

    cc = 0.0
    for k, v in train.items():
        cc += len(v)
    logging.info('Average sequence length: {:.2f}'.format(cc / len(train)))

    # TODO: Create PyTorch Dataloader, warp sampling(?)
    # training_generator = data.DataLoader(train)
    # for epoch in range(args.epochs):
    #     # Training
    #     for local_batch in training_generator:
    #         print(local_batch)