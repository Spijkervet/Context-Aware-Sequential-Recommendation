import logging
import argparse

from data_reader import DataReader

if __name__ == '__main__':
    logger = logging.getLogger('ir2')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("{0}/{1}.log".format('.', 'output')),
            logging.StreamHandler()
    ])
    
    parser = argparse.ArgumentParser()

    # DATASET PARAMETERS
    parser.add_argument('--raw_dataset', help='Raw gzip dataset, Amazon Product Review data')
    parser.add_argument('--type', required=True, type=str, help='Dataset type (amazon, movielens, amazon_ratings)')
    parser.add_argument('--dataset', required=True, help='Location of pre-processed dataset')
    parser.add_argument('--limit', default=None, type=int, help='Limit the number of datapoints')
    config = parser.parse_args()

      
    # Start the data reader and read the dataset
    dr = DataReader(config.raw_dataset, config.dataset, config.type, limit=config.limit)
    dr.preprocess()

