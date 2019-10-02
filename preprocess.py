import logging
import argparse

from data_reader import DataReader

def main(raw_dataset, out_dataset, dataset_type, limit, maxlen):
    # Start the data reader and read the dataset
    dr = DataReader(raw_dataset, out_dataset, dataset_type, limit=limit, maxlen=maxlen)
    dr.preprocess()

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
    parser.add_argument('--dataset', required=True, help='Location of pre-processed dataset')
    parser.add_argument('--type', required=True, type=str, help='Dataset type (amazon, movielens, amazon_ratings)')
    parser.add_argument('--limit', default=None, type=int, help='Limit the number of datapoints')
    parser.add_argument('--maxlen', default=50, type=int, help='Maximum length of user item sequence, for zero-padding')
    args = parser.parse_args()

    main(args.raw_dataset, args.dataset, args.type, args.limit, args.maxlen)