from data_reader import DataReader
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    limit = 5000
    reviews_books = DataReader('./data/reviews_Books_5.json.gz', limit=limit)
    reviews_books.preprocess('./data/preprocessed_books_data.csv')