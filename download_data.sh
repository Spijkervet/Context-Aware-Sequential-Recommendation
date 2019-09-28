cd data

# Amazon Product Review data
wget -nc http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz
# cd data && curl http://snap.stanford.edu/data/amazon/productGraph/metadata.json.gz -O

# Movielens 1M
wget -nc http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
rm ml-1m.zip