cd data

# Amazon Product Review data
# wget -nc http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz
# curl http://snap.stanford.edu/data/amazon/productGraph/metadata.json.gz -O

wget -nc http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty.json.gz


# Movielens 1M
wget -nc http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip -o ml-1m.zip
rm ml-1m.zip
