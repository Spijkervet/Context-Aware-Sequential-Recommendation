mkdir log
mkdir data
cd data
mkdir music
wget http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz
tar -xvf lastfm-dataset-1K.tar.gz -C music --strip-components=1

cd music

tail -192 user-item.lst > te_user-item.lst
tail -192 user-item-delta-time.lst > te_user-item-delta-time.lst
tail -192 user-item-accumulate-time.lst > te_user-item-accumulate-time.lst
head -800 user-item.lst > tr_user-item.lst
head -800 user-item-delta-time.lst > tr_user-item-delta-time.lst
head -800 user-item-accumulate-time.lst > tr_user-item-accumulate-time.lst