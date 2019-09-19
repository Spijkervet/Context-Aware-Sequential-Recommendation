# information_retrieval_2
IR2 course UvA

## Installation
Run the following command to initialize the environment:
```
pip3 install virtualenv
python3 -m venv ir2
source ir2/bin/activate
pip3 install -r requirements.txt
```

## Download the data
You can download the data by invoking:
```
sh download_data.sh
```

## Run the program:
The program accepts dataset/train/model parameters. An example:
```
python3 main.py --raw_dataset data/reviews_Books_5.json.gz --dataset data/Books.txt --preprocess --train_dir data/train --batch_size 128
```