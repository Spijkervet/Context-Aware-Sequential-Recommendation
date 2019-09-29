# Information Retrieval 2
This is the Github repository containing the code for the Context-Aware Sequential Recommendation project for the Information Retrieval 2 course at the University of Amsterdam

## Quick Start
The whole program, including initializing the environment, downloading the datasets and training the model, can be run with:
```
sh run.sh
```

# Manual Installation
Run the following commands to initialize the environment:
```
pip3 install virtualenv && \
python3 -m venv ir2 && \
source ir2/bin/activate && \
pip3 install -r requirements.txt
```

Activate the environment with: `source ir2/bin/activate`

## Download the data
You can download the datasets (Amazon Books, Movielens 1M) by invoking:
```
sh download_data.sh
```

## Preprocess the data
The data can be preprocessed with:

```
# AMAZON BOOKS
python3 preprocess.py --raw_dataset data/reviews_Books_5.json.gz --type amazon --dataset data/Books.txt

# MOVIELENS 1-M
python3 preprocess.py --raw_dataset data/ml-1m/ratings.dat --type movielens --dataset data/ml-1m.txt

```

## Run the program:
The program accepts dataset/train/model parameters. An example:
```
python3 main.py --dataset data/ml-1m.txt --train_dir maxlen_200_dropout_0.2 --maxlen=200 --dropout_rate=0.2
```

## Parameters:
```
usage: main.py [-h] --dataset DATASET [--limit LIMIT] [--maxlen MAXLEN]
               --train_dir TRAIN_DIR [--batch_size BATCH_SIZE] [--lr LR]
               [--num_epochs NUM_EPOCHS] [--max_norm MAX_NORM]
               [--hidden_units HIDDEN_UNITS] [--num_blocks NUM_BLOCKS]
               [--num_heads NUM_HEADS] [--dropout_rate DROPOUT_RATE]
               [--l2_emb L2_EMB] [--saved_model SAVED_MODEL]

optional arguments:
  -h, --help                  show this help message and exit
  --dataset DATASET           Location of pre-processed dataset
  --limit LIMIT               Limit the number of datapoints
  --maxlen MAXLEN             Maximum length of user item sequence, for zero-padding
  --train_dir TRAIN_DIR
  --batch_size BATCH_SIZE     Batch size
  --lr LR                     Learning rate
  --num_epochs NUM_EPOCHS     Number of epochs
  --max_norm MAX_NORM
  --hidden_units HIDDEN_UNITS
  --num_blocks NUM_BLOCKS
  --num_heads NUM_HEADS
  --dropout_rate DROPOUT_RATE
  --l2_emb L2_EMB
  --saved_model SAVED_MODEL   File to save model checkpoints
  ```
