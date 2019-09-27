# information_retrieval_2
IR2 course UvA

## Pulling this repository
You can either pull the main repository or with all submodules to run the baselines:
```
git pull
```
or
```
git pull --recurse-submodules
```

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
python3 main.py --raw_dataset data/reviews_Books_5.json.gz --dataset data/Books.txt --preprocess --batch_size 128
```

## Parameters:
```
usage: main.py [-h] [--raw_dataset RAW_DATASET] --dataset DATASET
               [--preprocess] [--limit LIMIT] --train_dir TRAIN_DIR
               [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
               [--train_steps TRAIN_STEPS] [--max_norm MAX_NORM]
               [--seq_length SEQ_LENGTH]
               [--dropout_keep_prob DROPOUT_KEEP_PROB]
               [--saved_model SAVED_MODEL] [--device DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  --raw_dataset RAW_DATASET
                        Raw gzip dataset, Amazon Product Review data
  --dataset DATASET     Location of pre-processed dataset
  --preprocess          Preprocess the raw dataset
  --limit LIMIT         Limit the number of datapoints
  --train_dir TRAIN_DIR
  --batch_size BATCH_SIZE
                        Batch size
  --learning_rate LEARNING_RATE
                        Learning rate
  --train_steps TRAIN_STEPS
                        Number of training steps
  --max_norm MAX_NORM   --
  --seq_length SEQ_LENGTH
                        Sequence length
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout
  --saved_model SAVED_MODEL
                        File to save model checkpoints
  --device DEVICE       Device to run model on
  ```
