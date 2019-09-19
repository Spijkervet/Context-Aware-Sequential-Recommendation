import sys
import os
import argparse
import logging
import time
from datetime import datetime
import numpy as np


# Deep Learning lib
import torch
from torch.utils.data import DataLoader

# Model-specific imports
from data_reader import DataReader
from util import data_partition
from data import AmazonDataset
from model import DummyModel

def load_dataset(config, data):
    # Initialize the dataset and data loader (note the +1)
    dataset = AmazonDataset(data, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    return dataset, data_loader

def create_model(config, dataset):
    # Initialize the model that we are going to use
    model = DummyModel(
        batch_size=config.batch_size,
        seq_length=config.seq_length,
        vocabulary_size=dataset.vocab_size,
        # lstm_num_hidden=config.lstm_num_hidden,
        # lstm_num_layers=config.lstm_num_layers,
        dropout=(1-config.dropout_keep_prob),
        device=config.device
    )

    if os.path.isfile(config.saved_model):
        print("### LOADING MODEL ###")
        model.load_state_dict(torch.load(config.saved_model, map_location=config.device))

    model.to(config.device)
    return model


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
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the raw dataset')
    parser.add_argument('--limit', default=None, type=int, help='Limit the number of datapoints')
    parser.add_argument('--train_dir', required=True)

    # TRAIN PARAMETERS
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--train_steps', default=100, type=int, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # MODEL PARAMETERS
    parser.add_argument('--seq_length', default=3, type=int, help='Sequence length')
    parser.add_argument('--dropout_keep_prob', default=1.0, type=float, help='Dropout')

    # MISC.
    parser.add_argument('--saved_model', default='model.pt', type=str, help='File to save model checkpoints')
    parser.add_argument('--device', default='cpu', type=str, help='Device to run model on') #TODO: GPU

    config = parser.parse_args()

    # Check if dataset exists
    if not os.path.exists(config.dataset):
        logger.info('Pre-process the data first using the --preprocess flag')
        sys.exit()

    # Check if training directory structure exists
    if not os.path.exists(config.train_dir):
        os.makedirs(config.train_dir)
  
    # Start the data reader and read the dataset
    dr = DataReader(config.raw_dataset, config.dataset, limit=config.limit)
    if config.preprocess:
        dr.preprocess()

    # Partition data
    dataset = data_partition(config.dataset)
    [train, valid, test, usernum, itemnum] = dataset
    num_batch = len(train) / config.batch_size

    cc = 0.0
    for k, v in train.items():
        cc += len(v)
    logging.info('Average sequence length: {:.2f}'.format(cc / len(train)))

    dataset, data_loader = load_dataset(config, train)
    
    model = create_model(config, dataset)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Time measurement
        t1 = time.time()

        """
        In PyTorch, we need to set the gradients to zero before starting to do backpropragation because 
        PyTorch accumulates the gradients on subsequent backward passes. https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        """
        optimizer.zero_grad()

        # Convert to Tensor
        batch_inputs = torch.stack(batch_inputs).to(config.device)
        batch_targets = torch.stack(batch_targets).to(config.device)
        out = model(batch_inputs)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

        # Calculate loss, perform backpropagation and evaluate (calculate accuracy)
        loss = criterion(out.permute(0,2,1), batch_targets)
        loss.backward()
        optimizer.step()

        accuracy = (out.argmax(2) == batch_targets).float().mean()

        # Time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)
        
        print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Batch/Sec = {:.2f}, "
                        "Accuracy = {:.2f}, Loss = {:.3f}".format(
                            datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                            config.train_steps, config.batch_size, examples_per_second,
                            accuracy, loss
        ))

        # Exit training
        if step == config.train_steps:
            break

