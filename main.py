import sys
import os
import argparse
import logging
import time


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf


from datetime import datetime
import numpy as np
from tqdm import tqdm

from data_reader import DataReader
from model import Model
from sampler import WarpSampler
from util import *


if __name__ == '__main__':

    MODEL_PATH = os.path.abspath('models')
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
    parser.add_argument('--dataset', required=True, help='Location of pre-processed dataset')
    parser.add_argument('--limit', default=None, type=int, help='Limit the number of datapoints')
    parser.add_argument('--maxlen', default=50, type=int, help='Maximum length of user item sequence, for zero-padding')

    parser.add_argument('--train_dir', required=True)
    
    # TRAIN PARAMETERS
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=201, help='Number of epochs')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # MODEL PARAMETERS
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)

    # MISC.
    parser.add_argument('--saved_model', default='model.pt', type=str, help='File to save model checkpoints')
    # parser.add_argument('--device', default='cuda', type=str, help='Device to run model on') #TODO: GPU

    args = parser.parse_args()

    # Check if dataset exists
    if not os.path.exists(args.dataset):
        logger.info('Pre-process the data first using the --preprocess flag')
        sys.exit()

    # Check if training directory structure exists
    TRAIN_FILES_PATH = os.path.join(MODEL_PATH, os.path.basename(args.dataset), args.train_dir)
    if not os.path.exists(TRAIN_FILES_PATH):
        os.makedirs(TRAIN_FILES_PATH)


    # Partition data
    """
    NOTE: Important - the timestamps are sorted ASCENDING, 
    so train[-1] is the most recent product in the sequence
    """

    dataset = data_partition(args.dataset)
    [train, valid, test, usernum, itemnum] = dataset
    num_batch = round(len(train) / args.batch_size)

    cc = sum([len(v) for v in train.values()])        
    logging.info('Average sequence length: {:.2f}'.format(cc / len(train)))

    # DONE: Understand WarpSampler (see explanation in Class)
    print('usernum', usernum, 'itemnum', itemnum)
    sampler = WarpSampler(train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=1)

    # Understand partitioning of train / validation / test data:
    # first_user = list(train.keys())[0]
    # print('first user train data', train[first_user])
    # print('first user valid data', valid[first_user])
    # print('first user valid data', test[first_user])
    # # print(u)


    # RESET GRAPH
    tf.compat.v1.reset_default_graph()

    # DISABLE EAGER EXECUTION FOR NOW
    tf.compat.v1.disable_eager_execution()


    
    # CONFIGURATION
    f = open(os.path.join(TRAIN_FILES_PATH, 'log.txt'), 'w')
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True


    # SESSION
    sess = tf.compat.v1.Session(config=config)

    # MODEL
    model = Model(usernum, itemnum, args)
    sess.run(tf.compat.v1.global_variables_initializer())

    # Add TensorBoard
    writer = tf.compat.v1.summary.FileWriter(TRAIN_FILES_PATH, sess.graph) 

    # Allow saving of model 
    MODEL_SAVE_PATH = os.path.join(TRAIN_FILES_PATH, 'model.ckpt')  
    saver = tf.compat.v1.train.Saver()
    if os.path.exists(MODEL_SAVE_PATH):
        saver.restore(sess, MODEL_SAVE_PATH) 

    T = 0.0
    t0 = time.time()

    try:
        for epoch in range(1, args.num_epochs + 1):

            for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                u, seq, pos, neg = sampler.next_batch()
                # auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                #                         {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                #                          model.is_training: True})

                auc, loss, _, summary = sess.run([model.auc, model.loss, model.train_op, model.merged],
                                        {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                        model.is_training: True})

            writer.add_summary(summary, epoch)
            writer.flush()
            save_path = saver.save(sess, MODEL_SAVE_PATH)
            print('Model saved in path: %s' % save_path)

            if epoch % 20 == 0:
                print('Evaluating')
                t1 = time.time() - t0
                T += t1
                t_test = evaluate(model, dataset, args, sess)
                t_valid = evaluate_valid(model, dataset, args, sess)
                print('')
                print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
                epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

                f.write(str(t_valid) + ' ' + str(t_test) + '\n')
                f.flush()

                summary = tf.compat.v1.Summary()
                summary.value.add(tag='VALID/NDCG@10', simple_value=float(t_valid[0]))
                summary.value.add(tag='VALID/HR@10', simple_value=float(t_valid[1]))
                summary.value.add(tag='TEST/NDCG@10', simple_value=float(t_test[0]))
                summary.value.add(tag='TEST/HR@10', simple_value=float(t_test[1]))
                writer.add_summary(summary, epoch)
                t0 = time.time()
    except:
        sampler.close()
        f.close()
        exit(1)

    f.close()
    sampler.close()
    print("Done")


