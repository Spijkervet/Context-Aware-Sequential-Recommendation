import argparse
import subprocess

parser = argparse.ArgumentParser()

# DATASET PARAMETERS
parser.add_argument('--dataset',
                help='Location of pre-processed dataset')
parser.add_argument('--limit', default=None, type=int,
                help='Limit the number of datapoints')
parser.add_argument('--maxlen', default=50, type=int,
                help='Maximum length of user item sequence, for zero-padding')

parser.add_argument('--train_dir') 

# TRAIN PARAMETERS
parser.add_argument('--batch_size', default=128,
                type=int, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--num_epochs', type=int,
                default=201, help='Number of epochs')
parser.add_argument('--max_norm', type=float, default=5.0, help='--')

# MODEL PARAMETERS
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--bin_in_hours', default=24, type=int)
parser.add_argument('--max_bins', default=200, type=int)

# MISC.
parser.add_argument('--saved_model', default='model.pt',
                type=str, help='File to save model checkpoints')
parser.add_argument('--test_baseline', default=False, action='store_true')
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--log_scale', default=False, action='store_true')
parser.add_argument('--input_context', default=False, action='store_true')
parser.add_argument('--model', default="cast", 
                help="model to use from {cast, sasrec, castsp}")
# parser.add_argument('--device', default='cuda', type=str, help='Device to run model on') #TODO: GPU

args = parser.parse_args()


# Experiment

args.dataset = 'data/ml-1m.txt'
# args.dataset = 'data/Beauty.txt'
for i in range(4, 8):
    model = 'cast_{}'.format(i)
    args.train_dir = model # + '_log'
    args.model = model

    try:
        # movielens params
        c = ['python3', 'main.py', '--dataset', args.dataset, 
            '--train_dir', args.train_dir, '--model', args.model,
            '--maxlen', '200', '--bin_in_hours', '48', '--dropout_rate', '0.2', 
            '--num_blocks', '2', '--seed', '42']

        # beauty params 
        # c = ['python3', 'main.py', '--dataset', args.dataset, 
        #     '--train_dir', args.train_dir, '--model', args.model,
        #     '--maxlen', '50', '--bin_in_hours', '48', '--dropout_rate', '0.5', 
        #     '--num_blocks', '2', '--seed', '42']#, '--log_scale']

        subprocess.call(c)
    except Exception as e:
        with open("experiment_errors.txt", "w") as f:
            print(e)
            f.write(str(e) + '\n')
        pass
