#!/bin/bash
#SBATCH -t 0:05:00
#SBATCH -N 1
#SBATCH -p gpu_short
#SBATCH --mem=60G

cd $HOME/time_lstm

module load python/3.5.0
module load cudnn
module load CUDA/8.0.44-GCCcore-5.4.0

pip install virtualenv
python3 -m virtualenv time_lstm
. time_lstm/bin/activate

pip install -v theano==0.9.0
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
pip install -v pandas==0.18.1

THEANO_FLAGS='floatX=float32' python main.py --model TLSTM3 --data amazon