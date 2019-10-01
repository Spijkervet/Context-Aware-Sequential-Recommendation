#!/bin/bash
#SBATCH --job-name=ir2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:05:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:2


module purge
module load eb

module load python/3.5.0
module load cudnn
module load CUDA/8.0.44-GCCcore-5.4.0

cd $HOME/time_lstm

pip install virtualenv
python3 -m virtualenv time_lstm
. time_lstm/bin/activate

pip3 install -v theano==0.9.0
pip3 install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
pip3 install -v pandas==0.18.1

THEANO_FLAGS='floatX=float32' python3 main.py --model TLSTM3 --data amazon