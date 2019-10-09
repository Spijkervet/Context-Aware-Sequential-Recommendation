#!/bin/bash
#SBATCH --job-name=ir2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=30000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1


module purge
module load eb
module load pre2019

module load python/3.5.0
module load cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0
module load CUDA/8.0.44-GCCcore-5.4.0
module load anaconda

cd $HOME/time_lstm

#conda create --name time_lstm -y
#source deactivate time_lstm
#conda clean --lock
#conda env update --file environment.yml

source activate time_lstm

pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

THEANO_FLAGS='device=cuda0, gpuarray.preallocate=1, floatX=float32' python main.py --model TLSTM3 --data movielens