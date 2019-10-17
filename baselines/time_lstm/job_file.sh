#!/bin/bash
#SBATCH --job-name=ir2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1


module purge
module load pre2019

module load python/2.7.9
#module load python/3.5.0
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

#THEANO_FLAGS='device=cuda0, gpuarray.preallocate=1, floatX=float32' python main.py --model TLSTM3 --data music --sigmoid_on --bn --grad_clip=1
BATCH=5
TEST_BATCH=5
VOCAB=3000 # 400 #$1
MLEN=200
DATA='music'
# DATA='movielens'
FIXED_EPOCHS=10
NUM_EPOCHS=50
NHIDDEN=128
PRETRAINED=""
LAST_EPOCH=9
SAMPLE_TIME=3
LEARNING_RATE=0.01 #$2 # 0.05
GRAD_CLIP=1
BATCH_NORM=1
SIGMOID=1

FLAGS="floatX=float32,device=cuda"
THEANO_FLAGS="${FLAGS}" python main.py --model TLSTM3 --data ${DATA} \
    --batch_size ${BATCH} --vocab_size ${VOCAB} --max_len ${MLEN} \
    --fixed_epochs ${FIXED_EPOCHS} --num_epochs ${NUM_EPOCHS} \
    --num_hidden ${NHIDDEN} --test_batch ${TEST_BATCH} \
    --learning_rate ${LEARNING_RATE} --sample_time ${SAMPLE_TIME} \
    --grad_clip ${GRAD_CLIP} --bn ${BATCH_NORM} --sigmoid_on ${SIGMOID}

