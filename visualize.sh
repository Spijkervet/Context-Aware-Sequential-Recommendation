#!/bin/bash

module purge
module load 2019
module load eb

module load Python/3.6.6-foss-2018b
module load CUDA/10.0.130
module load cuDNN/7.6.3-CUDA-10.0.130

export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH

# pip3 install tensorflow --user

. ir2/bin/activate

python3 -m tensorboard.main --logdir $1 --port $2
