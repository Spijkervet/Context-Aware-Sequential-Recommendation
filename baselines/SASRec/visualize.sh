#!/bin/bash

module purge
module load eb
module load python/3.4.2
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176
export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH

# pip3 install tensorflow --user

python3 -m tensorboard.main --logdir ml-1m_default 
