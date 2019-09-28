#!/bin/sh

#SBATCH --job-name=sasrec
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1


module purge
module load eb

module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176
export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH


pip3 install -r requirements.txt --user

sh download_data.sh

python3 preprocess.py --raw_dataset data/reviews_Books_5.json.gz --type amazon --dataset data/Books2.txt
python3 main.py --type amazon --raw_dataset data/reviews_Books_5.json.gz --dataset data/Books.txt --batch_size 128