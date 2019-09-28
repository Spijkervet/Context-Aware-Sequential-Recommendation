#!/bin/bash
# SASRec, from https://github.com/kang205/SASRec.git 

#SBATCH --job-name=sasrec
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=12000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

module purge
module load eb
module load python/2.7.9
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176
export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH

# Mail
echo "[SASRec] Job $SLURM_JOBID started at `date`" | mail $USER -s "Job $SLURM_JOBID"

# Program
pip2 install virtualenv 
python2 -m virtualenv sasrec

. sasrec/bin/activate

pip2 install -r requirements.txt

python2 main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 
# python2 main.py --dataset=Books --train_dir=default --maxlen=50 --dropout_rate=0.2 

echo "[SASRec] Job $SLURM_JOBID finished at `date`" | mail $USER -s "Job $SLURM_JOBID"