#!/bin/sh

#SBATCH --job-name=ir2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=12000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:2


module purge
module load 2019

module load eb
module load Python/3.6.6-foss-2018b
module load CUDA/10.0.130
module load cuDNN/7.6.3-CUDA-10.0.130

# Mail
mail_template "IR2" $SLURM_JOBID "STARTED" "$1"

pip3 install virtualenv 
python3 -m virtualenv ir2

. ir2/bin/activate

pip3 install -r requirements.txt

### DOWNLOAD DATA ###
# sh download_data.sh

### PREPROCESSING ###
# AMAZON BOOKS
# python3 preprocess.py --raw_dataset data/reviews_Books_5.json.gz --type amazon --output data/Books.txt

# AMAZON BEAUTY
# python3 preprocess.py --raw_dataset data/reviews_Beauty.json.gz --type amazon --output data/Beauty.txt

# MOVIELENS 1-M
# python3 preprocess.py --raw_dataset data/ml-1m/ratings.dat --type movielens --output data/ml-1m.txt #--limit 100000

# MOVIELENS 20-M
python3 preprocess.py --raw_dataset data/ml-20m/ratings.csv --type movielens --output data/ml-20m.txt #--limit 100000

### PROGRAM ###
python3 run_experiments.py
mail_template "IR2" $SLURM_JOBID "FINISHED" "$1"
