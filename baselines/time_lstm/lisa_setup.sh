module purge
module load pre2019
module load Anaconda2

#rm -rf ~/.conda/envs/time_lstm
conda create --name time_lstm -y
source activate time_lstm
conda install theano=0.9
pip install --user --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
echo "now start the job by calling: sbatch timelstm3.job"