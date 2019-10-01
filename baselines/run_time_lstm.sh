#!/bin/bash
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=janerikvw2006@gmail.com

cd time_lstm

pip3 install virtualenv
python3 -m virtualenv time_lstm

. time_lstm/bin/activate

pip3 install -r requirements.txt

pip install -v theano==0.9.0
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
pip install -v pandas==0.18.1


THEANO_FLAGS='floatX=float32' python main.py --model TLSTM3 --data amazon
cd ../