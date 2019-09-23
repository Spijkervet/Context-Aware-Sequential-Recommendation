# SASRec, from https://github.com/kang205/SASRec.git 

git clone https://github.com/kang205/SASRec.git

pip2 install virtualenv 
python2 -m virtualenv sasrec

. sasrec/bin/activate

pip2 install -r sasrec_requirements.txt

cd SASRec
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 