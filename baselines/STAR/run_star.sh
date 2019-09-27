pip3 install virtualenv 
python3 -m virtualenv star

. star/bin/activate

cd STAR 
pip3 install -r requirements.txt
python3 main.py cpu miniData STAR