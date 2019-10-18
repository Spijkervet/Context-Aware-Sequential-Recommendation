import sys
import os
import subprocess
import json
from pathlib import Path


pathlist = Path('saved_models/').glob('*/*/')
for p in pathlist:
    print('Testing model {}'.format(p))

    with open(os.path.join(p, 'params.txt'), 'r') as f:
        params = json.load(f)

    cmd = ['python3', 'main.py']

    for k, v in params.items():
        cmd.append('--{}={}'.format(k,v))

    cmd.append('--test_model={}'.format(p))
    cmd.append('--test_seq_len={}'.format(10))

    subprocess.call(cmd)