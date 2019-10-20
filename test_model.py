import sys
import os
import subprocess
import json
from pathlib import Path


pathlist = Path('saved_models/ml-1m.txt').glob('*/')
for p in pathlist:
    # if 'cast_1' in str(p):
    #     continue

    for ms in [10, 50, 60, 160]:
        print('Testing model {}'.format(p))

        with open(os.path.join(p, 'params.txt'), 'r') as f:
            params = json.load(f)

        cmd = ['python3', 'main.py']

        for k, v in params.items():
            cmd.append('--{}={}'.format(k,v))

        cmd.append('--test_model={}'.format(p))
        cmd.append('--test_seq_len={}'.format(ms))

        subprocess.call(cmd)