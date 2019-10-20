import sys
import os
import subprocess
import json
from pathlib import Path


pathlist = Path('saved_models/ml-1m.txt').glob('*/')
for p in pathlist:
    for ms in [200]: # [10, 50, 60, 160]:
        print('Testing model {}'.format(p))

        with open(os.path.join(p, 'params.txt'), 'r') as f:
            params = json.load(f)

        cmd = ['python3', 'main.py']

        for k, v in params.items():
            if k in ['test_model', 'test_seq_len']:
                continue
            cmd.append('--{}={}'.format(k,v))

        cmd.append('--test_model={}'.format(p))
        cmd.append('--test_seq_len={}'.format(ms))

        subprocess.call(cmd)