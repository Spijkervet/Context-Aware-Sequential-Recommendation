import sys
import os
import subprocess
import json
from pathlib import Path


# pathlist = Path('saved_models/ml-1m.txt').glob('*/')
for p in ['saved_models/ml-1m.txt/sasrec_baseline_10-19-2019-21-23-42']:
    for ms in [200]: #[10, 50, 60, 160]:
        print('Testing model {} at max_seq_len {}'.format(p, ms))

        with open(os.path.join(p, 'params.txt'), 'r') as f:
            params = json.load(f)

        cmd = ['python3', 'main.py']

        for k, v in params.items():
            cmd.append('--{}={}'.format(k,v))

        cmd.append('--test_model={}'.format(p))
        cmd.append('--test_seq_len={}'.format(ms))
        
        subprocess.call(cmd)