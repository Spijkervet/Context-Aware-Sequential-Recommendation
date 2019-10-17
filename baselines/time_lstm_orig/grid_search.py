import subprocess

for vocab in [200, 300, 400, 500, 600, 1000, 1500, 2000, 5000]:
    for lr in [0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3]:
        print('grid_search', vocab, lr)
        subprocess.call('sh train.sh {} {}'.format(vocab, lr), shell=True)
