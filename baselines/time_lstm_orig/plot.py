import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd



if __name__ == '__main__':
    # args = argparse.ArgumentParser()
    # # args.add_argument('--file', required=True, type=str, help='The results file located in ./log')
    # args = args.parse_args()

    if not os.path.exists('plots'):
        os.makedirs('plots')

    rootDir = 'log'
    for d, _, files in os.walk(rootDir):
        for f in files:
            if f == 'results.txt':
                f = os.path.join(d, f)
                with open(f, 'r'):
                    df = pd.read_csv(f)

                    if 'name' in df.columns:
                        df = df[df['name'] != 'additional_test']


                        out_dir = os.path.join('plots', os.path.dirname(f).split('/')[1])
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                            
                        plt.figure()
                        plt.plot(df['epoch'], df['recall10'])
                        plt.title('recalll10')
                        plt.xlabel('epoch')
                        plt.ylabel('recall10')
                        plt.savefig(os.path.join(out_dir, 'recall10.png'))

                        plt.figure()
                        plt.plot(df['epoch'], df['mrr10'])
                        plt.title('mrr10')
                        plt.xlabel('epoch')
                        plt.ylabel('mrr10')
                        plt.savefig(os.path.join(out_dir, 'mrr10.png'))