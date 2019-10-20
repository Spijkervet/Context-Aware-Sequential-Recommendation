import seaborn as sns
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import numpy as np

def main(filename='dummydata_heatmap.csv'):

    '''
    Creating heatmap plots from attention vectors.
    '''

    data = defaultdict(list)

    # get data from file
    with open(filename, 'r') as file:

        reader = csv.reader(file)

        for line in reader:

            # group data by sequence length
            data[len(line)].append([float(x) for x in line])

    maxlen = max(data.keys())

    # average vectors at each sequence length
    averaged_vectors = [average(data[k]) for k in sorted(data, reverse=True)]

    # append 0 to those vectors not of maximum length and store as np array
    averaged_arr = np.array([[0]*(maxlen-len(vec)) + vec for vec in averaged_vectors])
    

    # moet vec hier nog omgedraaid worden????
    sns.heatmap(averaged_arr.T, linewidth=0.5, square=True, cmap="YlGnBu")
    plt.xlabel("Position")
    plt.ylabel("Time step")
    plt.title("Attention weights over time steps")
    plt.show()
    plt.savefig('test.png')

def average(data_list):

    '''
    Average list of lists along first dimension.
    '''

    arr = np.array(data_list)
    return list(arr.mean(0))







if __name__ == '__main__':
    main()
