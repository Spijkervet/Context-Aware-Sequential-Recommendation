import os

def log_results(result_dir, current_epoch, recall10, MRR10_score, ndcg, avg_rate, loss, name):

    p = os.path.join(result_dir, 'results.txt')
    if not os.path.exists(p):
        with open(p, 'w') as f:
            f.write('epoch,recall10,mrr10,ndcg,avg_rate,loss,name\n')

    with open(p, 'a+') as f:
        f.write('{},{},{},{},{},{},{}\n'.format(current_epoch, recall10, MRR10_score, ndcg, avg_rate, loss, name))
    
    print(current_epoch, recall10, MRR10_score, ndcg, avg_rate, loss, name)