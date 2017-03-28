'''
This file implement the deep rbm model trained with deep MPF.
Optimization method: Adam
'''

from np_sgd_mpf import *

def train_deep_rbm(n_rbm,):

    epsilon = 0.01
    n_samples = 5
    learning_rate = 0.001
    epoches = 1000
    beta_list = [3]
    sparsity_list = [0.1]
    batch_list  = [20]

