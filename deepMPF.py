'''
This is the deep MPF. We implement the MPF-EM algorithm here for training RBMs.
Step 1: Binarize the input data, initialize weight and bias
Step 2: Feedforward and sampling to get the hidden activations
Step 3: Train the MPF to find the better weights and bias
Step 4: repeat Step 2 and 3 until weight converge.
Step 5: Sanity check, conduct generating tasks
'''

import numpy as np
import gzip
import timeit, pickle, sys, math
from sklearn import preprocessing
import theano
import theano.tensor as T
import os


def get_mpf_params(visible_units, hidden_units):

    '''
    :param visible_units: number of units in the visible layer
    :param hidden_units: number of units ni the hidden layer
    :return: The well structured MPF weight matrix
    The MPF weight matrix is of the form:
    [0,   W,
     W.T, 0]
    '''
    numpy_rng = np.random.RandomState(3333)

    W = numpy_rng.randn(visible_units, hidden_units) / np.sqrt(visible_units + hidden_units) / 100

    W_up = np.concatenate((np.zeros((visible_units,visible_units)), W), axis = 1)

    W_down = np.concatenate((W.T,np.zeros(hidden_units,hidden_units)), axis = 1 )

    W = np.concatenate((W_up,W_down), axis =1)

    print(W.shape)

    return W



def mpf_em(dataset,hidden_units,dynamic=False):


    ################################################################
    ################## Loading the Data        #####################
    ################################################################

    if dataset is not None:
        dataset = 'mnist.pkl.gz'
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
        f.close()

    else:
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = pickle.load(dataset,encoding="bytes")
        f.close()


    binarizer = preprocessing.Binarizer(threshold=0.5)
    data =  binarizer.transform(train_set[0])

    ################################################################
    ##################  Initialize Parameters  #####################
    ################################################################

    visible_units = train_set[0].shape[1]

    num_units = visible_units + hidden_units

    W = get_mpf_params(visible_units,hidden_units)

    # bias can be intialized in the MPF_optimizer class

    ################################################################
    ##################  Train MPF              #####################
    ################################################################

    # In one round, we feedfoward the and get the samples,
    # Compute the probability of each data samples,
    # call the minimum probability flow objective function













