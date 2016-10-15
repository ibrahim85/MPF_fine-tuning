'''
This is the deep MPF. We implement the MPF-EM algorithm here for training RBMs.
Step 1: Binarize the input data, initialize weight and bias
Step 2: Feedforward and sampling to get the hidden activations
Step 3: Train the MPF to find the better weights and bias
Step 4: repeat Step 2 and 3 until weight converge.
Step 5: Sanity check, conduct generating tasks
'''


from dmpf_optimizer import *
from sklearn import preprocessing
import numpy as np
import gzip
import timeit, pickle, sys, math
import theano
import theano.tensor as T
import Image
import copy
import os

from utils import tile_raster_images



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

    W_down = np.concatenate((W.T,np.zeros((hidden_units,hidden_units))), axis = 1 )

    W = np.concatenate((W_up,W_down), axis = 0)

    print(W.shape)

    return W



def mpf_em(dataset,hidden_units,dynamic=False):


    ################################################################
    ################## Loading the Data        #####################
    ################################################################

    if dataset is None:
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
    # k = data[:1,:]
    # print(k)
    # print(train_set[0][:1,:])

    #data = np.load('gibbs_samples.npy')


    ################################################################
    ##################  Initialize Parameters  #####################
    ################################################################

    #visible_units = train_set[0].shape[1]
    visible_units = data.shape[1]

    num_units = visible_units + hidden_units

    W_init = get_mpf_params(visible_units,hidden_units)
    W = copy.deepcopy(W_init)

    # np.save('rbm_W.npy',W)
    # bias can be intialized in the MPF_optimizer class

    ################################################################
    ##################  Train MPF              #####################
    ################################################################

    # In one round, we feedfoward the and get the samples,
    # Compute the probability of each data samples,
    # call the minimum probability flow objective function

    epsilon = 0.01
    learning_rate = 0.01
    connect_function = '1-bit-flip'
    index = T.lscalar()    # index to a mini batch
    x = T.matrix('x')
    batch_sz = 40

    mpf_optimizer = dmpf_optimizer(epsilon=epsilon, num_units = num_units, W = W, b = None,
                 input = x,batch_sz =batch_sz)


    n_train_batches = data.shape[0]//batch_sz

    data  = theano.shared(value=np.asarray(data, dtype=theano.config.floatX),
                                  name = 'train',borrow = True)

    cost,updates = mpf_optimizer.get_dmpf_cost(learning_rate= learning_rate,
                                               visible_units=visible_units,hidden_units=hidden_units)

    train_mpf = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: data[index * batch_sz: (index + 1) * batch_sz],
        },
        #on_unused_input='warn',
    )

    training_epochs = 8000

    start_time = timeit.default_timer()

    mean_epoch_error = []

    for epoch in range(training_epochs):

        # go through the training set
        mean_cost = []
        mean_batch_error = []
        norm_batch_error = []


        for batch_index in range(n_train_batches):
            weight = mpf_optimizer.W.get_value(borrow = True)
            bia = mpf_optimizer.b.get_value(borrow = True)
            mean_cost += [train_mpf(batch_index)]
            W2 = np.load('rbm_W.npy')

            # print(mean_cost)
            error_W = np.sum((W_init - mpf_optimizer.W.get_value(borrow=True))**2)
            error_bias = np.sum((bia - mpf_optimizer.b.get_value(borrow= True))**2)

            #print(error_bias)

            error = error_bias + error_W

            mean_batch_error += [error_W/(2*hidden_units*visible_units)]

            # norm_batch_error += [np.sum(( W/(np.sum(W**2)) -
            # mpf_optimizer.W.get_value(borrow=True)/(np.sum(mpf_optimizer.W.get_value(borrow=True)**2)) )**2 )]
        # image = Image.fromarray(
        #     tile_raster_images(
        #         X=W.T,
        #         img_shape=(28, 28),
        #         tile_shape=(10, 10),
        #         tile_spacing=(1, 1)
        #     )
        # )
        #
        # image.save('filters.png')
        mean_epoch_error += [np.mean((mean_batch_error))]

        print(mean_epoch_error[-1])

        if epoch > 2 and mean_epoch_error[-1] > mean_epoch_error[-2]:
            print('Ending epoch is %d .' % epoch)
            break

        # norm_epoch_error += [np.mean(np.sqrt(norm_batch_error))]


        #
        # print('Training epoch %d, cost is %f' % (epoch, np.mean(mean_cost) ) )

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time)

    print ('Training took %f minutes' % (pretraining_time / 60.))

    return mpf_optimizer.W.get_value(borrow = True)


if __name__ == '__main__':

    filter = mpf_em(hidden_units=200, dataset= None)


    test_image = tile_raster_images(
        X=filter.T,
        img_shape=(28, 28),
        tile_shape=(10, 10),
        tile_spacing=(1, 1)
    )
    print(test_image.shape)
    image = Image.fromarray(test_image)

    image.save('filters.png')




