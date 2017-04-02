'''
This file implement the deep rbm model trained with deep MPF.
Optimization method: Adam
'''
from sklearn import preprocessing
import numpy as np
import gzip
import timeit, pickle, sys, math
import theano
import theano.tensor as T
from PIL import Image
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from utils import tile_raster_images
from EM_MPF import em_mpf, get_mpf_params, propup, show_loss
from dmpf_optimizer import dmpf_optimizer
from KL_mpf import sigmoid



def rbm_mpf(hidden_units,decay,learning_rate,batch_sz,dataset = None,epsilon = 0.01):
    '''
    This function defines a general rbm training process. Starting from the data, together
    with the hyper-parameters, we train the rbm and return the associated weight, bias.

    :param data:
    :param hidden_units:
    :param decay:
    :param learning_rate:
    :param batch_sz:
    :param epsilon:
    :return:
    '''

    path = '../Thea_mpf/rbm_hidden_' + str(hidden_units) + '/decay_' + str(decay) + '/lr_' + str(learning_rate)
           # + '/bsz_' + str(batch_sz)
    if not os.path.exists(path):
        os.makedirs(path)

    data = np.load(dataset)
    visible_units = data.shape[1]

    n_train_batches = data.shape[0]//batch_sz

    num_units = visible_units + hidden_units

    W = get_mpf_params(visible_units, hidden_units)

    b = np.zeros(num_units)

    out_epoch = 500
    in_epoch = 1

    index = T.lscalar()    # index to a mini batch
    x = T.matrix('x')

    mpf_optimizer = dmpf_optimizer(
        epsilon=epsilon,
        decay=decay,
        explicit_EM= True,
        num_units = num_units,
        W = W,
        b = b,
        input = x,
        batch_sz =batch_sz)


    new_data  = theano.shared(value=np.asarray(np.zeros((data.shape[0],num_units)), dtype=theano.config.floatX),
                                  name = 'train',borrow = True)

    cost,updates = mpf_optimizer.get_dmpf_cost(
        learning_rate= learning_rate,
        visible_units=visible_units,
        hidden_units=hidden_units)

    train_mpf = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
        x: new_data[index * batch_sz: (index + 1) * batch_sz],
        },
        #on_unused_input='warn',
    )
    saveName_w = None
    saveName_b = None

    mean_epoch_error = []

    start_time = timeit.default_timer()

    for em_epoch in range(out_epoch):

        W = mpf_optimizer.W.get_value(borrow = True)
        b = mpf_optimizer.b.get_value(borrow = True)

        prop_W = W[:visible_units, visible_units:]
        prop_b = b[visible_units:]
        activations, sample_data = propup(data,prop_W,prop_b)
        #new_data.set_value(value=np.asarray(sample_data, dtype=theano.config.floatX),borrow = True)
        new_data.set_value(np.asarray(sample_data, dtype=theano.config.floatX))

        for mpf_epoch in range(in_epoch):
            mean_cost = []
            for batch_index in range(n_train_batches):
                mean_cost += [train_mpf(batch_index)]
            mean_epoch_error += [np.mean(mean_cost)]
        print('The cost for mpf in epoch %d is %f'% (em_epoch,mean_epoch_error[-1]))

        image_shape = (int(np.sqrt(visible_units)), int(np.sqrt(visible_units)))

        if em_epoch % 20 == 0:

            saveName = path + '/weights_' + str(em_epoch) + '.png'

            tile_shape = (20, hidden_units//20)

            #displayNetwork(W1.T,saveName=saveName)

            image = Image.fromarray(
                tile_raster_images(  X=(mpf_optimizer.W.get_value(borrow = True)[:visible_units,visible_units:]).T,
<<<<<<< HEAD
                        img_shape=(20,10),
=======
                        img_shape=image_shape,
>>>>>>> 0bcf54a01d657ae113dc4373f4d861472726d1d3
                        tile_shape=tile_shape,
                        tile_spacing=(1, 1)
                    )
                    )
            image.save(saveName)

        if (em_epoch+1) % 100 ==0:
            W = mpf_optimizer.W.get_value(borrow = True)
            W1 = W[:visible_units,visible_units:]
            b1 = mpf_optimizer.b.get_value(borrow = True)

            saveName_w = path + '/weights_' + str(em_epoch) + '.npy'
            saveName_b = path + '/bias_' + str(em_epoch) + '.npy'
            np.save(saveName_w,W1)
            np.save(saveName_b,b1)

    loss_savename = path + '/train_loss.eps'
    show_loss(savename= loss_savename, epoch_error= mean_epoch_error)

    end_time = timeit.default_timer()

    running_time = (end_time - start_time)

    print ('Training took %f minutes' % (running_time / 60.))


    return saveName_w, saveName_b



def train_deep_rbm(learning_rate, lay1_unit, lay2_unit,decay, batch_size=40, epoches=500):


    epsilon = 0.01
    n_samples = 1
    learning_rate = learning_rate

    layer_1_hid = lay1_unit
    layer_2_hid = lay2_unit

<<<<<<< HEAD
    decay = 0.0001
    batch_sz = 20
    epoches = 1
=======
    decay = decay
    batch_sz = batch_size
    epoches = epoches
>>>>>>> 0bcf54a01d657ae113dc4373f4d861472726d1d3


    dataset = 'mnist.pkl.gz'
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
    f.close()
    binarizer = preprocessing.Binarizer(threshold=0.5)
    data =  binarizer.transform(train_set[0])

    visible_units = data.shape[1]


    savename_w1, savename_b1 = em_mpf(hidden_units = layer_1_hid,learning_rate = learning_rate, epsilon = 0.01,decay=decay,
                                   batch_sz=batch_sz, epoch= epoches)

    print('This is the end of the first RBM................')
    W1 = np.load(savename_w1)

    b1 = np.load(savename_b1)

    b1 = b1[visible_units:]
    activation = sigmoid(np.dot(data,W1) + b1.reshape([1,-1]))

    hidden1_data = np.random.binomial(n=1,p = activation)

    path = '../Thea_mpf/rbm_hidden_' + str(layer_2_hid)
           # + '/decay_' + str(decay) + '/lr_' + str(learning_rate) \
           # + '/bsz_' + str(batch_sz)
    if not os.path.exists(path):
        os.makedirs(path)

    hidden1_data_path = path + '/hidden_data.npy'
    np.save(hidden1_data_path, hidden1_data)

    saveName_w2, saveName_b2 = rbm_mpf( dataset = hidden1_data_path,hidden_units=layer_2_hid,decay=decay,learning_rate= learning_rate,
            batch_sz= batch_sz,epsilon = 0.01)

    return savename_w1, savename_b1,saveName_w2, saveName_b2



if __name__ == '__main__':

<<<<<<< HEAD
    train_deep_rbm()
=======

    learning_list = [0.001]
    lay1_list = [196, 400, 100]
    lay2_list = [100, 60, 20]

    decay_list = [0.001, 0.0001, 0.01]
    epoches = 500
    for lr in learning_list:
        for decay in decay_list:
            for lay1_unit in lay1_list:
                for lay2_unit in lay2_list:
                    train_deep_rbm(learning_rate=lr,lay1_unit=lay1_unit,lay2_unit=lay2_unit,
                                   decay=decay,epoches=epoches)
>>>>>>> 0bcf54a01d657ae113dc4373f4d861472726d1d3
