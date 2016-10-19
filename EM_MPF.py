'''
The explicit version of EM-MPF.
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
from deepMPF import get_mpf_params
from display_network import displayNetwork

from utils import tile_raster_images

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def propup(data,W,b):

    activation = sigmoid(np.dot(data,W) + b)

    hidden_samples = np.random.binomial(n=1, p = activation)

    new_input = np.concatenate((data,hidden_samples), axis = 1)

    return activation, new_input


def get_sample_prob(activations):

    prob = np.ones(shape=(activations.shape[0]))
    for i in range(activations.shape[1]):
        prob *= activations[:,i]
    prob = prob / np.sum(prob)

    return prob



def em_mpf(hidden_units,learning_rate, epsilon, batch_sz = 20, dataset = None):

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
    displayNetwork(data[:100,:])
    # Binarize the mnist data doesnot hurt much to the input data.
    # displayNetwork(train_set[0][:100,:])


    ################################################################
    ##################  Initialize Parameters  #####################
    ################################################################

    #visible_units = train_set[0].shape[1]
    visible_units = data.shape[1]

    n_train_batches = data.shape[0]//batch_sz

    num_units = visible_units + hidden_units

    W = get_mpf_params(visible_units,hidden_units)

    b = np.zeros(num_units)

    out_epoch = 1000
    in_epoch = 11


    start_time = timeit.default_timer()

    for em_epoch in range(out_epoch):

        W_init = copy.deepcopy(W)
        b_init = copy.deepcopy(b)

        prop_W = W[:visible_units, visible_units:]
        prop_b = b[visible_units:]
        activations, new_data = propup(data,prop_W,prop_b)

        index = T.lscalar()    # index to a mini batch
        x = T.matrix('x')


        mpf_optimizer = dmpf_optimizer(epsilon=epsilon, explicit_EM= True,num_units = num_units, W = W, b = b,
                 input = x,batch_sz =batch_sz)

        new_data  = theano.shared(value=np.asarray(new_data, dtype=theano.config.floatX),
                                  name = 'train',borrow = True)

        cost,updates = mpf_optimizer.get_dmpf_cost(learning_rate= learning_rate,
                                               visible_units=visible_units,hidden_units=hidden_units)
        train_mpf = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
            x: new_data[index * batch_sz: (index + 1) * batch_sz],
        },
        #on_unused_input='warn',
        )

        mean_epoch_error = []
        for mpf_epoch in range(in_epoch):
            mean_cost = []
            for batch_index in range(n_train_batches):
                weight = mpf_optimizer.W.get_value(borrow = True)
                bia = mpf_optimizer.b.get_value(borrow = True)

                mean_cost += [train_mpf(batch_index)]

            mean_epoch_error += [np.mean(mean_cost)]
            print('The cost for inner mpf in epoch %d is %f'% (mpf_epoch,np.mean(mean_cost)))


            if mpf_epoch % 10 == 0:
                image = Image.fromarray(
                    tile_raster_images(
                    X=(mpf_optimizer.W.get_value(borrow = True)[:visible_units,visible_units:]).T,
                        img_shape=(28, 28),
                        tile_shape=(10, 10),
                        tile_spacing=(1, 1)
                    )
                    )
                image.show()
                image.save('filters_at_epoch_%i_%i.png' % (em_epoch,mpf_epoch))


            if mpf_epoch > 2 and mean_epoch_error[-1] > mean_epoch_error[-2]:
                print('Inner mpf ends at epoch %d'% mpf_epoch)
                break


        image = Image.fromarray(
                tile_raster_images(
                    X=(mpf_optimizer.W.get_value(borrow = True)[:visible_units,visible_units:]).T,
                        img_shape=(28, 28),
                        tile_shape=(10, 10),
                        tile_spacing=(1, 1)
                    )
                    )
        image.show()
        image.save('filters_at_epoch_%i.png' % (em_epoch))


        W = mpf_optimizer.W.get_value(borrow = True)
        b = mpf_optimizer.b.get_value(borrow = True)

        deviation = np.sum((W - W_init)**2)/(2*visible_units*hidden_units)
        print(deviation)

        if em_epoch % 10 == 0:
            n_chains = 20
            n_samples = 10
            rng = np.random.RandomState(123)
            test_set_x = test_set[0]
            number_of_test_samples = test_set_x.shape[0]
            test_set_x = theano.shared( value = np.asarray(test_set_x, dtype=theano.config.floatX),
                                        name = 'test', borrow = True)

            # pick random test examples, with which to initialize the persistent chain
            test_idx = rng.randint(number_of_test_samples - n_chains)
            persistent_vis_chain = theano.shared(
                np.asarray(
                    test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
                    dtype=theano.config.floatX
                )
            )
            # end-snippet-6 start-snippet-7
            plot_every = 1000
            # define one step of Gibbs sampling (mf = mean-field) define a
            # function that does `plot_every` steps before returning the
            # sample for plotting
            (
                [
                    presig_hids,
                    hid_mfs,
                    hid_samples,
                    presig_vis,
                    vis_mfs,
                    vis_samples
                ],
                updates
            ) = theano.scan(
                mpf_optimizer.gibbs_vhv,
                outputs_info=[None, None, None, None, None, persistent_vis_chain],
                n_steps=plot_every
            )

            # add to updates the shared variable that takes care of our persistent
            # chain :.
            updates.update({persistent_vis_chain: vis_samples[-1]})
            # construct the function that implements our persistent chain.
            # we generate the "mean field" activations for plotting and the actual
            # samples for reinitializing the state of our persistent chain
            sample_fn = theano.function(
                [],
                [
                    vis_mfs[-1],
                    vis_samples[-1]
                ],
                updates=updates,
                name='sample_fn'
            )

            # create a space to store the image for plotting ( we need to leave
            # room for the tile_spacing as well)
            image_data = np.zeros(
                (29 * n_samples + 1, 29 * n_chains - 1),
                dtype='uint8'
            )
            for idx in range(n_samples):
                # generate `plot_every` intermediate samples that we discard,
                # because successive samples in the chain are too correlated
                vis_mf, vis_sample = sample_fn()
                print(' ... plotting sample ', idx)
                image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
                    X=vis_mf,
                    img_shape=(28, 28),
                    tile_shape=(1, n_chains),
                    tile_spacing=(1, 1)
                )

            # construct image
            image = Image.fromarray(image_data)
            image.save('samples_%i.png' % em_epoch)
            # end-snippet-7
            # os.chdir('../')

    end_time = timeit.default_timer()

    running_time = (end_time - start_time)

    print ('Training took %f minutes' % (running_time / 60.))


if __name__ == '__main__':


    em_mpf(hidden_units = 200,learning_rate = 0.01, epsilon = 0.01)











