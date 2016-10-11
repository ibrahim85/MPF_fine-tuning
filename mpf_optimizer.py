'''
This optimizer is developed for solving the MPF objective function with binary inputs.
We will view the entire graph which needs to be finetuned as a fully-observable boltzmann machine.
The traintime could benefit a lot from the special form of MPF.
'''

import numpy as np
import gzip
import timeit, pickle, sys, math
import theano
import theano.tensor as T
import os
from data_generator import *
from weighted_data_generator import *
import os.path
from lasagne.updates import nesterov_momentum
from theano.tensor.shared_randomstreams import RandomStreams


'''Optimizes based on MPF for fully-observable boltzmann machine'''
class MPF_optimizer(object):

    def __init__(self,epsilon = 1, num_units = 100, W = None, b = None, input = None,batch_sz = 10, connect_function = '1-bit-flip' ):
        '''

        :param W: the weights of the graph
        :param b: the bias of the graph
        :param input: input binary data samples
        :param connect_function: connection type
        :return:
        '''
        #W = np.load(W_path)
        #b = np.load(b_path)
        self.num_neuron = num_units
        numpy_rng = np.random.RandomState(123456)

        if W is None:
            initial_W = np.asarray(numpy_rng.randn(num_units, num_units) / np.sqrt(self.num_neuron) / 100 *
                                   (np.ones((num_units,num_units)) - np.identity(num_units)),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            self.W = theano.shared(value= (initial_W + initial_W.T)/2, name='W', borrow=True)
        else:
            self.W = theano.shared(value=np.asarray(W,dtype=theano.config.floatX),name = 'Weight', borrow = True)

        if b is None:
            self.b = theano.shared(
                value=np.zeros(
                    self.num_neuron,
                    dtype=theano.config.floatX
                ),
                name='bias',
                borrow=True
            )
        else:
            self.b = theano.shared(value=np.asarray(b,dtype=theano.config.floatX),name = 'bias', borrow = True)

        self.epsilon = epsilon

        self.batch_sz = batch_sz
        if not input:
            self.input = T.matrix('input')
        else:
            self.input = input

        self.params = [self.W, self.b]

        #self.params = []


        ''' Computing energy '''
    def energy(self,data_samples):
        # assume a graph of 100 nodes, then input of N*100, and W of 100* 100, b of 1*100
        wx = T.dot(T.dot(data_samples, self.W),data_samples.T)
        k = T.eye(data_samples.shape[0])
        wx_term = T.sum(wx * k, axis = 1) # only consider the diagonal elements as the exact energy of certain sample

        bias_term = T.dot(self.b, data_samples.T)
        return -0.5 * wx_term - bias_term


    def get_cost_updates(self,learning_rate):

        # data = T.repeat(self.input, self.input.shape[1], axis=0)
        #
        # Y = T.tile(T.eye(self.input.shape[1]),(self.batch_sz,1))
        #
        # non_data = (data + Y) % 2


        # rs = np.random.RandomState(1234)
        #
        # rng = T.shared_randomstreams.RandomStreams(rs.randint(999999))
        #
        #
        # corrupt = rng.binomial(size=self.input.shape, n=1,
        #                                 p = 0.2,
        #                                 dtype=theano.config.floatX)
        #
        # non_data = (self.input + corrupt)%2



        z = 1/2 - self.input

        energy_difference = z * (T.dot(self.input,self.W)+ self.b.reshape([1,-1]))

        #cost = T.sum(p) * (self.epsilon/self.batch_sz)


        # Y = self.input.reshape((self.batch_sz, 1, self.num_neuron), 3)\
        #     * T.ones((1, self.num_neuron, 1)) #tile out data vectors (repeat each one D times)
        # non_data = (Y + T.eye(self.num_neuron).reshape((1, self.num_neuron, self.num_neuron), 3))%2 # flip each bit once


        #energy_difference = 0.5* (self.energy(self.input).dimshuffle(0, 'x') - self.energy(non_data))
        # energy_difference = 0.5*(self.energy(self.input) - self.energy(non_data))
        #
        cost = (self.epsilon/self.batch_sz) * T.sum(T.exp(energy_difference))

        #gparams = T.grad(cost, self.params)
            # generate the list of updates
        # updates = [
        #     (params, params - learning_rate * gparams)
        #     for params, gparams in zip(self.params, gparams)
        #     ]

        #updates = nesterov_momentum(cost, self.params, learning_rate = learning_rate, momentum=0.9)


        return cost


    def error(self,num_neuron_list,data_x,data_y):

        '''
        This function can feedfoward the current model and get the error predictions.

        :param num_neuron_list: the list of number of neurons in each layer, including the first input layer
        :param data_x: the input float data
        :param data_y: the data labels
        :return: the mean error in classification for all batches
        '''

        Weight = []
        bias = []
        column = num_neuron_list[0]
        row = 0
        bias_index = 0
        for i in range(int(len(num_neuron_list)-1)):

            Weight.append(self.W[row : row + num_neuron_list[i],
                          column: column + num_neuron_list[i + 1]])
            bias.append(self.b[bias_index:bias_index + num_neuron_list[i+1]])
            column += num_neuron_list[i + 1]
            row += num_neuron_list[i]
            bias_index += num_neuron_list[i+1]


        ## feedforward the neural network and get the softmax output
        activation = None
        for i in range( int(len(Weight) -1)):
            if activation is None:
                activation =T.nnet.sigmoid(T.dot(data_x,Weight[i]) + bias[i])
            else:
                activation = T.nnet.sigmoid(T.dot(activation,Weight[i]) + bias[i])

        self.p_y_given_x =  T.nnet.softmax(T.dot(activation,Weight[-1]) + bias[-1])

        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)

        ## get the prediction accuracy

        if data_y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', data_y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if data_y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, data_y))
        else:
            raise NotImplementedError()

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def mnist_mpf(data_dict,W=None,b=None, dataset = 'mnist.pkl.gz',
              num_neuron_list = None,n_samples = 100,epsilon = 0.01,learning_rate = 0.001,
              n_epochs=1000,batch_sz = 20,mnist = True, connect_function = '1-bit-flip'):

    '''

    :param data_dict: the activation/input dictionary of all the neurons
    :param W_path: the path of weight matrix
    :param b_path: path of bias matrix
    :param epsilon: the step epsilon in MPF
    :param learning_rate: learning rate in gradient descent
    :param batch_sz: batch size for SGD
    :param connect_function: 1-bit-flip, random-flip, or factorize or persistent or continuous
    :return: the predicted classification rate
    '''

    ################################################################
    ################## Loading the Data        #####################
    ################################################################

    data_path = 'binary_data_samples.npy'

    if not os.path.exists(data_path):

        print('Generating the binary data samples ......')

        data_gen = data_generator(data_dict = data_dict, n_samples = n_samples,savename=data_path)

        data_gen.data_generator(mnist = mnist)

    else:

        print('Binary data samples already exist ......')

    data_samples = np.load(data_path)

    n_train_batches = data_samples.shape[0]//batch_sz

    data_samples  = theano.shared(value=np.asarray(data_samples, dtype=theano.config.floatX),
                                  name = 'train',borrow = True)

    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='bytes')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)

    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_sz
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_sz

    ################################################################
    ##################Initialize Train Function#####################
    ################################################################

    index = T.lscalar()    # index to a mini batch
    x = T.matrix('x')
    y = T.ivector('y')

    mpf_optimizer = MPF_optimizer(
        epsilon = epsilon,
        W = W,
        b = b,
        input = x,
        batch_sz=batch_sz,
        connect_function = connect_function
    )


    cost, updates = mpf_optimizer.get_cost_updates(learning_rate= learning_rate)

    train_mpf = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: data_samples[index * batch_sz: (index + 1) * batch_sz],
        },
        on_unused_input='warn',
    )


    validate_error = mpf_optimizer.error(num_neuron_list=num_neuron_list,data_x= x,data_y=y)

    validate_mpf = theano.function(
        [index],
        validate_error,
        givens={
            x: valid_set_x[index * batch_sz: (index + 1) * batch_sz],
            y: valid_set_y[index * batch_sz: (index + 1) * batch_sz]
        },
        on_unused_input='warn',
    )


    test_error = mpf_optimizer.error(num_neuron_list=num_neuron_list,data_x = x, data_y = y)

    test_mpf = theano.function(
        [index],
        test_error,
        givens={
            x: test_set_x[index * batch_sz: (index + 1) * batch_sz],
            y: test_set_y[index * batch_sz: (index + 1) * batch_sz]
        },
        on_unused_input='warn',
    )

    ################################################################
    ##################  Training MPF Model     #####################
    ################################################################
    print('... training the MPF model')
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_mpf(minibatch_index)

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_mpf(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_mpf(i)
                                   for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(mpf_optimizer, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The fine-tuning run for %d epochs, with %.2fm' % (
        epoch, (end_time - start_time) / 60))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)








