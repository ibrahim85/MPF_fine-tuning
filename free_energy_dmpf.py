'''
This is the Deep MPF which used the free energy.
'''

import numpy as np
import gzip
import timeit, pickle, sys, math
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from lasagne.updates import nesterov_momentum
from lasagne.updates import adam, rmsprop
from theano.tensor.shared_randomstreams import RandomStreams
import sys
sys.setrecursionlimit(40000)


class free_energy_dmpf_optimizer(object):

    def __init__(self,epsilon = 1.0, visible_units = 16, hidden_units =8, W = None, b_vis = None,b_hid = None,
                 input = None,explicit_EM = False, batch_sz = 20, theano_rng = None, connect_function = '1-bit-flip' ):
        '''

        :param W: the weights of the graph
        :param b: the bias of the graph
        :param input: input binary data samples
        :param connect_function: connection type
        :return:
        '''
        #W = np.load(W_path)
        #b = np.load(b_path)
        self.visible_units = visible_units
        self.hidden_units = hidden_units

        numpy_rng = np.random.RandomState(12336)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

        if W is None:
            initial_W = np.asarray(numpy_rng.randn(self.visible_units, self.hidden_units) / np.sqrt(self.visible_units),
                dtype=theano.config.floatX)
            # theano shared variables for weights and biases
            self.W = theano.shared(value= initial_W, name='W', borrow=True)
        else:
            self.W = theano.shared(value=np.asarray(W,dtype=theano.config.floatX),name = 'Weight', borrow = True)

        if b_vis is None:
            self.b_vis = theano.shared(
                value=np.zeros(self.visible_units,dtype=theano.config.floatX),name='b_vis',borrow=True)
        else:
            self.b_vis = theano.shared(value=np.asarray(b_vis,dtype=theano.config.floatX),name = 'b)vis', borrow = True)


        if b_hid is None:
            self.b_hid = theano.shared(
                value=np.zeros(self.hidden_units,  dtype=theano.config.floatX),name='bias',borrow=True)
        else:
            self.b_hid = theano.shared(value=np.asarray(b_hid,dtype=theano.config.floatX),name = 'bias', borrow = True)

        self.epsilon = epsilon
        self.batch_sz = batch_sz
        if not input:
            self.input = T.matrix('input')
        else:
            self.input = input

        self.params = [self.W, self.b_vis, self.b_hid]


    def feedforward(self, data_samples):

        return T.exp(T.dot(data_samples, self.W) + self.b_hid)


    def free_energy(self, data_samples):

        wx = self.feedforward(data_samples)
        vbias_term = T.dot(data_samples, self.b_vis)
        hidden_term = T.sum(T.log(1 + wx), axis=1)
        return -hidden_term - vbias_term


    def get_cost_updates(self, learning_rate = 0.01,):
        '''
        In this function we compute the cost of deep MPF for the 1-bit-flip case.
        We use a for-loop to compute the flip for each bit.
        We do not use tensor or the tile functions, which can help reduce the for-loop to single matrix operation.
        '''
        base_energy = self.feedforward(data_samples=self.input)

        cost = 0



        for j in range(50):

            i = np.random.randint(low=0,high=self.visible_units,size=(1,))[0]

            # in every step, compute the MPF between x and a one-bit-flip non-data neighbor

            non_data_energy = 1 + (base_energy * \
                                  T.exp(T.dot((1-2*self.input[:,i].reshape([1,-1])).T, self.W[i,:].reshape([1,-1]))))

            data_energy = 1 + base_energy
            #energy_diff = T.sum(T.exp(0.5*T.log(T.prod(non_data_energy/data_energy, axis =1))))

            energy_diff = 0.5*T.sum(T.log(non_data_energy/data_energy),axis =1) \
                          - 0.5*self.b_vis[i]*(1-2*self.input[:,i].reshape([1,-1]))

            cost = cost + (self.epsilon/self.batch_sz)*T.sum(T.exp(energy_diff))

            #cost = T.sum(non_data_energy)
        gparams = T.grad(cost, self.params, consider_constant=[self.input])
        # W_grad = T.grad(cost=cost, wrt = self.W,consider_constant=[self.input])
        # b_vis_grad = T.grad(cost=cost, wrt=self.b_vis,consider_constant=[self.input])
        # b_hid_grad = T.grad(cost=cost, wrt=self.b_hid,consider_constant=[self.input])
        #
        # cache = W
        #
        # x += - learning_rate * dx / (np.sqrt(cache) + eps)

        #updates = [(self.params, self.params - learning_rate * gparams)]

        #updates = rmsprop(loss_or_grads=gparams,params=self.params, learning_rate=0.1, rho=0.9, epsilon=1e-08)

        updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(self.params, gparams)
            ]

        return cost, updates
    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(vis, self.W) \
                                 + self.b_hid
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(hid, self.W.T) \
                                 + self.b_vis
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state
            Thin function would be useful for performing CD and PCD'''
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state
            This function would be useful for sampling from the RBM'''
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]


if __name__ == '__main__':


    epsilon = 0.01
    learning_rate = 0.02
    vis_units = 10
    hid_units = 10
    num_units = vis_units + hid_units
    index = T.lscalar()    # index to a mini batch
    x = T.matrix('x')
    batch_sz = 10


    mpf_optimizer = free_energy_dmpf_optimizer(epsilon=epsilon, visible_units= vis_units, hidden_units= hid_units,
                 input = x,batch_sz =batch_sz)

    W = np.load('rbm_weight_10000.npy')
    print(W.shape)

    data = np.load('rbm_samples_10000.npy')
    n_train_batches = data.shape[0]//batch_sz

    data  = theano.shared(value=np.asarray(data, dtype=theano.config.floatX),
                                  name = 'train', borrow = True)


    cost, updates = mpf_optimizer.get_cost_updates(learning_rate=learning_rate)

    train_mpf = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: data[index * batch_sz: (index + 1) * batch_sz],
        },
        #on_unused_input='warn',
    )

    training_epochs = 400

    start_time = timeit.default_timer()

    mean_epoch_error = []

    for epoch in range(training_epochs):

        mean_cost = []
        mean_batch_error = []
        norm_batch_error = []


        for batch_index in range(n_train_batches):

            a = train_mpf(batch_index)
            mean_cost += [a]
            W_prime = mpf_optimizer.W.get_value(borrow = True)

            error_W = np.sum((W - W_prime)**2)

            mean_batch_error += [error_W/(vis_units*hid_units)]

        mean_epoch_error += [np.mean(mean_batch_error)]

        norm_batch_error += [np.mean(mean_cost)]
        print('The cost for dmpf in epoch %d is %f, rmse is %f' % (epoch, norm_batch_error[-1], mean_batch_error[-1]))

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time)

    print ('Training took %f minutes' % (pretraining_time / 60.))


    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title('SGD Diff between W and W_prime')
    plt.imshow(np.abs(W - W_prime), extent=[0,10,0,10],aspect = 'auto')
    plt.colorbar()
    plt.show()
    fig1.savefig('free_energy_Sgd_Imageseq.png')


    np.save('free_energy_Wprime.npy',W_prime)

    #index = np.random.random_integers(low=0,high=127,size = (100,))

    W1 = W_prime.ravel()
    W2 = W.ravel()
    #
    # index = np.random.random_integers(low=0,high=300,size = (100,))
    # W1 = W1[index]
    # W2 = W2[index]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title('SGD: Diff of Randomly 100 Weight')
    plt.plot(W1,'y')
    plt.plot(W2,'c')
    plt.legend(['Recover W', 'Original W'])
    plt.show()
    fig1.savefig('Free_Energy_Random_Diff.png')
    plt.close()

    print(mpf_optimizer.b_vis.get_value(borrow = True))

    print(mpf_optimizer.b_hid.get_value(borrow = True))







