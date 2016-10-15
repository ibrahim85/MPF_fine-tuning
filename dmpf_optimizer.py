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
class dmpf_optimizer(object):

    def __init__(self,epsilon = 1, num_units = 100, W = None, b = None,
                 input = None,batch_sz = 20, theano_rng = None, connect_function = '1-bit-flip' ):
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

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

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

        self.zero_grad = None



        #self.params = []


        ''' Computing energy '''
    def energy(self,data_samples):
        # assume a graph of 100 nodes, then input of N*100, and W of 100* 100, b of 1*100
        wx = T.dot(T.dot(data_samples, self.W),data_samples.T)
        k = T.eye(data_samples.shape[0])
        wx_term = T.sum(wx * k, axis = 1) # only consider the diagonal elements as the exact energy of certain sample

        bias_term = T.dot(self.b, data_samples.T)
        return -0.5 * wx_term - bias_term

    # def get_feedfowd_params(self,visible_units,hidden_units):
    #     # Get the feedforward weights from the big W matrix for MPF
    #     W_feedfowd = self.W[:visible_units,visible_units:]
    #
    #     return W_feedfowd


    def get_dmpf_cost(self,visible_units, hidden_units, learning_rate = 0.01, n_samples = 1):

        # In one round, we feed forward the and get the samples,
        # Compute the probability of each data samples,
        # call the minimum probability flow objective function


        W_feedfowd = self.W[:visible_units,visible_units:]

        b_feedfowd = self.b[visible_units:]

        H = T.nnet.sigmoid(T.dot(self.input,W_feedfowd) + b_feedfowd )

        # generate hidden samples

        # srng = RandomStreams(seed=234)
        # rv_u = srng.binomial(n=1,p = H)
        # f = theano.function([], rv_u)
        # hidden_samples = f()

        hidden_samples = self.theano_rng.binomial(size=H.shape,
                                             n=1, p=H,
                                             dtype=theano.config.floatX)

        # get the new input data for training MPF

        self.input = T.concatenate((self.input,hidden_samples), axis = 1)

        # compute the weight for each sample
        sample_prob = theano.shared(value= np.asarray(np.ones(self.batch_sz), dtype=theano.config.floatX), borrow = True)
        for i in range(visible_units):
            sample_prob *= H[:,i]

        sample_prob = sample_prob / T.sum(sample_prob)

        # compute the weighted MPF cost
        z = 1/2 - self.input
        energy_difference = z * (T.dot(self.input,self.W)+ self.b.reshape([1,-1]))
        cost = (self.epsilon) * T.sum(T.exp(energy_difference))

        # compute the weighted MPF grad

        W_grad = T.grad(cost, wrt = self.W, consider_constant=[hidden_samples,sample_prob])

        if self.zero_grad is None:
            a = np.ones((visible_units,hidden_units))
            b = np.zeros((visible_units,visible_units))
            c = np.zeros((hidden_units,hidden_units))
            zero_grad_u = np.concatenate((b,a),axis = 1)
            zero_grad_d = np.concatenate((a.T,c),axis=1)
            zero_grad = np.concatenate((zero_grad_u,zero_grad_d),axis=0)
            self.zero_grad = theano.shared(value=np.asarray(zero_grad,dtype=theano.config.floatX),
                                           name='zero_grad',borrow = True)
        W_grad *= self.zero_grad

        b_grad = T.grad(cost=cost, wrt=self.b, consider_constant=[hidden_samples,sample_prob])

        updates = [(self.W, self.W - learning_rate * W_grad),
                (self.b, self.b - learning_rate * b_grad)]


        return cost, updates








