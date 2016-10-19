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
import sys
import copy
sys.setrecursionlimit(40000)


'''Optimizes based on MPF for fully-observable boltzmann machine'''
class dmpf_optimizer(object):

    def __init__(self,epsilon = 1, num_units = 100, W = None, b = None,
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
        self.explicit_EM = explicit_EM
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


    def get_dmpf_cost(self,visible_units, hidden_units, learning_rate = 0.01, n_samples = 1, sample_prob = None):

        # In one round, we feed forward the and get the samples,
        # Compute the probability of each data samples,
        # call the minimum probability flow objective function

        self.visible_units = visible_units
        self.hidden_units = hidden_units

        if self.zero_grad is None:
            a = np.ones((visible_units,hidden_units))
            b = np.zeros((visible_units,visible_units))
            c = np.zeros((hidden_units,hidden_units))
            zero_grad_u = np.concatenate((b,a),axis = 1)
            zero_grad_d = np.concatenate((a.T,c),axis=1)
            zero_grad = np.concatenate((zero_grad_u,zero_grad_d),axis=0)
            self.zero_grad = theano.shared(value=np.asarray(zero_grad,dtype=theano.config.floatX),
                                           name='zero_grad',borrow = True)

        if not self.explicit_EM:

            # W_init = self.W.get_value(borrow = True)
            # b_init = self.b.get_value(borrow = True)
            # W_feed = copy.deepcopy(W_init)
            # b_feed = copy.deepcopy(b_init)
            #
            # W_feedfowd = W_feed[:visible_units,visible_units:]
            #
            # b_feedfowd=  b_feed[visible_units:]
            #
            # print(W_feed[1,784:800])
            # print(W_feedfowd[1,:16])
            #
            # W_feedfowd = theano.shared(value=W_feedfowd)
            # b_feedfowd = theano.shared(value=b_feedfowd)
            #
            # W_feedfowd = self.W[:visible_units,visible_units:]
            # b_feedfowd = self.b[visible_units:]
            # verify_W = W_feedfowd.get_value(borrow = True)
            # verify_diff = (verify_W - W_feedd)
            # print(verify_diff)

            # H = T.nnet.sigmoid(T.dot(self.input,W_feedfowd) + b_feedfowd )


            # generate hidden samples

            # hidden_samples = self.theano_rng.binomial(size=H.shape,
            #                                      n=1, p=H,
            #                                      dtype=theano.config.floatX)
            activation = self.gibbs_vhv(v0_sample=self.input)
            hidden_samples = activation[2]

            # get the new input data for training MPF

            self.input = T.concatenate((self.input,hidden_samples), axis = 1)

            self.sample_prob = T.prod(activation[1], axis = 1)
            self.sample_prob = self.sample_prob / T.sum(self.sample_prob)

            # compute the weight for each sample
            # self.sample_prob = theano.shared(value= np.asarray(np.ones(self.batch_sz), dtype=theano.config.floatX), borrow = True)
            # for i in range(visible_units):
            #     self.sample_prob *= H[:,i]
            #
            # self.sample_prob = self.sample_prob / T.sum(self.sample_prob)

        # compute the weighted MPF cost
            z = 1/2 - self.input
            energy_difference = z * (T.dot(self.input,self.W)+ self.b.reshape([1,-1]))
            self.sample_prob = self.sample_prob.reshape((1,-1)).T
            k = theano.shared(value= (np.asarray(np.ones(self.hidden_units),dtype=theano.config.floatX)).reshape((1,-1)))
            self.sample_prob = self.sample_prob * k
            cost = (self.epsilon/self.batch_sz) * T.sum(T.exp(energy_difference)*self.sample_prob)

        # compute the weighted MPF grad
            W_grad = T.grad(cost=cost, wrt = self.W,consider_constant=[activation[1],activation[2]])
            b_grad = T.grad(cost=cost, wrt=self.b,consider_constant=[activation[1],activation[2]])


        else:
            z = 1/2 - self.input
            energy_difference = z * (T.dot(self.input,self.W)+ self.b.reshape([1,-1]))
            #sample_prob = T.tile(sample_prob,(self.hidden_units,1)).T
            cost = (self.epsilon/self.batch_sz) * T.sum(T.exp(energy_difference))

            W_grad = T.grad(cost = cost, wrt = self.W)
            b_grad = T.grad(cost=cost, wrt=self.b)


        W_grad *= self.zero_grad

        updates = [(self.W, self.W - learning_rate * W_grad),
                (self.b, self.b - learning_rate * b_grad)]


        return W_grad[0,784:790],updates


    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(vis, self.W[:self.visible_units,self.visible_units:]) \
                                 + self.b[self.visible_units:]
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
        pre_sigmoid_activation = T.dot(hid, self.W[:self.visible_units,self.visible_units:].T) \
                                 + self.b[:self.visible_units]
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








