'''

'''

'''
This file is created by Zuozhu for the MPF project of fully visible boltzmann machine.
In this file, we split the whole process into five parts:
step 1: Initialize a fully-visible boltzmann machine with J, b
step 2: Starting from a randomly choosing sample, run gibbs sampling to get many training samples
step 3: Based on the training samples, optimize the MPF objective function to get J', b'
step 4: check whether MPF converge to the original parameters, i.e., J,b ~ J',b'.
Step 5: If step 4 holds, we can use MPF for finetuning.
'''

import numpy as np
import time, pickle, sys, math
import theano
import theano.tensor as T
import os
import theano.sandbox.rng_mrg as RNG_MRG
from theano.tensor.shared_randomstreams import RandomStreams

def sigmoid(x):
    return 1 / (1 + np.exp(-x))



'''The Fully-visible boltzmann machine.  '''
class gibbs_sample(object):

    def __init__(self, x_0 = None, num_units = 100,J = None,b = None,numpy_rng=None):

        """
         :param num_units: the number of units in the fully visible botzmann machine
         :param J: The weights in the fbm
         :param b: The bisa term in the fbm
         :param x_0: The starting point of the MCMC
         :param numpy_rng: random number generator
         :return:
        """
        self.num_units = num_units

        if numpy_rng is None:
            # create a number generator
            numpy_rng = np.random.RandomState(1234)

        if not J:
            # Since this is fully visible, so the weight matrix should be symmetric

            initial_J = np.asarray(numpy_rng.uniform(low= -1,high= 1,size=(num_units, num_units)) *
                                   (np.ones((num_units,num_units)) - np.identity(num_units)))
            J = 0.5 * (initial_J + initial_J.T)

            #J = theano.shared(value = sym_J, name='W', borrow=True)

        # The bias term may not be zeros.
        if not b:
            # b = theano.shared(value=np.zeros(num_units, dtype=theano.config.floatX),
            #                   name='bias', borrow=True)
            b = numpy_rng.uniform(low= -1,high= 1,size=(num_units))

        if not x_0:
            x_0 = numpy_rng.uniform(
                  low= -1,
                  high= 1,
                  size=(num_units,))
                  #dtype=theano.config.floatX)

            #x_0 = theano.shared(value=initial_x_0, name='x0', borrow=True)

        self.J = J
        self.b = b
        self.x_0 = x_0


    def onestep_gibbs(self, count = 0, x_count = None):

        '''
        :param count: which node we are going to draw a die
        :param x_count: the current status of the graph
        :return: the new state after draw a die with respect to the prob of the node
        '''
        prob = sigmoid(np.dot(x_count,self.J[:,count]) + self.b[count])

        gen_count_bit = np.random.binomial(n=1, p = prob)

        gen_sample = x_count

        #print(x_count)

        gen_sample[count] = gen_count_bit

        #print(gen_sample)

        return gen_sample

    def oneround_gibbs(self, step = 10, start = None):

    #We perform the one-step gibbs sampling here. Starting from x_0, we sample from all the nodes within the scan.
        count = np.random.randint(100)
        #one_round_samples = []
        real_samples = []

        gen_sample = start

        for i in range(self.num_units):

            gen_sample = self.onestep_gibbs(count=count,x_count = gen_sample)

            #one_round_samples.append(gen_sample)

            if count % step ==0:
                real_samples.append(gen_sample)

            count = (count + 1) % self.num_units


        return np.asarray(real_samples)

    def gibbs_sample(self,step=10,n_samples = 1000):
        """
        :param p_first: The position of the first sample that we need (After how many gibbs sampling steps)
        :param step: The sample frequency (sample every how many steps of gibbs sampling)
        :param n_samples: how many samples we need
        :return: the set of training samples

        """

        '''
        The Gibbs sampling can be decomposed into several steps:
        step 1: Initializing the weight and bias term of the graph (Finished in the above part), intialize X(0)
        step 2: Do gibbs sampling for all the nodes one by one, alternatively. And get the final one as
        the first sample
        step 3: Start the Monte Carlo Sampling from this sample, and save samples every 10 samples
        step 4: Sample until stop
        '''

        train_samples  = self.oneround_gibbs(start = self.x_0)

        print(train_samples.shape)


        for i in range(1, n_samples):


            real_samples = self.oneround_gibbs(start = train_samples[-1,:])

            train_samples = np.concatenate((train_samples,real_samples),axis = 0)

        return train_samples




if __name__ == '__main__':

    gibbs_samplor = gibbs_sample()

    train_samples = gibbs_samplor.gibbs_sample(step = 10, n_samples=10000)

    np.save('g_samples.npy',train_samples)

    np.save('g_weight.npy', gibbs_samplor.J)

    np.save('g_bias.npy',gibbs_samplor.b)


    # print(np.load('gibbs_weight.npy').shape)
    # print(np.load('gibbs_bias.npy'))
    print(np.load('g_samples.npy').shape)
    print(np.load('g_samples.npy')[:10,:])




























