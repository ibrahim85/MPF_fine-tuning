'''
This is the burn in version of gibbs sampling from a predefined weight and
bias parameters. We first run the gibbs sampling for burn in steps, and then
we get the samples every burn-in steps.
'''

import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class burnin_gibbs(object):

    def __init__(self, num_units = 100,J = None,b = None,numpy_rng=None):

        """
         :param num_units: the number of units in the fully visible botzmann machine
         :param J: The weights in the fbm
         :param b: The bisa term in the fbm
         :param numpy_rng: random number generator
         :return:
        """
        self.num_units = num_units

        if numpy_rng is None:
            # create a number generator
            numpy_rng = np.random.RandomState(1234)

        if not J:
            # Since this is fully visible, so the weight matrix should be symmetric

            initial_J = np.asarray(numpy_rng.randn(num_units, num_units) / np.sqrt(self.num_units) * 3 *
                                   (np.ones((num_units,num_units)) - np.identity(num_units)))
            J = 0.5 * (initial_J + initial_J.T)
        self.J = J
        # The bias term may not be zeros.
        if not b:
            b = - np.sum(self.J,axis = 1)
        self.b = b



    def sample(self,n_sample = 10000):
        independent_steps = 10 * self.num_units
        burnin_steps = 100 * self.num_units
        sample_steps = burnin_steps + (n_sample - 1) * independent_steps
        update_i = np.floor(np.random.uniform(low=0,high=1,size=(sample_steps,)) * self.num_units)
        update_i = update_i.astype(np.int64)
        treshhold_i = np.random.uniform(low=0,high=1,size=(sample_steps,))

        samples = np.zeros( (n_sample,self.num_units) )

        i_out = 0
        next_sample = burnin_steps

        numpy_rng = np.random.RandomState(12345)
        x = np.floor(numpy_rng.uniform(
                  low= 0,
                  high= 1,
                  size=(self.num_units,))*2)

        for i in range(sample_steps):

            E_act = 2 * np.dot(x, self.J[:,update_i[i]]) + self.b[update_i[i]]
            p_act = sigmoid(E_act)
            if p_act > treshhold_i[i]:
                x[update_i[i]] = 1
            else:
                x[update_i[i]] = 0

            if i == next_sample:
                next_sample = next_sample + independent_steps
                samples[i_out,:] = x
                i_out += 1

        return samples

if __name__ == '__main__':

    samplor = burnin_gibbs()
    samples = samplor.sample(n_sample=100000)
    weight = samplor.J
    bias = samplor.b

    np.save('gibbs_samples_100000.npy',samples)

    np.save('gibbs_weight_100000.npy', weight)

    np.save('gibbs_bias_100000.npy',bias)


    print(np.load('gibbs_weight_100000.npy'))
    print(np.load('gibbs_bias_100000.npy'))
    print(np.load('gibbs_samples_100000.npy').shape)
    print(np.load('gibbs_samples_100000.npy')[:10,:])
