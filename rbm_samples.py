'''
This is the burn in version of gibbs sampling from a predefined weight and
bias parameters. We first run the gibbs sampling for burn in steps, and then
we get the samples every burn-in steps.
'''


import numpy as np
from deepMPF import *

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class rbm_samples(object):

    def __init__(self, vis_units = 20, hid_units = 10, W = None,b_vis = None,b_hid = None, numpy_rng=None):

        numpy_rng = np.random.RandomState(66666)
        self.vis_units = vis_units
        self.hid_units = hid_units
        self.num_neuron = self.vis_units + self.hid_units
        if not W:
            self.W = np.asarray(numpy_rng.randn(self.vis_units, self.hid_units) / np.sqrt(self.num_neuron))

        if not b_vis:
            self.b_vis = numpy_rng.randn(self.vis_units)/np.sqrt(self.vis_units)

        if not b_hid:
            self.b_hid = numpy_rng.randn(self.hid_units)/np.sqrt(self.hid_units)


    def sample(self,n_sample = 10000):
        independent_steps = 10 * self.vis_units # this is the mix-in rate
        burnin_steps = 100 * self.vis_units
        sample_steps = burnin_steps + (n_sample - 1) * independent_steps

        samples = np.zeros((n_sample,self.vis_units))

        i_out = 0
        next_sample = burnin_steps

        numpy_rng = np.random.RandomState(12345)
        x = np.floor(numpy_rng.uniform(
                  low= 0,
                  high= 1,
                  size=(self.vis_units,))*2)

        for i in range(sample_steps):

            h_presigmoid = np.dot(x,self.W) + self.b_hid
            h_activation = sigmoid(-h_presigmoid)
            h_samples = np.random.binomial(n=1,p=h_activation)

            v_activation = sigmoid( -(np.dot(h_samples,self.W.T) + self.b_vis) )
            x = np.random.binomial(n=1, p=v_activation)

            if i == next_sample:
                next_sample = next_sample + independent_steps
                samples[i_out,:] = x
                i_out += 1

        return samples

if __name__ == '__main__':

    samplor = rbm_samples(vis_units= 10, hid_units= 10)
    samples = samplor.sample(n_sample=400)
    weight = samplor.W

    print(samples[:1,:])
    print(samples.shape)


    np.save('rbm_samples_10000.npy',samples)

    np.save('rbm_weight_10000.npy', weight)







