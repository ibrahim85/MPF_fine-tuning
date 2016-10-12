'''
In this file we implement the l-bfgs version optimization of MPF
'''
import numpy as np
from scipy.optimize import fmin_l_bfgs_b as minimize
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))

def unravel_params(theta,visible_size):
    W = theta[0:visible_size*visible_size].reshape(visible_size,visible_size)
    b = theta[visible_size*visible_size:].reshape(visible_size,1)
    return W,b

def ravel_params(W,b):
    return np.concatenate((W.ravel(),b.ravel()))

def init_params(visible_size,random_state=np.random.RandomState()):
    r = np.sqrt(6)/np.sqrt(2*visible_size+1)
    b = np.zeros((visible_size,1))
    W = random_state.rand(visible_size,visible_size)*2*r-r
    W = (W+W.T)/2
    np.fill_diagonal(W,0)
    return W,b

def costgrad(theta,data,cost_only=False):
    n_samples = data.shape[0]
    visible_size = data.shape[1]
    W,b = unravel_params(theta,visible_size)
    d = 1/2-data
    p = np.exp(d*(np.dot(data,W)+b.reshape([1,-1])))
    cost = np.sum(p)/n_samples
    if cost_only: return cost
    p = d*p
    bgrad = np.mean(p,axis=0)
    Wgrad = (np.dot(p.T,data)+np.dot(data.T,p))/n_samples
    np.fill_diagonal(Wgrad,0)
    return cost, ravel_params(Wgrad,bgrad)

def energy(W,b,x):
    return -np.dot(np.dot(x,W),x.T)/2 - np.dot(x,b)


def mpf_bfgs(data,W_init=None,b_init=None,factr=1e7,maxiter=15000,random_state=np.random.RandomState()):
    visible_size = np.shape(data)[1]
    if (W_init is not None and b_init is not None):
        W,b = W_init,b_init
    else:
        W,b = init_params(visible_size,random_state=random_state)

    theta = ravel_params(W,b)
    theta,cost,messages = minimize(costgrad,theta,args=(data,False),maxfun=maxiter,factr=factr)
    W,b = unravel_params(theta,visible_size)
    return W,b,cost,messages

if __name__ == '__main__':

    random_state=np.random.RandomState()
    data = np.load('gibbs_samples.npy')
    W_prime,b_prime,cost,messages = \
                mpf_bfgs(data,random_state=random_state)

    W = np.load('gibbs_weight.npy')
    b = np.load('gibbs_bias.npy')

    error = np.sum((W - W_prime)**2)/10000
    print(error)

    W_norm = np.sum(W**2)
    W_prime_norm = np.sum(W_prime**2)

    error2 =  np.sum((W/W_norm - W_prime/W_prime_norm)**2)/10000
    print(error2)

    W1 = W.ravel()
    W2 = W_prime.ravel()

    np.save('W_prime.npy',W_prime)
    np.save('b_prime.npy',b_prime)

    plt.plot(W1[900:1000])
    plt.plot(W2[900:1000])
    plt.show()

    plt.imshow(W- W_prime)
    plt.colorbar()
    plt.show()

    plt.imshow(np.abs(W- W_prime))
    plt.colorbar()
    plt.show()

    ###Fianl error 0.06#####



    #
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # ax1.imshow(W, extent=[0,1,0,1],aspect = 'auto')
    # ax1.set_title('Original W')
    # plt.show()
    #
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111)
    # ax2.imshow(W_prime - W, extent=[0,1,0,100],aspect = 'auto')
    # ax2.set_title('Learned W')
    # plt.show()


