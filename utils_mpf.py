import pickle
import numpy as np
import matplotlib.pyplot as plt


def load(filename):
    with open(filename,'rb') as f :
        bob = pickle.load(f,encoding="bytes")
    f.close()
    return bob


def save(filename, bob):
    f = open(filename,'wb')
    pickle.dump(bob,f)
    f.close()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def propup(data,W,b):

    activation = sigmoid(np.dot(data,W) + b)

    hidden_samples = np.random.binomial(n=1, p = activation)

    new_input = np.concatenate((data,hidden_samples), axis = 1)

    return activation, new_input


def get_mpf_params(visible_units, hidden_units):

    '''
    :param visible_units: number of units in the visible layer
    :param hidden_units: number of units ni the hidden layer
    :return: The well structured MPF weight matrix
    The MPF weight matrix is of the form:
    [0,   W,
     W.T, 0]
    '''
    numpy_rng = np.random.RandomState(555555)

    W = numpy_rng.randn(visible_units,hidden_units)/np.sqrt(visible_units*hidden_units)

   # W = np.random.uniform(low=-1, high=1,size = (visible_units,hidden_units))

    W_up = np.concatenate((np.zeros((visible_units,visible_units)), W), axis = 1)

    W_down = np.concatenate((W.T,np.zeros((hidden_units,hidden_units))), axis = 1 )

    W = np.concatenate((W_up,W_down), axis = 0)

    print(W.shape)

    return W


def show_loss(savename, epoch_error = None):

    x = np.arange(len(epoch_error))
    plt.plot(x, epoch_error, 'r')
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.title('Adam Training Loss')
    plt.grid(True)
    plt.savefig(savename)

