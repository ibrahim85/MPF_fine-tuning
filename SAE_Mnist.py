import os
import sys
import timeit

import numpy as np
import pickle
import gzip
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from dA import *
from logistic_sgd import *
from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image

def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))

def feedforward(ae,dataset,save_name):

    '''
    :param ae: ae is the model learned by the dA class, which contains the weight and bias of the neural network
    :param dataset: dataset is the numerical input to the model visible layer
    :param save_name: the path of the model hidden layer activations we want to save
    :return:
    '''

    ae = pickle.load(open(ae,'rb'),encoding="bytes")

    if dataset == 'mnist.pkl.gz':
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
        f.close()

    else:
        f = open(dataset,'rb')
        train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
        f.close()

    W = ae.W.get_value(borrow=True)
    b = ae.b.get_value(borrow=True)

    output_train = sigmoid_array(np.dot(train_set[0],W) + b)
    output_valid = sigmoid_array(np.dot(valid_set[0],W) + b)
    output_test = sigmoid_array(np.dot(test_set[0],W) + b)

    output =   [[output_train,train_set[1]], [output_valid, valid_set[1]], [output_test, test_set[1]]]
    with open(save_name, 'wb') as f:
        pickle.dump(output,f)


def softmax(x):

    e_x = np.exp(x - np.max(x,axis=1).reshape(x.shape[0], 1))
    return e_x/np.sum(e_x,axis=1).reshape(x.shape[0],1)


def get_softmax_prob(dataset,save_name,model = 'best_classifier.pkl'):

    '''
    :param dataset: dataset is the input of the softmax classifier
    :param save_name: save name is the path to save the softmax output probabilities
    :param model: model is the classifier we learned by the logistic_sgd class
    :return:
    '''

    softmax_model = pickle.load(open(model,'rb'),encoding="bytes")
    W = softmax_model.W.get_value(borrow=True)
    b = softmax_model.b.get_value(borrow=True)

    f = open(dataset,'rb')
    train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
    f.close()

    prob_train = softmax(np.dot(train_set[0],W) + b)
    prob_valid = softmax(np.dot(valid_set[0],W) + b)
    prob_test = softmax(np.dot(test_set[0],W) + b)

    prob = [prob_train, prob_valid, prob_test]

    with open(save_name, 'wb') as f:
        pickle.dump(prob,f)



def SAE(dataset = 'mnist.pkl.gz', num_neuron_list = None, learning_rate = 0.001):

    '''
    :param num_layer: the numer of layers in the neural network
    :param dataset: the input to the bottom visible layer
    :param num_neuron_list: the number of neurons in each hidden layer
    :return: paths to the activations and probability outputs and also the individual auto-encoder models
    '''

    model_list = []
    activation_list = []
    num_layer = len(num_neuron_list)

    for i in range(num_layer - 1):
        model_list.append('model_' + str(i +1) + '.pkl')
        activation_list.append('activation_' + str(i + 1) + '.pkl')


    for i in range(len(model_list)):

        if i ==0:
            data = dataset
        else:
            data = activation_list[i-1]

        if i < num_layer - 2:

            print('Pre-train layer %d' % i)

            ae = test_dA(learning_rate=learning_rate, training_epochs= 300,
                   dataset= data,batch_size=20,  n_visible = num_neuron_list[i] , n_hidden=num_neuron_list[i+1],
                   model= model_list[i] , output_folder='dA_plots')

            feedforward(ae, dataset=data, save_name= activation_list[i])

        else:

            print('Pre-train the classifier')

            sgd_optimization_mnist(learning_rate=learning_rate, n_epochs=1000,
                           dataset= data, model = model_list[i], n_in= num_neuron_list[i], n_out= num_neuron_list[i+1],
                           batch_size=20)

            get_softmax_prob(dataset = data,save_name = activation_list[i],model = model_list[i])

    return model_list, activation_list

if __name__ ==  '__main__':

    SAE(num_neuron_list=[784,500,300,10])
