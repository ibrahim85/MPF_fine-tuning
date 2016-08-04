from mpf_optimizer import *
from SAE_Mnist import SAE
from utils_mpf import load, save

def get_all_parameters(num_neuron, models = []):

    '''
    In this function, we combine all the pre-trained layer-by-layer neural network together and
    to get a all-in-one weight and bias matrix

    E.g.: the weight matrix is now of [num_neuron, num_neuron]

    [ zeros, W1, zeros, zeros
      W1.T,  zeros, W2, zeros
      zeros, W2.T, zeros, W3
      zeros, zeros, W3.T, zeros]

    :param num_neuron: the number of neurons in the whole neural network
    :param models: the list of pre-trained layer by layer network
    :return: the whole weight and bias matrix for the network
    '''

    W = np.zeros(num_neuron,num_neuron)
    b = np.zeros(num_neuron)

    row = 0
    column = 0

    for i in range(len(models)):
        ae = pickle.load(open(models[i],'rb'),encoding="bytes")
        Wi = ae.W.get_value(borrow=True)
        bi = ae.b.get_value(borrow=True)
        if column == 0:
            column += Wi.shape[0]
        W[row:row + Wi.shape[0], column:column + Wi.shape[1]] = Wi
        b[row:row + Wi.shape[0]] = bi

        row += Wi.shape[0]
        column += Wi.shape[1]

    W = 0.5*(W + W.T)

    return W, b


if __name__ == '__main__':

    ################################################################
    ##################  Layer-wise Pre-train   #####################
    ################################################################

    '''
    In this part, we finish the layer-wise pre-train and get the pre-train layer models and activations
    in each layer.
    '''
    num_layer = 4
    num_neuron_list = [784,500,300,10]
    num_neuron = np.sum(num_neuron_list)

    model_list, activation_list = SAE(num_neuron_list = num_neuron_list,learning_rate = 0.001)

    ###############################################################
    ############ Form the data_dict for data generator  ###########
    ###############################################################

    data_dict = dict()

    for i in range(num_layer):
        if i == 0:
            train_set = load('mnist.pkl')

        else:
            activation = load(activation_list[i])
            train_set = activation[0][0]

        filename = 'train_'+str(i+1) + '.pkl'
        save(filename,train_set)
        data_dict['layer_' + str(i+1)] = filename

    #Form the whole weight matrix

    W, b = get_all_parameters(num_neuron= num_neuron, models = model_list)

    ###############################################################
    ############ Train MPF fine-tuning with SGD  ##################
    ###############################################################

