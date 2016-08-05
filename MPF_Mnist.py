from mpf_optimizer import *
from SAE_Mnist import SAE
from utils_mpf import load, save
import os.path

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

    W = np.zeros((num_neuron,num_neuron))
    b = np.zeros(num_neuron)

    row = 0
    column = 0
    bias_index = 0

    for i in range(len(models)):
        ae = pickle.load(open(models[i],'rb'),encoding="bytes")
        Wi = ae.W.get_value(borrow=True)
        bi = ae.b.get_value(borrow=True)
        if column == 0:
            column += Wi.shape[0]
        W[row:row + Wi.shape[0], column:column + Wi.shape[1]] = Wi
        b[bias_index:bias_index + Wi.shape[1]] = bi

        row += Wi.shape[0]
        column += Wi.shape[1]
        bias_index += Wi.shape[1]

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
    num_neuron_list = [784,20,10,10]
    num_layer = len(num_neuron_list)
    num_neuron = np.sum(num_neuron_list)

    check_pretrain = 'model_1.pkl'
    if not os.path.exists(check_pretrain):
        model_list, activation_list = SAE(num_neuron_list = num_neuron_list,learning_rate = 0.01)

    else:
        model_list = []
        activation_list = []
        for i in range(num_layer - 1):
            model_list.append('model_' + str(i +1) + '.pkl')
            activation_list.append('activation_' + str(i + 1) + '.pkl')
        print("Pre-training already finished.......")

    ###############################################################
    ############ Form the data_dict for data generator  ###########
    ###############################################################

    data_dict = dict()

    check_trdata = 'train_1.pkl'

    if not os.path.exists(check_trdata):

        for i in range(num_layer):
            if i == 0:
                f = gzip.open('mnist.pkl.gz', 'rb')
                train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
                train_set = train_set[0]
                f.close()
            else:
                activation = load(activation_list[i-1])
                train_set = activation[0][0]

            filename = 'train_'+str(i+1) + '.pkl'
            save(filename,train_set)
            data_dict['layer_' + str(i+1)] = filename
    else:
        print('float train data already exists ......')
        for i in range(num_layer):
            filename = 'train_'+str(i+1) + '.pkl'
            data_dict['layer_' + str(i+1)] = filename

    #Form the whole weight matrix

    W, b = get_all_parameters(num_neuron= num_neuron, models = model_list)

    ###############################################################
    ############ Train MPF fine-tuning with SGD  ##################
    ###############################################################

    mnist_mpf(data_dict = data_dict, W=W,b=b,
              num_neuron_list = num_neuron_list,n_samples = 10,epsilon = 0.01,learning_rate = 0.1,
              n_epochs=1000,batch_sz = 20,mnist = True, connect_function = '1-bit-flip')