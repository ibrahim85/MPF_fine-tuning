from gibbs_sample import *
from mpf_optimizer import  *
import numpy as np
import matplotlib.pyplot as plt


'''
Given: The gibbs samples, the weight, the bias
Goal: Training the MPF to estimate the parameters and compare with the original parameters
'''

def check_sanity(W,bias,samples):
    '''
    :param W: The weight of the true model
    :param bias: The bias of the True model
    :param samples: The samples generated from gibbs sampling
    :return: The square error of the parameters
    '''
    epsilon = 0.01
    learning_rate = 0.001
    connect_function = '1-bit-flip'
    num_units = 100
    index = T.lscalar()    # index to a mini batch
    x = T.matrix('x')
    batch_sz = 40

    mpf_optimizer = MPF_optimizer(
        epsilon = epsilon,
        num_units=num_units,
        input = x,
        batch_sz=batch_sz,
        )

    data_samples = np.load(samples)

    n_train_batches = data_samples.shape[0]//batch_sz

    data_samples  = theano.shared(value=np.asarray(data_samples, dtype=theano.config.floatX),
                                  name = 'train',borrow = True)

    cost = mpf_optimizer.get_cost_updates(learning_rate= learning_rate)

    g_W = T.grad(cost=cost, wrt=mpf_optimizer.W)

    zero_dig = theano.shared(np.asarray((np.ones((num_units,num_units)) - np.identity(num_units)),
                dtype=theano.config.floatX) , name = 'zero_dig', borrow = True)

    g_W = g_W * zero_dig

    g_b = T.grad(cost=cost, wrt=mpf_optimizer.b)

    updates = [ (mpf_optimizer.W, mpf_optimizer.W - learning_rate * g_W),
                (mpf_optimizer.b, mpf_optimizer.b - learning_rate * g_b)]

    train_mpf = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: data_samples[index * batch_sz: (index + 1) * batch_sz],
        },
        on_unused_input='warn',
    )

    training_epochs = 8000

    start_time = timeit.default_timer()

    # go through training epochs
    mean_epoch_error = []
    norm_epoch_error = []
    bias_mean_epoch_error = []
    for epoch in range(training_epochs):

        # go through the training set
        mean_cost = []
        mean_batch_error = []
        norm_batch_error = []


        for batch_index in range(n_train_batches):
            mean_cost += [train_mpf(batch_index)]
            weight = mpf_optimizer.W.get_value(borrow = True)
            bia = mpf_optimizer.b.get_value(borrow = True)

            mean_batch_error += [np.sum((W - mpf_optimizer.W.get_value(borrow=True))**2)]

            # norm_batch_error += [np.sum(( W/(np.sum(W**2)) -
            # mpf_optimizer.W.get_value(borrow=True)/(np.sum(mpf_optimizer.W.get_value(borrow=True)**2)) )**2 )]

        mean_epoch_error += [np.mean((mean_batch_error))/10000]



        if epoch > 2 and mean_epoch_error[-1] < 0.011:
            print('Ending epoch is %d .' % epoch)
            break

        # norm_epoch_error += [np.mean(np.sqrt(norm_batch_error))]

        # print(mean_epoch_error[-1]/10000)
        #
        # print('Training epoch %d, cost is %f' % (epoch, np.mean(mean_cost) ) )

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time)

    print ('Training took %f minutes' % (pretraining_time / 60.))


    return mean_epoch_error, mpf_optimizer.W.get_value(borrow= True),mpf_optimizer.b.get_value(borrow= True)



if __name__ == '__main__':

    W = np.load('gibbs_weight_100000.npy')
    print(W[:10,:10])
    bias = np.load('gibbs_bias_100000.npy')
    print(bias[:10])
    samples = 'gibbs_samples_100000.npy'

    error,W_prime,b_prime = check_sanity(W,bias,samples)

    print(error[-2])


    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title('SGD Diff between W and W_prime')
    plt.imshow(np.abs(W - W_prime), extent=[0,100,0,100],aspect = 'auto')
    plt.colorbar()
    plt.show()
    fig1.savefig('01_1000_100000_Sgd_Imageseq.png')


    np.save('0.01_1000_100000_sgd_Wprime.npy',W_prime)

    index = np.random.random_integers(low=0,high=10000,size = (100,))

    W1 = W_prime.ravel()
    W2 = W.ravel()
    W11 = W1[index]
    W22 = W2[index]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title('SGD: Diff of Randomly 100 Weight')
    plt.plot(W11,'y')
    plt.plot(W22,'c')
    plt.legend(['Recover W', 'Original W'])
    plt.show()
    fig1.savefig('01_1000_100000_Sgd_Random_Diff.png')
    plt.close()

    ####Final error 0.01#####












