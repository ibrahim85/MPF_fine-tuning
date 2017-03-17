

import numpy as np
import matplotlib.pyplot as plt
from dmpf_optimizer import *
from deepMPF import *


def dmpf_sanity_check(W,samples):

    epsilon = 0.1
    learning_rate = 0.001
    vis_units = 10
    hid_units = 10
    num_units = vis_units + hid_units
    index = T.lscalar()    # index to a mini batch
    x = T.matrix('x')
    batch_sz = 20

    initial_W = get_mpf_params(vis_units,hid_units)
    mpf_optimizer = dmpf_optimizer(epsilon=epsilon, num_units = num_units, W = initial_W, b = None,
                 input = x,batch_sz =batch_sz)



    data = np.load(samples)

    n_train_batches = data.shape[0]//batch_sz
    data  = theano.shared(value=np.asarray(data, dtype=theano.config.floatX),
                                  name = 'train', borrow = True)

    cost,updates = mpf_optimizer.get_dmpf_cost(learning_rate= learning_rate,
                                               visible_units=vis_units,hidden_units=hid_units)


    train_mpf = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: data[index * batch_sz: (index + 1) * batch_sz],
        },
        #on_unused_input='warn',
    )

    training_epochs = 500

    start_time = timeit.default_timer()

    mean_epoch_error = []

    for epoch in range(training_epochs):

        # go through the training set
        mean_cost = []
        mean_batch_error = []
        norm_batch_error = []


        for batch_index in range(n_train_batches):

            # for chain in range(np.min([12, epoch])):
            #
            #     train_mpf(batch_index)

            a = train_mpf(batch_index)
            mean_cost += [a]

            weight = mpf_optimizer.W.get_value(borrow = True)

            W_prime = weight[:vis_units,vis_units:]

            bia = mpf_optimizer.b.get_value(borrow = True)



            # W1 = W_prime.ravel()
            # W2 = W.ravel()
            # # W11 = W1[index]
            # # W22 = W2[index]
            #
            # fig1 = plt.figure()
            # ax1 = fig1.add_subplot(111)
            # ax1.set_title('SGD: Diff of Randomly 100 Weight')
            # plt.plot(W1,'y')
            # plt.plot(W2,'c')
            # plt.legend(['Recover W', 'Original W'])
            # plt.show()
            # fig1.savefig('01_Sgd_Random_Diff.png')
            # plt.close()

            error_W = np.sum((W - mpf_optimizer.W.get_value(borrow = True)[:vis_units,vis_units:])**2)

            mean_batch_error += [error_W/(vis_units*hid_units)]

        mean_epoch_error += [np.mean(mean_batch_error)]

        norm_batch_error += [np.mean(mean_cost)]
        print('The cost for dmpf in epoch %d is %f, rmse is %f' % (epoch, norm_batch_error[-1],mean_batch_error[-1]))

        # if epoch > 0 and epoch %10 ==0 :
        #     learning_rate = learning_rate/2


    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time)

    print ('Training took %f minutes' % (pretraining_time / 60.))

    return mean_epoch_error, mpf_optimizer.W.get_value(borrow= True)[:vis_units,vis_units:],\
           mpf_optimizer.b.get_value(borrow= True)

if __name__ == '__main__':

    W = np.load('rbm_weight_10000.npy')

    print(W[:10,:])

    samples = 'rbm_samples_10000.npy'

    error,W_prime,b_prime = dmpf_sanity_check(W,samples)

    #print(error[-2])

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title('SGD Diff between W and W_prime')
    plt.imshow(np.abs(W - W_prime), extent=[0,10,0,10],aspect = 'auto')
    plt.colorbar()
    plt.show()
    fig1.savefig('01_1000_Sgd_Imageseq.png')


    np.save('wb_0.01_1000_sgd_Wprime.npy',W_prime)
    np.save('wb_0.01_1000_sgd_bprime.npy',b_prime)

    #index = np.random.random_integers(low=0,high=127,size = (100,))

    W1 = W_prime.ravel()
    W2 = W.ravel()
    # W11 = W1[index]
    # W22 = W2[index]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title('SGD: Diff of Randomly 100 Weight')
    plt.plot(W1,'y')
    plt.plot(W2,'c')
    plt.legend(['Recover W', 'Original W'])
    plt.show()
    fig1.savefig('01_Sgd_Random_Diff.png')
    plt.close()

    bias = np.zeros(24)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title('SGD: Diff of bias')
    plt.plot(b_prime,'y')
    plt.plot(bias,'c')
    plt.legend(['Recover b', 'Original b'])
    plt.show()
    fig1.savefig('01_Sgd_Random_Diff_b.png')
    plt.close()

    print(np.sum(b_prime-bias)**2)
