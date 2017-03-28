
from np_mpf import *
from KL_mpf import *
from adam import get_adam


'''
This is the SGD implementation of the np_mpf.
Basically, there are two types of sgd, the first one is: feedforward all the data, train for several epoches, and
do it alternatively,
The second one is, feedforward with the new parameters in each sgd step.
'''


#
# def adam(learning_rate, grad)
#
#     return new_grad
#
#
# def rms_prop(learning_rate, grad):
#
#
#
#     return new_grad



def forward_all_sgd(epsilon, n_samples, learning_rate, epoches = 1, beta=3, sparsity = 0.1, hidden_units = 200, batch_size=20, decay = 0.001):

    vis_units = 784
    hid_units = hidden_units
    epsilon = epsilon
    learning_rate = learning_rate
    n_samples = n_samples
    print(batch_size)

    path = '../Grid_Adam_SGD_filters/num_samples_' + str(n_samples) + '/hidden_' + str(hidden_units) + '/decay_' + str(decay)  + '/lr_' + str(learning_rate)
    #'/batch_' + str(batch_size)+
           #'/hidden_' + str(hidden_units) + '/beta_' + str(beta) + '/sparsity_' +str(sparsity)
    if not os.path.exists(path):
        os.makedirs(path)
    print(path)

    Image = load_mnist()

    print(Image.shape)

    mpf_optimizer = KL_dmpf_optimizer(vis_units = vis_units, hid_units= hid_units, epsilon = epsilon)


    batch_sz =batch_size
    n_batches = Image.shape[0] // batch_sz

    total_cost = []

    # theta = ravel_params(mpf_optimizer.W, mpf_optimizer.b)
    # sample_prob, data = mpf_optimizer.get_samples(theta=theta, data_samples = Image, n_samples = n_samples)
    # print(data.shape)
    m = 0
    v = 0
    eps = 1e-8
    beta_1_power = 0.9
    beta_2_power = 0.999

    theta = ravel_params(mpf_optimizer.W, mpf_optimizer.b)
    sample_prob, data = mpf_optimizer.get_samples(theta=theta,
                                                         data_samples = Image, n_samples = n_samples)

    for i in range(epoches):
        #
        # theta = ravel_params(mpf_optimizer.W, mpf_optimizer.b)
        # sample_prob, data = mpf_optimizer.get_samples(theta=theta,
        #                                               data_samples = Image, n_samples = n_samples)


        if i >0 and i % 1 == 0:
            theta = ravel_params(mpf_optimizer.W, mpf_optimizer.b)
            sample_prob, data = mpf_optimizer.get_samples(theta=theta,
                                                              data_samples = Image, n_samples = n_samples)

        # if i > 0 and i % 30 ==0 :
        #     learning_rate = learning_rate / 2
        # if i % 20 == 0:
        #     epsilon += 0.1

        mean_cost = []

        for j in range(n_batches):
            batch_data = data[j*batch_sz : (j+1)*batch_sz, :]
            batch_sample_prob = sample_prob[j*batch_sz:(j+1)*batch_sz]
            cost, grad = mpf_optimizer.get_cost_updates(theta, data=batch_data,
                                    sample_prob=batch_sample_prob, epsilon=epsilon, beta=beta,sparsityParam=sparsity, decay= decay)


            m, v, beta_1_power, beta_2_power = get_adam(grad, m_t=m,  v_t=v, beta_1_power=beta_1_power,
                                                        beta_2_power=beta_2_power)

            lr = learning_rate * np.sqrt(1 - beta_2_power) / (1 - beta_1_power)

            theta = theta - lr * m / (np.sqrt(v) + eps)
            # theta = theta - learning_rate * grad


            mean_cost += [cost]
        total_cost += [np.mean(mean_cost)]
        print('The cost for dmpf in epoch %d is %f'% (i, total_cost[-1]))

        W1,b1 = unravel_params(theta,visible_size= vis_units+hid_units)
        W1 = W1[:vis_units,vis_units:]
        saveName = path + '/weights_' + str(epsilon) + '_' + str(i) + '.png'
        display(W1.T,saveName=saveName)

        # if i > 2:
        #     improve = (total_cost[-2] - total_cost[-1]) / total_cost[-2]
        #     if  improve < 0.001 * total_cost[-2]:
        #         theta = ravel_params(mpf_optimizer.W, mpf_optimizer.b)
        #         sample_prob, data = mpf_optimizer.get_samples(theta=theta,
        #                                                        data_samples = Image, n_samples = n_samples)
        #         print('Start a new sampling process at epoch %d .................' % (i))
        if i % 99 == 0:
            W1,b1 = unravel_params(theta,visible_size= vis_units+hid_units)
            W1 = W1[:vis_units,vis_units:]
            saveName_w = path + '/weights_' + str(i) + '.npy'
            saveName_b = path + '/bias_' + str(i) + '.npy'
            np.save(saveName_w,W1)
            np.save(saveName_b,b1)



def forward_batch_sgd(epsilon, n_samples, learning_rate, epoches):

    vis_units = 64
    hid_units = 16
    epsilon = epsilon
    learning_rate = learning_rate
    n_samples = n_samples

    path = '../Grid_SGD_filters/num_samples_' + str(n_samples)
    if not os.path.exists(path):
        os.makedirs(path)
    Image = load_IMAGE()

    mpf_optimizer = KL_dmpf_optimizer(vis_units = vis_units, hid_units= hid_units, epsilon = epsilon)

    batch_sz = 20
    n_batches = Image.shape[0] // batch_sz
    total_cost = []

    for i in range(epoches):
        theta = ravel_params(mpf_optimizer.W, mpf_optimizer.b)
        mean_cost = []
        for j in range(n_batches):
            batch_data = Image[j*batch_sz: (j+1)*batch_sz]
            batch_sample_prob, batch_data = mpf_optimizer.get_samples(theta=theta,
                                                        data_samples = batch_data, n_samples = n_samples)
            cost, grad = mpf_optimizer.get_cost_updates(theta, data=batch_data,
                                                        sample_prob=batch_sample_prob, epsilon= epsilon)
            theta = theta - learning_rate*grad
            mean_cost += [cost]

        total_cost += [np.mean(mean_cost)]
        print('The cost for dmpf in epoch %d is %f'% (i, total_cost[-1]))

        W1,b1 = unravel_params(theta,visible_size= vis_units+hid_units)
        W1 = W1[:vis_units,vis_units:]
        saveName = path + '/weights_' + str(epsilon) + '_' + str(i) + '.png'
        display(W1.T,saveName=saveName)



if __name__ == '__main__':

    epsilon = 0.01
    n_samples = 1
    learning_rate_list = [0.001, 0.0008, 0.002, 0.0006]
    # hyper-parameters are: learning rate, num_samples, sparsity, beta, epsilon, batch_sz, epoches
    # Important ones: num_samples, learning_rate,
    epoches = 500
    n_samples_list = [1]
    hidden_units_list = [200]
    beta_list = [0]
    sparsity_list = [.1]
    batch_list = [20]
    decay_list = [0.001]

    for batch_size in batch_list:
        for n_samples in n_samples_list:
            for decay in decay_list:
                for hidden_units in hidden_units_list:
                    for learning_rate in learning_rate_list:
                        for beta in beta_list:
                            if beta !=0:
                                for sparsity in sparsity_list:
                                    forward_all_sgd(epsilon=epsilon,n_samples=n_samples,learning_rate=learning_rate,epoches=epoches,
                                            beta=beta, sparsity= sparsity, hidden_units=hidden_units, batch_size=batch_size, decay=decay)
                            else:
                                forward_all_sgd(epsilon=epsilon,n_samples=n_samples,learning_rate=learning_rate,epoches=epoches,
                                            beta=beta,hidden_units=hidden_units, batch_size=batch_size, decay=decay)


