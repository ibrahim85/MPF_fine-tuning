
from np_mpf import *
from KL_mpf import *


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



def forward_all_sgd(epsilon, n_samples, learning_rate, epoches = 1):

    vis_units = 64
    hid_units = 16
    epsilon = epsilon
    learning_rate = learning_rate
    n_samples = n_samples

    path = '../Grid_SGD_filters/num_samples_' + str(n_samples)
    if not os.path.exists(path):
        os.makedirs(path)

    Image = load_IMAGE()

    print(Image.shape)

    mpf_optimizer = KL_dmpf_optimizer(vis_units = vis_units, hid_units= hid_units, epsilon = epsilon)


    batch_sz = 100
    n_batches = Image.shape[0] // batch_sz

    total_cost = []

    # theta = ravel_params(mpf_optimizer.W, mpf_optimizer.b)
    # sample_prob, data = mpf_optimizer.get_samples(theta=theta, data_samples = Image, n_samples = n_samples)
    # print(data.shape)

    for i in range(epoches):
        #
        # theta = ravel_params(mpf_optimizer.W, mpf_optimizer.b)
        # sample_prob, data = mpf_optimizer.get_samples(theta=theta,
        #                                               data_samples = Image, n_samples = n_samples)

        if i % 5 == 0:
            theta = ravel_params(mpf_optimizer.W, mpf_optimizer.b)
            sample_prob, data = mpf_optimizer.get_samples(theta=theta,
                                                          data_samples = Image, n_samples = n_samples)

        if i > 0 and i % 50 ==0 :
            learning_rate = learning_rate / 10
        # if i % 20 == 0:
        #     epsilon += 0.1

        mean_cost = []
        for j in range(n_batches):
            batch_data = data[j*batch_sz : (j+1)*batch_sz, :]
            batch_sample_prob = sample_prob[j*batch_sz:(j+1)*batch_sz]
            cost, grad = mpf_optimizer.get_cost_updates(theta, data=batch_data,
                                                        sample_prob=batch_sample_prob, epsilon=epsilon)
            theta = theta - learning_rate*grad
            mean_cost += [cost]
        total_cost += [np.mean(mean_cost)]
        print('The cost for dmpf in epoch %d is %f'% (i, total_cost[-1]))

        W1,b1 = unravel_params(theta,visible_size= vis_units+hid_units)
        W1 = W1[:vis_units,vis_units:]
        saveName = path + '/weights_' + str(epsilon) + '_' + str(i) + '.png'
        display(W1.T,saveName=saveName)


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

    batch_sz = 100
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
    n_samples = 5
    learning_rate = 0.1
    epoches = 200
    forward_all_sgd(epsilon=epsilon,n_samples=n_samples,learning_rate=learning_rate,epoches=epoches)


