from EM_MPF import *
from KL_mpf import load_IMAGE


def natural_image_mpf(hidden_units,learning_rate, epsilon, decay =0.001,  batch_sz = 20, dataset = None):

    data = load_IMAGE()

    visible_units = data.shape[1]

    n_train_batches = data.shape[0]//batch_sz

    num_units = visible_units + hidden_units

    W = get_mpf_params(visible_units, hidden_units)

    b = np.zeros(num_units)

    out_epoch = 500
    in_epoch = 1

    index = T.lscalar()    # index to a mini batch
    x = T.matrix('x')

    mpf_optimizer = dmpf_optimizer(
        epsilon=epsilon,
        decay=decay,
        explicit_EM= True,
        num_units = num_units,
        W = W,
        b = b,
        input = x,
        batch_sz =batch_sz)


    new_data  = theano.shared(value=np.asarray(np.zeros((data.shape[0],num_units)), dtype=theano.config.floatX),
                                  name = 'train',borrow = True)

    cost,updates = mpf_optimizer.get_dmpf_cost(
        learning_rate= learning_rate,
        visible_units=visible_units,
        hidden_units=hidden_units)

    train_mpf = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
        x: new_data[index * batch_sz: (index + 1) * batch_sz],
        },
        #on_unused_input='warn',
    )


    mean_epoch_error = []

    path = '../natural_image/hidden_' + str(hidden_units) + '/decay_' + str(decay) + '/lr_' + str(learning_rate) \
           + '/bsz_' + str(batch_sz)
    if not os.path.exists(path):
        os.makedirs(path)

    start_time = timeit.default_timer()

    for em_epoch in range(out_epoch):

        W = mpf_optimizer.W.get_value(borrow = True)
        b = mpf_optimizer.b.get_value(borrow = True)

        prop_W = W[:visible_units, visible_units:]
        prop_b = b[visible_units:]
        activations, sample_data = propup(data,prop_W,prop_b)

        #sample_prob = get_sample_prob(activations) # This is a vector, each entry stands for the probability of
        #the respected sample
        #y = T.vector('y')
	    #new_data.set_value(np.asarray(sample_data, dtype=theano.config.floatX))
        # sample_prob = theano.shared(value = np.asarray(sample_prob, dtype= theano.config.floatX),
        #                             name='prob',borrow = True)
        new_data.set_value(np.asarray(sample_data, dtype=theano.config.floatX))

        for mpf_epoch in range(in_epoch):
            mean_cost = []
            for batch_index in range(n_train_batches):
                mean_cost += [train_mpf(batch_index)]
            mean_epoch_error += [np.mean(mean_cost)]
        print('The cost for mpf in epoch %d is %f'% (em_epoch,mean_epoch_error[-1]))

        if em_epoch % 10 ==0:
            saveName = path + '/weights_' + str(em_epoch) + '.png'
            tile_shape = (int(np.sqrt(hidden_units)), int(np.sqrt(hidden_units)))

                #displayNetwork(W1.T,saveName=saveName)

            image = Image.fromarray(
                tile_raster_images(  X=(mpf_optimizer.W.get_value(borrow = True)[:visible_units,visible_units:]).T,
                            img_shape=(8, 8),
                            tile_shape=tile_shape,
                            tile_spacing=(1, 1)
                        )
                        )
            image.save(saveName)
            # W1 = mpf_optimizer.W.get_value(borrow = True)[:visible_units,visible_units:]
            # displayNetwork(W1.T,saveName=saveName)


    loss_savename = path + '/train_loss.eps'
    show_loss(savename= loss_savename, epoch_error= mean_epoch_error)

    end_time = timeit.default_timer()

    running_time = (end_time - start_time)

    print ('Training took %f minutes' % (running_time / 60.))


if __name__ == '__main__':


    learning_rate_list = [0.001]
    # hyper-parameters are: learning rate, num_samples, sparsity, beta, epsilon, batch_sz, epoches
    # Important ones: num_samples, learning_rate,
    hidden_units_list = [25]
    n_samples_list = [1]
    beta_list = [0]
    sparsity_list = [.1]
    batch_list = [40]
    decay_list = [0.001]

    for batch_size in batch_list:
        for n_samples in n_samples_list:
            for hidden_units in hidden_units_list:
                for decay in decay_list:
                    for learning_rate in learning_rate_list:
                            natural_image_mpf(hidden_units = hidden_units,learning_rate = learning_rate, epsilon = 0.01,decay=decay,
                                   batch_sz=batch_size)