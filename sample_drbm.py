import numpy as np
from utils import tile_raster_images
import gzip, pickle, os
import theano
import Image
from utils_mpf import sigmoid


dataset = 'mnist.pkl.gz'
f = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
f.close()

path = '../Thea_mpf/Generated_samples'
if not os.path.exists(path):
    os.makedirs(path)

def generate_from_rbm(W_file, b_file):
    W = np.load(W_file)

    print(W.shape)
    b = np.load(b_file)
    visible_units = W.shape[0]

    b_vis = b[:visible_units].reshape([1,-1])
    b_hid = b[visible_units:].reshape([1,-1])

    n_chains = 20
    n_samples = 10
    rng = np.random.RandomState(123)
    test_set_x = test_set[0]
    number_of_test_samples = test_set_x.shape[0]

    # pick random test examples, with which to initialize the persistent chain
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = np.asarray(test_set_x[test_idx:test_idx + n_chains])
    print(test_set[1][test_idx:test_idx + n_chains])

    # end-snippet-6 start-snippet-7
    plot_every = 100
    image_data = np.zeros(
        (29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8'
    )

    for idx in range(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        v_samples = persistent_vis_chain
        vis_mf = None

        for j in range(plot_every):
            ### Experiment shows that

            upact = sigmoid(np.dot(v_samples,W) + b_hid)
            up_sample = np.random.binomial(n=1, p= upact)
            vis_mf = sigmoid(np.dot(up_sample, W.T) + b_vis)
            v_samples = np.random.binomial(n=1,p=vis_mf)
            #v_samples = vis_mf


        print(' ... plotting sample ', idx)
        image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(28, 28),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )

        # image_binary_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
        #     X=v_samples,
        #     img_shape=(28, 28),
        #     tile_shape=(1, n_chains),
        #     tile_spacing=(1, 1)
        # )

    # construct image
    image = Image.fromarray(image_data)
    # image_binary = Image.fromarray(image_binary_data)
    image.save(path + '/samples.png')
    # image_binary.save(path + '/binary_samples.eps')



W = path + '/weights_499.npy'
b = path + '/bias_499.npy'

generate_from_rbm(W_file=W, b_file=b)
