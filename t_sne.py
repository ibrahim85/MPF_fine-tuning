import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, preprocessing
from KL_mpf import load_mnist, sigmoid
import gzip, pickle, time


dataset = 'mnist.pkl.gz'
f = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = pickle.load(f,encoding="bytes")
f.close()
binarizer = preprocessing.Binarizer(threshold=0.5)
data =  binarizer.transform(train_set[0][:20000])
targets = train_set[1]
print(data.shape)
print(targets.shape)

def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    for i in range(1000):
        plt.text(X[i, 0], X[i, 1], str(targets[i]),
                 color=plt.cm.Set1(targets[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})



def t_sne(weight,bias,hidden_units,perplexity,savename):

    visible_units = data.shape[1]
    W = np.load(weight)
    #W = W[:visible_units,visible_units:]
    b = np.load(bias)
    b = b[visible_units:]
    activation = sigmoid(np.dot(data,W) + b.reshape([1,-1]))

    print(activation.shape)
    print(activation[1])

    tsne = manifold.TSNE(n_components=2, perplexity= perplexity, early_exaggeration=4.0, learning_rate=1000.0,
                         n_iter = 2000, init='pca')

    X_tsne = tsne.fit_transform(activation)

    print(X_tsne.shape)

    #t0 = time()

    plot_embedding(X_tsne,
               "t-SNE embedding of the digits")
    plt.savefig(savename)

    plt.show()

    return X_tsne

def search_tsne():
    perplexity = [5, 10, 20, 30, 50, 70]

    # weight = '../mpf_results/40/weights_499.npy'
    # bias = '../mpf_results/40/bias_499.npy'
    # hidden_units = 40
    #
    # for i in perplexity:
    #     path = '../mpf_results/40/tsne_perp_' + str(i) + '.png'
    #     t_sne(weight=weight,bias=bias,hidden_units=hidden_units, perplexity=i, savename=path)


    weight = '../mpf_results/100/weights_499.npy'
    bias = '../mpf_results/100/bias_499.npy'
    hidden_units = 100

    for i in perplexity:
        path = '../mpf_results/100/tsne_perp_' + str(i) + '.png'
        t_sne(weight=weight,bias=bias,hidden_units=hidden_units, perplexity=i, savename=path)

search_tsne()