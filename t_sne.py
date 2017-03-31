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
data =  binarizer.transform(train_set[0])
targets = train_set[1]
print(data.shape)
print(targets.shape)

def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(targets[i]),
                 color=plt.cm.Set1(targets[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})


def t_sne(weight,bias,hidden_units):

    visible_units = data.shape[1]
    W = np.load(weight)
    W = W[:visible_units,visible_units:]
    b = np.load(bias)
    b = b[visible_units:]
    activation = sigmoid(np.dot(data,W) + b.reshape([1,-1]))

    tsne = manifold.TSNE(n_components=2, perplexity= 40, random_state=0)

    X_tsne = tsne.fit_transform(activation)

    t0 = time()

    plot_embedding(X_tsne,
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))

    plt.show()


