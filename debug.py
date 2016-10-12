import numpy as np
import pickle
import os.path
import gzip
import matplotlib.pyplot as plt


W_prime = np.load('W_prime.npy')
W = np.load('gibbs_weight.npy')
b_prime = np.load('b_prime.npy')
bias = np.load('gibbs_bias.npy')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title('Diffrence between W and W_prime')
kk = plt.imshow(np.abs(W_prime - W), extent=[0,100,0,100],aspect = 1)
plt.colorbar()
kk.set_clim(0.0, .8)
plt.show()
fig1.savefig('Diff_Imageseq.png')
plt.close()

index = np.random.random_integers(low=0,high=10000,size = (100,))

W1 = W_prime.ravel()
W2 = W.ravel()
W11 = W1[index]
W22 = W2[index]

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title('Difference of Randomly 100 Weight')
plt.plot(W11,'y')
plt.plot(W22,'c')
plt.legend(['Recover W', 'Original W'])
plt.show()
fig1.savefig('Random_Diff.png')
plt.close()

###################Scale W and W_prime with bias as the Diagonal#######

W_diag = np.identity(100)
W_prime_diag = np.identity(100)

W = W + W_diag
W_prime = W_prime + W_prime_diag
ratio = (np.sum(W/W_prime) - 100) /10000
W = W - W_diag
W_prime = W_prime - W_prime_diag
print(ratio)
W_prime  = W_prime * ratio

k = np.abs(W_prime - W)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title('Scaled Diffrence between W and W_prime')
kk= plt.imshow(np.abs(W_prime - W), extent=[0,100,0,100],aspect = 'auto')
plt.colorbar()
kk.set_clim(0.0, .8)
plt.show()
fig1.savefig('Scaled_Diff_Imageseq.png')
plt.close()

#index = np.random.random_integers(low=0,high=10000,size = (100,))

W1 = W_prime.ravel()
W2 = W.ravel()

W11 = W1[index]
W22 = W2[index]

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title('Scaled Diff of Randomly 100 Weight')
plt.plot(W11,'y')
plt.plot(W22,'c')
plt.legend(['Recover W', 'Original W'])
plt.show()
fig1.savefig('Scaled_Random_Diff.png')
plt.close()




