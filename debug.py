import numpy as np
import pickle
import os.path
import gzip
import matplotlib.pyplot as plt


W_prime = np.load('wb_0.01_1000_sgd_Wprime.npy')
W = np.load('rbm_weight_50000.npy')
b_prime = np.load('wb_0.01_1000_sgd_bprime.npy')
#bias = np.load('rmb_bias_50000.npy')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title('Diffrence between W and W_prime')
kk = plt.imshow(np.abs(W_prime - W), extent=[0,16,0,8],aspect = 1)
plt.colorbar()
#kk.set_clim(0.0, .8)
plt.show()
fig1.savefig('Diff_Imageseq.png')
plt.close()

#index = np.random.random_integers(low=0,high=255,size = (100,))

W1 = W_prime.ravel()
W2 = W.ravel()
# W11 = W1[index]
# W22 = W2[index]

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title('Difference of Randomly 100 Weight')
plt.plot(W1,'y')
plt.plot(W2,'c')
plt.legend(['Recover W', 'Original W'])
plt.show()
fig1.savefig('Random_Diff.png')
plt.close()

###################Scale W and W_prime with bias as the Diagonal#######

W_diag = np.identity(100)
W_prime_diag = np.identity(100)

W = W + W_diag
W_prime = W_prime + W_prime_diag
ratio = (np.sum(W/W_prime) - 16) /256
W = W - W_diag
W_prime = W_prime - W_prime_diag
print(ratio)
W_prime  = W_prime * ratio

k = np.abs(W_prime - W)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title('Scaled Diffrence between W and W_prime')
kk= plt.imshow(np.abs(W_prime - W), extent=[0,16,0,16],aspect = 'auto')
plt.colorbar()
#kk.set_clim(0.0, .8)
plt.show()
fig1.savefig('Scaled_Diff_Imageseq.png')
plt.close()


W1 = W_prime.ravel()
W2 = W.ravel()
W11 = W1[index]
W22 = W2[index]

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title('Scaled Difference of Randomly 100 Weight')
plt.plot(W11,'y')
plt.plot(W22,'c')
plt.legend(['Recover W', 'Original W'])
plt.show()
fig1.savefig('Random_Diff.png')
plt.close()

#index = np.random.random_integers(low=0,high=10000,size = (100,))

# W1 = W_prime.ravel()
# W2 = W.ravel()
#
# W11 = W1[index]
# W22 = W2[index]

b_prime1 = np.load('wb_0.01_1000_sgd_bprime.npy')

print(np.sum(b_prime1-bias)**2)
print( np.sum(b_prime1-bias)**2 + np.sum(W_prime-W)**2)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.set_title('Not Scaled Diff of Bias')
plt.plot(b_prime1,'y')
plt.plot(bias,'c')
plt.legend(['Recover b', 'Original b'])
plt.show()
fig1.savefig('Scaled_Random_Diff.png')
plt.close()




