import numpy as np

import matplotlib.pyplot as plt


data = np.load('rbm_samples_50000.npy')

print(data[:10])

a = [1,2,4,5,1,1,1,1,1,1]
x = np.arange(len(a))
plt.plot(x,a, 'r')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.title('Adam Training Loss')
plt.grid(True)
#plt.show()
plt.savefig('myfig.eps')
