import numpy as np

a= dict()

a['layer_1'] = np.arange(9).reshape(3,3)

a['layer_2'] = np.arange(9).reshape(3,3)

b = None

for k,v in a.items():
    print(v)
    if b is None:
        b = v
    else:
        b = np.concatenate((b,v), axis = 1)

print(b)