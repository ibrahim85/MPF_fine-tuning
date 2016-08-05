import pickle

def load(filename):
    with open(filename,'rb') as f :
        bob = pickle.load(f,encoding="bytes")
    f.close()
    return bob


def save(filename, bob):
    f = open(filename,'wb')
    pickle.dump(bob,f)
    f.close()

