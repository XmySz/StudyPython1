import random

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
data = sio.loadmat('ex3data1.mat')

X = data['X']
y = data['y']

def show_data():
    m = X.shape[0]
    n = X.shape[1]
    flags = np.zeros((m,), bool)
    res = False
    for i in range(100):
        index = random.randint(0, m - 1)
        while flags[index]:
            index = random.randint(0, m)
        if type(res) == bool:
            res = X[index].reshape(1, n)
        else:
            res = np.concatenate((res, X[index].reshape(1, n)), axis=0)

    image_dimension = int(np.sqrt(X.shape[-1]))
    image = False
    im = False
    for i in res:
        if type(image) == bool:
            image = i.reshape(image_dimension, image_dimension)
        else:
            if image.shape[-1] == 200:
                if type(im) == bool:
                    im = image
                else:
                    im = np.concatenate((im, image), axis=0)
                image = i.reshape(image_dimension, image_dimension)
            else:
                image = np.concatenate((image, i.reshape(image_dimension, image_dimension)), axis=1)

    s = np.concatenate((im, image), axis=0)
    plt.imshow(s.T)
    plt.axis('off')
    plt.show()

show_data()