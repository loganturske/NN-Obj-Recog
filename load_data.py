import pickle
import gzip

import numpy as np
import matplotlib.pyplot as plt


def load_data():
    f = gzip.open('./mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='bytes', fix_imports=True )
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    #split training data
    patches_data = [im.reshape(28,28) for im, __ in training_data[:1000]]
    training_data = training_data[1000:]
    
    return (training_data, patches_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def devectorize_key(e):
    return list(e).index(1.0)

def random_maniputlate_image(img, shift_lim=5, image_shape=(28,28)):
    img = img.reshape(image_shape)
#    #apply random rotation by 90 deg
#    img = np.rot90(img, np.random.randint(4))
    #apply random translation up to shift_lim pixels in each axis
    img = np.pad(img, shift_lim, 'constant', constant_values=0)
    img = np.roll(img, np.random.randint(-shift_lim, shift_lim), axis=0)
    img = np.roll(img, np.random.randint(-shift_lim, shift_lim), axis=1)
    img = img[shift_lim:-shift_lim, shift_lim:-shift_lim]

    return img.reshape(img.size,1)

if __name__ is '__main__':
    __, __, __, test_data = load_data_wrapper()
    
    plt.figure(figsize=(4.2, 4.5))
    for i, patch in enumerate(test_data[:49]):
        plt.subplot(7, 7, i + 1)
        plt.imshow(patch[0].reshape(28,28), cmap=plt.cm.gray,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

    plt.suptitle('49 MNIST\nHandwritten Digits', fontsize=16)
    plt.show()
    
    np.random.seed(seed=0)
    plt.figure(figsize=(4.2, 4.5))
    for i, patch in enumerate(test_data[:49]):
        plt.subplot(7, 7, i + 1)
        plt.imshow(random_maniputlate_image(patch[0]).reshape(28,28), cmap=plt.cm.gray,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

    plt.suptitle('49 Shifted MNIST\nHandwritten Digits', fontsize=16)
    plt.show()
    
