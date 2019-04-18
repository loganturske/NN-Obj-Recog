try:
    import cPickle 
except:
    import _pickle as cPickle 

import gzip
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    f = gzip.open('./mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding='iso-8859-1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():

    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_results = [vectorized_result(y) for y in va_d[1]]
    validation_data = list(zip(validation_inputs, validation_results))

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_results = [vectorized_result(y) for y in te_d[1]]
    test_data = list(zip(test_inputs, test_results))

    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def devectorize_result(e):
    return list(e).index(1.0)

if __name__ is '__main__':

    training_data, validation_data, test_data = load_data_wrapper()

    for im, result in training_data[0:5]:
        plt.imshow(im.reshape([28,28])); plt.show()
        print(devectorize_result(result))
