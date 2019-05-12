import load_data
from Convolutional_Network import CNNetwork
from Network import Network
import numpy as np

training_data, patch_data, validation_data, test_data = load_data.load_data_wrapper()

netff0 = Network([784, 100, 25, 10])
netff0.SGD(training_data[:10000], 10, 10, .5, test_data=test_data[:100])
netcff0 = CNNetwork([100, 25, 10], patch_data, n_clusters=16, patch_size=(8,8), pool_size=(5,5))
netcff0.SGD(training_data[:10000], 10, 10, .5, test_data=test_data[:100], convolve=True)


np.random.seed(seed=0)
altered_training_data = [(load_data.random_maniputlate_image(img), key) for img, key in training_data[:10000]]
altered_test_data = [(load_data.random_maniputlate_image(img), key) for img, key in test_data[:100]]

netcff1 = CNNetwork([100, 25, 10], patch_data, n_clusters=16, patch_size=(8,8), pool_size=(5,5))
netff1 = Network([784, 100, 25, 10])

netcff1.SGD(altered_training_data, 10, 10, .5, test_data=altered_test_data, convolve=True)
