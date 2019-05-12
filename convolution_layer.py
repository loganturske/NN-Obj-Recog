# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:11:58 2019

@author: jdixon
"""
import random
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d
from scipy import signal
import load_data
import time
import matplotlib.pyplot as plt

class Convolution_Layer(object):
    '''
    packaing funtion abocve into a useful object
    '''
    def __init__(self, training_set, n_clusters, patch_size):
        self.training_set = training_set
        self.build_kernls(n_clusters, patch_size=patch_size)

    
    def build_kernls(self, n_clusters, patch_size, max_patches=10, epocs=6, batch_size=25, seed=0, rs=0):
        self.n_clusters = n_clusters
        self.patch_size=patch_size
        self.max_patches=max_patches
        self.epocs=epocs
        self.batch_size=batch_size
        self.seed=seed
        self.rs=rs
        self.kernels = self.get_convolutional_kernels_kmeans_clusters(self.training_set,
                                                                      n_clusters=self.n_clusters,
                                                                      patch_size=self.patch_size,
                                                                      max_patches=self.max_patches,
                                                                      epocs=self.epocs,
                                                                      batch_size=self.batch_size,
                                                                      seed=self.seed,
                                                                      rs=self.rs)
    
    def feed_forward(self, image, pool=(5,5), image_size=(28,28)):
        self.pool_bins = pool
        self.image_size = image_size
        self.convolutions = self.convolution_layer(image, self.image_size, self.kernels)
        self.pool = self.pool_layer(self.convolutions, self.pool_bins)
        self.output = [self.sigmoid(p) for p in self.pool]
        self.vector_out = self.vectorize(self.output)
        return self.vector_out

    def convolution_layer(self, image, image_size, convolution_kernels,
                          operation='convolve'):
        '''
        Implentatnion of convolutinal layer using [scipy.signal.convolve2d] or
        scipy.signal.correlate2d as the convolution operation
        
        inputs:
            image: vectorized 1d numpy array
            image_size: tuple containing dimensions of image
            convolution_kernels: list of image patches(2d numpy arrays) to be used as convolutonal kernels
            operation{'convolve'}:  'correlate' or 'convolve' whcih convolution operation to use
        
        output:
            list of convolution outputs (2d numpy arrays)
        
        Written J. Dixon 5-3-2019
        '''
        # reshape image initalize output
        img = image.copy()
        img = img.reshape(image_size)
        output = []
        
        # loop over convolutional kernls
        for kernel in convolution_kernels:
            if operation=='correlate':
                convolutional_output = signal.correlate2d(img, kernel, mode='valid')
            elif operation=='convolve':
                convolutional_output = signal.convolve2d(img, kernel, mode='valid')
            else:
                print('invalid convolutional method Try: {"correlate" or "convolve"}')
                return 
            
            
            output.append(convolutional_output)
        
        return output  
    
    def pool_layer(self, convolutions, bin_size):
        '''
        Implentatnion of convolutinal max pooling layer
        
        inputs:
            convolutions
            bin_size: tuple of bin size for X and Y 
        
        output:
            list of pooled convolution outputs (2d numpy arrays)
        
        Written J. Dixon 5-3-2019
        '''
        # upack and initilize outputs
        M, N = bin_size
        output = []
        # Pool each convolution 
        for conv in convolutions:
            x_bins = range(0, conv.shape[0], M)
            y_bins = range(0, conv.shape[1], N)
            
            #symetricly extend convoltion to avoid bin size errors
            dummy_conv = np.pad(conv,((0,M),(0,N)), 'symmetric')
            out = np.zeros([len(x_bins), len(y_bins)])
            
            # bin outputs 
            for i, x in enumerate(x_bins):
                for j, y in enumerate(y_bins):
                    out[i,j]= np.max(dummy_conv[x:x+M, y:y+M])

            output.append(out)
        return output
    
    def vectorize(self, list_arrays):
        '''
        Vectorizes a lsit of numpy arrays to a n,1 numpy array
        
        Written J. Dixon 5-3-2019
        '''
        input_array = np.stack(list_arrays)
        return input_array.reshape(input_array.size,1)
        
    def sigmoid(self, z):
    	return 1.0/(1.0+np.exp(-z))
    
    def get_convolutional_kernels_kmeans_clusters(self, image_set, n_clusters, patch_size, max_patches, epocs, batch_size,  seed, rs):
        '''    
        This unsupervised traning of the convolutional kernels adaped form scikit-leran 
        "online learning of a dictonary of parts of faces" totorial found at
        https://scikit-learn.org/stable/auto_examples/cluster/plot_dict_face_patches.html#sphx-glr-auto-examples-cluster-plot-dict-face-patches-py
        
        Bibtex entry for scikit-learn in a scientific publication
        @article{scikit-learn,
         title={Scikit-learn: Machine Learning in {P}ython},
         author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
                 and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
                 and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
                 Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
         journal={Journal of Machine Learning Research},
         volume={12},
         pages={2825--2830},
         year={2011}
        }
        
        uses sklearn.cluster.MiniBatchKMeans to find the cluster centers of 
        the patches extracted by sklearn.feature_extraction.image.extract_patches_2d
        
        also depenednt on numpy
        
        inputs:
            image set: for training (list of 2d numpy arrays)
            n_clusters{49}: nubmer of clusters to be found
            patch_size{(7, 7)}: size of image patches
            max_patches{50}: maximumn number of patches extracted from each inage
            epocs{6}: number of epocs
            batch_size{10}: number of images to use for a training cycle
            seed{0}: seed for load_data.random_maniputlate_image
            rs{0}: seed for random state for MiniBatchKMeans and extract_patches_2d
        
        output:
            kernels: list of image patches(2d numpy arrays) to be used as convolutonal kernels
        
        written by J. Dixon 5-3-2019    
        '''
        
        # seed ranodom staes and initilize K-means ovject, buffers, index
        rng = np.random.RandomState(rs)
        np.random.seed(seed=seed)
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=rng)
        buffer = []
        index = 0
        #begin unsupervised training on the test set
        random.seed(0)
        for _ in range(epocs):
            random.shuffle(image_set)
            for img in image_set:
                # extract image patches from training image and add to buffer
                data = extract_patches_2d(img.reshape(28,28), patch_size, max_patches=max_patches, random_state=rng)
                data = np.reshape(data, (len(data), -1))
                buffer.append(data)
                index += 1 
                
                # fit data when batch size criteria is met   
                if index % batch_size  == 0:
                    data = np.concatenate(buffer, axis=0)
                    # fit data using kmeans partial fit and clear buffer
                    kmeans.partial_fit(data)
                    buffer = []
        # extract and cluster centers as list of convolutional kernles
        kernels = [self.hlf((patch.reshape(patch_size)-np.min(patch))/(np.max(patch) - np.min(patch)),0.5) for patch in kmeans.cluster_centers_]
        return kernels
    
    def hlf(self, in_array, thr=0.0):
        array_out = in_array.copy()
        array_out[array_out > thr] = 1
        array_out[array_out <= thr] = -1
        return array_out

if __name__ is '__main__':
    '''
    make figures for convolutional layre
    '''
    __, patches_data, __, test_data = load_data.load_data_wrapper()
    
#%%
    np.random.seed(seed=0)
    test_data = [(load_data.random_maniputlate_image(img.reshape(28,28)), key) for img, key in test_data[:49]]
    

    t0 = time.time()
    cnn = Convolution_Layer(patches_data, 25, (8,8))
    dt = time.time() - t0

        
    plt.figure(figsize=(4.2, 4.5))
    for i, patch in enumerate(cnn.kernels):     
        plt.subplot(int(np.sqrt(cnn.n_clusters)), int(np.sqrt(cnn.n_clusters)), i + 1)
        plt.imshow(patch, cmap=plt.cm.gray,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

    plt.suptitle('%d (%dx%d) MNIST Digits Patches\nTrain time %.1fs from %d Images' %
                 (cnn.n_clusters, cnn.patch_size[0], cnn.patch_size[1], dt, cnn.epocs * len(cnn.training_set)), fontsize=16)
    plt.show()
#%%
    for img, key in test_data[0:1]:
#        plt.figure(figsize=(4.2, 4.5))
#        plt.imshow(img.reshape(28,28), cmap=plt.cm.gray,
#                   interpolation='nearest')
#        plt.suptitle('Randomly adjusted MNIST\n Digits %d' % key, fontsize=16)
#        plt.xticks(())
#        plt.yticks(())
#        plt.show()        
        cnn.feed_forward(img)
        plt.figure(figsize=(4.2, 4.5))
        for i, patch in enumerate(cnn.convolutions):
            plt.subplot(int(np.sqrt(cnn.n_clusters)), int(np.sqrt(cnn.n_clusters)), i + 1)
            plt.imshow(patch, cmap=plt.cm.gray,
                       interpolation='nearest')
            plt.xticks(())
            plt.yticks(())          
        plt.suptitle('cnn _convolutional for digit %d' %
                     key, fontsize=16)        
        plt.figure(figsize=(4.2, 4.5))
        for i, patch in enumerate(cnn.pool):
            plt.subplot(int(np.sqrt(cnn.n_clusters)), int(np.sqrt(cnn.n_clusters)), i + 1)
            plt.imshow(patch, cmap=plt.cm.gray,
                       interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
        plt.suptitle('cnn _pool for digit %d' %
                     key, fontsize=16)
        
        
        plt.figure(figsize=(4.2, 4.5))
        for i, patch in enumerate(cnn.output):
            plt.subplot(int(np.sqrt(cnn.n_clusters)), int(np.sqrt(cnn.n_clusters)), i + 1)
            plt.imshow(patch, cmap=plt.cm.gray,
                       interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
        plt.suptitle('cnn _activation for digit %d' %
                     key, fontsize=16)
        
