# -*- coding: utf-8 -*-

"""
Created on Mon Jul 21 09:53:29 2014

@author: Sai Ganesh (edited By Huiling, Shaowei)
"""

from __future__ import division
import numpy as np
from numpy.linalg import norm
from numpy.random import rand
from numpy.random import randint
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import PIL.Image
import os
import struct
from scipy.optimize import fmin_l_bfgs_b as minimize
import os, struct
from array import array
from cvxopt.base import matrix

## ---------------------------------------------------------------

def read(digits, dataset, path = "."):
    """
    Python function for importing the MNIST data set.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')


    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    #images =  matrix(0, (len(ind), rows*cols))
    #labels = matrix(0, (len(ind), 1))
    N = len(ind)
    images = np.zeros((N, rows*cols))
    labels = np.zeros((N, 1))
    for i in range(len(ind)):
        # images[i, :] = img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]
        # labels[i] = lbl[ind[i]]
        images[i] = np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])#.reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

def sampleIMAGES(patchsize, dataset):
    # sampleIMAGES 
    # Returns 10000 patches for training
    digits = np.arange(10)

    IMAGES, labels = read(digits, dataset, path = ".")  # load images from disk

    numpatches = len(labels)

    # Initialize patches with zeros.  Your code will fill in this matrix--one
    # row per patch, 10000 rows. 
    #patches = np.zeros((numpatches, patchsize*patchsize))
    patches = IMAGES[0:numpatches, 0:patchsize*patchsize]
    labels = labels[0:numpatches]
    ## ---------- YOUR CODE HERE --------------------------------------
    #  Instructions: Fill in the variable called "patches" using data 
    #  from IMAGES.  
    #
    #  IMAGES is a 3D array containing 10 images,
    #  and Python indexes arrays by starting from 0.
    #  For instance, IMAGES[:,:,0] is a 512x512 array containing the 1st image,
    #  and to visualize it you can type 
    #      from matplotlib import pyplot as plt
    #      plt.imshow(IMAGES[:,:,0])
    #      plt.set_cmap('gray')
    #      plt.show()
    #  (The contrast on these images look a bit off because they have
    #  been preprocessed using using "whitening."  See the lecture notes for
    #  more details.) As a second example, IMAGES[20:30,20:30,1] is an image
    #  patch corresponding to the pixels in the block (20,20) to (29,29) of
    #  Image 2
    # for i in range(0,numpatches):
    #     pix_index = np.random.randint(504,size=2)
    #     im_index = np.random.randint(10)
    #     patches[i,:] = patches[i,:] + IMAGES[pix_index[0]:pix_index[0]+8,pix_index[1]:pix_index[1]+8,im_index].reshape(1,64)


    ## ---------------------------------------------------------------
    # For the autoencoder to work well we need to normalize the data
    # Specifically, since the output of the network is bounded between [0,1]
    # (due to the sigmoid activation function), we have to make sure 
    # the range of pixel values is also bounded between [0,1]
    patches = normalizeData(patches)
    return patches, labels


def partition_data(data, label, digit):

    ## we firstly merge the data and label into a ndarray, and then
    #partition it into two sub-array according to the labels,
    # and then split the two sub-array into data and label again.

    mergedata = np.concatenate((label,data), axis = 1)


    mergeLabeled = mergedata[mergedata[:,0] < digit]
    mergeUnlabeled  = mergedata[mergedata[:,0] >= digit]

    trainUnlabeled = np.delete(mergeUnlabeled, 0, 1)

    # did not randomly choose samples here, check will this partition work or not
    numTrain = int(mergeLabeled.shape[0]/2)

    train = mergeLabeled[0:numTrain,:]
    test = mergeLabeled[numTrain:,:]

    labelTrain = train[:,0]
    labelTest = test[:,0]

    trainLabeled = np.delete(train,0,1)
    test = np.delete(test,0,1)

    return trainUnlabeled, labelTrain, trainLabeled, test, labelTest

## ---------------------------------------------------------------
def normalizeData(patches):
    # Squash data to [0.1, 0.9] since we use sigmoid as the activation
    # function in the output layer
    
    # Remove DC (mean of images). 
    patches = patches-np.array([np.mean(patches,axis=1)]).T
        
    # Truncate to +/-3 standard deviations and scale to -1 to 1
    pstd = 3*np.std(patches)
    patches = np.fmax(np.fmin(patches,pstd),-pstd)/pstd
    
    # Rescale from [-1,1] to [0.1,0.9]
    patches = (patches+1)*0.4+0.1    
    return patches


## ---------------------------------------------------------------
def sparseAutoencoderCost(theta,visibleSize,hiddenSize,decayWeight,sparsityParam,beta,data):
    # visibleSize: the number of input units (probably 64) 
    # hiddenSize: the number of hidden units (probably 25) 
    # decayWeight: weight decay parameter lambda
    # sparsityParam: The desired average activation for the hidden units (denoted in the lecture
    #                           notes by the greek alphabet rho, which looks like a lower-case "p").
    # beta: weight of sparsity penalty term
    # data: Our 10000x64 matrix containing the training data.  So, data[i-1,:] is the i-th training example. 
      
    # The input theta is a vector (because scipy.optimize.fmin_l_bfgs_b expects the parameters to be a vector). 
    # We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
    # follows the notation convention of the lecture notes.     
    W1,W2,b1,b2 = unravelParameters(theta,hiddenSize,visibleSize)

    # Cost and gradient variables (your code needs to compute these values). 
    # Here, we initialize them to zeros. 
    cost = 0
    W1grad = np.zeros(np.shape(W1))
    W2grad = np.zeros(np.shape(W2))
    b1grad = np.zeros(np.shape(b1))
    b2grad = np.zeros(np.shape(b2))
    # print('Original weight and bias done....................................')
    ## ---------- YOUR CODE HERE --------------------------------------
    #  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
    #                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
    #
    # W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
    # Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
    # as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
    # respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
    # with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
    # [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
    # of the lecture notes (and similarly for W2grad, b1grad, b2grad).
    # 
    # Stated differently, if we were using batch gradient descent to optimize the parameters,
    # the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2
    #for i in range(0,data.shape[0]):
    z2 = np.dot(data, W1) + b1
    a2 = sigmoid(z2)
    storeActivation = np.sum(a2, axis = 0)
    rho = storeActivation/data.shape[0]
    z3 = np.dot(a2,W2) + b2 #
    a3 = sigmoid(z3)  #
    delta_3 = -(data - a3) * (a3*(1-a3))
    KL = beta*((-sparsityParam/rho) + ((1 - sparsityParam)/(1 - rho)))

    delta_2 = (np.dot(delta_3,W2.T) + KL )* (a2*(1-a2))

    dev_w2 = np.dot(a2.T,delta_3)
        #print(dev_w2)#25*64
    dev_w1 = np.dot(data.T,delta_2)

    W1grad = dev_w1/data.shape[0] + decayWeight*W1
    W2grad = dev_w2/data.shape[0] + decayWeight*W2
    b1grad = np.sum(delta_2, axis = 0)/data.shape[0]
    b2grad = np.sum(delta_3, axis = 0)/data.shape[0]


    cost = 0.5*np.sum((data - a3)**2)/data.shape[0] + 0.5*decayWeight*(np.sum(W1*W1)+np.sum(W2*W2))
    k = np.sum(sparsityParam*np.log(sparsityParam/rho)+(1 - sparsityParam)*np.log((1 - sparsityParam)/(1 - rho)))
    cost = cost + beta*k
    #-------------------------------------------------------------------
    # After computing the cost and gradient, we will convert the gradients back
    # to a vector format (suitable for scipy.optimize.fmin_l_bfgs_b).  
    grad = ravelParameters(W1grad,W2grad,b1grad,b2grad)
    return cost,grad


def feedForward(samples, W, b):
        # this function returns the hidden layer activations
        z2 = np.dot(samples,W) + b
        a2 = sigmoid(z2)
        return a2

def softmaxCost(theta, numClasses, inputSize,label, decayWeight,data):
    # initialize theta firstly
    # While theta should be [numClasses, inputSize]
    # label: the labels of the input data, which is in the range of [0,9], and the size is 10000*1
    # numClasses is 10
    # inputSize is 28*28
    # data: data is the input training data, and the dimension is 10000*inputSize
    #
    theta = theta.reshape(numClasses,inputSize+1)
    y = np.zeros((data.shape[0], numClasses))

    for i in range(len(label)):
        k = np.zeros(numClasses)
        k[label[i,0]] = 1
        y[i] = y[i] + k


    theta_1 =  np.dot(data, theta.T)
    theta_1 = theta_1 - np.amax(theta_1, axis = 1).reshape(data.shape[0],1)
    prob = np.exp(theta_1) # 10000*724 * 724*10 = 10000*10
    sum_prob = np.sum(prob, axis = 1).reshape(data.shape[0],1) # 10000*1
    prob = prob/sum_prob  #10000*10
    delta = prob - y
    grad = np.zeros(theta.shape)

    for i in range(numClasses):
        grad[i] = np.sum(data * delta[:,i].reshape(data.shape[0],1), axis = 0)/data.shape[0]

    grad = grad + decayWeight*theta

    prob = np.log(prob)
    cost = y*prob
    cost = -np.sum(cost)/y.shape[0] + 0.5*decayWeight*np.sum(theta**2)

    #cost = -np.sum(y * np.log(prob))/data.shape[0]
    grad = grad.reshape(1,numClasses*(inputSize+1))
    grad = np.asfortranarray(grad)

    return cost, grad


## ---------------------------------------------------------------

def predictSoftmax(theta,data,label, numClasses,inputSize):

    theta = theta.reshape(numClasses,inputSize+1)
    y = np.zeros((data.shape[0], numClasses))

    for i in range(len(label)):
        k = np.zeros(numClasses)
        k[label[i]] = 1
        y[i] = (y[i] + k).astype(int)

    theta_1 =  np.dot(data, theta.T)
    theta_1 = theta_1 - np.amax(theta_1, axis = 1).reshape(data.shape[0],1)
    prob = np.exp(theta_1) # 10000*724 * 724*10 = 10000*10
    sum_prob = np.sum(prob, axis = 1).reshape(data.shape[0],1) # 10000*1
    prob = prob/sum_prob  #10000*10
    predict = prob/np.amax(prob, axis = 1).reshape(data.shape[0],1)
    predict = (predict == 1.0 ).astype(float)
    k = 0
    for i in range(len(label)):
        if np.array_equal(predict[i,:],y[i,:]):
            k = k+1
    correctness = k/(len(label))
    return correctness

## ---------------------------------------------------------------
def finetuning(theta,data,label,decayWeight,hiddenSizeL1,hiddenSizeL2,visibleSize,numClasses):

    thetaSoftmax, W11, W21, b11, b21 = unravelDeepParameters(theta,visibleSize,hiddenSizeL1,hiddenSizeL2,numClasses)


    cost = 0
    W1grad = np.zeros(np.shape(W11))
    W2grad = np.zeros(np.shape(W21))
    b1grad = np.zeros(np.shape(b11))
    b2grad = np.zeros(np.shape(b21))
    softmaxgrad = np.zeros(np.shape(thetaSoftmax))

    #-----------------------------------------------
    ## Start computing the gradient here, since we already have the output prob. we do not need to
    #compute it again.
    #Step1: we compute the delta_3 for the softmax classifier layer
    #Think z4 here is theta.T*x, and a4 = prob
    y = np.zeros((data.shape[0], numClasses))
    for i in range(len(label)):
        k = np.zeros(numClasses)
        k[label[i,0]] = 1
        y[i] = (y[i] + k).astype(int)

    z2 = np.dot(data, W11) + b11
    a2 = sigmoid(z2)
    z3 = np.dot(a2,W21) + b21 #
    a3 = sigmoid(z3)
    #Note that a3 has  10000*201 dimension, since softmax added a scalar x_0 into the input layer
    #To compute delta_2, we only need the 200 hidden units but not the scalar unit

    intercept = np.ones(a3.shape[0])
    intercept = intercept.reshape(a3.shape[0],1)
    a3_1 = np.concatenate((intercept, a3),axis=1)

    #thetaSoftmax has 10*201 size, so, we delete the first column, which refers to the x_0
    thetaSoftmax_1 = thetaSoftmax[:,1:]

    theta_1 =  np.dot(a3_1, thetaSoftmax.T)
    theta_1 = theta_1 - np.amax(theta_1, axis = 1).reshape(data.shape[0],1)
    prob = np.exp(theta_1) # 10000*724 * 724*10 = 10000*10
    sum_prob = np.sum(prob, axis = 1).reshape(data.shape[0],1) # 10000*1
    prob = prob/sum_prob

    delta_4 = prob - y # 10000*10
    delta_3 = np.dot(delta_4,thetaSoftmax_1)*(a3*(1-a3))
    delta_2 = np.dot(delta_3,W21.T,)*(a2*(1-a2))

    ##Now we compute the gradient for each layer
    for i in range(numClasses):
        softmaxgrad[i] = np.sum(a3_1 * delta_4[:,i].reshape(a3_1.shape[0],1), axis = 0)/a3_1.shape[0]

    softmaxgrad = softmaxgrad + decayWeight*thetaSoftmax

    W2grad = np.dot(a2.T,delta_3)/data.shape[0]
    W1grad = np.dot(data.T,delta_2)/data.shape[0]
    b1grad = np.sum(delta_2, axis = 0)/data.shape[0]
    b2grad = np.sum(delta_3, axis = 0)/data.shape[0]

    grad = ravelDeepParameters(W1grad,W2grad,b1grad,b2grad,softmaxgrad)

    prob = np.log(prob)
    cost = y*prob
    cost = -np.sum(cost)/y.shape[0] + 0.5*decayWeight*np.sum(thetaSoftmax**2)

    return cost,grad

## ---------------------------------------------------------------
def predictFinetuning(theta,data,label,visibleSize,hiddenSizeL1,hiddenSizeL2,numClasses):

    thetaSoftmax, W11, W21, b11, b21 = unravelDeepParameters(theta,visibleSize,hiddenSizeL1,hiddenSizeL2,numClasses)
    y = np.zeros((data.shape[0], numClasses))

    for i in range(len(label)):
        k = np.zeros(numClasses)
        k[label[i,0]] = 1
        y[i] = (y[i] + k).astype(int)

    z2 = np.dot(data, W11) + b11
    a2 = sigmoid(z2)
    z3 = np.dot(a2,W21) + b21 #
    a3 = sigmoid(z3)

    intercept = np.ones(a3.shape[0])
    intercept = intercept.reshape(a3.shape[0],1)
    a3_1 = np.concatenate((intercept, a3),axis=1)


    theta_1 =  np.dot(a3_1, thetaSoftmax.T)
    theta_1 = theta_1 - np.amax(theta_1, axis = 1).reshape(data.shape[0],1)
    prob = np.exp(theta_1) # 10000*724 * 724*10 = 10000*10
    sum_prob = np.sum(prob, axis = 1).reshape(data.shape[0],1) # 10000*1
    prob = prob/sum_prob

    predict = prob/np.amax(prob, axis = 1).reshape(data.shape[0],1)
    predict = (predict == 1.0 ).astype(float)
    k = 0
    for i in range(len(label)):
        if np.array_equal(predict[i,:],y[i,:]):
            k = k+1
    correctness = k/(len(label))
    return correctness

## ---------------------------------------------------------------
def sigmoid(x):
    # Here's an implementation of the sigmoid function, which you may find useful
    # in your computation of the costs and the gradients.  This inputs a (row or
    # column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 
    return 1/(1+np.exp(-x))



def unravelDeepParameters(theta,visibleSize,hiddenSizeL1,hiddenSizeL2,numClasses):
    thetaSoftmax = theta[0:numClasses*(hiddenSizeL2+1)].reshape(numClasses,hiddenSizeL2+1)
    W11 = theta[numClasses*(hiddenSizeL2+1):numClasses*(hiddenSizeL2+1)+
                                        visibleSize*hiddenSizeL1].reshape(visibleSize,hiddenSizeL1)

    k = numClasses*(hiddenSizeL2+1)+visibleSize*hiddenSizeL1

    W21 = theta[k:k+hiddenSizeL2*hiddenSizeL1].reshape(hiddenSizeL1,hiddenSizeL2)
    b11 = theta[k+hiddenSizeL2*hiddenSizeL1:k+hiddenSizeL2*hiddenSizeL1+hiddenSizeL1]
    b21 = theta[k+hiddenSizeL2*hiddenSizeL1+hiddenSizeL1:k+hiddenSizeL2*hiddenSizeL1+hiddenSizeL1+hiddenSizeL2]

    return thetaSoftmax, W11, W21, b11, b21
## ---------------------------------------------------------------
def unravelParameters(theta,hiddenSize,visibleSize):
    # Convert theta to the (W1, W2, b1, b2) matrix/vector format
    W1 = theta[0:hiddenSize*visibleSize].reshape(visibleSize,hiddenSize)
    W2 = theta[hiddenSize*visibleSize:2*hiddenSize*visibleSize].reshape(hiddenSize,visibleSize)
    b1 = theta[2*hiddenSize*visibleSize:2*hiddenSize*visibleSize+hiddenSize]
    b2 = theta[2*hiddenSize*visibleSize+hiddenSize:]
    return W1,W2,b1,b2


## ---------------------------------------------------------------
def ravelParameters(W1,W2,b1,b2):
    # Unroll the (W1, W2, b1, b2) matrix/vector format to the theta format.
    return np.concatenate((W1.ravel(),W2.ravel(),b1.ravel(),b2.ravel()))

def ravelDeepParameters(W1,W2,b1,b2,softmax):
    # Unroll the (W1, W2, b1, b2) matrix/vector format to the theta format.
    return np.concatenate((softmax.ravel(),W1.ravel(),W2.ravel(),b1.ravel(),b2.ravel()))


## ---------------------------------------------------------------
def checkNumericalGradient():
    # This code can be used to check your numerical gradient implementation 
    # in computeNumericalGradient()
    # It analytically evaluates the gradient of a very simple function called
    # simpleQuadraticFunction (see below) and compares the result with your numerical
    # solution. Your numerical gradient implementation is incorrect if
    # your numerical solution deviates too much from the analytical solution.

    # Evaluate the function and gradient at x = [4; 10]; (Here, x is a 2d vector.)
    x = np.array([4,10])
    value,grad = simpleQuadraticFunction(x)
    
    # Use your code to numerically compute the gradient of simpleQuadraticFunction at x.
    # (The notation "lambda x: simpleQuadraticFunction(x)[0]" creates a function
    # that only returns the cost and not the grad of simpleQuadraticFunction.)
    numgrad = computeNumericalGradient(lambda x: simpleQuadraticFunction(x)[0],x)
    
    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar. 
    print(np.array([numgrad,grad]).T)
    print("The above two columns you get should be very similar.")
    print("(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n")
    
    # Evaluate the norm of the difference between two solutions.  
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
    # in computeNumericalGradient.m, then diff below should be 2.1452e-12 
    diff = norm(numgrad-grad)/norm(numgrad+grad)
    print(diff)
    print("Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n")


## ---------------------------------------------------------------
def simpleQuadraticFunction(x):
    # this function accepts a 2D vector as input. 
    # Its outputs are:
    #   value: h(x1, x2) = x1^2 + 3*x1*x2
    #   grad: A 2-dim vector that gives the partial derivatives of h with respect to x1 and x2
    value = x[0]**2+3*x[0]*x[1]
    grad = np.zeros(np.shape(x))
    grad[0] = 2*x[0]+3*x[1]
    grad[1] = 3*x[0]
    return value, grad


## ---------------------------------------------------------------
def computeNumericalGradient(J,theta):
    # numgrad = computeNumericalGradient(J, theta)
    # theta: a vector of parameters
    # J: a function that outputs r.
    # Calling y = J(theta) will return the function value at theta. 
      
    # Initialize numgrad with zeros
    numgrad = np.zeros(np.shape(theta))

    ## ---------- YOUR CODE HERE --------------------------------------
    # Instructions: 
    # Implement numerical gradient checking, and return the result in numgrad.  
    # (See Section 2.3 of the lecture notes.)
    # You should write code so that numgrad(i) is (the numerical approximation to) the 
    # partial derivative of J with respect to the i-th input argument, evaluated at theta.  
    # I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
    # respect to theta(i).
    #               
    # Hint: You will probably want to compute the elements of numgrad one at a time.
    for i in range(0,numgrad.shape[0]):
        k = np.zeros(np.shape(theta))
        k[i] = 0.0001
        y1 = J(theta+k)
        y2 = J(theta - k)
        numgrad[i] = (y1-y2)/0.0002

    ## ---------------------------------------------------------------
    return numgrad


## ---------------------------------------------------------------
def initializeParameters(hiddenSize,visibleSize):
    # Initialize parameters randomly based on layer sizes.
    r = np.sqrt(6)/np.sqrt(hiddenSize+visibleSize+1)
    W1 = rand(visibleSize,hiddenSize)*2*r-r
    W2 = rand(hiddenSize,visibleSize)*2*r-r
    b1 = np.zeros((hiddenSize,1))
    b2 = np.zeros((visibleSize,1))

    # Convert weights and bias gradients to the vector form.
    # This step will "unroll" (flatten and concatenate together) all 
    # your parameters into a vector, which can then be used 
    # with scipy.optimize.fmin_l_bfgs_b. 
    theta = ravelParameters(W1,W2,b1,b2)    
    return theta


def initializeSoftmaxParameters(numClasses,inputSize):
    # Initialize parameters randomly based on layer sizes.
    #r = np.sqrt(6)/np.sqrt(hiddenSize+visibleSize+1)
    theta = rand(numClasses*(inputSize+1))


    # Convert weights and bias gradients to the vector form.
    # This step will "unroll" (flatten and concatenate together) all
    # your parameters into a vector, which can then be used
    # with scipy.optimize.fmin_l_bfgs_b.
    return theta.ravel('F')



## ---------------------------------------------------------------
def displayNetwork(A,optNormEach=False,optNormAll=True,numColumns=None,imageWidth=None,cmapName='gray',
                   borderColor='blue',borderWidth=1,verbose=True,graphicsLibrary='matplotlib',saveName=''):
    # This function visualizes filters in matrix A. Each row of A is a
    # filter. We will reshape each row into a square image and visualizes
    # on each cell of the visualization panel. All other parameters are
    # optional, usually you do not need to worry about them.
    #
    # optNormEach: whether we need to normalize each row so that 
    # the mean of each row is zero.
    #
    # optNormAll: whether we need to normalize all the rows so that 
    # the mean of all the rows together is zero.
    #
    # imageWidth: how many pixels are there for each image
    # Default value is the squareroot of the number of columns in A.
    #
    # numColumns: how many columns are there in the display. 
    # Default value is the squareroot of the number of rows in A.
 
    # compute number of rows and columns    
    nr,nc = np.shape(A) 
    if imageWidth==None:
        sx = np.ceil(np.sqrt(nc))
        sy = np.ceil(nc/sx)
    else:
        sx = imageWidth
        sy = np.ceil(nc/sx)
    if numColumns==None:
        n = np.ceil(np.sqrt(nr))
        m = np.ceil(nr/n)
    else:
        n = numColumns
        m = np.ceil(nr/n)
    n = np.uint8(n)
    m = np.uint8(m)
    if optNormAll:    
        A = A-A.min()
        A = A/A.max()
     
    # insert data onto squares on the screen
    k = 0
    buf = borderWidth
    array = -np.ones([buf+m*(sy+buf),buf+n*(sx+buf)])
    for i in range(1,m+1):
        for j in range(1,n+1):
            if k>=nr: 
                continue
            B = A[k,:]
            if optNormEach:
                B = B-B.min()
                B = B/float(B.max())           
            B = np.reshape(np.concatenate((B,-np.ones(sx*sy-nc)),axis=0),(sx,-1))
            array[(buf+(i-1)*(sy+buf)):(i*(sy+buf)),(buf+(j-1)*(sx+buf)):(j*(sx+buf))] = B
            k = k+1       
     
    # display picture and save it
    cmap = plt.cm.get_cmap(cmapName)
    cmap.set_under(color=borderColor)
    cmap.set_bad(color=borderColor) 
    if graphicsLibrary=='PIL':
        im = PIL.Image.fromarray(np.uint8(cmap(array)*255))
        if verbose:
            im.show()
        if saveName != '':
            im.save(saveName)
    elif graphicsLibrary=='matplotlib':
        plt.imshow(array,interpolation='nearest',
                   norm=colors.Normalize(vmin=0.0,vmax=1.0,clip=False))
        plt.set_cmap(cmap)
        if verbose:
            plt.show()
        if saveName != '':
            plt.savefig(saveName)


## ---------------------------------------------------------------
class SparseAutoencoder():
    
    def __init__(self,visibleSize,hiddenSize=25,sparsityParam=0.01,beta=3,decayWeight=0.0001):
        # hyperparameters
        self.visibleSize = visibleSize      # number of input units
        self.hiddenSize = hiddenSize        # number of hidden units
        self.sparsityParam = sparsityParam  # desired average activation of hidden units
        self.beta = beta                    # weight of sparsity penalty term
        self.decayWeight = decayWeight      # weight decay parameter
        # parameters
        self.W1 = None  # array of shape (visibleSize, hiddenSize)
        self.W2 = None  # array of shape (hiddenSize, visibleSize)
        self.b1 = None  # vector of length (hiddenSize)
        self.b2 = None  # vector of length (visibleSize)

    def fit(self,patches,maxiter=400):
        # this function trains the weights of the sparse autoencoder. 
        if (self.W1 is None or self.W2 is None or self.b1 is None or self.b2 is None):
            theta = initializeParameters(self.hiddenSize,self.visibleSize)
        else:
            theta = ravelParameters(self.W1,self.W2,self.b1,self.b2)
        sparseAutoencoderArgs = (self.visibleSize,self.hiddenSize,self.decayWeight,
                                 self.sparsityParam,self.beta,patches)
        opttheta,cost,messages = minimize(sparseAutoencoderCost,theta,fprime=None,
                                          args=sparseAutoencoderArgs,maxiter=400)
        self.W1,self.W2,self.b1,self.b2 = unravelParameters(opttheta,self.hiddenSize,self.visibleSize)
        
    def predict(self,samples):
        # this function returns the output layer activations (estimates)
        z2 = np.dot(samples,self.W1)+np.array([self.b1])
        a2 = sigmoid(z2)
        z3 = np.dot(a2,self.W2)+np.array([self.b2])
        a3 = sigmoid(z3)
        return a3

    def score(self,patches):
        # computes the cost function of the sparseAutoencoder
        theta = ravelParameters(self.W1,self.W2,self.b1,self.b2)
        return sparseAutoencoderCost(theta,self.visibleSize,self.hiddenSize,self.decayWeight,
                                     self.sparsityParams,self.beta,patches)[0]



if __name__ == "__main__":
    #  This function contains code that helps you get started on the
    #  programming assignment. You will need to complete the code in
    #  sampleIMAGES(), sparseAutoencoderCost() and computeNumericalGradient().
    #  For the purpose of completing the assignment, you do not need to
    #  change the code in this function.
    #
    ##======================================================================
    ## STEP 0: Here we provide the relevant parameters values that will
    #  allow your sparse autoencoder to get good filters; you do not need to
    #  change the parameters below.
    patchsize = 28
    visibleSize = patchsize*patchsize      # number of input units                     # looks like a lower-case "p" in the lecture notes).
    decayWeight = 3e-3   # weight decay parameter
    numClasses = 10             # weight of sparsity penalty term

    # number of input units
    hiddenSizeL1 = 4
    hiddenSizeL2 = 5
    inputSize = hiddenSizeL2# number of hidden units
    sparsityParam = 0.1   # desired average activation of the hidden units.
                           # (This was denoted by the Greek alphabet rho, which                  # looks like a lower-case "p" in the lecture notes).
    beta = 3
    ##======================================================================
    #STEP 1: Load data from the MNIST database
    #This loads our training and test data from the MNIST database files.
    # We have sorted the data for you in this so that you will not have to
    # change it.

    patches,labels = sampleIMAGES(patchsize,"training")

    thetaSparse1 = initializeParameters(hiddenSizeL1,visibleSize)

    labels = np.asfortranarray(labels)
    patches = np.asfortranarray(patches)

    displayNetwork(patches[:100,:])

    # ======================================================================
    # STEP 2: Train the sparse autoencoder
    # This trains the sparse autoencoder on the unlabeled training
    # images.

    opttheta1,cost,messages = minimize(sparseAutoencoderCost,thetaSparse1,fprime=None,maxiter=400,
                                      args=(visibleSize,hiddenSizeL1,decayWeight,sparsityParam,beta,patches))

    W11,W12,b11,b12 = unravelParameters(opttheta1,hiddenSizeL1,visibleSize)

    displayNetwork(W11.T,saveName='weights.png')

    # ======================================================================
    #  STEP 3: Extract Features from the Supervised Dataset
    # feedForward Autoencoder to train the second Autoencoder

    trainFeature = feedForward(patches,W11,b11)

    trainFeature = np.asfortranarray(trainFeature)

    # ======================================================================
    #  STEP 4: Training the second auto encoder with the hidden layer output of the first layer
    thetaSparse2 = initializeParameters(hiddenSizeL2,hiddenSizeL1)
    opttheta2,cost,messages = minimize(sparseAutoencoderCost,thetaSparse2,fprime=None,maxiter=400,
                                      args=(hiddenSizeL1,hiddenSizeL2,decayWeight,sparsityParam,beta,trainFeature))

    W21,W22,b21,b22 = unravelParameters(opttheta2,hiddenSizeL2,hiddenSizeL1)
    displayNetwork(W21.T,saveName='weights.png')

    # ======================================================================
    #  STEP 5: Extract Features from the Supervised Dataset
    # feedForward Autoencoder to train the softmax classifier


    trainFeature2 = feedForward(trainFeature,W21,b21)

    trainFeature2 = np.asfortranarray(trainFeature2)
    # ======================================================================
    # STEP 6: Train the softmax classifier
    thetaSoftmax = 0.005 * np.random.randn(numClasses*(inputSize+1))
    thetaSoftmax = np.asfortranarray(thetaSoftmax)
    intercept = np.ones(trainFeature2.shape[0])
    intercept = intercept.reshape(trainFeature2.shape[0],1)
    trainFeature3 = np.concatenate((intercept, trainFeature2),axis=1)

    opttheta_softmax,cost,messages = minimize(softmaxCost,thetaSoftmax,fprime=None,
                                      args=(numClasses,inputSize,labels,decayWeight,trainFeature3),maxiter=400)


    # # ======================================================================
    # # STEP 7: According to the opttheta1, opttheta2, opttheta_softmax, do
    # # fine tuning, and get the best opttheta1, opttheta2, opttheta_softmax to minimize the cost function.
    # # so, we need to define a new gradient and cost computation like function here.
    #
    #
    theta = ravelDeepParameters(W11,W21,b11,b21,opttheta_softmax)

    opttheta,cost,messages = minimize(finetuning,theta,fprime=None,args=(patches,labels,
                                        decayWeight,hiddenSizeL1,hiddenSizeL2,visibleSize,numClasses),maxiter=400)



    data, label = sampleIMAGES(patchsize,"testing")

    correctness = predictFinetuning(theta,data,label,visibleSize,hiddenSizeL1,hiddenSizeL2,numClasses)
    print(correctness)
    # ##==========================================================
    # ## Doing gradient check of finetuning here
    # k = numClasses*(inputSize+1) + visibleSize*hiddenSizeL1 + hiddenSizeL1*hiddenSizeL2 + hiddenSizeL1 + hiddenSizeL2
    # theta = 0.005 * np.random.randn(k)
    # print(labels.shape)
    #
    # cost, grad = finetuning(theta,patches,labels,decayWeight,hiddenSizeL1,hiddenSizeL2,visibleSize,numClasses)
    # print("Computing numerical gradient of sparseAutoencoderCost...")
    # numgrad = computeNumericalGradient(lambda x: finetuning(x, patches,labels,decayWeight,hiddenSizeL1,
    #                                                         hiddenSizeL2,visibleSize,numClasses)[0],theta)
    #
    # # # Use this to visually compare the gradients side by side
    # #print(np.array([numgrad,grad]).T)
    # # print(grad)
    # # print('============')
    # # print(numgrad)
    # # Compare numerically computed gradients with the ones obtained from backpropagation
    # diff = norm(numgrad-grad)/norm(numgrad+grad)
    #
    # print(diff)# Should be small. In our implementation, these values are
    # ## usually less than 1e-9.
    # ## When you got this working, Congratulations!!!
    # print('diff is computed')
    # ##==========================================================
    # ## Doing gradient check of finetuning here


    # opttheta,cost,messages = minimize(sparseAutoencoderCost,theta,fprime=None,maxiter=400,
    #                                   args=(visibleSize,hiddenSize,decayWeight,sparsityParam,beta,patches))



    # opttheta,cost,messages = minimize(softmaxCost,theta,fprime=None,
    #                                   args=(numClasses,inputSize,labels,decayWeight,patches),maxiter=400)
    #
    #
    #
    # correctness = predictSoftmax(opttheta,test,label_test,numClasses,inputSize)
    # print(correctness)
    # print('We are done')
    # #
    # # ##======================================================================
    # # ## STEP 5: Visualization
    # # # save the visualization to a file
    # #
    # W1,W2,b1,b2 = unravelParameters(opttheta,hiddenSize,visibleSize)
    # displayNetwork(W1.T,saveName='weights.png')
    # #
    # # ##======================================================================
    # # ## STEP 6: Classes and objects
    # # # bind the functions together into a SparseAutoencoder class
    # #sae = SparseAutoencoder(visibleSize=64,hiddenSize=25,sparsityParam=0.01,
    # #                       beta=3,decayWeight=0.0001)
    # sae = SparseAutoencoder(visibleSize,hiddenSize,sparsityParam=0.1,
    #                         beta=3,decayWeight=0.003)
    # sae.fit(patches,maxiter=2000)
    # displayNetwork(sae.W1.T,saveName='weights_2000.png')
    # estimates = sae.predict(patches[:100,:])
    # displayNetwork(estimates)
    # print('We are done')
