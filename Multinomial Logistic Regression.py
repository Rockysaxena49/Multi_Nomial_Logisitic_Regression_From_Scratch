import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')
# load the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# the images are shaped (60000,28,28) so each training sample is 28*28 ex r0 -> 28X28 r1 -> 28X28 ... r5999 -> 28X28
# This is to print out the handwritten digit in each of the training sample
plt.imshow(x_train[2])
plt.show()

# we want to reshape to (60000,784) for training and (1000, 784) for testing
print('xtrain shape ', x_train.shape)
print('xtest shape ', x_test.shape)
x_train_rs = x_train.reshape([60000, 28 * 28])
x_train_rs_scaled = x_train_rs / 255
x_test_rs = x_test.reshape([10000, 28 * 28])
x_test_rs_scaled = x_test_rs/ 255



k = 10 # number of classes

# create softmax regression function
def softmax(x, w, b):
    z = x.dot(w) + b
    h = np.exp(z)
    y_pred = h/np.sum(h)
    return z, h, y_pred

# create onehot encoding function
def oneHot(y_pred):
    y_predOneHot = np.full_like(y_pred, fill_value= 0) # will create a row vector filled with 0's of length y_pred
    y_predOneHot[np.argmax(y_pred)] = 1 # the largest element of y_predict get it's index and use that index in
    # y_predOnehot and label it 1 all other postions will be 0 in y_predOneHot
    return y_predOneHot

# this will take the maximum elements index of y_predOneHot + 1 and that will be the assigned class for training sample
def labelY_pred(y_predOneHot):
    classifiedY_pred = np.argmax(y_predOneHot) #+ 1  # add one because we indexed at 0
    return classifiedY_pred

# Multinomial cross entropy or cost function
def multiCrossEntropy(x, lossFn, trueLabel_enc, y_predProbabilty, lam):

    for i in range(x.shape[0]):
        lossFn[i] = -1*trueLabel_enc[i].dot(np.log(y_predProbabilty[i]))

    avgloss = ((np.sum(lossFn) / x.shape[0]))
    #avgloss = ((np.sum(lossFn) / x.shape[0])) + (lam/2)*np.sum(w) # L2 equation
    return avgloss

def updateWeights(w, learningRate, x, y_predProbabilty, trueLabel_enc, lam):

    w = w - learningRate * ((1 / x.shape[1]) * np.dot(x.T, (y_predProbabilty - trueLabel_enc)))
    #w = w - learningRate*(((1/x.shape[1])*(np.dot(x.T, (y_predProbabilty - trueLabel_enc))) + lam*w)) #L2 weights
    return w

def updateBias(y_predProbabilty, trueLabel_enc):
    bias = (1 / y_predProbabilty.shape[0]) * np.sum(y_predProbabilty - trueLabel_enc, axis=0)
    return bias

##################################################### Main function begins here ########################################
#####################################################Initalize variables#### ###########################################
lam = 1000
accuracyTrain = []
accuracyTest = []

trueLabel_enc = (np.arange(np.max(y_train) + 1) == y_train[:, None]).astype(float) # one hot encode the true label

w = np.ones((x_train_rs_scaled.shape[1], k))
#print('w shape \n', w.shape)
bias = np.ones(k)
#print('bias shape ', bias.shape)


y_predProbabilty = np.empty([x_train_rs_scaled.shape[0], k])
test_predProbabilty = np.empty([x_test_rs_scaled.shape[0], k])

oneHotY_predict = np.empty([x_train_rs_scaled.shape[0], k])
oneHotTest_predict = np.empty([x_test_rs_scaled.shape[0], k])

classifiedY_predict = np.empty([x_train_rs_scaled.shape[0]])
classifiedTest_predict = np.empty([x_test_rs_scaled.shape[0]])

avgloss = []
learningRate = 0.003
lossFn = np.empty(x_train_rs_scaled.shape[0])
niter = 0

########################################################################################################################
# This part runs through the training algorithm and testing algorithm utilizing the above functions
while niter < 100:
    #training
    for i in range(x_train_rs_scaled.shape[0]):
        z, h, y_predProbabilty[i] = softmax(x_train_rs_scaled[i, :], w, bias)
        oneHotY_predict[i] = oneHot(y_predProbabilty[i])
        classifiedY_predict[i] = labelY_pred(oneHotY_predict[i])

    # Testing
    for i in range(x_test_rs_scaled.shape[0]):
        temp1, temp2, test_predProbabilty[i] = softmax(x_test_rs_scaled[i, :], w, bias)
        oneHotTest_predict[i] = oneHot(test_predProbabilty[i])
        classifiedTest_predict[i] = labelY_pred(oneHotTest_predict[i])
    accuracyTest.append(accuracy_score(y_test, classifiedTest_predict))

    avgloss.append(multiCrossEntropy(x_train_rs_scaled, lossFn, trueLabel_enc, y_predProbabilty, lam))
    #print('avgloss \n', avgloss)

    # update the weights
    w = updateWeights(w, learningRate, x_train_rs_scaled, y_predProbabilty, trueLabel_enc, lam)

    # update the bias
    bias = updateBias(y_predProbabilty, trueLabel_enc)


    accuracyTrain.append(accuracy_score(y_train, classifiedY_predict))

    niter = niter + 1




print('Training accuracy no lambda ',accuracyTrain[-1])
print('Testing accuracy no lambda ',accuracyTest[-1])

num_iter = np.linspace(1,niter,num= niter, endpoint = True)

# graph log_likely hood vs iterations
plt.plot(num_iter,avgloss)
plt.title('Number of Iterations vs Average Loss')
plt.xlabel('Number of Iterations ')
plt.ylabel('Average Loss ')
plt.show()

# graph training accuracy over iterations
plt.plot(num_iter,accuracyTrain)
plt.title('Number of Iterations vs Training Accuracy')
plt.xlabel('Number of iterations ')
plt.ylabel('Training Accuracy ')
plt.show()

# graph testing accuracy over iterations
plt.plot(num_iter,accuracyTest)
plt.title('Number of Iterations vs Testing Accuracy')
plt.xlabel('Number of iterations ')
plt.ylabel('Testing Accuracy ')
plt.show()