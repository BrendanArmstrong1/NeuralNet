import numpy as np
from pathlib import Path
from tqdm import tqdm
from NeuralNet import *
import time
from numba import njit, prange


class Layer_Conv2D:
    def __init__(self, n_filters, kernel, stride=1, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.stride = stride
        self.filters = n_filters
        self.kernel = kernel
        self.weights = np.random.normal(0, 0.07, size=(n_filters,*kernel))
        self.biases = np.zeros((1, n_filters))
        #################################################
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l2 = bias_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1

    def forward(self, inputs, training=True, padding=0):
        if len(inputs.shape) == 3:
            self.inputs = inputs.reshape(1,*inputs.shape)
        else:
            self.inputs = inputs
        self.output = self.convolve2D(self.inputs, self.weights)

    def backward(self, dvalues):
        self.dbiases = np.sum(dvalues, axis=(0,2,3)).reshape(*self.biases.shape)
        self.dweights = np.zeros((dvalues.shape[0],*self.weights.shape))
        self.dinputs = np.zeros_like(self.inputs)
        drot = np.rot90(dvalues, 2, axes=(2,3))
        self.dweights = self.convolve2D_Weights(self.inputs, drot)
        '''
        for i in range(self.kernel[1]):
            ii = i * self.stride
            for j in range(self.kernel[2]):
                jj = j * self.stride
                strip = self.inputs[:,:,ii:ii+dvalues.shape[2], jj:jj+dvalues.shape[3]]
                for l in range(self.weights.shape[1]):
                    for f in range(dvalues.shape[1]):
                        self.dweights[:,f,l,i,j] = np.einsum('byx,byx->b',strip[:,l], drot[:,f])
                        '''
        self.dweights = np.sum(np.rot90(self.dweights,2,axes=(3,4)), axis=0)
        ####################################################################
        wrot = np.rot90(self.weights, 2, axes=(2,3))
        py = self.kernel[1] - 1
        px = self.kernel[2] - 1
        dpadded = np.pad(dvalues, [(0,0),(0,0),(py,py),(px,px)])
        self.dinputs = self.convolve2D_Dinputs(dpadded, wrot)
        '''
        for i in range(self.dinputs.shape[2]):
            ii = i * self.stride
            for j in range(self.dinputs.shape[3]):
                jj = j * self.stride
                strip = dpadded[:,:, ii:ii+self.kernel[1], jj:jj+self.kernel[2]]
                for l in range(self.kernel[0]):
                    self.dinputs[:,l,i,j] = np.einsum('fyx,bfyx->b', wrot[:,l], strip)
        '''
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases< 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases
    
    @staticmethod
    @njit(parallel=True, nogil=True)
    def convolve2D(inputs, kernel, stride=1, padding=0):
        outy = (inputs.shape[-2] + 2*padding - kernel.shape[-2]) // stride + 1
        outx = (inputs.shape[-1] + 2*padding - kernel.shape[-1]) // stride + 1
        batch_number = inputs.shape[0]
        filter_number = kernel.shape[0]
        output = np.zeros((batch_number, filter_number, outy, outx))
        for b in prange(batch_number):
            for i in prange(outy):
                ii = i * stride
                for j in prange(outx):
                    jj = j * stride
                    strip = inputs[:,:,ii:ii+kernel.shape[2], jj:jj+kernel.shape[3]]
                    for f in prange(filter_number):
                            output[b,f,i,j] = np.sum(kernel[f] * strip[b])
        return output

    @staticmethod
    @njit(parallel=True, nogil=True)
    def convolve2D_Weights(inputs, kernel, stride=1, padding=0):#inputs are [self.inputs, dvalues(rotated)]
        outy = (inputs.shape[-2] + 2*padding - kernel.shape[-2]) // stride + 1
        outx = (inputs.shape[-1] + 2*padding - kernel.shape[-1]) // stride + 1
        batch_number = inputs.shape[0]
        filter_number = kernel.shape[1]
        layer_number = inputs.shape[1]
        output = np.zeros((batch_number, filter_number, layer_number, outy, outx))
        for b in prange(batch_number):
            for i in prange(outy):
                ii = i * stride
                for j in prange(outx):
                    jj = j * stride
                    strip = inputs[:,:,ii:ii+kernel.shape[2], jj:jj+kernel.shape[3]]
                    for f in prange(filter_number):
                        for l in prange(layer_number):
                            output[b,f,l,i,j] = np.sum(kernel[:,f,i,j] * strip[b,l,i,j])
        return output



    @staticmethod
    @njit(parallel=True, nogil=True)
    def convolve2D_Dinputs(inputs, kernel, stride=1, padding=0):# inputs are [dvalues(padded),self.weights(rotated)]
        outy = (inputs.shape[-2] + 2*padding - kernel.shape[-2]) // stride + 1
        outx = (inputs.shape[-1] + 2*padding - kernel.shape[-1]) // stride + 1
        batch_number = inputs.shape[0]
        filter_number = kernel.shape[0]
        layer_number = kernel.shape[1]
        output = np.zeros((batch_number, layer_number, outy, outx))
        for b in prange(batch_number):
            for i in prange(outy):
                ii = i * stride
                for j in prange(outx):
                    jj = j * stride
                    strip = inputs[:,:,ii:ii+kernel.shape[2], jj:jj+kernel.shape[3]]
                    for l in prange(layer_number):
                            output[b,l,i,j] = np.sum(kernel[:,l] * strip[b])
        return output

class Layer_Maxpooling:
    def __init__(self, kernel=2):
        self.kernel = kernel

    def forward(self, inputs, training=True):
        self.Y = ((inputs.shape[-2] - self.kernel) // self.kernel) + 1
        self.X = ((inputs.shape[-1] - self.kernel) // self.kernel) + 1
        self.inputs = np.zeros_like(inputs) 
        self.output = np.zeros((inputs.shape[0], inputs.shape[-3], self.Y, self.X))
        for i in range(self.Y):
            ii = i * self.kernel
            for j in range(self.X):
                jj = j * self.kernel
                strip = inputs[:,:, ii:ii+self.kernel, jj:jj+self.kernel]
                maxx = np.amax(strip, axis=(2,3)) 
                eva = ((np.sum((strip == maxx[:,:,None,None])*1,axis=(2,3)) - 1) == 0) * 1
                adder = ((strip == maxx[:,:,None,None])*1)*eva[:,:,None,None]
                self.output[:,:,i,j] = np.amax(strip,axis=(2,3))
                self.inputs[:,:,ii:ii+self.kernel, jj:jj+self.kernel] += adder

    def backward(self, dvalues):
        self.dinputs = self.inputs
        for i in range(self.Y):
            ii = i * self.kernel
            for j in range(self.X):
                jj = j * self.kernel
                self.dinputs[:,:, ii:ii+self.kernel, jj:jj+self.kernel] *= dvalues[:,:,i,j][:,:,None,None] 


class Layer_Flattening:
    def forward(self, inputs, training=True):
        self.inputs_shape = inputs.shape
        self.output = inputs.reshape(inputs.shape[0], -1)

    def backward(self, dvalues):
        self.dinputs = dvalues.reshape(self.inputs_shape)


np.set_printoptions(linewidth=600)
p = Path().cwd().parent / 'Data/Fashion_Mnist'

files = []
for file in p.iterdir():
    if file.suffix == '.npy':
        files.append(file)

y_val = np.load(files[0])
y = np.load(files[1])
X = (np.load(files[2]).astype(np.float32) - 127.5) / 127.5
X_val = (np.load(files[3]).astype(np.float32) - 127.5) / 127.5

X = X.reshape(X.shape[0],1,28,28)
X_val = X_val.reshape(X_val.shape[0],1,28,28)
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]


sampln = X[:64]
Den1 = Layer_Dense(384,10)
ActF = Activation_Softmax()

Conv1 = Layer_Conv2D(12,(1,5,5))
Re1 = Activation_RelU()
D1 = Layer_Dropout(0.25)
P1 = Layer_Maxpooling()
Conv2 = Layer_Conv2D(24,(12,5,5))
Re2 = Activation_RelU()
P2 = Layer_Maxpooling()
F = Layer_Flattening()

t2 = time.time()
for i in range(10):
    t1 = time.time()
    Conv1.forward(sampln)
    Re1.forward(Conv1.output, training = True)
    P1.forward(Re1.output, training = True)

    Conv2.forward(P1.output)
    Re2.forward(Conv2.output, training=True)
    P2.forward(Re2.output, training = True)

    F.forward(P2.output)
    Den1.forward(F.output, training = True)
    ActF.forward(Den1.output, training = True)

    F.backward(F.output)

    P2.backward(F.dinputs)
    Re2.backward(P2.dinputs)
    Conv2.backward(Re2.dinputs)

    P1.backward(Conv2.dinputs)
    Re1.backward(P1.dinputs)
    Conv1.backward(Re1.dinputs)
    print(f'{time.time() - t1:.6f} seconds')
print(f'{time.time() - t2:.6f} seconds')

