from scipy.special import expit
import numpy as np
def softmax_function(signal,brop=False):#as Output Layer,I direct calculate dL/dx=out-target,not use dL/dy*dy/dx.so i dont calculate dy/dx.
    #if use this for immediatelayer,there will be some numerical problem in Bprop.
        maxes=np.amax(signal,axis=1)
        maxes=maxes.reshape(maxes.shape[0],1)
        e_x=np.exp(signal-maxes)
        y=e_x/np.sum(e_x,axis=1).reshape(e_x.shape[0],1)
        return y
def sigmoid_function( signal, brop=False ):
    signal=np.clip(signal,-500,500)
    signal = 1/(1+np.exp(-signal))
    if brop:
        return np.multiply(signal, 1-signal)
    else:

        return signal
def elliot_function( signal, brop=False ):
    """ A fast approximation of sigmoid """
    s = 1 # steepness
    
    abs_signal = (1 + np.abs(signal * s))
    if brop:
        return 0.5 * s / abs_signal**2
    else:
        # Return the activation signal
        return 0.5*(signal * s) / abs_signal + 0.5
def symmetric_elliot_function( signal, brop=False ):
    """ A fast approximation of tanh """
    s = 1.0 # steepness
    
    abs_signal = (1 + np.abs(signal * s))
    if brop:
        return s / abs_signal**2
    else:
        # Return the activation signal
        return (signal * s) / abs_signal
def ReLU_function( signal, brop=False ):
    if brop:
        derivative = np.maximum( 0, signal )
        derivative[derivative!= 0] = 1.
        return derivative
    else:
        # Return the activation signal
        return np.maximum( 0, signal )
def LReLU_function( signal, brop=False):
    if brop:
        derivate = np.copy( signal )
        derivate[ derivate < 0 ] = 0.01
        derivate[ derivate > 0 ] = 1.0
        return derivate
    else:
        # Return the activation signal
        output = np.copy( signal )
        output[ output < 0 ] *= 0.01
        return output
def tanh_function( signal, brop=False ):
    signal=np.tanh(signal)
    if brop:
        return 1-np.power(signal,2)
    else:
        return signal
def linear_function( signal, derivative=False ):
    if derivative:
        # Return the partial derivation of the activation function
        return 1
    else:
        # Return the activation signal
        return signal
def dropout( X, p = 0.2 ):
    if p > 0:
        retain_p = 1 - p
        X = np.multiply(retain_p,X)
    return X
#end  
def add_bias(A):
    return np.hstack(( np.ones((A.shape[0],1)), A )) # Add 1 as bias.
#end addBias




