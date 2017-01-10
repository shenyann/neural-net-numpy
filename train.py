from activation_functions import sigmoid_function, tanh_function, linear_function,\
                                 LReLU_function, ReLU_function, elliot_function, symmetric_elliot_function,softmax_function,dropout,add_bias
from neuralnet import NeuralNet
import numpy as np
import cPickle
f=open('/Users/yanshen/Downloads/mnist.gz','rb')
train_set,valid_set,test_set=cPickle.load(f)
train_set_x,train_set_y=train_set
valid_set_x,valid_set_y=valid_set

train_set_x=train_set_x[:]
train_set_y=train_set_y[:]


groundtruthlabel=np.zeros((train_set_y.shape[0],10))#convert labels to groundtruth label matrix
for i in range(train_set_y.shape[0]):
    groundtruthlabel[i,train_set_y[i]]=1
settings = {
    # Required settings
    "n_inputs"              : 28*28,
    "n_outputs"             : 10,
    "n_hidden_layers"       : 1,        # Number of nodes in each hidden layer,I only write n=0 or 1.
    "n_hiddens"             : 200,        # Number of hidden layers in the network
    "activation_functions"  : [tanh_function,softmax_function],
    # Optional settings
    "weights_low"           : -np.sqrt(6/(28*28+200)),
    "weights_high"          : +np.sqrt(6/(28*28+200)),
    "save_trained_network"  : False,
    
    "input_layer_dropout"   : 0.0,
    "hidden_layer_dropout"  : 0.,
    
    "batch_size"            : train_set_x.shape[0],        # must greater than 0
}
# initialize the neural network
network = NeuralNet( settings )


network.brop(
                train_set_x,
                train_set_y,
                groundtruthlabel,
                ERROR_LIMIT     = 1e-6,
                learning_rate   = 0.01,
                momentum_factor = 0.,
            )

network.save_to_file( "trained_configuration.pkl" )
