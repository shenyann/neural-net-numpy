from activation_functions import sigmoid_function, tanh_function, linear_function,\
                                 LReLU_function, ReLU_function, elliot_function, symmetric_elliot_function,softmax_function
from neuralnet import NeuralNet
import numpy as np
import cPickle
f=open('/Users/yanshen/Downloads/mnist.gz','rb')
train_set,valid_set,test_set=cPickle.load(f)
train_set_x,train_set_y=train_set
valid_set_x,valid_set_y=valid_set
test_set_x,test_set_y=test_set

filename=open("trained_configuration.pkl")
import cPickle
store_dict = cPickle.load(filename)
settings = {
    # Optional settings
    "n_inputs"           : store_dict["n_inputs"],
    "n_outputs"          : store_dict["n_outputs"] ,
    "n_hiddens"          :store_dict["n_hiddens"],
    "n_hidden_layers"    :store_dict["n_hidden_layers"],
    "n_weights" :           store_dict["n_weights"] ,
    "n_weights":               store_dict["weights"] ,
    "activation_functions" :store_dict["activation_functions"]

}

network = NeuralNet( settings )
y_pred=network.fprop(valid_set_x,trace=False)
y_pred_test=network.fprop(test_set_x,trace=False)
print"valid_set error:" ,np.mean(1-np.not_equal(y_pred,valid_set_y))
print"test_set error:" ,np.mean(1-np.not_equal(y_pred_test,test_set_y))