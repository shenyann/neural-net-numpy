import math
import random
import itertools
import collections
from activation_functions import add_bias,dropout
import numpy as np
default_settings = {
    # Optional settings
    "weights_low"           : -0.01,     # Lower bound on initial weight range
    "weights_high"          : 0.01,      # Upper bound on initial weight range
    
    "input_layer_dropout"   : 0.0,      # dropout fraction of the input layer
    "hidden_layer_dropout"  : 0.,      # dropout fraction in all hidden layers
    
    "batch_size"            : 128,        #batch learning size
}
class NeuralNet:
    def __init__(self, settings ):
        self.__dict__.update( default_settings )
        self.__dict__.update( settings )
        assert len(self.activation_functions) == (self.n_hidden_layers + 1), \
            "Expected {n_expected} activation functions, but was initialized with {n_received}.".format(
                n_expected = (self.n_hidden_layers + 1),
                n_received = len(self.activation_functions)
            )
        if self.n_hidden_layers == 0:
            # input -> [] -> output
            self.n_weights = (self.n_inputs + 1) * self.n_outputs
        else:
            # input -> [n_hiddens] -> output
            self.n_weights = (self.n_inputs + 1) * self.n_hiddens +\
                             (self.n_hiddens * self.n_outputs) + self.n_outputs
        # Initialize the network with new randomized weights
        self.set_list_of_weight_matrix(self.generate_weights( self.weights_low, self.weights_high ) )
    def generate_weights(self, low = -0.1, high = 0.1):
        # Generate new random weights for all the connections in the network
        if not False:
            return [random.uniform(low,high) for _ in xrange(self.n_weights)]
        else:
            return np.random.uniform(low, high, size=(1,self.n_weights)).tolist()[0]
    def list_of_weight_matrix(self, weight_list ):
        # This method create a list of weight matrices. Each list element
        # corresponds to the connection between two layers.
        if self.n_hidden_layers == 0:
            return [np.array(weight_list).reshape(self.n_inputs+1,self.n_outputs)]
        else:
            weight_layers = [ np.array(weight_list[:(self.n_inputs+1)*self.n_hiddens]).reshape(self.n_inputs+1,self.n_hiddens) ]
            weight_layers += [ np.array(weight_list[(self.n_inputs+1)*self.n_hiddens:]).reshape(self.n_hiddens+1,self.n_outputs) ]
        return weight_layers
    def set_list_of_weight_matrix(self, weight_list ):
        # This is a helper method for setting the network weights to a previously defined list.
        # This is useful for utilizing a previously optimized neural network weight set.
        self.weights = self.list_of_weight_matrix( weight_list )
    def get_flatten_weights(self, ):
        # This will stack all the weights in the network on a list, which may be saved to the disk.
        return [w for l in self.weights for w in l.flat]
    def fprop(self, input_values, trace=False ):
        output = input_values
        if trace: tracelist = [ output ]
        for i, weight_layer in enumerate(self.weights):
            if i == 0:
                output = np.dot( output, weight_layer[1:,:] ) + weight_layer[0:1,:] # implicit bias
            else:
                output = np.dot( output, weight_layer[1:,:] ) + weight_layer[0:1,:] # implicit bias
            if trace: tracelist.append( output )
            output = self.activation_functions[i]( output)

        if trace: return tracelist

        if trace is False:
            y_pred            =  np.argmax(output,axis=1)
            return y_pred
    def brop(self, training_set_x,training_set_y,groundtruthlabel,ERROR_LIMIT = 1e-3, learning_rate = 0.3, momentum_factor = 0.95,epochs=100 ):
        print"training:......."
        training_data    = training_set_x
        groundtruthlabel = groundtruthlabel
        training_targets=training_set_y
        MSE              = ( ) # inf
        neterror         = None
        momentum         = collections.defaultdict( int )
        batch_size       = self.batch_size
        epoch = 0
        while epoch < epochs:
            epoch += 1
            input_list      = self.fprop(training_data,trace=True)
            out               = self.activation_functions[-1](input_list[-1])
            delta            =  (1-out)*groundtruthlabel #dL/dx
            Loss               = -np.mean(np.log(out+1e-5)[np.arange(training_targets.shape[0]),training_targets])
            y_pred            =  np.argmax(out,axis=1)
            loop  = itertools.izip(
                                xrange(len(self.weights)-1, -1, -1),
                                reversed(self.weights),
                                reversed(input_list[:-1])

                                )

            for i, weight_layer,inputs in loop:
                    # Loop over the weight layers in reversed order to calculate the deltas
                if i == 0:
                    dropped = dropout( add_bias(inputs).T, self.input_layer_dropout)
                else:
                    dropped = self.activation_functions[i-1](dropout(add_bias(inputs).T,self.hidden_layer_dropout))

                dW = learning_rate * np.dot((dropped), delta) + momentum_factor * momentum[i]

                if i!=0:
                    weight_delta = np.dot( delta, weight_layer[1:,:].T )
                    delta = np.multiply(weight_delta, self.activation_functions[i-1](inputs, brop=True) )
                       # Store the momentum
                momentum[i] = dW
                    # Update the weights
                self.weights[ i ] -= dW


            if epoch%1==0:
                # Show the current training status
                print "* current network error per epoch(loss):", Loss
                print"epoch:",epoch
                print"current error:" ,np.mean(1-np.not_equal(y_pred,training_targets))
        
        print "* Converged to error bound (%.4g) with Loss = %.4g." % ( ERROR_LIMIT, Loss )
        print "* Trained for %d epochs." % epoch


    def save_to_file(self, filename = "network.pkl" ):
        import cPickle
        with open( filename , 'wb') as file:
            store_dict = {
                "n_inputs"             : self.n_inputs,
                "n_outputs"            : self.n_outputs,
                "n_hiddens"            : self.n_hiddens,
                "n_hidden_layers"      : self.n_hidden_layers,
                "activation_functions" : self.activation_functions,
                "n_weights"            : self.n_weights,
                "weights"              : self.weights

            }
            cPickle.dump( store_dict, file, 2 )
    @staticmethod
    def load_from_file( filename = "network.pkl" ):
        network = NeuralNet( 0,0,0,0,[0] )
        
        with open( filename , 'rb') as file:
            import cPickle
            store_dict = cPickle.load(file)
            
            network.n_inputs             = store_dict["n_inputs"]
            network.n_outputs            = store_dict["n_outputs"]
            network.n_hiddens            = store_dict["n_hiddens"]
            network.n_hidden_layers      = store_dict["n_hidden_layers"]
            network.n_weights            = store_dict["n_weights"]
            network.weights              = store_dict["weights"]
            network.activation_functions = store_dict["activation_functions"]

        return network