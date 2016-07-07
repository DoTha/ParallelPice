import sys
import os

import numpy as np
import theano
import theano.tensor as T
rng = np.random
import lasagne
 
class Controller(object): #a neural network ... 

    def _build_mlp(self, input_var,depth, width,input_dim,target_dim):
        # This creates an MLP of two hidden layers of 800 units each, followed by
        # a softmax output layer of 10 units. It applies 20% dropout to the input
        # data and 50% dropout to the hidden layers.

        # Input layer, specifying the expected input shape of the network
        # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
        # linking it to the given Theano variable `input_var`, if any:
        network = lasagne.layers.InputLayer(shape=(None,input_dim),input_var=input_var)
        #network=batchnormalization.BatchNormLayer(network) #I took the batchnormalization out because it is not yet compatible with the parallel framework
        ''' 
        nonlin = lasagne.nonlinearities.rectify
        drop_hidden=0.0
        
        for _ in range(depth):
            network = lasagne.layers.DenseLayer(network, width, nonlinearity=nonlin,W=lasagne.init.Orthogonal('relu'))
            if drop_hidden:
                network = lasagne.layers.dropout(network, p=drop_hidden)
                
        network_drop = lasagne.layers.DropoutLayer(network, p=0.5)
        
        
        l_out = lasagne.layers.DenseLayer(network_drop, num_units=target_dim, nonlinearity=lasagne.nonlinearities.identity)
        '''
        # Each layer is linked to its incoming layer(s), so we only need to pass
        # the output layer to give access to a network in Lasagne:
        
        l_out = lasagne.layers.DenseLayer(network, num_units=target_dim, nonlinearity=lasagne.nonlinearities.identity,W=lasagne.init.Constant(val=0.0),b=lasagne.init.Constant(val=0.0))
        
        return l_out

    def __init__(self, depth, width,input_dim,target_dim,learning_rate,trainer):
        self.input_dim = input_dim
        self.target_dim = target_dim
        
        #write here some lasagne code
        input_var = T.matrix('inputs')
        target_var = T.matrix('targets')
        weights_var = T.vector('weights')

        # Create neural network model (depending on first command line parameter)
        print("Building model and compiling functions...")
      
        network = self._build_mlp(input_var,depth, width,input_dim,target_dim)
      
        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(network)
        loss = (prediction-target_var)**2
        loss = (loss.T*weights_var).T.sum()
        # We could add some weight decay as well here, see lasagne.regularization.

        # Create update expressions for training
        params = lasagne.layers.get_all_params(network, trainable=True)
        grads = lasagne.updates.get_or_compute_grads(loss,params)
        
        if trainer == "adam":
                updates = lasagne.updates.adam(grads,params,learning_rate=learning_rate)
        elif trainer == "sgd":
                updates = lasagne.updates.sgd(grads,params,learning_rate=learning_rate)
        elif trainer == "momentum":
                updates = lasagne.updates.momentum(grads, params, learning_rate=learning_rate, momentum=1-learning_rate)
        elif trainer == "nestrov":
                updates = lasagne.updates.nesterov_momentum(grads, params, learning_rate=learning_rate, momentum=1-learning_rate)
        elif trainer == "adagrad":
                updates=lasagne.updates.adagrad(grads, params, learning_rate=learning_rate,epsilon=1e-06)
        else:
                assert(1==2), "error: do not know this trainer"
                
        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = (test_prediction-target_var)**2
        test_loss = (test_loss.T*weights_var).T.sum()
        
        #############################
        #### Compile functions ######
        #############################
        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        grad_fn = theano.function([weights_var, input_var, target_var],grads) #gives gradient
        train_fn = theano.function([weights_var, input_var, target_var]+grads, loss, updates=updates) #updates weights
        
        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([weights_var,input_var, target_var], test_loss)
        #Compile a third function computing the output of the NN
        predict_fn = theano.function([input_var], prediction)
        
        train_params=[aa for aa in updates.keys() if aa not in params]; #extract the train parameters
        
        self.predict_function = predict_fn #the predict function will be called from outside. 
        self._network = network
        self.train_function = train_fn #does a single training update: updates parameters and training_parameters
        self.validation_function = val_fn
        self.gradient_function = grad_fn

    def set_values(self,values):
        params = lasagne.layers.get_all_params(self._network, trainable=True)
        for i in range(len(params)): 
            params[i].set_value(values[i])
                    
    def get_values(self, get_keys = False):
        params = lasagne.layers.get_all_params(self._network, trainable=True)
        if get_keys:
            return [(param,param.get_value()) for param in params]
        else:
            return [param.get_value() for param in params]
