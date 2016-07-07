import numpy as np
import copy
import itertools
rng = np.random
from mpi4py import MPI

class Parallel_Trainer(object): #takes a neural network and Parallel_DataGenerator and trains the neural network in a distributed manner
    
    def _merger(self,mergearray):
        #uses reduce2all to update parameters and trainparameters with the average over all workers
        #private function
        #has no input
        _comm_buffer_p=copy.deepcopy(mergearray) #create buffer for MPI process... do not know if this has a memory leak
        _comm_buffer_p/=self._comm_size #normalization to get avg instead of sum
        
        self._comm.Allreduce(_comm_buffer_p,mergearray, op=MPI.SUM)
        return mergearray
        
    def _zerobroadcast(self,mergearray):
        _comm_buffer_p=copy.deepcopy(mergearray) 
        return self._comm.bcast(_comm_buffer_p, root=0)
    
    def __init__(self,comm, Parallel_DataGenerator, controller):
        #trainfunction_merger is a function which uses reduce2all in order to merge the partial parameter updates done by the different workers. default is to just take the average for the parameters and the training_parameters
        self._comm = comm #pointer to the mpi communicator
        self._mpi_rank = comm.Get_rank()
        self._comm_size = comm.Get_size() #variable which stores the number of workers
        self._parallel_DataGenerator = Parallel_DataGenerator
        self.controller = controller

        #communicate initialization of weights and traiings parameters from zero to all other workers
        controller.set_values(self._zerobroadcast(controller.get_values()))
        

    def train_network(self,num_epochs=1500, size_minibatch=1000, stopping_criterion=lambda x:False): #stopping_criterion must be a function, which uses the train and test error (and potentially their history by saving it.. ) in order to determine if to stop the training (stopping criterion can use MPI to get to a consensus over all the workers)
        #in each epoch we sample over batch_size minibatches
        for epoch in range(num_epochs):
            training_batch, validation_batch = self._parallel_DataGenerator.get_minibatch_iterators(size_minibatch,10)  #Parallel_DataGenerator gives back two iterators one for the training and one for the validation
            train_err = np.array(0.)
            train_batches = 0.
            for batch in training_batch:
                inputs, targets, weights = batch
                _gradients=self.controller.gradient_function(weights, inputs, targets) #compute gradients
                for param in _gradients: self._merger(param)
                train_err += self.controller.train_function(weights, inputs, targets, *_gradients) #update local parameters and train parameters using the gradient. method can be adgrad, rmsprop or whatever gradient based optimizer
                train_batches += 1
            
            #Synchronization of training error
            train_err=self._merger(train_err)
            
            
            
            val_err = np.array(0.)
            val_batches = 0.
            for batch in validation_batch:
                inputs, targets, weights = batch
                val_err += self.controller.validation_function(weights, inputs, targets) #compute local validation error
                val_batches += 1
            val_err=self._merger(val_err)
                
            #a=self.controller.get_values(True)      
            #print epoch, 'validation_error: ', val_err, 'trainings_error: ', train_err, ' ', a
                
            if stopping_criterion(val_err):  #stopping criterions can be implemented outside. input are local validation and train error
                return val_err
                
            return val_err
