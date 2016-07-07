import numpy as np
import copy
import itertools
rng = np.random


class Parallel_DataGenerator(object): #this one takes a controllproblem and generates data in parallel. it has a method which creates an iterator for a training and validation set. and it has a function to generate rollouts
#training_batch, validation_batch = Parallel_DataGenerator.get_minibatch_iterators(batch_size)
#it also generates statistics like Effective Sampling Size      
    def _organize_data(self): #prepares the data for the neural network

        self._localize_weights() #all weights are transfered to worker 0. there they are normalized and exponentiated
        
#       self._resample() #currently empty needs to be written
        
        self._redistribute_data_and_store_in_numpy_matrix() #load balancing and bringing data in correct format -> at the moment it just redistributes the weights (I update the neural network so that it can deal with weights samples)
        
    def _localize_weights(self): #localizes all weights and normalizes them
        #copy all weights to worker 0 -> the weights come in a numpy matrix self._weights_global[i] is thereby a vector of the weights of the ith worker
        log_weights_global = np.asarray(self._comm.gather(self._log_weights_local, root=0))
        if self._mpi_rank==0:
            log_weights_global -= log_weights_global.max() #this is for the nummerics
            self._weights_global = np.exp(log_weights_global) #exponetiation
            self._weights_global /= self._weights_global.sum() #normalization
    
    def _resample(self): #resample the weights.. needs to be written (TODO)
        pass #pass is necessary for an empty function
    
    def _redistribute_data_and_store_in_numpy_matrix(self): #.. does no load balancing yet. TODO
        #at the moment we have not resampling  so we just redistribute the weights and store the data in a numpy matrix
        self._weights_local = self._comm.scatter(self._weights_global, root=0) #rank 0 is sending the exponetiated weights back to the workers
        self.weights_of_timepoints_local =  np.reshape(self._weights_local*np.ones([self.number_of_rollouts_per_worker,self.timelength]),self.number_of_rollouts_per_worker*self.timelength)
        #now we write the data in local numpy matrices
        self._targets_local = self._reshape_data(self._control_problem.Upn_storage)
        self._inputs_local = self._reshape_data(self._control_problem.C_inp_storage)
    
    def _reshape_data(self,x):
        #reshapes the data from dimension x particlenumber x timelength -> dimension x particlenumber*timelength. thus the different times are seen as different datapoints
        (particlenumber,dimension,timelength)=np.shape(x)
        x = np.swapaxes(x,0,1)
        x = np.reshape(x,(dimension,particlenumber*timelength))
        x = x.T
        return x

    def _systematic_resample(self, weights,N): #I copied this function here for convenience. it is not integrated yet! (TODO)
        """ Performs the systemic resampling algorithm used by particle filters.

        This algorithm separates the sample space into N divisions. A single random
        offset is used to to choose where to sample from for all divisions. This
        guarantees that every sample is exactly 1/N apart.

        Parameters
        ----------
        weights : list-like of float
            list of weights as floats

        Returns
        -------

        indexes : ndarray of ints
            array of indexes into the weights defining the resample. i.e. the
            index of the zeroth resample is indexes[0], etc.
        """

        # make N subdivisions, and choose positions with a consistent random offset
        positions = (rng() + np.arange(N)) / N

        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes
    
    def __init__(self,comm, control_problem):

        self.number_of_rollouts_per_worker = control_problem.num_particles
        self.timelength = control_problem.timelength
        
        self._mpi_rank = comm.Get_rank() #create local variable which has the MPI rank
        self._comm_size = comm.Get_size() #variable which stores the number of workers
        self._comm = comm #pointer to the mpi communicator (why?)
        
        np.random.seed(seed = self._mpi_rank) #set the random seeds
        
        self._control_problem = control_problem
        
        self._log_weights_local = np.zeros([self.number_of_rollouts_per_worker,1]) #create vector in which the local weights are stored
        self._weights_local = np.ones([self.number_of_rollouts_per_worker,1])

        self._weights_global = np.zeros([self._comm_size,self.number_of_rollouts_per_worker,1]) #create vector to store all weights
        self.weights_of_timepoints_local = np.zeros(self.number_of_rollouts_per_worker*self.timelength)
                
   
    def do_rollouts(self,uncontrolled = 0):
        #the output of the rollout are the weights which are written into the local list for log-weights
        print "Starting multiple_rollouts in process ", self._mpi_rank, "out of ", self._comm_size
        self._log_weights_local = self._control_problem.do_rollouts(uncontrolled)
        self._organize_data() #call organize data function (it uncommented, it doesn't work, cleaner if this happens outside explicitly)
        
    def get_minibatch_iterators(self, batchsize, val_percent=10.): #gives back two iterator-generators. one for training and one for validation. the iterator-generators generate an iteration over the tripple [inputs, targets, weights]
        assert len(self._inputs_local) == len(self._targets_local) #checks if input and target length are consistent
        
        def minibatch_iterator(inputs, targets, weights, batchsize): #function which generates the iterator-generator -> it iterates over the whole (local) dataset
            if batchsize > len(weights):
                print "WARNING: Batchsize larger than data set; iterator is empty!"
                
            indices = np.arange(len(inputs))
            #np.random.shuffle(indices)
            indices = (indices + rng.randint(len(inputs)))%len(inputs)
            for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
                    excerpt = indices[start_idx:start_idx + batchsize]
                    yield inputs[excerpt], targets[excerpt], weights[excerpt]
                
        #split in validation and training set:
        indices = np.arange(len(self._inputs_local))
        val_indices=indices[:int(np.floor(len(self._inputs_local)*val_percent/100.))]
        train_indices=indices[int(np.floor(len(self._inputs_local)*val_percent/100.)):]
        
        
        training_weights = self.weights_of_timepoints_local[train_indices]/(self.timelength*(1.-val_percent/100.)) #normalized training weights
        validation_weights = self.weights_of_timepoints_local[val_indices]/(self.timelength*val_percent/100.) #normalized validation weights
        
        train_iterator = minibatch_iterator(self._inputs_local[train_indices],self._targets_local[train_indices],training_weights,batchsize)
        
        validation_iterator = minibatch_iterator(self._inputs_local[val_indices],self._targets_local[val_indices],validation_weights,batchsize)
        
        return train_iterator, validation_iterator 
        
    def EffSS(self):
        EffSS=(np.sum(self._weights_global)**2)/np.sum(self._weights_global**2)
        if self._mpi_rank > 0:
            print "warning: only worker 0 has all weights and can compute the global EffSS"
        #computes effective sampling size for the themerature lambd
        return EffSS


            
