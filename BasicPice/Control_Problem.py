import numpy as np
import copy
import itertools
rng = np.random


class Control_Problem(object):
    
    def test_dimensionalities(self):
    #Tests if the output dimensionalities of the functions Phi, V, single_rollout_step and the controller have the right dimensionalities
        def my_error(output, shouldbe, message,alternative = False):
            assert(np.shape(output) == shouldbe or np.shape(output) == alternative), message+str(shouldbe)+" It has dimensions: "+str(np.shape(output))+" isparallel = "+str(self.isparallel)
        
        if self.isparallel:
            dummy = self.single_rollout_step(self.X_initial,np.zeros([self.num_particles,self.control_dimensions]),0,self.dt)
            assert(len(dummy)==2), "single_rollout_step should have two outputs."
            my_error(dummy[0],(self.num_particles,self.state_dimensions),"the first output of single_rollout_step has wrong output dimension. Should be [number_of_particles,dim]=")
            my_error(dummy[1],(self.num_particles,self.c_inp_dimensions),"the second output of single_rollout_step has wrong output dimension. Should be [number_of_particles,c_inp_dimensions]=")
            
            my_error(self.controller.predict_function(self.C_inp_initial),(self.num_particles,self.control_dimensions),"Output of controller.predict_function has wrong output dimension. Should be [num_particles,control_dimensions]=")
            
            
            my_error(self.V(self.X_initial,self.C_inp_initial,0),(self.num_particles,),"V has wrong output dimension. Should be [num_particles,]=")
            my_error(self.Phi(self.X_initial,self.C_inp_initial),(self.num_particles,),"Phi has wrong output dimension. Should be [num_particles,]=")
            self.reset_dynamics() #resets the simulator after testing.
        else:
            dummy = self.single_rollout_step(self.X_initial[0,:],np.zeros(self.control_dimensions),0,self.dt)
            assert(len(dummy)==2), "single_rollout_step should have two outputs."
            my_error(dummy[0],(self.state_dimensions,),"the first output of single_rollout_step has wrong output dimension. Should be [dim,]=")
            my_error(dummy[1],(self.c_inp_dimensions,),"the second output of single_rollout_step has wrong output dimension. Should be [c_inp_dimensions,]=")
            
            my_error(self.controller.predict_function(np.array([self.C_inp_initial[0,:]])),(1,self.control_dimensions),"Output of controller.predict_function has wrong output dimension. Should be [1,control_dimensions]=")
            
            my_error(self.V(self.X_initial[0,:],self.C_inp_initial[0,:],0),(1,),"V has wrong output dimension. Should be () or",())
            my_error(self.Phi(self.X_initial[0,:],self.C_inp_initial[0,:]),(1,),"Phi has wrong output dimension. Should be () or",())
            self.reset_dynamics() #resets the simulator after testing.
    
    def __init__(self,*args, **kwargs):
        
        if len(args) == 12:
             #set parameters
            self.V = args[0] #PathCost. takes state and observations as input
            self.Phi = args[1] #End cost, takes state and observations as input
            self.lambd=args[2] #temperature of control problem
            self.T = args[3] #total time
            self.timelength = args[4] #number of time steps
            self.control_dimensions = args[5]
            self.X_initial = args[6] #initial conditions
            self.C_inp_initial = args[7] # control input
            self.single_rollout_step = args[8] #function (x_new, observation_new) <- f(x_old,u+noise, t, dt)
            self.controller = args[9] #controller which serves as importance sampler takes state and returns control proposal. it is afunction with input-format [self.num_particles,self.c_inp_dimensions] and output-format [self.num_particles,self.control_dimensions]
            self.isparallel = args[10]
            self.reset_dynamics = args[11] #set a reset for a potential simulator
        else:
            assert(len(args)==0), 'Wrong number of arguments, should be 11. V, Phi, lambda, T, timelength, control_dimensions, X_initial, C_inp_initial, single_rollout_step, controller, isparallel'
        
        #initialize arrays:
        self.dt = float(self.T)/self.timelength # timelength are the number of steps (why not call it that way)
        self.state_dimensions = np.size(self.X_initial,axis = 1)
        self.c_inp_dimensions = np.size(self.C_inp_initial,axis = 1)
        self.num_particles = np.size(self.X_initial,axis = 0)
        self.accumStateCost = np.zeros([self.num_particles,1]) #Phi+Integral over V
        self.accumControlCost = np.zeros([self.num_particles,1]) #Integral over 0.5*u'noise_covariance^-1u + u'noise_covariance^-1dW
        
        
        self.X_storage = np.zeros([self.num_particles,self.state_dimensions,self.timelength]) #here the rollout is stored
        self.C_inp_storage = np.zeros([self.num_particles,self.c_inp_dimensions,self.timelength]) #here the visible part of rollout is stored (control input)
        self.Upn_storage = np.zeros([self.num_particles,self.control_dimensions,self.timelength]) #here the control+noise (U Plus Noise) is stored
        
        self.test_dimensionalities()

        
    def do_rollouts(self,uncontrolled = 0):
        if self.isparallel:
            self.multiple_rollouts(uncontrolled)
        else:
            self.do_single_rollout(uncontrolled)
        return self.get_log_weight()
        
    def do_single_rollout(self,uncontrolled = 0):
        for j in range(self.num_particles):
            #here we need a r-loop which creates the rollout and stores all the variables
            #set initial conditions
            self.reset_dynamics()
            self.X_storage[j,:,0] = self.X_initial[j,:]
            self.C_inp_storage[j,:,0] = self.C_inp_initial[j,:]
            self.accumStateCost[j,:] = 0
            self.accumControlCost[j,:] = 0
            
            #precompute the noise
            xi = rng.randn(self.control_dimensions,self.timelength-1)/np.sqrt(self.dt)
            t=0;
            for i in range(self.timelength-1):
        
                u = (1-uncontrolled)*self.controller.predict_function(np.array([self.C_inp_storage[j,:,i]])) #controller gets a state and time dependent signal as input
                u=u[0,:]            
                self.Upn_storage[j,:,i] = u + xi[:,i]; #saves the control+noise
                
                self.X_storage[j,:,i+1], self.C_inp_storage[j,:,i+1] =  self.single_rollout_step(self.X_storage[j,:,i], self.Upn_storage[j,:,i],t,self.dt) #this does one rollout step 
                
                self.accumStateCost[j,:] += self.V(self.X_storage[j,:,i+1], self.C_inp_storage[j,:,i+1],t)#this could also be a function on the observations! it gets 2 inputs in general
                self.accumControlCost[j,:] += 0.5*np.dot(u.T,u)*self.dt + np.dot(u.T,xi[:,i]*self.dt)
                t += self.dt;
            self.accumStateCost[j,:] += self.Phi(self.X_storage[j,:,i], self.C_inp_storage[j,:,i])

        
        
    def multiple_rollouts(self,uncontrolled = 0):
        #here we need a r-loop which creates the rollout and stores all the variables
        self.reset_dynamics()
        self.X_storage[:,:,0] = self.X_initial
        self.C_inp_storage[:,:,0] = self.C_inp_initial
        self.accumStateCost[:,:] = 0
        self.accumControlCost[:,:] = 0
        
        #precompute noise
        xi = rng.randn(self.num_particles,self.control_dimensions,self.timelength-1)/np.sqrt(self.dt)
        t=0;
        for i in range(self.timelength-1):
            u = (1-uncontrolled)*self.controller.predict_function(self.C_inp_storage[:,:,i]) #controller gets a state and time dependent signal as input
            
            self.Upn_storage[:,:,i] = u + xi[:,:,i]; #saves the control+noise
            
            self.X_storage[:,:,i+1], self.C_inp_storage[:,:,i+1] =  self.single_rollout_step(self.X_storage[:,:,i], self.Upn_storage[:,:,i],t,self.dt) #this does one rollout step 
            
            self.accumStateCost[:,0] += self.V(self.X_storage[:,:,i+1], self.C_inp_storage[:,:,i+1],t)#this could also be a function on the observations! it gets 2 inputs in general
            self.accumControlCost[:,0] += 0.5*np.sum(u*u,axis=1)*self.dt + np.sum(u*xi[:,:,i]*self.dt,axis=1)
            t += self.dt;
        
        self.accumStateCost[:,0] += self.Phi(self.X_storage[:,:,-1], self.C_inp_storage[:,:,-1])

    def get_log_weight(self,lambd=-1): # Define this function extra for annealing capability 
        if lambd == -1:
            lambd = self.lambd
        return -(1./lambd)*(self.accumStateCost) - self.accumControlCost 
