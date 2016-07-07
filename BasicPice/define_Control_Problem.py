import numpy as np
from BasicPice import Control_Problem


'''
This is a raw form of defining a control problem.
When called please give at least (controller, number_of_rollouts).
A basic LQ control problem with zero costs is default.
To define a new control problem inherit from this class and define your own functions.
If you inherit from this class you have an initializer with which you can set the parameters number_of_rollouts, lambda, T, timelength
default values are set:
    control_dimensions = 1
    dim = 1
    isparallel = 1
'''
class define_Control_Problem(Control_Problem):
    

    dim = 1
    isparallel = 1 #option if the rollouts on a single core are implemented sequentiallty or in parallel. Parallel means here, that at each time point all simulations are updated 1 timestep. This can be much more efficient than the sequentiall option. 
    
    def __init__(self,*args):
        
        self.controller = args[0] #put here a handle to a function which computes the optimal control. Takes state and returns control proposal. It is a function with input-format [self.num_particles,self.c_inp_dimensions] and output-format [self.num_particles,self.control_dimensions]. If isparallel=0 then input is and [1,c_inp_dimensions] output is [1,control_dimensions].
        
        self.control_dimensions = self.controller.target_dim #read out the dimension of the controller outout from the controller
        self.controller_input_dim = self.controller.input_dim #read out the the input dimension of the controller
        
        self.number_of_rollouts = args[1]
        
        if len(args) == 5:
            #controll parameters
            self.lambd = args[2] #the temperature of the control problem
            self.T = args[3] #The Time Horizon
            #simulation parameters
            self.timelength = args[4] #number of timesteps
        elif len(args)>2:
            print "warning: in define_Control_Problem. not enough or too many arguments given"
        
        self._set_initial_conditions()
        
        super(define_Control_Problem, self).__init__() #this calles the initializer of the parent class. There some arrays are initialized
        
    def _set_initial_conditions(self): #if you want to set your own initial conditions 
        self.X_initial = np.zeros([self.number_of_rollouts,self.dim]) #the initial conditions. They can be set for each rollout on this core independently. (for tensegrity this is not important and will be likely ignored)
        self.C_inp_initial = np.zeros([self.number_of_rollouts,self.controller_input_dim]) #the corresponding observations of the initial conditions.
    
    def V(self,state, observations,t):
        '''
        define here the Path-cost as a function.
        x is thereby the full state while y are the observations.
        If you use the isparallel=1 option:
                Input: [number_of_rollouts,dim]
                Output: [self.num_particles,].
        If isparallel=0 
                Input: [1,]
                Output: [1,] or []
        '''
        return state*0.0 #default implementation: no Path-cost
    
    def Phi(self,state, observations):
        '''
        define here the End-cost as a function.
        x is thereby the full state while y are the observations.
        If you use the isparallel=1 option:
                Input: [number_of_rollouts,dim]
                Output: [self.num_particles,].
        If isparallel=0 
                Input: [1,]
                Output: [1,] or []
        '''
        return state*0.0 #simple implementation: no End-Cost
    
    def single_rollout_step(self,state,uplusnoise,t,dt):
        '''
        The dynamics.
        function (x_new, observation_new) <- f(x_old,u+noise, t, dt). 
        If you use the isparallel=1 option:
                Input: [number_of_rollouts,state_dimensions],[number_of_rollouts,control_dimensions], scalar, scalar
                Output: [number_of_rollouts,state_dimensions],[number_of_rollouts,c_inp_dim]
        If isparallel=0 
                Input: [state_dimensions,],[control_dimensions,], scalar, scalar
                Output: [state_dimensions,], [c_inp_dimensions,]
        '''
        return (state+np.dot(dt,uplusnoise), state+dt*np.dot(dt,uplusnoise)) #default implementation: wiener process in one dimension
        
    def reset_dynamics(self):
        '''if the dynamics are done by a simulation (tensegrity) please put here a handle to the function which resets the dynamics. This will be called everytime before a tollout is simulated.'''
        pass #default implementation: no reset

















