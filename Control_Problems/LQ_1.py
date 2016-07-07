import sys
sys.path.append("../../")
import numpy as np
from BasicPice.define_Control_Problem import define_Control_Problem


#TODO: I think the best would be if this class just inherits from Controller
class specific_Control_Problem(define_Control_Problem):
    
    dim = 1
    isparallel = 1
    '''
    These variables you have to define otherwise they are set to there standard values
        control_dimensions = 1
        dim = 1
        isparallel = 1 #option if the rollouts on a single core are implemented sequentiallty or in parallel. Parallel means here, that at each time point all simulations are updated 1 timestep. This can be much more efficient than the sequentiall option. 
    '''
    '''
    You can additionally define standard values for lambda, T and timelength. 
    These values will however be overwritten if they are given at initialization.
    '''
    
    
    #model-specific parameters
    var_obs = 0.01
    noise_covariance = 1.0

    '''
    def _set_initial_conditions(self):
        self.X_initial = np.zeros([self.number_of_rollouts,self.dim]) #the initial conditions. They can be set for each rollout on this core independently. (for tensegrity this is not important and will be likely ignored)
        self.C_inp_initial = self.X_initial #the corresponding observations of the initial conditions.
    '''
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
        return np.sum(state*0.0,axis = 1) #default implementation: no Path-cost
        

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
        return np.sum((state-1.0)**2/(2*self.var_obs),axis = 1)

    
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
        return (state+np.dot(dt*self.noise_covariance,uplusnoise), state+dt*np.dot(dt*self.noise_covariance,uplusnoise))
        
    def reset_dynamics(self):
        '''if the dynamics are done by a simulation (tensegrity) please put here a handle to the function which resets the dynamics. This will be called everytime before a tollout is simulated.'''
        pass
