import theano
import theano.tensor as T
import sys
sys.path.append("../../")
import numpy as np
from BasicPice.define_Theano_Control_Problem import define_Theano_Control_Problem
from theano import shared

class specific_Control_Problem(define_Theano_Control_Problem):
    
    def set_initial_conditions(self): #if you want to set your own initial conditions 
        x_initial = np.ones([self.number_of_rollouts,1]) #the initial conditions. They can be set for each rollout on this core independently. (for tensegrity this is not important and will be likely ignored)
        y_initial = np.ones([self.number_of_rollouts,1])
        
        return [x_initial,y_initial]

    def symbolic_V(self,symbolic_X_list, symbolic_c_inp_list,t):
        x = symbolic_X_list[0]
        y = symbolic_X_list[1]
        return 0.

    def symbolic_Phi(self,symbolic_X_list, symbolic_c_inp_list):
        x = symbolic_X_list[0]
        y = symbolic_X_list[1]
        return T.sum(x**2+y**2)

    def symbolic_f(self,symbolic_X_list,t):
        x = symbolic_X_list[0]
        y = symbolic_X_list[1]
        fx = -x+y #T.as_tensor(0.)
        fy = -x**2 #T.as_tensor(0.)
        return [fx,fy]

    def symbolic_g(self, symbolic_X_list,t):
        '''
        the gx for every state x must be a matrix with dimensions [number_of_rollouts,x_dim, control_dim]
        with x.shape = [number_of_rollouts, x_dim]
        '''
        x = symbolic_X_list[0]
        y = symbolic_X_list[1]
        gx = T.as_tensor(np.ones([1,self.control_dimensions]))
        gy = T.as_tensor(np.ones([1,self.control_dimensions]))
        return [gx,gy]

    def symbolic_c_inp_function(self, symbolic_X_list,t):
        x = symbolic_X_list[0]
        y = symbolic_X_list[1]
        return [x,y]
