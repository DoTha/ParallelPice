import theano
import theano.tensor as T
import numpy as np
from BasicPice import Control_Problem
from theano import shared
import collections

class define_Theano_Control_Problem(Control_Problem):
    
    ######################################################################
    ######## private functions:      #####################################
    ######################################################################
    
    def _compile_theano_functions(self):
        dummy_x = T.matrix("dummy_x") #dummy variable for input. will be ignored
        dummy_c_inp = T.matrix("dummy_c_inp") #dummy variable for input. will be ignored

        sym_uplusnoise = T.matrix("uplusnoise")
        t = T.scalar("t")   
        dt = T.scalar("dt")  
        self.symbolic_X_list = [shared(state) for state in self.X_init_list]
        symbolic_c_inp_list = self.symbolic_c_inp_function(self.symbolic_X_list,t)

        V_sym = self._batch_vectorization(fun = self.symbolic_V,symbolic_X_list = self.symbolic_X_list,symbolic_c_inp_list = symbolic_c_inp_list,t = t)
        print "compiling V..."
        self.V = theano.function([dummy_x, dummy_c_inp, t],V_sym, on_unused_input = 'ignore')
        print "...done"

        Phi_sym = self._batch_vectorization(fun = self.symbolic_Phi,symbolic_X_list = self.symbolic_X_list,symbolic_c_inp_list = symbolic_c_inp_list)
        print "compiling Phi..."
        self.Phi = theano.function([dummy_x, dummy_c_inp],Phi_sym, on_unused_input = 'ignore')
        print "...done"

        stacked_new_symbolic_X, stacked_new_symbolic_c_inp, update = self._symbolic_single_step_update(self.symbolic_X_list,symbolic_c_inp_list,sym_uplusnoise,t,dt)
        print "compiling single_rollout_step..."
        self.single_rollout_step = theano.function([dummy_x,sym_uplusnoise,t,dt],[stacked_new_symbolic_X,stacked_new_symbolic_c_inp],updates = update,on_unused_input = 'ignore')
        print "...done"
        
        stacked_symbolic_X,stacked_symbolic_c_inp = self._sym_get_stacked_X_C_inp(self.symbolic_X_list,t)
        self._get_stacked_X_C_inp = theano.function([t],[stacked_symbolic_X,stacked_symbolic_c_inp], on_unused_input = 'ignore')
        
        g = self._batch_vectorization(fun = self.symbolic_g,symbolic_X_list = self.symbolic_X_list,t = t)
        self._g_test_function = theano.function([t],g, on_unused_input = 'ignore')
        
    def _symbolic_single_step_update(self,symbolic_X_list,symbolic_c_inp_list,uplusnoise,t,dt):
        f_list = self._batch_vectorization(fun = self.symbolic_f,symbolic_X_list = self.symbolic_X_list,t = t)
        g_list = self._batch_vectorization(fun = self.symbolic_g,symbolic_X_list = self.symbolic_X_list,t = t)

        new_symbolic_X_list = [symbolic_X+dt*(f_it+T.batched_dot(g_it,uplusnoise)) for symbolic_X,f_it,g_it in zip(symbolic_X_list,f_list,g_list)]
        
        stacked_new_symbolic_X,stacked_new_symbolic_c_inp = self._sym_get_stacked_X_C_inp(new_symbolic_X_list,t)

        update = collections.OrderedDict()
        for key, value in zip(symbolic_X_list,new_symbolic_X_list):
            update[key] = value
        
        return stacked_new_symbolic_X, stacked_new_symbolic_c_inp, update
    
    def _sym_get_stacked_X_C_inp(self,symbolic_X_list,t):
        symbolic_c_inp_list = self.symbolic_c_inp_function(symbolic_X_list,t)
        stacked_symbolic_c_inp = T.concatenate(symbolic_c_inp_list,axis = 1)
        stacked_symbolic_X = T.concatenate(symbolic_X_list,axis = 1)
        return stacked_symbolic_X,stacked_symbolic_c_inp
    
    def _sym_reset(self,symbolic_X_list,state_list_init,symbolic_c_inp_list,sym_c_inp_list):
        update = collections.OrderedDict()
        for key, value in zip(symbolic_X_list,state_list_init):
            update[key] = value
        sym_c_inp_list = self.c_inp_function(new_state_list,t)
        for key, value in zip(symbolic_c_inp_list,sym_c_inp_list):
            update[key] = value
        return update
    
    def _batch_vectorization(self,**args):
        fun_in = args["fun"]
        symbolic_X_list = args["symbolic_X_list"]
        if "symbolic_c_inp_list" in args and "t" in args:
            t = args["t"]
            symbolic_c_inp_list = args["symbolic_c_inp_list"]
            fun = lambda x,y: fun_in(x,y,t)
        elif "symbolic_c_inp_list" in args and "t" not in args:
            symbolic_c_inp_list = args["symbolic_c_inp_list"]
            fun = fun_in
        elif "symbolic_c_inp_list" not in args and "t" in args:
            t = args["t"]
            symbolic_c_inp_list = []
            fun = lambda x,y: fun_in(x,t)

        fun_list = []
        for i in np.arange(self.number_of_rollouts):
            symbolic_X_list_i = [a[i] for a in symbolic_X_list]
            symbolic_c_inp_list_i = [a[i] for a in symbolic_c_inp_list]
            out_list = fun(symbolic_X_list_i,symbolic_c_inp_list)
            fun_list.append(out_list)
        if type(fun_list[0]) != list:
            return T.stack(fun_list,axis = 0)
        else:
            ziped_list = [list(a) for a in zip(*fun_list)]
            return [T.stack(a,axis = 0) for a in ziped_list]
    
    def _consistency_tests(self):
        test_g = self._g_test_function(0.)
        for i,a in enumerate(test_g):
            print a.shape[2]
            assert(a.shape[2] == self.control_dimensions), \
            "number of control dimensions in the controller and in the definition of symbolic_g is inconsistent. For the controller it is "\
            +str(self.control_dimensions)+" and for the "+str(i)+"th dimension of g it is "+str(a.shape[2])
            
        assert(self.controller_input_dim ==  self.C_inp_initial.shape[1]), "C_inp_dimensions of controller (="+str(self.controller_input_dim)+") and symbolic_c_inp_function (="+str(self.C_inp_initial.shape[1])+") are inconsistent."
    
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
        
        self.isparallel = 1
        self.X_init_list = self.set_initial_conditions()
        self._compile_theano_functions()
        self.X_initial, self.C_inp_initial = self._get_stacked_X_C_inp(0.)
        self._consistency_tests()
        super(define_Theano_Control_Problem, self).__init__() #this calles the initializer of the parent class. There some arrays are initialized
        
    ######################################################################
    ###########public function which are not set by user #################
    ######################################################################
    
    def reset_dynamics(self):
        #reset symbolic_state to initial conditions
        for symbolic_X,X_init in zip(self.symbolic_X_list,self.X_init_list):
            symbolic_X.set_value(X_init)
    
    ######################################################################
    ######## User defined functions: #####################################
    ######################################################################
    '''
    will be overwritten in a children class
    '''
    def set_initial_conditions(self): #if you want to set your own initial conditions 
        x_initial = np.ones([self.number_of_rollouts,3]) #the initial conditions. They can be set for each rollout on this core independently. (for tensegrity this is not important and will be likely ignored)
        y_initial = np.ones([self.number_of_rollouts,1])
        z_initial = np.ones([self.number_of_rollouts,1])
        
        return [x_initial,y_initial,z_initial]

    def symbolic_V(self,symbolic_X_list, symbolic_c_inp_list,t):
        x = symbolic_X_list[0]
        y = symbolic_X_list[1]
        z = symbolic_X_list[2]
        return T.sum(x**2)+T.sum(y**2)+T.sum(z**2)

    def symbolic_Phi(self,symbolic_X_list, symbolic_c_inp_list):
        x = symbolic_X_list[0]
        y = symbolic_X_list[1]
        z = symbolic_X_list[2]
        return T.sum(x**2)+T.sum(y**2)+T.sum(z**2)

    def symbolic_f(self,symbolic_X_list,t):
        x = symbolic_X_list[0]
        y = symbolic_X_list[1]
        z = symbolic_X_list[2]
        fx = -x
        fy = -y
        fz = -z
        return [fx,fy,fz]

    def symbolic_g(self, symbolic_X_list,t):
        '''
        the gx for every state x must be a matrix with dimensions [number_of_rollouts,x_dim, control_dim]
        with x.shape = [number_of_rollouts, x_dim]
        '''
        x = symbolic_X_list[0]
        y = symbolic_X_list[1]
        z = symbolic_X_list[2]
        gx = T.as_tensor(np.zeros([3,self.control_dimensions]))
        gy = T.as_tensor(np.ones([1,self.control_dimensions]))
        gz = T.as_tensor(np.zeros([1,self.control_dimensions]))
        return [gx,gy,gz]

    def symbolic_c_inp_function(self, symbolic_X_list,t):
        x = symbolic_X_list[0]
        y = symbolic_X_list[1]
        z = symbolic_X_list[2]
        return [x,y+z]
