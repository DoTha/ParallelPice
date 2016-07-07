import sys
sys.path.append("../")
import numpy as np
import copy
from mpi4py import MPI

import BasicPice as BP
import Controllers.LinearTheano as NN
import Control_Problems.LQ_1 as DCP

comm = MPI.COMM_WORLD
from matplotlib import pyplot as plt 

#for saving
file_identifier = sys.argv[1]+"_"+sys.argv[2] #creates a identifier to store the file
def my_save_local(variable,file_idx):
    if comm.Get_rank() == 0:
        np.save("../Data/"+variable+"__"+file_idx,eval(variable))

number_of_rollouts = 10

lambd = 10
T = 10
timelength = 1000

#instantiate objects
controller = NN.Controller(10,100,1,1,0.001,"adam") #build a neural network which serves as controller. parameters: [depth,width,c_inp_dim,control_dim,learning_rate]
d_control_problem = DCP.specific_Control_Problem(controller,number_of_rollouts, lambd, T, timelength) #define a control problem

ParDataGen = BP.Parallel_DataGenerator(comm,d_control_problem)  #instantiate the parallel data-generator
parallel_trainer = BP.Parallel_Trainer(comm,ParDataGen, controller) #instantiate the parallel trainer



#loop
rollout_epochs = 5
if comm.Get_rank()==0:
    EffSS_store = np.zeros(rollout_epochs)
    val_err_store = np.zeros(rollout_epochs)

for i in range(rollout_epochs):
    ParDataGen.do_rollouts(1)
    if comm.Get_rank() == 0:
        EffSS_store[i] = ParDataGen.EffSS()
    val_err = parallel_trainer.train_network(2, 300, lambda x: x<0.01)
    if comm.Get_rank() == 0: val_err_store[i] = val_err


print "Test 2D, parallel passed"
