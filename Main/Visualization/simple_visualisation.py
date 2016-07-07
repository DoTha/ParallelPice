import sys
import numpy as np
import matplotlib.pyplot as plt
my_line=[]
which_file = int(sys.argv[1]) #which file do you want? 1 for the latest which was created
for line in open('../Data/list_of_data.dat','r'):
    my_line.append(line[:-1])
latest_file_identifier=my_line[-which_file] #find latest file_identifier
print latest_file_identifier
def my_load_local(variable,file_idx):
    return np.load("../Data/"+variable+"__"+file_idx+".npy")

#load variables
EffSS_store = my_load_local("EffSS_store", latest_file_identifier)
val_err_store = my_load_local("val_err_store", latest_file_identifier)

#plot
plt.figure(1)
plt.subplot(211)
plt.plot(EffSS_store)
plt.xlabel('iterations')
plt.ylabel('EffSS')
plt.subplot(212)
plt.plot(val_err_store)
plt.xlabel('iterations')
plt.ylabel('val_err')

plt.show()
