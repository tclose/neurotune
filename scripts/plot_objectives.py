from matplotlib import pyplot as plt
import cPickle as pkl
with open('/home/tclose/Documents/objective_functions.pkl') as f:
    data = pkl.load(f)
plt.figure()    
plt.plot(data[3], data[2])
plt.title('Vanilla')
plt.figure()    
plt.plot(data[3], data[1])
plt.title('Resampled')
plt.figure()    
plt.plot(data[3], data[2])
plt.title('Convolved & Resampled')
plt.show()