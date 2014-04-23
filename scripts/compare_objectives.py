import os.path
from nineline.cells.neuron import NineCellMetaClass, simulation_controller
import neurotune.objective
import neurotune.simulation
from matplotlib import pyplot as plt
import numpy
import cPickle as pickle

# The path to the original golgi cell 9ml file
cell_9ml=os.path.join('/home', 'tclose', 'git', 'kbrain', '9ml', 'neurons', 'Golgi_Solinas08.9ml')

# Generate the reference trace from the original class
cell = NineCellMetaClass(cell_9ml)()
cell.record('v')
simulation_controller.run(simulation_time=2000.0, timestep=0.05)
reference_trace = cell.get_recording('v')

with open('/home/tclose/Documents/reference_trace.pkl', 'w') as f:
    pickle.dump(reference_trace, f)

#Instantiate the tuner
plain_objective = neurotune.objective.phase_plane.PhasePlaneHistObjective(reference_trace, 
                                                                          resample=False)
resamp_objective = neurotune.objective.phase_plane.PhasePlaneHistObjective(reference_trace)
conv_objective = neurotune.objective.phase_plane.ConvPhasePlaneHistObjective(reference_trace)

objectives = [resamp_objective, conv_objective, plain_objective]

resamp_objective.plot_d_dvdt(reference_trace, show='/home/tclose/Documents/v_dvdt.pkl') # show=False

for obj, title in zip(objectives, ('resampled', 'convolved', 'plain')):
    obj.plot_hist(reference_trace, show=os.path.join('/home', 'tclose', 'Documents', title + '.pkl')) # show=False
    plt.title(title)

parameters = [neurotune.Parameter('diam', 'um', 10.0, 40.0)]

grid_length = 11
param_range = numpy.linspace(-15.0, 15.0, grid_length) + 27.0

to_pickle = []
for j, obj in enumerate(objectives):
    simulation = neurotune.simulation.nineline.NineLineSimulation(cell_9ml)
    simulation._set_tuneable_parameters(parameters)
    simulation.process_requests(obj.get_recording_requests())
    fitnesses = numpy.empty(grid_length)    
    for i, param in enumerate(param_range):
        recordings = simulation.run([param])
        fitnesses[i] = obj.fitness(recordings)
        print "finished param {} of objective {}".format(i, j)
    to_pickle.append(fitnesses)
    plt.figure()
    plt.plot(param_range, fitnesses)
to_pickle.append(param_range)
with open('/home/tclose/Documents/objective_functions.pkl', 'w') as f:
    pickle.dump(to_pickle, f)
plt.show()