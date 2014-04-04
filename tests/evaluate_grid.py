#!/usr/bin/python
import os.path
from nineline.cells.neuron import NineCellMetaClass, simulation_controller
from neurotune import Parameter
# from neurotune import Tuner
from neurotune.tuner.mpi import MPITuner as Tuner
from neurotune.objective.phase_plane import ConvPhasePlaneHistObjective
from neurotune.algorithm.grid import GridAlgorithm
from neurotune.simulation.nineline import NineLineSimulation
import cPickle as pkl
from matplotlib import pyplot as plt

# The path to the original golgi cell 9ml file
cell_9ml = os.path.join('/home', 'tclose', 'git', 'kbrain', '9ml', 'neurons', 'Golgi_Solinas08.9ml')

# Generate the reference trace from the original class
cell = NineCellMetaClass(cell_9ml)()
cell.record('v')
simulation_controller.run(simulation_time=2000.0, timestep=0.05)
reference_trace = cell.get_recording('v')

parameters = [Parameter('diam', 'um', 10.0, 40.0),
              Parameter('soma.Lkg.gbar', 'S/cm^2', 1e-5, 3e-5)]

# Instantiate the tuner
tuner = Tuner(parameters,
              ConvPhasePlaneHistObjective(reference_trace),
              GridAlgorithm(num_steps=11),
              NineLineSimulation(cell_9ml))

# Run the tuner
pop, grid = tuner.tune()

if tuner.is_master():
    print "Fittest candidate {}".format(pop)
    
    with open('/home/tclose/Data/NeuroTune/tune_grid.pkl', 'w') as f:
        pkl.dump(grid, f)
    
    plt.imshow(grid, interpolation='nearest', origin='lower', aspect='auto',
               extent=(parameters[0][2], parameters[0][3], parameters[1][2], parameters[1][3]))
    plt.show()


