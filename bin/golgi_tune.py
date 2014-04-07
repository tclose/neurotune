import os.path
from nineline.cells.neuron import NineCellMetaClass, simulation_controller
from neurotune import Tuner, Parameter
from neurotune.objective.phase_plane import ConvPhasePlaneHistObjective
from neurotune.algorithm.genetic import EDAAlgorithm
from neurotune.simulation.nineline import NineLineSimulation

# The path to the original golgi cell 9ml file
cell_9ml=os.path.join('/home', 'tclose', 'git', 'kbrain', '9ml', 'neurons', 'Golgi_Solinas08.9ml')

# Generate the reference trace from the original class
cell = NineCellMetaClass(cell_9ml)()
cell.record('v')
simulation_controller.run(simulation_time=2000.0, timestep=0.05)
reference_trace = cell.get_recording('v')

#Instantiate the tuner
tuner = Tuner([Parameter('diam', 'um', 10.0, 40.0)], 
              ConvPhasePlaneHistObjective(reference_trace), 
              EDAAlgorithm(),
              NineLineSimulation(cell_9ml))

# Run the tuner
pop, ea = tuner.tune()
