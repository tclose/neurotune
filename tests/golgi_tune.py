import os.path
from nineline.cells.neuron import NineCellMetaClass, simulation_controller
import neurotune as nt

# The path to the original golgi cell 9ml file
cell_9ml=os.path.join('/home', 'tclose', 'git', 'kbrain', '9ml', 'neurons', 'Golgi_Solinas08.9ml')

# Generate the reference trace from the original class
cell = NineCellMetaClass(cell_9ml)()
cell.record('v')
simulation_controller.run(simulation_time=2000.0, timestep=0.25)
reference_trace = cell.get_recording('v')

#Instantiate the tuner
parameters = [nt.Parameter('diam', 'um', 10.0, 40.0)]
objective = nt.objective.PhasePlaneHistObjective(reference_trace, resample=True)
algorithm = nt.algorithm.EDAAlgorithm()
simulation = nt.simulation.NineLineSimulation(cell_9ml)
tuner = nt.Tuner(parameters, objective, algorithm, simulation)


objective.plot_hist(reference_trace)

# Run the tuner
#pop, ea = tuner.tune(10, 100)
