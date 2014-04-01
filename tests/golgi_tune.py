import os.path
from nineline.cells.neuron import NineCellMetaClass, simulation_controller
import neurotune as nt

# The path to the original golgi cell 9ml file
cell_9ml=os.path.join('/home', 'tclose', 'git', 'kbrain', '9ml', 'neurons', 'Golgi_Solinas08.9ml')

# Generate the reference trace from the original class
cell = NineCellMetaClass(cell_9ml)()
cell.record('v')
simulation_controller.run(simulation_time=2000.0, timestep=0.05)
reference_trace = cell.get_recording('v')

#Instantiate the tuner
parameters = [nt.Parameter('diam', 'um', 10.0, 40.0)]
# objective = nt.objective.PhasePlaneHistObjective(reference_trace, resample=True)
objective = nt.objective.ConvPhasePlaneHistObjective(reference_trace, kernel_width=(5.25, 18.75),
                                                     resample_type='cubic')
algorithm = nt.algorithm.EDAAlgorithm()
simulation = nt.simulation.NineLineSimulation(cell_9ml)
tuner = nt.Tuner(parameters, objective, algorithm, simulation)

# Run the tuner
pop, ea = tuner.tune(10, 100)
