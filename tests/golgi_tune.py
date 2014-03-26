import os.path
from nineline.cells.neuron import NineCellMetaClass, simulation_controller
import neurotune as nt

cell_9ml=os.path.join('/home', 'tclose', 'git', 'kbrain', '9ml', 'neurons', 'Golgi_Solinas08.9ml')

cell = NineCellMetaClass(cell_9ml)()
cell.record('v')
simulation_controller.run(simulation_time=2000.0, timestep=0.25)
reference_trace = cell.get_recording('v')

tuner = nt.Tuner([nt.Parameter('diam', 'um', 10.0, 40.0)],
                 nt.objective.PhasePlaneHistObjective(reference_trace, simulation_time=2000.0), 
                 nt.algorithm.EDAAlgorithm(), 
                 nt.simulation.NineLineSimulation(cell_9ml))
pop, ea = tuner.tune(10, 100)
