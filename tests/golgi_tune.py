import os.path
# try:
#     import mpi4py #@UnusedImport
#     from neurotune import MPITuner as Tuner
# except ImportError:
#     from neurotune import Tuner as Tuner
from nineline.cells.neuron import NineCellMetaClass, simulation_controller
from neurotune import Tuner
from neurotune.simulation import NineLineSimulation
from neurotune.algorithm import EDAAlgorithm
from neurotune.objective import PhasePlaneHistObjective

cell_9ml=os.path.join('/home', 'tclose', 'git', 'kbrain', '9ml', 'neurons', 'Golgi_Solinas08.9ml')
genome_keys='{soma}diam'
constraints=[(10.0, 40.0)]
simulation_time = 2000.0

cell = NineCellMetaClass(cell_9ml)()
cell.record('v')
simulation_controller.run(simulation_time)
reference_trace = cell.get_recording('v')

tuner = Tuner(PhasePlaneHistObjective(reference_trace, simulation_time), EDAAlgorithm(constraints), 
              NineLineSimulation(cell_9ml, genome_keys))
pop, ea = tuner.tune(10, 100)

print pop
