import os.path
# try:
#     import mpi4py #@UnusedImport
#     from neurotune import MPITuner as Tuner
# except ImportError:
#     from neurotune import Tuner as Tuner
import nineline.pyNN.neuron
from neurotune import Tuner
from neurotune.simulation import NineLineSimulation
from neurotune.algorithm import EDAAlgorithm
from neurotune.objective import PhasePlaneHistObjective

cell_9ml=os.path.join('/home', 'tclose', 'git', 'kbrain', '9ml', 'neurons', 'Golgi_Solinas08.9ml')
genome_keys='{soma}diam'
constraints=[(10.0, 40.0)]
record_time = 2000.0

pop, cell = nineline.pyNN.neuron.create_singleton_population(cell_9ml, {})
pop.record('{soma}v')
nineline.pyNN.neuron.setup()
nineline.pyNN.neuron.run(record_time)
reference_traces = pop.get_data('{soma}v')

tuner = Tuner(PhasePlaneHistObjective(reference_traces, record_time), EDAAlgorithm(constraints), 
              NineLineSimulation(cell_9ml, genome_keys))
pop, ea = tuner.tune(10, 100)

print pop
