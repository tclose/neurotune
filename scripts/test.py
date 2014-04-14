import cPickle as pkl
from nineline.cells.neuron import NineCellMetaClass, simulation_controller
from neurotune.objective.phase_plane import ConvPhasePlaneHistObjective
from neurotune.simulation.nineline import NineLineSimulation
  
cell9ml = '/home/t/tclose/git/kbrain/9ml/neurons/Golgi_Solinas08.9ml'
# Generate the reference trace from the original class
cell = NineCellMetaClass(cell9ml)()
cell.record('v')
simulation_controller.run(simulation_time=2000.0, timestep=0.025)
reference_trace = cell.get_recording('v')
# Select which objective function to use
objective = ConvPhasePlaneHistObjective(reference_trace)
simulation = NineLineSimulation(cell9ml)
print "done"
#candidate, recordings = pkl.load(open('/home/tclose/Data/NeuroTune/failed_tune/evaluation_exception.pkl'))
#print objective.fitness(recordings)
# print candidate
# for (obj, _), rec in recordings.iteritems():
#     print obj.fitness(rec)

# from nineline.cells.neuron import NineCellMetaClass, simulation_controller
# from neurotune.objective.phase_plane import PhasePlaneHistObjective, ConvPhasePlaneHistObjective, PhasePlanePointwiseObjective
# from neurotune.objective.multi import MultiObjective
# from neurotune.simulation.nineline import NineLineSimulation
# from neurotune.algorithm.grid import GridAlgorithm
# from neurotune.tuner import Tuner
# from neurotune import Parameter
# 
# cell_9ml = '/home/tclose/git/kbrain/9ml/neurons/Golgi_Solinas08.9ml'
# cell = NineCellMetaClass(cell_9ml)()
# cell.record('v')
# simulation_controller.run(simulation_time=2000.0, timestep=0.025)
# reference_trace = cell.get_recording('v')
#     
# parameters = [Parameter('diam', 'um', 10.0, 40.0),
#               Parameter('soma.Lkg.gbar', 'S/cm^2', 1e-5, 3e-5)]
# 
# objective = MultiObjective(PhasePlaneHistObjective(reference_trace), 
#                            ConvPhasePlaneHistObjective(reference_trace),
#                            PhasePlanePointwiseObjective(reference_trace, (20, -20), 100))
# 
# simulation = NineLineSimulation(cell_9ml)
# 
# # Instantiate the tuner
# tuner = Tuner(parameters,
#               objective,
#               GridAlgorithm(num_steps=11),
#               simulation)
# 
# recording = simulation.run([  3.70000000e+01,   2.40000000e-05])
# print objective.fitness(recording)
