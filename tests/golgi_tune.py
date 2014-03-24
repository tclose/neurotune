import os.path
from neurotune import Tuner
from neurotune.simulation import NineLineController
from neurotune.algorithm import EDAAlgorithm
from neurotune.objective import PhasePlaneHistObjective

cell_9ml=os.path.join('/home', 'tclose', 'git', 'kbrain', '9ml', 'neurons', 'Golgi_Solinas08.9ml')
reference_traces_filename=os.path.join('/home', 'tclose', 'traces_file.pkl')
genome_keys='{soma}diam'
constraints=[(10.0, 40.0)]




tuner = Tuner(PhasePlaneHistObjective(reference_traces_filename), EDAAlgorithm(constraints), 
              NineLineController(cell_9ml, genome_keys))
pop, ea = tuner.tune(10, 100)

print pop
