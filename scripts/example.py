import os.path
import quantities as pq
from neurotune import Parameter
from neurotune.algorithm.inspyred import NSGA2Algorithm, ec
from neurotune.objective.multi import MultiObjective
from neurotune.objective.spike import (SpikeFrequencyObjective,
                                       SpikeAmplitudeObjective)

from neurotune.simulation.nineline import NineLineSimulation

#try:
#    from neurotune.tuner.mpi import MPITuner as Tuner
#except ImportError:
#    from neurotune.tuner import Tuner
from neurotune.tuner import Tuner

# Parameter(name, units, lbound, ubound, log_scale=False)
parameters = [Parameter('param1', 'V', -70, -50),
              Parameter('param2', 'S/cm^2', -3, -2, log_scale=True)]
algorithm = NSGA2Algorithm(args.population_size,
                     	   max_generations=args.num_generations,
                           observer=[ec.observers.population_observer],
                           output_dir=os.path.join(os.environ['HOME'],
                                                   'neurotune_out'))
                                                   
freq_objective = SpikeFrequencyObjective(100 * pq.Hz, time_start=500.0 * pq.ms,
                                time_stop=2000.0 * pq.ms)
amp_objective = SpikeAmplitudeObjective(10 * pq.mV, time_start=500.0 * pq.ms,
                                        time_stop=2000.0 * pq.ms)                                                   
objective = MultiObjective(freq_objective, amp_objective)
simulation = ?

tuner = Tuner(parameters,
			  objective,
			  algorithm,
			  simulation,
			  verbose=args.verbose)
			  
best_population, tuner_output = tuner.tune()                                  