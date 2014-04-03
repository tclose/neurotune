from __future__ import absolute_import


class Tuner(object):
    """
    Base Tuner object that contains the three components (objective function, algorithm and 
    simulation objects) and runs the algorithm
    """    
    
    def __init__(self, tuneable_parameters, objective, algorithm, simulation, verbose=False):
        """
        `objective`  -- The objective function to be tuned against [neurotune.objectives.*Objective]
        `algorithm`  -- The algorithm used to tune the cell with [neurotune.algorithms.*Algorithm]
        `simulation` -- The interface to the neuronal simulator used [neurotune.simulations.*Controller] 
        """
        # Set members
        self.tuneable_parameters = tuneable_parameters
        self.objective = objective
        self.algorithm = algorithm
        self.simulation = simulation
        self.verbose = verbose
        # Register tuneable parameters and recording requests
        self.algorithm._set_tuneable_parameters(tuneable_parameters)
        self.simulation._set_tuneable_parameters(tuneable_parameters)
        self.simulation.process_requests(objective.get_recording_requests())
        
    def tune(self, **kwargs):
        """
        Runs the optimisation algorithm and returns the final population and algorithm state
        """
        
        return self.algorithm.optimize(self._evaluate_all_candidates, **kwargs)
    
    def _evaluate_candidate(self, candidate):
        """
        Evaluate the fitness of a single candidate
        """
        if self.verbose:
            print "Evaluating candidate {}".format(candidate)
        return self.objective.fitness(self.simulation.run(candidate))
            
    def _evaluate_all_candidates(self, candidates, args=None): #@UnusedVariable args
        """
        Evaluate each candidate and return the evaluations in a numpy array. To be passed to 
        inspyred optimisation algorithm (overridden in MPI derived class)
        
        `candidates` -- a list of candidates (themselves an iterable of float parameters) 
                        [list(list(float))]
        `args`       -- unused but provided to match inspyred API
        """
        return [self._evaluate_candidate(c) for c in candidates]
