from __future__ import absolute_import
import traceback


class EvaluationException(Exception):
    
    def __init__(self, candidate, tback=None):
        self.candidate = candidate
        if tback is None:
            self.traceback = traceback.format_exc()
        else:
            self.traceback = tback
        self.message = ("Error while evaluating candidate {} with the following error:\n\n{}"
                        .format(candidate, tback))

    def __str__(self):
        return self.message

    def save_candidate(self, filename):
        with open(filename, 'w') as f:
            f.write(', '.join([str(c) for c in self.candidate]))
        print "Saving candidate that caused exception to file at '{}'".format(filename)
            
    
class Tuner(object):
    """
    Base Tuner object that contains the three components (objective function, algorithm and 
    simulation objects) and runs the algorithm
    """    
    
    def __init__(self, *args, **kwargs):
        self.set(*args, **kwargs)
    
    def set(self, tuneable_parameters, objective, algorithm, simulation, verbose=False):
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
        try:
            fitness = self.objective.fitness(self.simulation.run(candidate))
        except Exception:
            raise EvaluationException(candidate)
        return fitness
            
    def _evaluate_all_candidates(self, candidates, args=None): #@UnusedVariable args
        """
        Evaluate each candidate and return the evaluations in a numpy array. To be passed to 
        inspyred optimisation algorithm (overridden in MPI derived class)
        
        `candidates` -- a list of candidates (themselves an iterable of float parameters) 
                        [list(list(float))]
        `args`       -- unused but provided to match inspyred API
        """
        return [self._evaluate_candidate(c) for c in candidates]
    
    @classmethod
    def is_master(self):
        """
        Provided for convenient interoperability with the MPITuner class
        """
        return True
