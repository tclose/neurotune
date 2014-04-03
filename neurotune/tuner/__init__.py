

class Tuner(object):
    
    def __init__(self, tuneable_parameters, objective, algorithm, simulation):
        """
        Initialises the Tuner object
        
        `objective`  -- The objective function to be tuned against [neurotune.objectives.*Objective]
        `algorithm`  -- The algorithm used to tune the cell with [neurotune.algorithms.*Algorithm]
        `simulation` -- The interface to the neuronal simulator used [neurotune.simulations.*Controller] 
        """
        self.tuneable_parameters = tuneable_parameters
        self.objective = objective
        self.algorithm = algorithm
        self.algorithm._set_tuneable_parameters(tuneable_parameters)
        self.simulation = simulation
        self.simulation._set_tuneable_parameters(tuneable_parameters)
        self.simulation.process_requests(objective.get_recording_requests())
        
    def tune(self, num_candidates, max_iterations, seeds=None, random_seed=None, 
             stats_filename=None, indiv_filename=None, **kwargs):
        """
        Runs the optimisation algorithm and returns the final population and algorithm state
        
        `num_candidates`  -- the number of candidates to use in the algorithm
        `max_iterations`  -- the maximum number of iterations to perform
        `random_seed`     -- the seed to initialise the candidates with
        `stats_filename`  -- the name of the file to save the generation-based statistics in
        `indiv_filename`  -- the name of the file to save the candidate parameters in
        `kwargs`          -- optional arguments to be passed to the optimisation algorithm
        """
        self._open_readout_files(stats_filename, indiv_filename, kwargs)
        result = self.algorithm.optimize(num_candidates, self._evaluate_all_candidates, 
                                         max_iterations, seeds=seeds, random_seed=random_seed,
                                         **kwargs)
        self._close_readout_files()
        return result
        
    def _evaluate_all_candidates(self, candidates, args): #@UnusedVariable args
        """
        Evaluate each candidate and return the evaluations in a numpy array. To be passed to inspyred
        optimisation algorithm
        """
        return [self._evaluate_candidate(c) for c in candidates]
    
    def _evaluate_candidate(self, candidate):
        """
        Evaluate the fitness of a single candidate
        """
        return self.objective.fitness(self.simulation.run(candidate))
    
    def _open_readout_files(self, stats_filename, indiv_filename, kwargs):
        if stats_filename:
            self.stats_file = kwargs['statistics_file'] = open(stats_filename, 'w')
        else:
            self.stats_file = None
        if indiv_filename:
            self.indiv_file = kwargs['individual_file'] = open(indiv_filename, 'w')
        else:
            self.indiv_file = None
            
    def _close_readout_files(self):
        if self.stats_file:
            self.stats_file.close()
        if self.indiv_file:
            self.indiv_file.close()
