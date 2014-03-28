import numpy
from collections import deque, namedtuple
from abc import ABCMeta # Metaclass for abstract base classes
import neurotune.simulation as simulation
import neurotune.algorithm as algorithm
import neurotune.objective as objective
try:
    from mpi4py import MPI
except ImportError:
    pass


Parameter = namedtuple('Parameter', 'name units lbound ubound')
       

class BaseTuner(object):
    
    __metaclass__ = ABCMeta # Declare this class abstract to avoid accidental construction
    
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
        
    def tune(self, pop_size, max_iterations, random_seed=None, stats_filename=None, #@UnusedVariable
             indiv_filename=None, **kwargs): #@UnusedVariable
        """
        Runs the optimisation algorithm and returns the final population and algorithm state
        """
        raise NotImplementedError("'tune' method should be implemented by derived class")
    
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
    

class Tuner(BaseTuner):
    """
    A basic tuner class that runs the complete algorithm on the current node (good for debugging)
    """
    
    def tune(self, num_candidates, max_iterations, seeds=None, random_seed=None, 
             stats_filename=None, indiv_filename=None, **kwargs):
        """
        Runs the optimisation algorithm and returns the final population and algorithm state
        
        `num_candidates`  -- the number of candidates to use in the algorithm
        `max_iterations`  -- the maximum number of iterations to perform
        `random_seed`     -- the seed to initialize the candidates with
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
    

class MPITuner(BaseTuner):
    """
    A tuner class that runs the optimisation algorithm on a master node and distributes candidates
    to slave nodes for simulation and evaluation of their fitness
    """ 
    
    MASTER = 0      # The processing node to be used for the master node
    COMMAND_MSG = 1 # A message tag signifying that the passed message is a command to a slave node
    DATA_MSG = 2    # A message tag signifying that the passed message is data returned by a slave node
    
    # Try to use the mpi4py import but fail silently if it wasn't imported. __init__ will check to 
    # see if it was successful before initializing a MPITuner object. Allows other tuner types to be
    # used if mpi4py is not installed.
    try:
        comm = MPI.COMM_WORLD           # The MPI communicator object
        rank = comm.Get_rank()          # The ID of the current process
        num_processes = comm.Get_size() # The number of processes available
    except NameError:
        pass
    
    def __init__(self, *args, **kwargs):
        try:
            MPI
        except NameError:
            raise Exception("MPITuner cannot be used because import 'mpi4py' was not found.")
        super(MPITuner, self).__init__(*args, **kwargs)
       
    @classmethod
    def is_master(cls):
        """
        Checks if current processing node is the master
        """
        return cls.rank == cls.MASTER       
       
    def tune(self, num_candidates, max_iterations, random_seed=None, stats_filename=None, 
             indiv_filename=None, **kwargs):
        """
        Runs the optimisation algorithm and returns the final population and algorithm state
        
        `num_candidates`  -- the number of candidates to use in the algorithm
        `max_iterations`  -- the maximum number of iterations to perform
        `random_seed`     -- the seed to initialize the candidates with
        `stats_filename`  -- the name of the file to save the generation-based statistics in
        `indiv_filename`  -- the name of the file to save the candidate parameters in
        `kwargs`          -- optional arguments to be passed to the optimisation algorithm
        """
        
        if self.is_master():
            self._open_readout_files(stats_filename, indiv_filename, kwargs)
            result = self.algorithm.optimize(num_candidates, 
                                             self._distribute_candidates_for_evaluation, 
                                             max_iterations, random_seed, **kwargs)
            self._close_readout_files()
            self._release_slaves()
            return result
        else:
            self._listen_for_candidates_to_evaluate()
            return None

    def _distribute_candidates_for_evaluation(self, candidates, args): #@UnusedVariable args
        """
        Run on the master node, this method distributes candidates to to the slave nodes to be 
        evaluated then collates their results into a single numpy vector
        
        `candidates`  -- candidates to be evaluated
        `args`        -- unused but supplied for compatibility with inspyred library
        """
        assert self.is_master(), "Distribution of candidate jobs should only be performed by master node"
        candidate_jobs = deque(enumerate(candidates))
        free_processes = deque(xrange(1, self.num_processes))
        # Create a list of empty lists the same length as the candidate list
        evaluations = [[]] * len(candidates) 
        while candidate_jobs:
            if free_processes:
                self.comm.send(candidate_jobs.pop(), dest=free_processes.pop(), 
                               tag=self.COMMAND_MSG) 
            else:    
                processID, jobID, result = self.comm.recv(source=MPI.ANY_SOURCE, tag=self.DATA_MSG)
                evaluations[jobID] = result
                free_processes.append(processID)
        while len(evaluations) < len(candidates):
            processID, jobID, result = self.comm.recv(source=MPI.ANY_SOURCE, tag=self.DATA_MSG)
            evaluations[jobID] = result
        assert not all([e != [] for e in evaluations]), "One or more evaluations were not set"
        return evaluations

    def _listen_for_candidate_to_evaluate(self):
        """
        Run on the slave nodes, this method receives candidates to evaluate from the master node,
        evaluates them and sends back the master
        """
        assert not self.is_master(), "Evaluation of candidates should only be performed by slave nodes"
        command = self.comm.recv(source=self.MASTER, tag=self.COMMAND_MSG)
        while command != 'stop':
            jobID, candidate = command
            evaluation = self._evaluate_candidate(candidate)
            self.comm.send((self.rank, jobID, evaluation), dest=self.MASTER, tag=self.DATA_MSG)
            command = self.comm.recv(source=self.MASTER, tag=self.COMMAND_MSG)
        print "Stopping listening on process {}".format(self.rank)

        
    def _release_slaves(self):
        """
        Release slave nodes from listening to new candidates to evaluate
        """
        for processID in xrange(1, self.num_processes):
            self.comm.send('stop', dest=processID, tag=self.COMMAND_MSG)   

