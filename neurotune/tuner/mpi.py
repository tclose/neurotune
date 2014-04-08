from __future__ import absolute_import
import sys
import os
import time
from copy import deepcopy
import shutil
import subprocess
from copy import copy
from collections import deque
from mpi4py import MPI
from .__init__ import Tuner


class MPITuner(Tuner):
    """
    A tuner class that runs the optimisation algorithm on a master node and distributes candidates
    to slave nodes for simulation and evaluation of their fitness
    """ 
    
    MASTER = 0      # The processing node to be used for the master node
    COMMAND_MSG = 1 # A message tag signifying that the passed message is a command to a slave node
    DATA_MSG = 2    # A message tag signifying that the passed message is data returned by a slave node
    
    comm = MPI.COMM_WORLD           # The MPI communicator object
    rank = comm.Get_rank()          # The ID of the current process
    num_processes = comm.Get_size() # The number of processes available
       
    def set(self, *args, **kwargs):
        self.evaluate_on_master = kwargs.pop('evaluate_on_master', self.num_processes < 10)
        self.mpi_verbose = kwargs.pop('verbose', True)
        super(MPITuner, self).set(*args, **kwargs)         
    
    @classmethod
    def is_master(cls):
        """
        Checks if current processing node is the master
        """
        return cls.rank == cls.MASTER       
       
    def tune(self, **kwargs):
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
            result = self.algorithm.optimize(self._distribute_candidates_for_evaluation, **kwargs)
            self._release_slaves()
        else:
            self._listen_for_candidates_to_evaluate()
            result = (None, None)
        return result
    
    def __del__(self):
        """
        Calls MPI finalize, so care should be taken not to let MPITuner objects to go out of scope
        if you have multiple instantiations (why you would have multiple instantiations I am not 
        sure though as you can use 'set' to re-purpose an existing Tuner)
        """
        MPI.Finalize()

    def _distribute_candidates_for_evaluation(self, candidates, args=None): #@UnusedVariable args
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
        remaining_evaluations = len(candidates)
        if self.evaluate_on_master:
            since_master_evaluation = 0
        while candidate_jobs:
            if free_processes:
                self.comm.send(candidate_jobs.pop(), dest=free_processes.pop(), 
                               tag=self.COMMAND_MSG)
                if self.evaluate_on_master:
                    since_master_evaluation += 1
                    if since_master_evaluation == self.num_processes - 1:
                        jobID, candidate = candidate_jobs.pop()
                        if self.mpi_verbose:
                            print ("Evaluating jobID: {}, candidate: {} on process {}"
                                   .format(jobID, candidate, self.rank))
                        evaluations[jobID] = self._evaluate_candidate(candidate)
                        remaining_evaluations -= 1
                        since_master_evaluation = 0
            else:    
                processID, jobID, result = self.comm.recv(source=MPI.ANY_SOURCE, tag=self.DATA_MSG)
                evaluations[jobID] = result
                remaining_evaluations -= 1
                free_processes.append(processID)
        while remaining_evaluations:
            processID, jobID, result = self.comm.recv(source=MPI.ANY_SOURCE, tag=self.DATA_MSG)
            evaluations[jobID] = result
            remaining_evaluations -= 1
        return evaluations

    def _listen_for_candidates_to_evaluate(self):
        """
        Run on the slave nodes, this method receives candidates to evaluate from the master node,
        evaluates them and sends back the master
        """
        assert not self.is_master(), "Evaluation of candidates should only be performed by slave nodes"
        command = self.comm.recv(source=self.MASTER, tag=self.COMMAND_MSG)
        while command != 'stop':
            jobID, candidate = command
            if self.mpi_verbose:
                print "Evaluating jobID: {}, candidate: {} on process {}".format(jobID, candidate, 
                                                                                 self.rank)
            evaluation = self._evaluate_candidate(candidate)
            self.comm.send((self.rank, jobID, evaluation), dest=self.MASTER, tag=self.DATA_MSG)
            command = self.comm.recv(source=self.MASTER, tag=self.COMMAND_MSG)
        if self.mpi_verbose:
            print "Stopping listening on process {}".format(self.rank)

        
    def _release_slaves(self):
        """
        Release slave nodes from listening to new candidates to evaluate
        """
        for processID in xrange(1, self.num_processes):
            self.comm.send('stop', dest=processID, tag=self.COMMAND_MSG)


        