from __future__ import absolute_import
import sys
from collections import deque
from mpi4py import MPI
from . import Tuner, EvaluationException


class MPITuner(Tuner):
    """
    A tuner class that runs the optimisation algorithm on a master node and distributes candidates
    to slave nodes for simulation and evaluation of their fitness
    """

    MASTER = 0  # The processing node to be used for the master node
    COMMAND_MSG = 1  # A message tag signifying that the passed message is a command to a slave node
    DATA_MSG = 2  # A message tag signifying that the passed message is data returned by a slave node

    comm = MPI.COMM_WORLD  # The MPI communicator object
    rank = comm.Get_rank()  # The ID of the current process
    num_processes = comm.Get_size()  # The number of processes available

    def set(self, *args, **kwargs):
        if self.num_processes == 1:
            self.evaluate_on_master = True
        else:
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
            try:
                result = self.algorithm.optimize(self._distribute_candidates, **kwargs)
            finally:
                self._release_slaves()
        else:
            self._listen_for_candidates()
            result = (None, None)
        return result

    def __del__(self):
        """
        Calls MPI finalize, so care should be taken not to let MPITuner objects to go out of scope
        if you have multiple instantiations (why you would have multiple instantiations I am not 
        sure though as you can use 'set' to re-purpose an existing Tuner)
        """
        MPI.Finalize()

    def _distribute_candidates(self, candidates, args=None):  # @UnusedVariable args
        """
        Run on the master node, this method distributes candidates to to the slave nodes to be 
        evaluated then collates their results into a single numpy vector
        
        `candidates`  -- candidates to be evaluated
        `args`        -- unused but supplied for compatibility with inspyred library
        """
        assert self.is_master(), "Distribution of candidate jobs should only be performed by master node"
        candidate_jobs = deque(enumerate(candidates))
        free_processes = deque(xrange(1, self.num_processes)) if self.num_processes > 1 else [0]
        # Create a list of None values the same length as the candidate list
        evaluations = [None] * len(candidates)
        # Record the number of evaluations that are yet to be performed
        remaining_evaluations = len(candidates)
        # If evaluate on master is true, the number of evaluations since an evaluation occurred on
        # the master is counted and when it reaches the number of slave processes another evaluation
        # is performed on the master
        until_master_eval = self.num_processes - 1
        while remaining_evaluations:
            # If there are remaining candidates and free processes then distribute the
            # candidates to the processes
            if free_processes and candidate_jobs:
                if self.num_processes > 1:
                    self.comm.send(candidate_jobs.pop(), dest=free_processes.pop(),
                                   tag=self.COMMAND_MSG)
                    until_master_eval -= 1
                # If evaluate_on_master is set, check to see how many evaluations have been sent
                # since the last evaluation on the master node and if it equals the number of
                # processes evaluate another candidate on the master node
                if self.evaluate_on_master and until_master_eval == 0:
                    jobID, candidate = candidate_jobs.pop()
                    if self.mpi_verbose:
                        print ("Evaluating jobID: {}, candidate: {} on process {}"
                               .format(jobID, candidate, self.rank))
                    evaluations[jobID] = self._evaluate_candidate(candidate)
                    remaining_evaluations -= 1
                    until_master_eval = self.num_processes - 1
            # Once all slave processes are busy wait for them to finish and record their result
            else:
                # Receive evaluation from slave node
                received = self.comm.recv(source=MPI.ANY_SOURCE, tag=self.DATA_MSG)
                try:
                    processID, jobID, result = received
                except ValueError:  # If the slave raised an evaluation error it sends 4-tuple
                    raise EvaluationException(*received)
                evaluations[jobID] = result
                free_processes.append(processID)
                remaining_evaluations -= 1
        return evaluations

    def _listen_for_candidates(self):
        """
        Run on the slave nodes, this method receives candidates to evaluate from the master node,
        evaluates them and sends back the master
        """
        assert not self.is_master(), "Evaluation of candidates should only be performed by slave nodes"
        command = self.comm.recv(source=self.MASTER, tag=self.COMMAND_MSG)
        while command != 'stop':
            jobID, candidate = command
            if self.mpi_verbose:
                print ("Evaluating jobID: {}, candidate: {} on process {}"
                       .format(jobID, candidate, self.rank))
            try:
                evaluation = self._evaluate_candidate(candidate)
            except EvaluationException as e:
                # Check to see that the size of the recordings isn't very large before attempting
                # to pass it back over MPI
                if sys.getsizeof(e.recordings) > 100000:
                    e.recordings = 'Too large to pass over MPI'
                # This will tell the master node to raise an EvaluationException and release all
                # slaves
                self.comm.send((e.objective, e.candidate, e.recordings, e.traceback), 
                               dest=self.MASTER, tag=self.DATA_MSG)
                break
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



