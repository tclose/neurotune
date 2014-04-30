from __future__ import absolute_import
import os
import traceback
import cPickle as pkl
import neo.io


class EvaluationException(Exception):
    
    def __init__(self, objective, candidate, recordings, tback=None):
        self.objective = objective
        self.candidate = candidate
        self.recordings = recordings
        self.traceback = tback if tback is not None else traceback.format_exc()

    def __str__(self):
        return ("Evaluating candidate {} caused the following error:\n\n{}"
                .format(self.candidate, self.traceback))

    def save(self, filename):
        with open(filename, 'w') as f:
            pkl.dump((self.objective, self.candidate, self.recordings), f)
        print "Saving failed candidate and recordings to file at '{}'".format(filename)
            
    
class Tuner(object):
    """
    Base Tuner object that contains the three components (objective function, algorithm and 
    simulation objects) and runs the algorithm
    """
    num_processes = 1
    
    def __init__(self, *args, **kwargs):
        self.set(*args, **kwargs)
    
    def set(self, tune_parameters, objective, algorithm, simulation, verbose=False,
            save_recordings=None):
        """
        `objective`  -- The objective function to be tuned against [neurotune.objectives.*Objective]
        `algorithm`  -- The algorithm used to tune the cell with [neurotune.algorithms.*Algorithm]
        `simulation` -- The interface to the neuronal simulator used [neurotune.simulations.*Controller]
        `verbose`    -- flags whether to print out which candidate is being evaluated
        `save_recordings` -- the location of the directory where the recordings will be saved. If None (the default) recordings are not saved
        """
        # Set members
        self.tune_parameters = tune_parameters
        self.objective = objective
        self.algorithm = algorithm
        self.simulation = simulation
        self.verbose = verbose
        self.save_recordings = save_recordings
        if self.save_recordings:
            if not os.path.exists(self.save_recordings):
                os.makedirs(self.save_recordings)
            print "Recordings will be saved to '{}' directory".format(save_recordings)
        # Register tuneable parameters
        self.algorithm.set_tune_parameters(tune_parameters)
        self.simulation.set_tune_parameters(tune_parameters)
        # Pass recording requests from objective to simulation
        recording_requests = objective.get_recording_requests()
        self.simulation._process_requests(recording_requests)
        
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
            recordings, requests_dict = self.simulation._get_requested_recordings(candidate)
            if self.save_recordings:
                fname = '_'.join(['{}={}'.format(p.name, c)
                                  for p, c in zip(self.tune_parameters, candidate)]) + '.neo.pkl'
                fpath = os.path.join(self.save_recordings, fname)
                if os.path.exists(fpath):
                    os.remove(fpath)
                io = neo.io.pickleio.PickleIO(fpath)
                io.write(recordings)
            fitness = self.objective.fitness(requests_dict)
        except Exception:
            if __debug__:
                raise
            else:
                # Check to see whether the candidate was recorded properly before the error
                if 'requests_dict' not in locals().keys():
                    requests_dict = None
                raise EvaluationException(self.objective, candidate, requests_dict)
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
