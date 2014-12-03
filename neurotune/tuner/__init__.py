from __future__ import absolute_import
import os
import collections
import traceback
import cPickle as pkl
import neo.io
from ..analysis import AnalysedRecordings


class EvaluationException(Exception):
    """
    This exception is thrown by the Tuner class when there is an unexpected
    error in the evaluation of the candidate (an unhandled corner case),
    which allows the tuning to fail gracefully and the case that cause the
    exception to be examined
    """

    def __init__(self, objective, simulation, candidate, analysis, tback=None):
        self.objective = objective
        self.simulation = simulation
        self.candidate = candidate
        self.analysis = analysis
        self.traceback = tback if tback is not None else traceback.format_exc()

    def __str__(self):
        return ("Evaluating candidate {} caused the following error:\n\n{}"
                .format(self.candidate, self.traceback))

    def save(self, filename):
        with open(filename, 'w') as f:
            pkl.dump((self.objective, self.simulation, self.candidate,
                      self.analysis), f)
        print ("Saving failed candidate along with objective, simulation "
               "and analysis objects to file at '{}'".format(filename))


class BadCandidateException(Exception):
    """
    This exception is thrown when a candidate causes a known error (such as
    simulation that doesn't converge) and which needs to be handled by the
    algorithm
    """

    def __init__(self, candidate):
        self.candidate = candidate


# This tuple collects all information needed to save the recordings into a
# single tuple
SaveRecordingsInfo = collections.namedtuple('SaveRecordingsInfo',
                                            'dir prefix ext io')


class Tuner(object):
    """
    Base Tuner object that contains the three components (objective function,
    algorithm and simulation objects) and runs the algorithm
    """
    num_processes = 1
    num_tuners = 0

    def __init__(self, *args, **kwargs):
        if self.num_tuners:
            raise Exception("Only one tuner object should be used at a time, "
                            "please use the 'set' method to repurpose an "
                            "existing tuner for consecutive tunings")
        self.num_tuners = 1
        self.tune_parameters = None
        self.objective = None
        self.algorithm = None
        self.simulation = None
        self.set(*args, **kwargs)

    def __del__(self):
        self.num_tuners = 0

    def set(self, tune_parameters=None, objective=None, algorithm=None,
            simulation=None, verbose=False, save_recordings=None):
        """
        Only on tuner instance should be used at a time as NEURON will
        otherwise simulate both each time it is run. Therefore this set
        method should be use to repurpose an existing simulator if performing
        subsequent tunings.

        `tune_parameters` -- The list of parameters to be tuned
        `objective`       -- The objective function to be tuned against
                             [neurotune.objectives.*Objective]
        `algorithm`       -- The algorithm used to tune the cell with
                             [neurotune.algorithms.*Algorithm]
        `simulation`      -- The interface to the neuronal simulator used
                             [neurotune.simulations.*Controller]
        `verbose`         -- flags whether to print out which candidate is
                             being evaluated
        `save_recordings` -- the location of the directory where the recordings
                             will be saved. If None (the default) recordings
                             are not saved
        """
        # Set members
        if tune_parameters is not None:
            if not len(tune_parameters):
                raise Exception("No parameters to tune were provided "
                                "(empty list)")
            self.tune_parameters = tune_parameters
        elif not self.tune_parameters:
            raise Exception("tune_parameters is not set and therefore must be "
                            "provided")
        if objective is not None:
            self.objective = objective
        elif not self.objective:
            raise Exception("objective is not set and therefore must be "
                            "provided")
        if algorithm is not None:
            self.algorithm = algorithm
        elif not self.algorithm:
            raise Exception("algorithm is not set and therefore must be "
                            "provided")
        if simulation is not None:
            self.simulation = simulation
        elif not self.simulation:
            raise Exception("simulation is not set and therefore must be "
                            "provided")
        self.objective.tuner = self
        self.algorithm.tuner = self
        self.simulation.tuner = self
        self.verbose = verbose
        self.bad_candidates = []
        if save_recordings:
            rec_dir = os.path.abspath(os.path.dirname(save_recordings))
            rec_prefix = os.path.basename(save_recordings)
            if rec_prefix.startswith('.'):
                rec_ext = rec_prefix
                rec_prefix = ''
            else:
                rec_prefix, rec_ext = os.path.splitext(rec_prefix)
                if not rec_ext:
                    rec_ext = '.pkl'
                    rec_dir = os.path.join(rec_dir, rec_prefix)
                    rec_prefix = ''
            if rec_ext == '.pkl':
                rec_io = neo.io.pickleio.PickleIO
            elif rec_ext == '.h5':
                rec_io = neo.io.hdf5io.NeoHdf5IO
            else:
                raise Exception("Unrecognised Neo extention '{}' for saving "
                                "recordings".format(rec_ext))
            self.save_recordings = SaveRecordingsInfo(rec_dir, rec_prefix,
                                                      '.neo' + rec_ext, rec_io)
            try:
                os.makedirs(rec_dir)
            except OSError as e:
                # If directory already exists ignore the exception
                if e.errno != 17:
                    raise
            print "Recordings will be saved to '{}' directory".format(rec_dir)
        else:
            self.save_recordings = None
        # Register tuneable parameters
        self.algorithm.set_tune_parameters(tune_parameters)
        self.simulation.set_tune_parameters(tune_parameters)
        # Pass recording requests from objective to simulation
        recording_requests = objective.get_recording_requests()
        self.simulation._process_requests(recording_requests)

    def tune(self, **kwargs):
        """
        Runs the optimisation algorithm and returns the final population and
        algorithm state
        """
        return self.algorithm.optimize(self._evaluator, **kwargs)

    def _evaluator(self, candidates, args=None):  # @UnusedVariable
        """
        Evaluate each candidate and return the evaluations in a numpy array. To
        be passed to inspyred optimisation algorithm (overridden in MPI derived
        class)

        `candidates` -- a list of candidates (themselves an iterable of float
                        parameters) [list(list(float))]
        `args`       -- unused but provided to match inspyred API
        """
        return [self._evaluate_candidate(c) for c in candidates]

    def _evaluate_candidate(self, candidate):
        """
        Evaluate the fitness of a single candidate
        """
        if self.verbose:
            print "Evaluating candidate {}".format(candidate)
        try:
            recordings = self.simulation.run_all(candidate)
            if self.save_recordings:
                fname = (self.save_recordings.prefix +
                         ','.join(['{}={}'.format(p.name, c)
                                   for p, c in zip(self.tune_parameters,
                                                   candidate)]) +
                         self.save_recordings.ext)
                fpath = os.path.join(self.save_recordings.dir, fname)
                if os.path.exists(fpath):
                    os.remove(fpath)
                self.save_recordings.io(fpath).write(recordings)
            analysis = AnalysedRecordings(recordings, self.simulation.setups)
            fitness = self.objective.fitness(analysis)
        except BadCandidateException:
            print ("WARNING! Candidate {} caused a BadCandidateException. "
                   "This typically means there was an instability in the "
                   "simulation for these parameters".format(candidate))
            fitness = self.algorithm.BAD_FITNESS_VALUE
            self.bad_candidates.append(candidate)
        except:
            # Check to see if using distributed processing, in which case
            # raise an EvaluationException (allows the MPI tuner to fail
            # gracefully). Otherwise the assumption is that you are debugging
            # and would prefer to raise the exception normally to debug in an
            # IDE.
            if self.num_processes == 1 and __debug__:
                raise
            else:
                raise EvaluationException(self.objective, self.simulation,
                                          candidate,
                                          locals().get('analysis', None))
        return fitness

    @classmethod
    def is_master(self):
        """
        Provided for convenient interoperability with the MPITuner class
        """
        return True

    @property
    def num_parameters(self):
        return len(self.tune_parameters)
