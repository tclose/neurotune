"""
Run the simulation
"""
from __future__ import absolute_import
from collections import namedtuple
from itertools import groupby
from abc import ABCMeta  # Metaclass for abstract base classes
import neo
import quantities as pq


class RecordingRequest(object):
    """"
    RecordingRequests are raised by objective functions and are passed to
    Simulation objects so they can set up the required simulation conditions
    (eg IClamp, VClamp, spike input) and recorders
    """

    def __init__(self, record_time=2000.0, record_variable=None,
                 conditions=None):
        """
        `record_time`     -- the length of the recording required by the
                             simulation
        `record_variable` -- the name of the section/synapse/current
                             to record from (simulation specific)
        `conditions`      -- the experimental conditions
                             required (eg. initial voltage, current clamp)
        """
        self.record_time = record_time
        self.record_variable = record_variable
        self.conditions = conditions


class Setup(object):
    """
    Groups together all the simulation set-up information to interface to and
    from a requested simulation
    """
    def __init__(self, time, conditions, record_variables, request_keys):
        self.time = time
        self.conditions = conditions
        self.record_variables = record_variables
        self.request_keys = request_keys


class ExperimentalConditions(object):
    """
    Defines the experimental conditions an objective function requires to make
    its evaluation. Can be extended for specific conditions required by novel
    objective functions but may not be supported by all simulations, especially
    custom ones
    """

    def __init__(self, initial_v=None, clamps=[]):
        """
        `initial_v` -- the initial voltage of the membrane
        """
        self.initial_v = initial_v
        self.clamps = set(clamps)

    def __eq__(self, other):
        return (self.initial_v == other.initial_v and
                self.clamps == other.clamps)


class StepCurrentSource(object):

    def __init__(self, amplitudes, times):
        self.amplitudes = amplitudes
        self.times = times

    def __eq__(self, other):
        return (self.amplitudes == other.amplitudes and
                self.times == other.times)


class Simulation():
    "Base class of Simulation objects"

    # Declare this class abstract to avoid accidental construction
    __metaclass__ = ABCMeta

    supported_clamp_types = []

    def _process_requests(self, recording_requests):
        """
        Merge recording requests so that the same recording/simulation doesn't
        get performed multiple times

        `recording_requests`  -- a list of recording requests from the
                                 objective functions [RecordingRequest]
        """
        # Group into requests by common experimental conditions
        try:
            request_items = recording_requests.items()
        except AttributeError:
            request_items = [(None, recording_requests)]
        request_items.sort(key=lambda x: x[1].conditions)
        common_conditions = groupby(request_items,
                                    key=lambda x: x[1].conditions)
        # Merge the common requests into simulation setups
        self._simulation_setups = []
        for conditions, requests_iter in common_conditions:
            if conditions is not None:
                for c in conditions.clamps:
                    if type(c) not in self.supported_clamp_types:
                        raise Exception("Condition of type {} is not supported"
                                        " by this Simulation class ({})"
                                        .format(type(c), self.__class__))
            requests = [r for r in requests_iter]
            # Get the maxium record time in the group
            record_time = max([r[1].record_time for r in requests])
            # Group the requests by common recording sites
            requests.sort(key=lambda x: x[1].record_variable)
            common_record_variables = groupby(requests,
                                            key=lambda x: x[1].record_variable)
            # Get the common recording sites
            record_variables, requests_iters = zip(*[(rv, list(requests))
                                                      for rv, requests in
                                                      common_record_variables])
            # Get the list of request keys for each requested recording
            request_keys = [zip(*com_record)[0]
                            for com_record in requests_iters]
            # Append the simulation request to the
            self._simulation_setups.append(Setup(record_time, conditions,
                                                 list(record_variables),
                                                 request_keys))
        # Do initial preparation for simulation (how much preparation can be
        # done depends on whether the same experimental conditions are used
        # throughout the evaluation process.
        self.prepare_simulations()

    @property
    def setups(self):
        try:
            return self._simulation_setups
        except AttributeError:
            raise Exception("Simulations have not been requested by objective "
                            "function yet")

    def _get_requested_recordings(self, candidate):
        """
        Return the recordings in a dictionary to be returned to the objective
        functions, so each objective function can access the recording it
        requested

        `candidate` -- a list of parameters [list(float)]
        """
        recordings = self.run(candidate)
        requests_dict = {}
        for seg, setup in zip(recordings.segments, self._simulation_setups):
            assert len(seg.analogsignals) == len(setup.request_keys)
            for signal, request_keys in zip(seg.analogsignals,
                                            setup.request_keys):
                requests_dict.update([(key, signal) for key in request_keys])
        return recordings, requests_dict

    def prepare_simulations(self):
        """
        Allows subclasses to prepares the simulations that are required by the
        chosen objective functions after they have been initially processed
        """
        pass

    def set_tune_parameters(self, tune_parameters):
        """
        Sets the parameters in which the candidate arrays passed to the 'run'
        method will map to in respective order

        `tune_parameters` -- list of parameter names which correspond to the
                             candidate order
        """
        self.tune_parameters = tune_parameters

    def run_all(self, candidate):
        """
        Runs all simulation reuquired by requested simulation setups
        """
        recordings = neo.Block(name=','.join(['{}={}'.format(p.name, c)
                                              for p, c in
                                              zip(self.tune_parameters,
                                                  candidate)]),
                               candidate=candidate)
        for setup in self.setups:
            recordings.segments.append(self.run(candidate, setup))
        return recordings

    def run(self, candidate, setup):
        """
        At a high level - accepts a candidate (a list of cell parameters that
        are being tuned)

        `candidate`         -- a list of parameters [list(float)]
        `setup`             -- a simulation setup [Setup]

        Returns:
            A Neo block object with a segment for each Setup in
            self._simulation_setups containing an analogsignal for each
            requested recording
        """
        raise NotImplementedError("Derived Simulation class '{}' does not "
                                  "implement _run_simulation method"
                                  .format(self.__class__.__name__))
