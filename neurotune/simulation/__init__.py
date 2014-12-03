"""
Run the simulation
"""
from __future__ import absolute_import
from collections import namedtuple
from itertools import groupby
from abc import ABCMeta  # Metaclass for abstract base classes
import neo
from copy import deepcopy
import quantities as pq
from collections import Sequence, Mapping, Container, Set
import numpy


class ExperimentalConditions(object):

    def __init__(self, **kwargs):
        self._conds = kwargs

    def __getitem__(self, key):
        return self._conds[key]

    def __iter__(self):
        return self.keys()

    def keys(self):
        return self._conds.iterkeys()

    def items(self):
        return self._conds.iteritems()

    def sort_key(self):
        """
        Returns a key with which the experimental conditions can be compared,
        sorted and grouped by
        """
        return self._convert_numpy_arrays_to_tuples(self._conds)

    @classmethod
    def _convert_numpy_arrays_to_tuples(cls, item):
        """
        This methods is a work-around for numpy's equality testing (which
        screws up sorting and grouping by returning an array of truth values
        instead of a single truth value) by converting all numpy elements to
        tuples
        """
        if isinstance(item, Sequence) or isinstance(item, Mapping):
            item = deepcopy(item)
            if isinstance(item, Sequence):
                keys = xrange(len(item))
            elif isinstance(item, Mapping):
                keys = item.iterkeys()
            for k in keys:
                if isinstance(item[k], numpy.ndarray):
                    item[k] = cls._cnvrt_np_to_tpl(item[k])
                elif isinstance(item[k], Container):
                    item[k] = cls._convert_numpy_arrays_to_tuples(item[k])
        elif isinstance(item, Set):
            item = deepcopy(item)
            for e in item:
                if isinstance(e, numpy.ndarray):
                    item.remove(e)
                    item.add(cls._cnvrt_np_to_tpl(e))
                elif isinstance(e, Container):
                    item.remove(e)
                    item.add(cls._convert_numpy_arrays_to_tuples(e))
        return item

    @classmethod
    def _cnvrt_np_to_tpl(cls, a):
        if isinstance(a, neo.IrregularlySampledSignal):
            a = (tuple(a.times), tuple(a))
        else:
            a = tuple(a)
        return a


class RecordingRequest(object):
    """"
    RecordingRequests are raised by objective functions and are passed to
    Simulation objects so they can set up the required simulation conditions
    (eg IClamp, VClamp, spike input) and recorders
    """

    def __init__(self, time_start=0.0, time_stop=2000.0, record_variable=None,
                 conditions=ExperimentalConditions()):
        """
        `time_stop`     -- the length of the recording required by the
                             simulation
        `record_variable` -- the name of the section/synapse/current
                             to record from (simulation specific)
        `conditions`      -- the experimental conditions required in a
                             dictionary (eg. initial voltage, current clamp)
        """
        self.time_start = time_start
        self.time_stop = time_stop
        self.record_variable = record_variable
        self.conditions = conditions
        self.tuner = None


# Only used within simulation.Setup class (would make it a class method but had
# issues with pickling.
RequestRef = namedtuple('RequestRef', 'key time_start time_stop')


class Setup(object):
    """
    Groups together all the simulation set-up information to interface to and
    from a requested simulation
    """

    def __init__(self, record_time, conditions=ExperimentalConditions(),
                 record_variables=[], var_request_refs=[]):
        self.record_time = record_time
        self.conditions = conditions
        self.record_variables = record_variables
        self.var_request_refs = var_request_refs


class Simulation():
    "Base class of Simulation objects"

    # Declare this class abstract to avoid accidental construction
    __metaclass__ = ABCMeta

    supported_conditions = []

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
        request_items.sort(key=lambda i: i[1].conditions.sort_key())
        common_conditions = groupby(request_items,
                                    key=lambda i: i[1].conditions.sort_key())
        # Merge the common requests into simulation setups
        self._simulation_setups = []
        for _, requests_iter in common_conditions:
            # Convert the requests to a list so it can be read multiple times
            requests = list(requests_iter)
            # Get the common conditions for the group, can't use the one
            # returned by groupby as it the numpy arrays have been converted
            # to tuples
            conditions = requests[0][1].conditions
            for key in conditions.keys():
                if key not in self.supported_conditions:
                    raise Exception("Condition of type '{}' is not supported "
                                    "by the '{}' Simulation class (supported "
                                    "condition types: {})"
                                    .format(key, self.__class__,
                                            "', '"
                                            .join(self.supported_conditions)))
            # Get the maxium record time in the group
            record_time = max([r[1].time_stop for r in requests])
            # Group the requests by common recording sites
            requests.sort(key=lambda x: x[1].record_variable)
            common_record_variables = groupby(requests,
                                              key=(lambda x:
                                                   x[1].record_variable))
            # Get the common recording sites
            record_variables, req_iters = zip(*[(rv, list(requests))
                                                      for rv, requests in
                                                      common_record_variables])
            # Get the list of request keys for each requested recording
            req_refs = [[RequestRef(key, req.time_start, req.time_stop)
                         for key, req in com_record]
                        for com_record in req_iters]
            # Append the simulation request to the
            self._simulation_setups.append(Setup(record_time, conditions,
                                                 list(record_variables),
                                                 req_refs))
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

    @property
    def tune_parameter_names(self):
        if not self.tune_parameters:
            raise Exception("Tuning parameters have not been set")
        return (p.name for p in self.tune_parameters)

    def run_all(self, candidate):
        """
        Runs all simulations reuquired by the requested simulation setups

        `candidate`         -- a list of parameters [list(float)]
        """
        recordings_name = ','.join(['{}={}'.format(p.name, c)
                                    for p, c in zip(self.tune_parameters,
                                                    candidate)])
        recordings = neo.Block(name=recordings_name,
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
            A Neo Segment containing an AnalogSignal object for each requested
            recording in the passed setup given the experimental conditions
            provided within the setup object
        """
        raise NotImplementedError("Derived Simulation class '{}' does not "
                                  "implement 'run(self, candidate, setup)' "
                                  "method" .format(self.__class__.__name__))
