from __future__ import absolute_import
from abc import ABCMeta  # Metaclass for abstract base classes
import numpy
import quantities as pq
from ..simulation import RecordingRequest


class Objective(object):
    """
    Base Objective class
    """

    # Declare this class abstract to avoid accidental construction
    __metaclass__ = ABCMeta

    def __init__(self, time_start=500.0 * pq.ms, time_stop=2000.0 * pq.ms,
                 exp_conditions=None, record_sites=[None]):
        """
        `time_stop` -- the required length of the recording required to
                       evaluate the objective
        """
        self.time_start = time_start
        self.time_stop = time_stop
        self.exp_conditions = exp_conditions
        self.record_sites = record_sites
        self.tuner = None

    def fitness(self, recordings):
        """
        Evaluates the fitness function given the simulated data

        `recordings` -- a dictionary containing the simulated data to be
                        assess, with the keys corresponding to the keys of the
                        recording request dictionary returned by 'get_recording
                        requests'
        """
        raise NotImplementedError("Derived Objective class '{}' does not "
                                  "implement fitness method"
                                  .format(self.__class__.__name__))

    def get_recording_requests(self):
        """
        Returns a RecordingRequest object or a dictionary of RecordingRequest
        objects with unique keys representing the recordings that are required
        from the simulation controller
        """
        requests = {}
        for site in self.record_sites:
            requests[site] = RecordingRequest(time_start=self.time_start,
                                              time_stop=self.time_stop,
                                              conditions=self.exp_conditions,
                                              record_variable=site)
        return requests


class DummyObjective(Objective):
    """
    A dummy objective that returns a constant value of 1 (useful for initial
    grid searches, which only need to see the recorded traces
    """

    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop('name', None)
        super(DummyObjective, self).__init__(*args, **kwargs)

    def fitness(self, recordings):  # @UnusedVariable
        return 1
