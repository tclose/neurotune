from __future__ import absolute_import
from abc import ABCMeta  # Metaclass for abstract base classes
import numpy
from ..simulation.__init__ import RecordingRequest


class Objective(object):
    """
    Base Objective class
    """

    __metaclass__ = ABCMeta  # Declare this class abstract to avoid accidental construction

    def __init__(self, time_start=0, time_stop=2000.0):
        """
        `time_stop` -- the required length of the recording required to evaluate the objective
        """
        self.time_start = time_start
        self.time_stop = time_stop

    def fitness(self, recordings):
        """
        Evaluates the fitness function given the simulated data
        
        `recordings` -- a dictionary containing the simulated data to be assess, with the keys 
                            corresponding to the keys of the recording request dictionary returned 
                            by 'get_recording requests'
        """
        raise NotImplementedError("Derived Objective class '{}' does not implement fitness method"
                                  .format(self.__class__.__name__))

    def get_recording_requests(self):
        """
        Returns a RecordingRequest object or a dictionary of RecordingRequest objects with unique keys
        representing the recordings that are required from the simulation controller
        """
        return RecordingRequest(time_stop=self.time_stop)


class DummyObjective(Objective):
    """
    A dummy objective that returns a constant value of 1 (useful for initial grid searches, which only
    need to see the recorded traces
    """
    
    def fitness(self, recordings):
        return 1

    