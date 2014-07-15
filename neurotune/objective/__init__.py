from __future__ import absolute_import
from abc import ABCMeta  # Metaclass for abstract base classes
import numpy
import quantities as pq
import neo.io
from ..simulation import Simulation, RecordingRequest
from ..analysis import AnalysedSignal


class Objective(object):
    """
    Base Objective class
    """

    # Declare this class abstract to avoid accidental construction
    __metaclass__ = ABCMeta

    def __init__(self, time_start=500.0 * pq.ms, time_stop=2000.0 * pq.ms,
                 conditions={}, record_sites=[None]):
        """
        `time_start`   -- the time given for the system to reach steady state
                          before starting to record [pq.Quantity]
        `time_stop`    -- the required length of the recording required to
                          evaluate the objective [pq.Quantity]
        `conditions`   -- the conditions required to run the simulation under
        `record_sites` -- the record sites that are required for the fitness
                          function
        """
        self.time_start = time_start
        self.time_stop = time_stop
        self.conditions = conditions
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
                                              conditions=self.conditions,
                                              record_variable=site)
        return requests

    def _set_reference(self, reference):
        recording_requests = self.get_recording_requests()
        # Get the number of references that are required
        num_refs = len(recording_requests)
        if isinstance(reference, str):  # Assume path of Neo file
            f = neo.io.PickleIO(reference)
            seg = f.read_segment()
            if len(seg.analogsignals) != num_refs:
                raise Exception("Number of loaded AnalogSignals ({}) does not "
                                "match number of recording requests ({})"
                                .format(len(seg.analogsignals), num_refs))
            if num_refs == 1:
                self.reference = AnalysedSignal(seg.analogsignals[0]).\
                                               slice(self.t_start, self.t_stop)
            else:
                self.reference = [AnalysedSignal(sig).slice(self.t_start,
                                                            self.t_stop)
                                  for sig in seg.analogsignals]
        elif isinstance(reference, Simulation):
            reference.process_recording_requests(recording_requests)
            recordings = reference.run_all(candidate=None)
            if num_refs == 1:
                self.reference = AnalysedSignal(
                                    recordings.segments[0].analogsignals[0]).\
                                               slice(self.t_start, self.t_stop)
            else:
                self.reference = [AnalysedSignal(seg.analogsignals[0]).\
                                               slice(self.t_start, self.t_stop)
                                  for seg in recordings.segments]
        elif num_refs == 1:
            if isinstance(reference, neo.AnalogSignal):
                self.reference = AnalysedSignal(reference).slice(self.t_start,
                                                                 self.t_stop)
            else:
                raise Exception("Unrecognised type of reference signal {}, "
                                "must be either a path to a Neo file, an "
                                "AnalogSignal or Simulation object"
                                .format(str(reference.__class__)))
        else:
            raise NotImplementedError


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
