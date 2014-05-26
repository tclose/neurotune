from __future__ import absolute_import
from abc import ABCMeta  # Metaclass for abstract base classes
import quantities as pq
import neo.io
from .__init__ import Objective
from ..simulation import RecordingRequest, ExperimentalConditions, \
                         StepCurrentSource


class CurrentClampObjective(Objective):

    __metaclass__ = ABCMeta

    def __init__(self, reference_trace, injected_current,
                 record_variable=None, time_start=500.0 * pq.ms,
                 time_stop=2000.0 * pq.ms):
        super(CurrentClampObjective, self).__init__(time_start, time_stop)
        # Save reference trace(s) as a list, converting if a single trace or
        # loading from file if a valid filename
        if isinstance(reference_trace, str):
            f = neo.io.PickleIO(reference_trace)
            seg = f.read_segment()
            self.reference_traces = seg.analogsignals[0]
        elif isinstance(reference_trace, neo.AnalogSignal):
            self.reference_traces = reference_trace
        # Save members
        self.record_variable = record_variable
        self.injected_current = injected_current
        step_source = StepCurrentSource([0, injected_current],
                                        [0.0, time_start])
        self.exp_conditions = ExperimentalConditions(clamps=step_source)

    def get_recording_requests(self):
        """
        Gets all recording requests required by the objective function
        """
        return RecordingRequest(record_variable=self.record_variable,
                                time_start=self.time_start,
                                time_stop=self.time_stop,
                                conditions=self.exp_conditions)


class CurrentClampTimeConstantObjective(CurrentClampObjective):

    pass


class CurrentClampPeakHeightObjective(CurrentClampObjective):

    pass
