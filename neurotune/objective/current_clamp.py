from __future__ import absolute_import
from abc import ABCMeta  # Metaclass for abstract base classes
import neo.io
from .__init__ import Objective
from ..simulation import RecordingRequest
from ..conditions import ExperimentalConditions, StepCurrentSource


class CurrentClampObjective(Objective):

    __metaclass__ = ABCMeta

    def __init__(self, reference_traces, injected_current,
                 record_variable=None, time_start=500.0, time_stop=2000.0):
        super(CurrentClampObjective, self).__init__(time_start, time_stop)
        # Save reference trace(s) as a list, converting if a single trace or
        # loading from file if a valid filename
        if isinstance(reference_traces, str):
            f = neo.io.PickleIO(reference_traces)
            seg = f.read_segment()
            self.reference_traces = seg.analogsignals
        elif isinstance(reference_traces, neo.AnalogSignal):
            self.reference_traces = [reference_traces]
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
                                record_time=self.time_stop,
                                conditions=self.exp_conditions)


class CurrentClampTimeConstantObjective(CurrentClampObjective):

    pass


class CurrentClampPeakHeightObjective(CurrentClampObjective):

    pass
