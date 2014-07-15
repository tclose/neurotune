from __future__ import absolute_import
from abc import ABCMeta  # Metaclass for abstract base classes
import numpy
import quantities as pq
import neo.io
from .__init__ import Objective
from ..simulation import RecordingRequest
from ..analysis import AnalysedSignal


class PassivePropertiesObjective(Objective):

    __metaclass__ = ABCMeta

    def __init__(self, inject_location=None, inject_amplitude=-100 * pq.nA,
                 time_start=250.0 * pq.ms, time_stop=500.0 * pq.ms):
        """
        `inject_location`  -- segment in which to inject the current into
        `inject_amplitude` -- the strength of the current
        `time_start`       -- start of the recording (after transients have
                              decayed)
        `time_stop`        -- end of the recording
        """
        super(PassivePropertiesObjective, self).__init__(time_start, time_stop)
        # Save members
        self.inject_location = inject_location
        self.inject_amplitude = inject_amplitude
        step_times = [0.0, self.time_start, self.time_stop]
        step_amps = [0.0, self.injected_current, self.injected_current]
        step_source = neo.IrregularlySampledSignal(step_times, step_amps)
        inject_dict = {self.inject_location: step_source}
        self.conditions = {'injected_currents': inject_dict}

    def _get_recording_requests(self, record_site=None):
        """
        Gets all recording requests required by the objective function
        """
        return RecordingRequest(record_variable=record_site + '.v',
                                time_start=self.time_start,
                                time_stop=self.time_stop,
                                conditions=self.conditions)


class RCCurveObjective(PassivePropertiesObjective):

    def __init__(self, reference, record_site, **kwargs):
        """
        `reference`        -- either an Analog signal or the location of a file
                      in Neo format containing a single AnalogSignal
        """
        super(RCCurveObjective, self).__init__(**kwargs)
        # Load reference trace from file
        # loading from file if a valid filename
        if isinstance(reference, str):
            f = neo.io.PickleIO(reference)
            seg = f.read_segment()
            reference = seg.analogsignals[0]
        # Convert to analysed signal and slice accoring to time_start and stop
        self.reference_trace = AnalysedSignal(reference).slice(self.time_start,
                                                               self.time_stop)
        self.record_site = record_site

    def get_recording_requests(self):
        self._get_recording_requests(self.record_variable)

    def fitness(self, analysis):
        signal = analysis.get_signal()
        return (numpy.sum((self.reference - signal) ** 2) /
                float(self.time_stop - self.time_start))


class SteadyStateVoltagesObjective(PassivePropertiesObjective):

    def __init__(self, references, record_sites, time_stop=750.0, **kwargs):
        if len(references) != len(record_sites):
            raise Exception("Number of references ({}) should match number of "
                            "record sites ({})".format(len(references),
                                                       len(record_sites)))
        super(RCCurveObjective, self).__init__(time_stop=time_stop, **kwargs)
        # Load reference traces as a list, converting if a trace or
        # loading from file if a valid filename
        if isinstance(references, str):
            f = neo.io.PickleIO(references)
            seg = f.read_segment()
            references = seg.analogsignals
        # Get the voltage at self.time_stop for each of the reference signals
        self.ref_vs = numpy.array([AnalysedSignal(s).slice(0.0,
                                                           self.time_stop)[-1]
                                   for s in references])
        self.record_sites = record_sites

    def get_recording_requests(self):
        return dict([(site, self._get_recording_requests(site))
                     for site in self.record_sites])

    def fitness(self, analysis):
        vs = numpy.array([analysis.get_signal(s)[-1]
                          for s in self.record_sites])
        return numpy.sum((self.ref_vs - vs) ** 2)
