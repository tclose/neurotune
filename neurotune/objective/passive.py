from __future__ import absolute_import
from abc import ABCMeta  # Metaclass for abstract base classes
from scipy.interpolate import UnivariateSpline
import numpy
import quantities as pq
import neo
from .__init__ import Objective
from ..simulation import RecordingRequest


class PassivePropertiesObjective(Objective):

    __metaclass__ = ABCMeta

    def __init__(self, inject_location=None, inject_amplitude=-2 * pq.nA,
                 time_start=250.0 * pq.ms, time_stop=500.0 * pq.ms):
        """
        `inject_location`  -- segment in which to inject the current into
                              if None it (should) default to the source_section
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

    def __init__(self, reference, **kwargs):
        """
        `reference`        -- either an Analog signal or the location of a file
                              in Neo format containing a single AnalogSignal
        """
        super(RCCurveObjective, self).__init__(**kwargs)
        self._set_reference(reference)

    def get_recording_requests(self):
        self._get_recording_requests(self.inject_location)

    def fitness(self, analysis):
        signal = analysis.get_signal()
        return (numpy.sum((self.reference - signal) ** 2) /
                float(self.time_stop - self.time_start))


class SteadyStateVoltagesObjective(PassivePropertiesObjective):

    def __init__(self, references, record_sites, ref_inject_dists,
                 rec_inject_dists, time_stop=750.0, interp_order=3,
                 **kwargs):
        if len(references) != len(record_sites):
            raise Exception("Number of references ({}) should match number of "
                            "record sites ({})".format(len(references),
                                                       len(record_sites)))
        super(RCCurveObjective, self).__init__(time_stop=time_stop, **kwargs)
        self.record_sites = record_sites
        self.ref_inject_dists = ref_inject_dists
        self.rec_inject_dists = rec_inject_dists
        self.interp_order = interp_order
        self._set_reference(references)
        # Get the steady-state voltages for each of the reference recordings
        steady_state_v = [r[-1] for r in self.reference]
        # Get an interpolated spline relating steady-state voltage to distance
        # from the root segment
        self.ss_interpolator = UnivariateSpline(ref_inject_dists,
                                                steady_state_v, k=interp_order)

    def get_recording_requests(self):
        return dict([(site, self._get_recording_requests(site))
                     for site in self.record_sites])

    def fitness(self, analysis):
        ss_v = numpy.array([analysis.get_signal(s)[-1]
                            for s in self.record_sites])
        # Get the reference distance function interpolated to the recorded
        # distances
        ref_ss_v = self.ss_interpolator(self.rec_inject_dists)
        return numpy.sum((ref_ss_v - ss_v) ** 2)
