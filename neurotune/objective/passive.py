from __future__ import absolute_import
from abc import ABCMeta  # Metaclass for abstract base classes
from scipy.interpolate import UnivariateSpline
import numpy
import quantities as pq
import neo
from .__init__ import Objective
from ..simulation import RecordingRequest, ExperimentalConditions


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
        step_amps = [0.0, self.inject_amplitude, self.inject_amplitude]
        step_source = neo.IrregularlySampledSignal(
                                               step_times, step_amps,
                                               units=inject_amplitude.units,
                                               time_units=self.time_stop.units)
        inject_dict = {self.inject_location: step_source}
        self.conditions = ExperimentalConditions(injected_currents=inject_dict)

    def _get_recording_request(self, record_site=None):
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
        return {None: self._get_recording_request(self.inject_location)}

    def fitness(self, analysis):
        signal = analysis.get_signal()
        return float(numpy.sum((self.reference - signal) ** 2) /
                     float(self.time_stop - self.time_start))


class SteadyStateVoltagesObjective(PassivePropertiesObjective):

    def __init__(self, references, record_sites, ref_inject_dists,
                 rec_inject_dists, time_stop=750.0 * pq.ms, coeff_order=6,
                 **kwargs):
        if len(references) != len(ref_inject_dists):
            raise Exception("Number of references ({}) should match number of "
                            "ref inject distances ({})"
                            .format(len(references),
                                                       len(ref_inject_dists)))
        super(SteadyStateVoltagesObjective, self).__init__(time_stop=time_stop,
                                                           **kwargs)
        self.record_sites = record_sites
        self.ref_inject_dists = ref_inject_dists
        self.rec_inject_dists = rec_inject_dists
        self.coeff_order = coeff_order
        if type(references) is numpy.ndarray:
            steady_state_v = references
        else:
            self._set_reference(references)
            # Get the steady-state voltages for each of the reference
            # recordings
            steady_state_v = [r[-1] for r in self.reference]
        # Get an interpolated spline relating steady-state voltage to distance
        # from the root segment
        self.ss_poly_fit = numpy.poly1d(numpy.polyfit(ref_inject_dists,
                                                      steady_state_v,
                                                      coeff_order))

    def get_recording_requests(self):
        return dict([(site, self._get_recording_request(site))
                     for site in self.record_sites])

    def fitness(self, analysis):
        ss_v = numpy.array([analysis.get_signal(s)[-1]
                            for s in self.record_sites])
        # Get the reference distance function interpolated to the recorded
        # distances
        ref_ss_v = self.ss_poly_fit(self.rec_inject_dists)
        return float(numpy.sum((ref_ss_v - ss_v) ** 2) /
                     len(self.rec_inject_dists))
