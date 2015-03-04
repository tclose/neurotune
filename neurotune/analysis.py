# -*- coding: utf-8 -*-
"""
Module for mathematical analysis of voltage traces from electrophysiology.

AUTHOR: Mike Vella vellamike@gmail.com

"""
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from scipy.optimize import brentq
import numpy
from copy import copy
import quantities as pq
import neo
from __builtin__ import classmethod
from simulation import Setup, RequestRef
from neurotune.simulation import ExperimentalConditions


def _all(vals):
    try:
        return all(vals)
    except TypeError:
        return bool(vals)


class AnalysedRecordings(object):

    def __init__(self, recordings, simulation_setups=None):
        if isinstance(recordings, neo.Segment):
            recordings = [recordings]
        if isinstance(recordings, list):
            block = neo.Block()
            block.segments.extend(recordings)
            recordings = block
        if simulation_setups is None:
            assert len(recordings.segments[0].analogsignals) == 1
            simulation_setups = self._get_dummy_setups(recordings)
        self.recordings = recordings
        self._simulation_setups = simulation_setups
        self._requests = {}
        for seg, setup in zip(recordings.segments, self._simulation_setups):
            assert len(seg.analogsignals) == len(setup.var_request_refs)
            for sig, request_refs, in zip(seg.analogsignals,
                                          setup.var_request_refs):
                # wrap the returned _signal in an AnalysedSignal wrapper
                signal = AnalysedSignal(sig)
                # For each request reference for the recorded variable
                # add the full _signal or a sliced version to the requests
                # dictionary
                sliced_signals = {}
                for key, t_start, t_stop in request_refs:
                    if t_start != signal.t_start or t_stop != signal.t_stop:
                        # Try to reuse the AnalysedSignalSlices as much as
                        # possible by storing in temporary dictionary using
                        # time start and stop as the key
                        dict_key = (float(pq.Quantity(t_start, 'ms')),
                                    float(pq.Quantity(t_stop, 'ms')))
                        try:
                            req_signal = sliced_signals[dict_key]
                        except KeyError:
                            req_signal = signal.slice(t_start, t_stop)
                            sliced_signals[dict_key] = req_signal
                    else:
                        req_signal = signal
                    self._requests[key] = req_signal
        self._objective_key = None

    def get_analysed_signal(self, key=None):
        """
        Returns the analysed _signal that matches the specified key
        """
        # If this is an objective specific analysis (i.e. one with an objective
        # key preset) prepend the objective key to the _signal key for the
        # complete "request" key
        if self._objective_key is not None:
            key = self._objective_key + (key,)
        return self._requests[key]

    def objective_specific(self, objective_key):
        """
        Returns a shallow copy of the current analysis in which the provided
        objective key is automatically prepended to the key requests. This
        makes it transparent to objective components of multi-objective
        functions that they are part of a multi-objective object.
        """
        specific_analysis = copy(self)
        specific_analysis._objective_key = (objective_key,)
        return specific_analysis

    @classmethod
    def _get_dummy_setups(cls, recordings):
        """
        Get default simulated setups to allow AnalysedRecordings to be
        constructed in tests outside of a Simulation object.
        """
        conditions = ExperimentalConditions()
        simulation_setups = []
        for seg in recordings.segments:
            record_variables = []
            request_refs = []
            rec_time = None
            for i, sig in enumerate(seg.analogsignals):
                if rec_time is None:
                    rec_time = sig.t_stop
                elif rec_time != sig.t_stop:
                    raise ValueError("Recording times are not equal ({} and "
                                     "{})".format(rec_time, sig.t_stop))
                record_variables.append(str(i))
                request = RequestRef(None, time_start=0.0,
                                     time_stop=sig.t_stop)
                request_refs.append([request])
            setup = Setup(rec_time, conditions=conditions,
                          record_variables=record_variables,
                          var_request_refs=request_refs)
            simulation_setups.append(setup)
        return simulation_setups


class AnalysedSignal(object):
    """
    A thin wrapper around the AnalogSignal class to keep all of the analysis
    with the _signal so it can be shared between multiple objectives (or even
    within a single more complex objective)
    """

    @classmethod
    def _argkey(cls, kwargs):
        """
        Returns keyword argument values, sorted by key names to be used for
        dictionary lookups
        """
        return tuple([val for _, val in sorted(kwargs.items(),
                                               key=lambda item: item[0])])

    @classmethod
    def _interpolate_v_dvdt(cls, v, dvdt, dvdt2v_scale=0.25, order=3):
        dv = numpy.diff(v)
        d_dvdt = numpy.diff(dvdt)
        interval_lengths = numpy.sqrt(numpy.asarray(dv) ** 2 +
                                      (numpy.asarray(d_dvdt) *
                                       dvdt2v_scale) ** 2)
        # Calculate the "positions" of the samples in terms of the fraction
        # of the length of the v-dv/dt path
        s = numpy.concatenate(([0.0], numpy.cumsum(interval_lengths)))
        # Save the original (non-sparsified) value to be returned
        return (InterpolatedUnivariateSpline(s, v, k=order),
                InterpolatedUnivariateSpline(s, dvdt, k=order), s)

    def __init__(self, signal):
        # Check if we can convert an irregularly sampled signal into a
        # vanilla analog signal (if it was done by mistake for example...)
        if isinstance(signal, neo.IrregularlySampledSignal):
            dt = signal.times[1:] - signal.times[:-1]
            if all(dt - dt[0] < 1e-6):
                signal = neo.AnalogSignal(numpy.array(signal),
                                          sampling_period=dt[0],
                                          t_start=signal.t_start,
                                          units=signal.units)
            else:
                raise Exception("Could not convert IrreguarlySampleSignal into"
                                " an AnalogSignal (too irreguarly sampled)")
        elif not isinstance(signal, neo.AnalogSignal):
            raise Exception("Can only analyse neo.AnalogSignals (not {})"
                            .format(type(signal)))
        self._signal = signal
        self._dvdt = None
        self._spike_periods = {}
        self._spike_amplitudes = {}
        self._spikes = {}
        self._splines = {}

    @property
    def name(self):
        return self.signal.name

    def smooth(self, order=3, smoothing_factor=None):
        self._signal[:] = UnivariateSpline(self.times, self, k=order,
                                           s=smoothing_factor)

    def __eq__(self, other):
        '''
        Equality test (==).
        '''
        return (_all(self._signal == other._signal) and
                _all(self._dvdt == other._dvdt) and
                _all(self._spike_periods == other._spike_periods) and
                _all(self._spikes == other._spikes) and
                _all(self._spike_amplitudes == other._spike_amplitudes) and
                _all(self._splines == other._splines))

    @property
    def signal(self):
        return self._signal

    @property
    def times(self):
        return self.signal.times

    @property
    def sampling_period(self):
        return self._signal.sampling_period

    @property
    def sampling_rate(self):
        return self._signal.sampling_rate

    @property
    def t_start(self):
        return self.signal.t_start

    @property
    def t_stop(self):
        return self.signal.t_stop

    @property
    def units(self):
        return self._signal.units

    @property
    def dvdt(self):
        if self._dvdt is None:
            dv = numpy.diff(numpy.asarray(self._signal))
            dt = numpy.diff(numpy.asarray(self.times))
            # Get the dvdt at the intervals between the samples
            dvdt = dv / dt
            # Linearly interpolate the dV/dt values back to the time points of
            # the original time course.
            dvdt = numpy.hstack((dvdt[0],
                                 (dvdt[1:] + dvdt[:-1]) / 2.0,
                                 dvdt[-1]))
            self._dvdt = neo.AnalogSignal(dvdt,
                                          sampling_period=self.sampling_period,
                                          units=self.units / self.times.units)
        return self._dvdt

    def _spike_period_indices(self, threshold='dvdt',
                              dvdt_crossing=50.0 * pq.mV / pq.ms,
                              dvdt_return=-50.0 * pq.mV / pq.ms,
                              volt_crossing=-35.0 * pq.mV,
                              volt_return=-35.0 * pq.mV,
                              index_buffer=0):
        """
        Find sections of the trace where it crosses the given dvdt threshold
        until it loops around and crosses the dvdt_return threshold in the
        positive direction again or alternatively if threshold=='v' when it
        crosses the dvdt_crossing threshold in the positive direction and
        crosses the dvdt_return threshold in the negative direction.

        `threshold`      -- can be either 'dvdt' or 'v', which determines the
                            type of threshold used to classify the spike
                            dvdt_crossing/dvdt_return
        `dvdt_crossing`  -- the starting dV/dt or v threshold of the spike
        `dvdt_return`    -- the stopping threshold, which needs to be crossed
                            in the positive direction for dV/dt and negative
                            for V
        `index_buffer`   -- the number of indices to pad the periods out by on
                            either side of the spike
        """
        argkey = (threshold, dvdt_crossing, dvdt_return, index_buffer)
        try:
            return self._spike_periods[argkey]
        except KeyError:
            if threshold not in ('dvdt', 'v'):
                raise Exception("Unrecognised threshold type '{}'"
                                .format(threshold))
            if threshold == 'dvdt':
                if dvdt_return > dvdt_crossing:
                    raise Exception("Stop threshold ({}) must be lower than or"
                                    " equal to the crossing threshold ({}) "
                                    "for dV/dt threshold crossing detection"
                                    .format(dvdt_return, dvdt_crossing))
                start_inds = numpy.where((self.dvdt[1:] >= dvdt_crossing) &
                                         (self.dvdt[:-1] < dvdt_crossing))[0]
                stop_inds = numpy.where((self.dvdt[1:] > dvdt_return) &
                                        (self.dvdt[:-1] <= dvdt_return))[0]
            else:
                if volt_return < volt_crossing:
                    raise Exception("Stop threshold ({}) must be higher than "
                                    "or equal to the crossing threshold ({}) "
                                    "for voltage threshold crossing detection"
                                    .format(dvdt_return, dvdt_crossing))
                start_inds = numpy.where((self._signal[1:] >= volt_crossing) &
                                        (self._signal[:-1] < volt_crossing))[0]  #@IgnorePep8
                stop_inds = numpy.where((self._signal[1:] < volt_return) &
                                        (self._signal[:-1] >= volt_return))[0]
            # Adjust the indices by 1
            start_inds += 1
            stop_inds += 1
            if len(start_inds) == 0 or len(stop_inds) == 0:
                periods = numpy.array([])
            else:
                # Ensure the dvdt_crossing and dvdt_return indices form regular
                # pairs where every dvdt_crossing has a dvdt_return that comes
                # after directly after it (i.e. before another dvdt_crossing)
                # and vice-versa
                periods = []
                for dvdt_crossing in start_inds:
                    try:
                        dvdt_return = stop_inds[numpy.where(stop_inds >
                                                            dvdt_crossing)][0]
                    # If the end of the loop is outside the time window
                    except IndexError:
                        continue
                    # Ensure that two spike periods don't overlap, which can
                    # occasionally occur in spike doublets for dV/dt thresholds
                    if (dvdt_crossing >= numpy.array(periods)).all():
                        periods.append((dvdt_crossing, dvdt_return))
                periods = numpy.array(periods)
                if index_buffer:
                    periods[:, 0] -= index_buffer
                    periods[:, 1] += index_buffer
                    periods[numpy.where(periods < 0)] = 0
                    periods[numpy.where(periods >
                                        len(self._signal))] = len(self._signal)
            self._spike_periods[argkey] = periods
            return periods

    def spikes(self, **kwargs):
        # Get unique dictionary key from keyword arguments
        args_key = self._argkey(kwargs)
        try:
            return self._spikes[args_key]
        except KeyError:
            spikes = []
            for start_i, end_i in self._spike_period_indices(**kwargs):
                dvdt = self.dvdt[start_i:end_i]
                times = self.times[start_i:end_i]
                # Get the indices before dV/dt zero crossings
                cross_indices = numpy.where((dvdt[:-1] >= 0) &
                                            (dvdt[1:] < 0))[0]
                # Pick the highest voltage zero crossing
                i = cross_indices[numpy.argmax(self._signal[cross_indices])]
                # Get the interpolated point where dV/dt crosses 0
                exact_cross = dvdt[i] / (dvdt[i] - dvdt[i + 1])
                # Calculate the exact spike time by interpolating between
                # the points straddling the zero crossing
                spike_time = times[i] + (times[i + 1] - times[i]) * exact_cross
                spikes.append(spike_time)
            spikes = neo.SpikeTrain(spikes, self._signal.t_stop,
                                    units=self.times.units)
            self._spikes[args_key] = spikes
            return spikes

    def spike_amplitudes(self, threshold_cross=100.0 * pq.mV / pq.ms,
                         threshold_return=-100.0 * pq.mV / pq.ms):
        # Get unique dictionary key from keyword arguments
        args_key = self._argkey({'threshold_cross': threshold_cross,
                                 'threshold_return': threshold_return})
        try:
            return self._spike_amplitudes[args_key]
        except KeyError:
            amplitudes = []
            for start_i, end_i in self._spike_period_indices(
                                                threshold='dvdt',               # @IgnorePep8
                                                dvdt_crossing=threshold_cross,
                                                dvdt_return=threshold_return):
                amplitudes.append(max(self._signal[start_i:end_i]))
            amplitudes = numpy.array(amplitudes) * self.units
            self._spike_amplitudes[args_key] = amplitudes
            return amplitudes

    def spike_periods(self, **kwargs):
        """
        Returns the times associated with teh spike_period_indices
        """
        # TODO: Could interpolate to find the exact time of crossings if
        #       required. Probably a bit OTT though
        return self.times[self._spike_period_indices(**kwargs)]

    def interspike_intervals(self, **kwargs):
        periods = self.spike_periods(**kwargs)
        return periods[:, 1] - periods[:, 0]

    def spike_frequency(self, **kwargs):
        """
        The average interspike interval. This is done instead of the number of
        spikes in the window divided by the interval width to stop incremental
        jumps in spike frequency when a spike falls outside of the window
        """
        spikes = self.spikes(**kwargs)
        num_spikes = len(spikes)
        if num_spikes >= 2:
            freq = (num_spikes - 1) / (spikes[-1] - spikes[0])
        elif num_spikes == 1:
            freq = 1 / (self._t_stop - self._t_start)
        else:
            freq = 0.0 * pq.Hz
        return freq

    def evenly_sampled_v_dvdt(self, resample_length, dvdt2v_scale=0.25,
                              interp_order=3):
        """
        Resamples traces at intervals along their path of length one taking
        given the axes scaled by dvdt2v_scale

        `resample_length` -- the new length between the samples
        """
        v_spl, dvdt_spl, s = self._interpolate_v_dvdt(self, self.dvdt,
                                                     dvdt2v_scale=dvdt2v_scale,  # @IgnorePep8
                                                     order=interp_order)         # @IgnorePep8
        # Get a regularly spaced array of new positions along the phase-plane
        # path to interpolate to
        new_s = numpy.arange(s[0], s[-1], resample_length)
        return v_spl(new_s), dvdt_spl(new_s)

    def spike_v_dvdt(self, num_samples, dvdt2v_scale=0.25, interp_order=3,
                     start_thresh=10.0, stop_thresh=-10.0, index_buffer=5):
        """
        Cuts outs loops (either spikes or sub-threshold oscillations) from the
        v-dV/dt trace based on the provided threshold values a

        `num_samples`     -- the number of samples to place around the V-dV/dt
                             spike
        `interp_order`    -- the order of the spline interpolation used
        `start_thresh`    -- the start dV/dt threshold
        `stop_thresh`     -- the stop dV/dt threshold
        `index_buffer`    -- the number of indices either side of the spike
                             period to include in the fitting of the spline
                             to avoid boundary effects
        """
        # Cut up the traces in between where the interpolated curve exactly
        # crosses the start and end thresholds
        spikes = []
        for start_i, stop_i in self._spike_period_indices(
                                  threshold='dvdt', dvdt_crossing=start_thresh,          # @IgnorePep8
                                  dvdt_return=stop_thresh,
                                  index_buffer=index_buffer):
            v_spl, dvdt_spl, s = self._interpolate_v_dvdt(
                                                   self._signal[start_i:stop_i],  # @IgnorePep8
                                                   self.dvdt[start_i:stop_i],
                                                   dvdt2v_scale, interp_order)
            try:
                buff = min(index_buffer * 2, len(s) - 1)
                start_s = brentq(lambda x: (dvdt_spl(x) - start_thresh),
                                 s[0], s[buff])
                end_s = brentq(lambda x: (dvdt_spl(x) - stop_thresh),
                               s[-buff], s[-1])
                # Over the loop length interpolate the splines at a fixed
                # number of points
                spike_s = numpy.linspace(start_s, end_s, num_samples)
                spike = numpy.array((v_spl(spike_s), dvdt_spl(spike_s)))
                spikes.append(spike)
            except ValueError:
                # If the integration got screwy at some point ignore this spike
                if not (self.dvdt > 10 ** 4).any():
                    raise
        return spikes

    def slice(self, t_start, t_stop):
        return AnalysedSignalSlice(self, t_start=t_start, t_stop=t_stop)

    def plot(self, show=True):
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(self.times, self._signal)
        plt.xlabel('time ({})'.format(self.times.units))
        plt.ylabel('V ({})'.format(self.units))
        plt.title('Voltage trace{}'.format('of ' + self.name
                                           if self.name else ''))
        if show:
            plt.show()

    def plot_v_dvdt(self, show=True):
        """
        Used in debugging to plot a histogram from a given trace

        `show`  -- whether to call the matplotlib 'show' function
                   (depends on whether there are subsequent plots to compare or
                   not) [bool]
        """
        from matplotlib import pyplot as plt
        # Plot original positions and interpolated traces
        plt.figure()
        plt.plot(self._signal, self.dvdt)
        plt.xlabel('v ({})'.format(self.units))
        plt.ylabel('dv/dt ({})'.format(self.units / self.times.units))
        plt.title('v-dv/dt{}'.format('of ' + self.name
                                     if self.name else ''))
        if show:
            plt.show()


class AnalysedSignalSlice(AnalysedSignal):
    """
    A thin wrapper around the AnalogSignal class to keep all of the analysis
    with the _signal so it can be shared between multiple objectives (or even
    within a single more complex objective)
    """

    def __init__(self, signal, t_start=0.0, t_stop=None):
        if not isinstance(signal, AnalysedSignal):
            raise Exception("Can only analyse AnalysedSignals (not {})"
                            .format(type(signal)))
        if t_start < signal.t_start:
            raise Exception("Slice t_start ({}) is before _signal t_start ({})"
                            .format(t_start, signal.t_start))
        if t_stop > signal.t_stop:
            raise Exception("Slice t_stop ({}) is after _signal t_stop ({})"
                            .format(t_stop, signal.t_stop))
        indices = numpy.where((signal.times >= t_start) &
                              (signal.times <= t_stop))[0]
        self._unsliced = signal
        self._start_index = indices[0]
        self._stop_index = indices[-1] + 1
        self._t_start = t_start
        self._t_stop = t_stop

    @property
    def signal(self):
        return self._unsliced._signal[self._start_index:self._stop_index]

    @property
    def t_start(self):
        return self._t_start

    @property
    def t_stop(self):
        return self._t_stop

    def __eq__(self, other):
        '''
        Equality test (==)
        '''
        return (self._unsliced == other._unsliced and
                self._start_index == other._start_index and
                self._stop_index == other._stop_index and
                self.t_start == other.t_start and
                self.t_stop == other.t_stop)

    @property
    def dvdt(self):
        return self._unsliced.dvdt[self._start_index:self._stop_index]

    def spikes(self, **kwargs):
        spikes = self._unsliced.spikes(**kwargs)
        spike_slice = spikes[numpy.where((spikes >= self.t_start) &
                                         (spikes <= self.t_stop))]
        spike_slice.t_stop = self.t_stop
        return spike_slice

    def v_dvdt_splines(self, **kwargs):
        v, dvdt, s = self._unsliced.v_dvdt_splines(**kwargs)
        return (v, dvdt, s[self._start_index:self._stop_index])

if __name__ == '__main__':
    seg = neo.PickleIO('/Users/tclose/git/neurotune/test/data/analysis/spiking_neuron.pkl').read()
    recs = AnalysedRecordings(seg)
    sig = recs.get_analysed_signal()
    print sig.spikes()
    print "done"
    

