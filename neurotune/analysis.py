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
import neo.core
from __builtin__ import classmethod


def _all(vals):
    try:
        return all(vals)
    except TypeError:
        return bool(vals)


class Analysis(object):

    def __init__(self, recordings, simulation_setups):
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

    def get_signal(self, key=None):
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
        if not isinstance(signal, neo.core.AnalogSignal):
            raise Exception("Can only analyse neo.coreAnalogSignals (not {})"
                            .format(type(signal)))
        self._signal = signal
        self._dvdt = None
        self._spike_periods = {}
        self._spikes = {}
        self._splines = {}

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
                _all(self._splines == other._splines))

    @property
    def signal(self):
        return self._signal

    @property
    def times(self):
        return self._signal.times

    @property
    def t_start(self):
        return self._signal.t_start

    @property
    def t_stop(self):
        return self._signal.t_stop

    @property
    def units(self):
        return self._signal.units

    @property
    def dvdt(self):
        if self._dvdt is None:
            dv = numpy.diff(self._signal)
            dt = numpy.diff(self.times)
            # Get the dvdt at the intervals between the samples
            dvdt = dv / dt
            # Linearly interpolate the dV/dt values back to the time points of
            # the original time course.
            spline = UnivariateSpline(self.times[:-1] + dt / 2.0, dvdt, s=1)
            self._dvdt = numpy.hstack((dvdt[0],
                                       pq.Quantity(spline(self.times[1:-1]),
                                                   units=dvdt.units),
                                       dvdt[-1]))
        return self._dvdt

    def _spike_period_indices(self, threshold='dvdt', start=10.0, stop=-10.0,
                              index_buffer=0):
        """
        Find sections of the trace where it crosses the given dvdt threshold
        until it loops around and crosses the stop threshold in the positive
        direction again or alternatively if threshold=='v' when it crosses the
        start threshold in the positive direction and crosses the stop
        threshold in the negative direction.

        `threshold` -- can be either 'dvdt' or 'v', which determines the
                            type of threshold used to classify the spike
                            start/stop
        `start`          -- the starting dV/dt or v threshold of the spike
        `stop`           -- the stopping threshold, which needs to be crossed
                            in the positive direction for dV/dt and negative
                            for V
        `index_buffer`   -- the number of indices to pad the periods out by on
                            either side of the spike
        """
        argkey = (threshold, start, stop, index_buffer)
        try:
            return self._spike_periods[argkey]
        except KeyError:
            if threshold not in ('dvdt', 'v'):
                raise Exception("Unrecognised threshold type '{}'"
                                .format(threshold))
            if threshold == 'dvdt':
                if stop > start:
                    raise Exception("Stop threshold ({}) must be lower than "
                                    "start threshold ({}) for dV/dt threshold "
                                    " crossing detection" .format(stop, start))
                start_inds = numpy.where((self.dvdt[1:] >= start) &
                                         (self.dvdt[:-1] < start))[0] + 1
                stop_inds = numpy.where((self.dvdt[1:] > stop) &
                                        (self.dvdt[:-1] <= stop))[0] + 1
            else:
                start_inds = numpy.where((self._signal[1:] >= start) &
                                         (self._signal[:-1] < start))[0] + 1
                stop_inds = numpy.where((self._signal[1:] < stop) &
                                        (self._signal[:-1] >= stop))[0] + 1
            if len(start_inds) == 0 or len(stop_inds) == 0:
                periods = numpy.array([])
            else:
                # Ensure the start and stop indices form regular pairs where
                # every start has a stop that comes after directly after it
                # (i.e. before another start) and vice-versa
                periods = []
                for start in start_inds:
                    try:
                        stop = stop_inds[numpy.where(stop_inds > start)][0]
                    # If the end of the loop is outside the time window
                    except IndexError:
                        continue
                    # Ensure that two spike periods don't overlap, which can
                    # occasionally occur in spike doublets for dV/dt thresholds
                    if (start >= numpy.array(periods)).all():
                        periods.append((start, stop))
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
                i = cross_indices[numpy.argmax(self[cross_indices])]
                # Get the interpolated point where dV/dt crosses 0
                exact_cross = dvdt[i] / (dvdt[i] - dvdt[i + 1])
                # Calculate the exact spike time by interpolating between
                # the points straddling the zero crossing
                spike_time = times[i] + (times[i + 1] - times[i]) * exact_cross
                spikes.append(spike_time)
            spikes = neo.SpikeTrain(spikes, self._t_stop,
                                    units=self.times.units)
            self._spikes[args_key] = spikes
            return spikes

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
                                  threshold='dvdt', start=start_thresh,          # @IgnorePep8
                                  stop=stop_thresh, index_buffer=index_buffer):
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
        return self._unsliced[self._start_index:self._stop_index]

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
