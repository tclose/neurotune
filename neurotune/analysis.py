# -*- coding: utf-8 -*-
"""
Module for mathematical analysis of voltage traces from electrophysiology.

AUTHOR: Mike Vella vellamike@gmail.com

"""
import scipy.stats
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from scipy.optimize import brentq
import numpy
import math
from copy import copy
import quantities as pq
import neo.core


class Analysis(object):

    def __init__(self, recordings, simulation_setups):
        self.recordings = recordings
        self._simulation_setups = simulation_setups
        self._requests = {}
        for seg, setup in zip(recordings.segments, self._simulation_setups):
            assert len(seg.analogsignals) == len(setup.var_request_refs)
            for sig, request_refs, in zip(seg.analogsignals,
                                              setup.var_request_refs):
                # wrap the returned signal in an AnalysedSignal wrapper
                signal = AnalysedSignal(sig)
                # For each request reference for the recorded variable
                # add the full signal or a sliced version to the requests
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
        # If this is an objective specific analysis (i.e. one with an objective
        # key preset) prepend the objective key to the signal key for the
        # complete "request" key
        if self._objective_key is not None:
            key = self._objective_key + (key,)
        return self._requests[key]

    def objective_specific(self, objective_key):
        """
        Returns a copy of the current analysis in which the provided objective
        key is automatically prepended to the key requests. This makes it
        transparent to objective components of multi-objective functions that
        they are part of a multi-objective object.
        """
        specific_analysis = copy(self)
        specific_analysis._objective_key = (objective_key,)
        return specific_analysis


class AnalysedSignal(neo.core.AnalogSignal):
    """
    A thin wrapper around the AnalogSignal class to keep all of the analysis
    with the signal so it can be shared between multiple objectives (or even
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

    def __new__(cls, signal):
        if isinstance(signal, AnalysedSignal):
            return signal
        elif not isinstance(signal, neo.core.AnalogSignal):
            raise Exception("Can only analyse neo.coreAnalogSignals (not {})"
                            .format(type(signal)))
        # Make a shallow copy of the original AnalogSignal object
        obj = copy(signal)
        # "Cast" the new AnalogSignal object to the AnalysedSignal derived
        # class
        obj.__class__ = AnalysedSignal
        obj._dvdt = None
        obj._spike_periods = {}
        obj._spikes = {}
        obj._splines = {}
        return obj

    def __reduce__(self):
        '''
        Map the __new__ function onto _new_AnalysedSignal, so that pickle
        works
        '''
        # Pass the unpickling function along with the neo_signal and members
        return _unpickle_AnalysedSignal, (self.__class__, self._base(),
                                          self._spikes, self._dvdt,
                                          self._spike_periods, self._splines)

    def _base(self):
        """
        Uncovers the neo.core.AnalogSignal object beneath
        """
        # Make a shallow copy of the AnalysedSignal
        neo_obj = copy(self)
        # Cast the copy back to a neo.core.AnalogSignal
        neo_obj.__class__ = neo.core.AnalogSignal
        return neo_obj

    def __eq__(self, other):
        '''
        Equality test (==).
        '''
        return ((self._base() == other._base()).all() and
                self._spikes == other._spikes and
                self._dvdt == other._dvdt)

    @property
    def dvdt(self):
        if self._dvdt is None:
            dv = numpy.diff(self)
            dt = numpy.diff(self.times)
            # Get the dvdt at the intervals between the samples
            dvdt = dv / dt
            # Linearly interpolate the dV/dt values back to the time points of
            # the original time course.
            spline = UnivariateSpline(self.times[:-1] + dt / 2.0, dvdt, s=1)
            self._dvdt = numpy.hstack((dvdt[0],
                                       pq.Quantity(spline(self.times[1:-1]),
                                                  units=dvdt.units), dvdt[-1]))
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
                start_inds = numpy.where((self[1:] >= start) &
                                         (self[:-1] < start))[0] + 1
                stop_inds = numpy.where((self[1:] < stop) &
                                        (self[:-1] >= stop))[0] + 1
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
                    periods[numpy.where(periods > len(self))] = len(self)
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
            spikes = neo.SpikeTrain(spikes, self.t_stop,
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
            freq = 1 / (self.t_stop - self.t_start)
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
                                                     dvdt2v_scale=dvdt2v_scale,
                                                     order=interp_order)
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
                                  threshold='dvdt', start=start_thresh,
                                  stop=stop_thresh, index_buffer=index_buffer):
            v_spl, dvdt_spl, s = self._interpolate_v_dvdt(
                                                    self[start_i:stop_i],
                                                    self.dvdt[start_i:stop_i],
                                                    dvdt2v_scale, interp_order)
            start_s = brentq(lambda x: (dvdt_spl(x) - start_thresh),
                             s[0], s[index_buffer * 2])
            end_s = brentq(lambda x: (dvdt_spl(x) - stop_thresh),
                           s[-index_buffer * 2], s[-1])
            # Over the loop length interpolate the splines at a fixed number of
            # points
            spike_s = numpy.linspace(start_s, end_s, num_samples)
            spike = numpy.array((v_spl(spike_s), dvdt_spl(spike_s)))
            spikes.append(spike)
        return spikes

    def slice(self, t_start, t_stop):
        return AnalysedSignalSlice(self, t_start=t_start, t_stop=t_stop)

    def plot(self, show=True):
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(self.times, self)
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
        plt.plot(self, self.dvdt)
        plt.xlabel('v ({})'.format(self.units))
        plt.ylabel('dv/dt ({})'.format(self.units / self.times.units))
        plt.title('v-dv/dt{}'.format('of ' + self.name
                                     if self.name else ''))
        if show:
            plt.show()


def _unpickle_AnalysedSignal(cls, signal, spikes, dvdt, spike_periods={},
                             splines={}):
    '''
    A function to map BaseAnalogSignal.__new__ to function that
        does not do the unit checking. This is needed for pickle to work.
    '''
    obj = cls(signal)
    obj._spikes = spikes
    obj._dvdt = dvdt
    obj._spike_periods = spike_periods
    obj._splines = splines
    return obj


class AnalysedSignalSlice(AnalysedSignal):
    """
    A thin wrapper around the AnalogSignal class to keep all of the analysis
    with the signal so it can be shared between multiple objectives (or even
    within a single more complex objective)
    """

    def __new__(cls, signal, t_start=0.0, t_stop=None):
        if not isinstance(signal, AnalysedSignal):
            raise Exception("Can only analyse AnalysedSignals (not {})"
                            .format(type(signal)))
        if t_start < signal.t_start:
            raise Exception("Slice t_start ({}) is before signal t_start ({})"
                            .format(t_start, signal.t_start))
        if t_stop > signal.t_stop:
            raise Exception("Slice t_stop ({}) is after signal t_stop ({})"
                            .format(t_stop, signal.t_stop))
        indices = numpy.where((signal.times >= t_start) &
                              (signal.times <= t_stop))[0]
        start_index = indices[0]
        end_index = indices[-1] + 1
        obj = AnalysedSignal.__new__(cls, signal[start_index:end_index])
        obj.__class__ = AnalysedSignalSlice
        obj.parent = signal
        obj._start_index = start_index
        obj._stop_index = end_index
        return obj

    def __eq__(self, other):
        '''
        Equality test (==)
        '''
        return (super(AnalysedSignalSlice, self).__eq__(other) and
                self.parent == other.parent and
                self._indices == other._indices)

    def __reduce__(self):
        '''
        Reduce the sliced analysedSignal for pickling
        '''
        return self.__class__, (self.parent, self.t_start, self.t_stop)

    @property
    def dvdt(self):
        return self.parent.dvdt[self._start_index:self._stop_index]

    def spikes(self, **kwargs):
        spikes = self.parent.spikes(**kwargs)
        return spikes[numpy.where((spikes >= self.t_start) &
                                  (spikes <= self.t_stop))]

    def _spike_period_indices(self, **kwargs):
        periods = self.parent._spike_period_indices(**kwargs)
        if len(periods):
            return (periods[numpy.where((periods[:, 0] >= self._start_index) &
                                        (periods[:, 1] <= self._stop_index))] -
                    self._start_index)
        else:
            return numpy.array([])

    def v_dvdt_splines(self, **kwargs):
        v, dvdt, s = self.parent.v_dvdt_splines(**kwargs)
        return (v, dvdt, s[self._start_index:self._stop_index])


def smooth(x, window_len=11, window='hanning'):
    """Smooth the data using a window with requested size.
    
    This function is useful for smoothing out experimental data.
    This method utilises the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    :param x: the input signal 
    :param window_len: the dimension of the smoothing window; should be an odd integer
    :param window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman', flat window will produce a moving average smoothing.

    :return: smoothed signal
        
    example:

    .. code-block:: python
       
       t=linspace(-2,2,0.1)
       x=sin(t)+randn(len(t))*0.1
       y=smooth(x)
    
    .. seealso::

       numpy.hanning
       numpy.hamming
       numpy.bartlett
       numpy.blackman
       numpy.convolve
       scipy.signal.lfilter
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = numpy.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode='valid')

    edge = window_len / 2
    return y[edge:-edge]

def linear_fit(t, y):
    """ Fits data to a line
        
    :param t: time vector
    :param y: variable which varies with time (such as voltage)
    :returns: Gradient M for a formula of the type y=C+M*x
    """

    vals = numpy.array(y)
    m, _ = numpy.polyfit(t, vals, 1)
    return m


def three_spike_adaptation(t, y):
    """ Linear fit of amplitude vs time of first three AP spikes

    Initial action potential amplitudes may very substaintially in amplitude
    and then settle down.
    
    :param t: time vector (AP times)
    :param y: corresponding AP amplitude
    :returns: Gradient M for a formula of the type y=C+M*x for first three action potentials
    """

    t = numpy.array(t)
    y = numpy.array(y)

    t = t[0:3]
    y = y[0:3]

    m = linear_fit(t, y)

    return m


def exp_fit(t, y):
    """
    Fits data to an exponential.
        
        Returns K for a formula of the type y=A*exp(K*x)
        
        :param t: time vector
        :param y: variable which varies with time (such as voltage)
    
    """

    vals = numpy.array(y)
    C = numpy.min(vals)
    vals = vals - C + 1e-9  # make sure the data is all positive
    vals = numpy.log(vals)
    K, _ = numpy.polyfit(t, vals, 1)

    return K


def max_min(a, t, delta=0, peak_threshold=0):
    """
    Find the maxima and minima of a voltage trace.
    
    :param a: time-dependent variable (usually voltage)
    :param t: time-vector
    :param delta: the value by which a peak or trough has to exceed its
        neighbours to be considered outside of the noise
    :param peak_threshold: peaks below this value are discarded
        
    :return: turning_points, dictionary containing number of max, min and 
        their locations
        
    .. note::

       minimum value between two peaks is in some ways a better way
       of obtaining a minimum since it guarantees an answer, this may be
       something which should be implemented.
        
    """

    gradients = numpy.diff(a)

    maxima_info = []
    minima_info = []

    count = 0

    for i in gradients[:-1]:
        count += 1

        if ((cmp(i, 0) > 0) & (cmp(gradients[count], 0) < 0) & (i != gradients[count])):
            # found a maximum
            maximum_value = a[count]
            maximum_location = count
            maximum_time = t[count]
            preceding_point_value = a[maximum_location - 1]
            succeeding_point_value = a[maximum_location + 1]

            # filter:
            maximum_valid = False  # logically consistent but not very pythonic..
            if ((maximum_value - preceding_point_value) > delta) * ((maximum_value - succeeding_point_value) > delta):
                maximum_valid = True
            if maximum_value < peak_threshold:
                maximum_valid = False
            if maximum_valid:
                maxima_info.append((maximum_value, maximum_location, maximum_time))

    maxima_num = len(maxima_info)

    if maxima_num > 0:
        minima_num = maxima_num - 1
    else:
        minima_num = 0

    import operator

    values_getter = operator.itemgetter(0)
    location_getter = operator.itemgetter(1)
    time_getter = operator.itemgetter(2)

    maxima_locations = map(location_getter, maxima_info)
    maxima_times = map(time_getter, maxima_info)
    maxima_values = map(values_getter, maxima_info)

    for i in range(maxima_num - 1):
        maximum_0_location = maxima_locations[i]
        maximum_1_location = maxima_locations[i + 1]

        interspike_slice = a[maximum_0_location:maximum_1_location]
        minimum_value = min(interspike_slice)
        minimum_location = list(interspike_slice).index(minimum_value) + maximum_0_location
        minimum_time = t[minimum_location]

        minima_info.append((minimum_value, minimum_location, minimum_time))

    minima_locations = map(location_getter, minima_info)
    minima_times = map(time_getter, minima_info)
    minima_values = map(values_getter, minima_info)

    # need to construct the dictionary here:
    turning_points = {'maxima_locations':maxima_locations, 'minima_locations':minima_locations, 'maxima_number':maxima_num, 'minima_number':minima_num, 'maxima_times':maxima_times, 'minima_times':minima_times, 'maxima_values':maxima_values, 'minima_values':minima_values}

    return turning_points

def spike_frequencies(t):
    """
    Calculate frequencies associated with interspike times
    
    :param t: a list of spike times in ms
        
    :return: list of frequencies in Hz associated with interspike times and
        times associated with the frequency (time of first spike in pair)
    
    """
    spike_times = numpy.array(t)
    interspike_times = numpy.diff(spike_times)
    interspike_frequencies = 1000 / interspike_times

    return [t[:-1], interspike_frequencies]


def mean_spike_frequency(t):
    """
    Find the average frequency of spikes
    
    :param t: a list of spike times in ms
        
    :return: mean spike frequency in Hz, calculated from mean interspike time
    
    """
    interspike_times = numpy.diff(t)
    mean_interspike_time = numpy.mean(interspike_times)
    mean_frequency = 1000.0 / (mean_interspike_time)  # factor of 1000 to give frequency in Hz

    if (math.isnan(mean_frequency)):
        mean_frequency = 0
    return mean_frequency


def y_from_x(y, x, y_to_find):
    """
    Returns list of x values corresponding to a y after a doing a 
    univariate spline interpolation

    :param x: x-axis numerical data
    :param y: corresponding y-axis numerical data
    :param y_to_find: x value for desired y-value,
        interpolated from nearest two measured x/y value pairs
    
    :return: interpolated y value
    
    """

    from scipy import interpolate

    yreduced = numpy.array(y) - y_to_find
    freduced = interpolate.UnivariateSpline(x, yreduced, s=3)

    return freduced.roots()


def single_spike_width(y, t, baseline):
    """ Find the width of a spike at a fixed height
    
    calculates the width of the spike at height baseline. If the spike shape
    does not intersect the height at both sides of the peak the method
    will return value 0. If the peak is below the baseline 0 will also 
    be returned.
    
    The input must be a single spike or nonsense may be returned.
    Multiple-spike data can be handled by the interspike_widths method.
    
    :param y: voltage trace (array) corresponding to the spike
    :param t: time value array corresponding to y
    :param baseline: the height (voltage) where the width is to be measured.        
    
    :return: width of spike at height defined by baseline

    :think about - set default baseline to none and calculate half-width
    
    """
    try:

        value = max(y)
        location = y.index(value)

        # moving left:
        while value > baseline:
            location -= 1
            value = y[location]
            undershoot_value = y[location + 1]
            overshoot_time = t[location]
            undershoot_time = t[location + 1]
            interpolated_left_time = numpy.interp(baseline, [value, undershoot_value], [overshoot_time, undershoot_time])

            if location < 0:
                raise Exception('Baseline does not intersect spike')

        # now go right
        value = max(y)
        location = y.index(value)

        while value > baseline :
            location += 1
            value = y[location]
            undershoot_value = y[location - 1]
            overshoot_time = t[location]
            undershoot_time = t[location - 1]
            interpolated_right_time = numpy.interp(baseline, [value, undershoot_value], [overshoot_time, undershoot_time])

            if location > len(y) - 1:
                raise Exception('Baseline does not intersect spike')

        width = interpolated_right_time - interpolated_left_time

    except:
        width = 0

    return width


def spike_widths(y, t, baseline=0, delta=0):
    """
    Find the widths of each spike at a fixed height in a train of spikes.
    
    Returns the width of the spike of each spike in a spike train at height 
    baseline. If the spike shapes do not intersect the height at both sides
    of the peak the method will return value 0 for that spike.
    If the peak is below the baseline 0 will also be returned for that spike.
    
    :param y: voltage trace (array) corresponding to the spike train
    :param t: time value array corresponding to y
    :param baseline: the height (voltage) where the width is to be measured.
        
    :return: width of spike at height defined by baseline
    
    """

    # first get the max and min data:
    max_min_dictionary = max_min(y, t, delta)

    max_num = max_min_dictionary['maxima_number']
#     maxima_locations=max_min_dictionary['maxima_locations']
    maxima_times = max_min_dictionary['maxima_times']
    minima_locations = max_min_dictionary['minima_locations']
#     maxima_values=max_min_dictionary['maxima_values']


    spike_widths = []
    for i in range(max_num):
        # need to splice down the y:
        if i == 0:
            left_min_location = 0
            right_min_location = minima_locations[i] + 1
        elif i == max_num - 1:
            left_min_location = minima_locations[i - 1]
            right_min_location = len(y)
        else:
            left_min_location = minima_locations[i - 1]
            right_min_location = minima_locations[i] + 1

        spike_shape = y[left_min_location:right_min_location]
        spike_t = t[left_min_location:right_min_location]

        try:
            width = single_spike_width(spike_shape, spike_t, baseline)
        except:
            width = 0

        spike_widths.append(width)

    maxima_times_widths = [maxima_times, spike_widths]
    return maxima_times_widths

def burst_analyser(t):
    """ Pearson's correlation coefficient applied to interspike times
        
    :param t: Rank-1 array containing spike times

    :return: pearson's correlation coefficient of interspike times 
    """

    x = numpy.arange(len(t))
    pearsonr = scipy.stats.pearsonr(x, t)[0]
    return pearsonr

def spike_covar(t):
    """ Calculates the coefficient of variation of interspike times 
        
    :param t: Rank-1 array containing spike times

    :return: coefficient of variation of interspike times 
    """

    interspike_times = numpy.diff(t)
    covar = scipy.stats.variation(interspike_times)
    return covar

def elburg_bursting(spike_times):
    """ bursting measure B as described by Elburg & Ooyen 2004

    :param spike_times: sequence of spike times

    :return: bursting measure B as described by Elburg & Ooyen 2004
    """

    interspikes_1 = numpy.diff(spike_times)

    num_interspikes = len(spike_times) - 1

    interspikes_2 = []
    for i in range(num_interspikes - 1):
        interspike = interspikes_1[i] + interspikes_1[i + 1]
        interspikes_2.append(interspike)

    mean_interspike = numpy.mean(interspikes_1)

    var_i_1 = numpy.var(interspikes_1)
    var_i_2 = numpy.var(interspikes_2)

    B = (2 * var_i_1 - var_i_2) / (2 * mean_interspike ** 2)

    return B

def alpha_normalised_cost_function(value, target, base=10):
    """Fitness of a value-target pair from 0 to 1 
    
    For any value/target pair will give a normalised value for
    agreement 1 is complete value-target match and 0 is 0 match.
    A mirrored exponential function is used.
    The fitness is given by the expression :math:`fitness = base^{-x}`

    where:

    .. math::
          x = {\dfrac{(value-target)}{(target + 0.01)^2}}
      
    :param value: value measured
    :param t: target
    :param base: the value 'base' in the above mathematical expression for x

    :return: fitness - a real number from 0 to 1
    
    """

    value = float(value)
    target = float(target)

    x = ((value - target) / (target + 0.01)) ** 2  # the 0.01 thing is a bit of a hack at the moment.
    fitness = base ** (-x)
    return fitness

def normalised_cost_function(value, target, Q=None):
    """ Returns fitness of a value-target pair from 0 to 1 
    
    For any value/target pair will give a normalised value for
    agreement 0 is complete value-target match and 1 is "no" match.
    
    If no Q is assigned, it is set such that it satisfies the condition
    fitness=0.7 when (target-valu)e=10*target. This is essentially 
    empirical and seems to work. Mathematical derivation is on Mike Vella's 
    Lab Book 1 p.42 (page dated 15/12/11).
             
    :param value: value measured
    :param t: target
    :param Q: This is the sharpness of the cost function, higher values correspond
        to a sharper cost function. A high Q-Value may lead an optimizer to a solution
        quickly once it nears the solution.
        
    :return: fitness value from 0 to 1
    
    """

    value = float(value)
    target = float(target)

    if Q == None:
        Q = 7 / (300 * (target ** 2))

    fitness = 1 - 1 / (Q * (target - value) ** 2 + 1)

    return fitness

def load_csv_data(file_path, plot=False):
    """Extracts time and voltage data from a csv file
    
    Data must be in a csv and in two columns, first time and second 
    voltage. Units should be SI (Volts and Seconds).

    :param file_path: full file path to file e.g /home/mike/test.csv
        
    :return: two lists - time and voltage

    """
    import csv

    csv_file = file(file_path, 'r')
    csv_reader = csv.reader(csv_file)

    v = []
    t = []

    i = 0
    for row in csv_reader:

        try:

            t_value = float(row[0]) * 1000  # convert to ms
            v_value = float(row[1]) * 1000  # convert to mV

            t.append(t_value)
            v.append(v_value)

        except:
            print 'row ', i, ' invalid'

        i += 1

    if plot:
        from matplotlib import pyplot
        pyplot.plot(t, v)
        pyplot.title('Raw data')
        pyplot.xlabel('Time (ms)')
        pyplot.ylabel('Voltage (mV)')
        pyplot.show()

    return t, v


def phase_plane(t, y, plot=False):  # plot should be here really
    """
    Return a tuple with two vectors corresponding to the phase plane of
    the tracetarget
    """
    dv = numpy.diff(y)
    dt = numpy.diff(t)
    dy_dt = dv / dt

    y = list(y)
    y = y[:-1]

    if plot:
        from matplotlib import pyplot
        pyplot.title('Phase Plot')
        pyplot.ylabel('dV/dt')
        pyplot.xlabel('Voltage (mV)')
        pyplot.plot(y, dy_dt)
        pyplot.show()

    return [y, dy_dt]

# def filter(t,v): #still experimental
#
#     import scipy
#
#     fft=scipy.fft(v) # (G) and (H)
#     bp=fft[:]
#     for i in range(len(bp)): # (H-red)
#         if i>=500:bp[i]=0
#     ibp=scipy.ifft(bp) # (I), (J), (K) and (L)
#
#     return ibp

def pptd(t, y, bins=10, xyrange=None, dvdt_threshold=None, plot=False):
    """
    Returns a 2D map of x vs y data and the xedges and yedges. 
    in the form of a vector (H,xedges,yedges) Useful for the 
    PPTD method described by Van Geit 2007.
    """

    phase_space = phase_plane(t, y)

    # filter the phase space data
    phase_dvdt_new = []
    phase_v_new = []
    if dvdt_threshold != None:
        i = 0
        for dvdt in phase_space[1]:
            if dvdt > dvdt_threshold:
                phase_dvdt_new.append(phase_space[1][i])
                phase_v_new.append(phase_space[0][i])
            i += 1
        phase_space[1] = phase_dvdt_new
        phase_space[0] = phase_v_new

    if xyrange != None:
        density_map = numpy.histogram2d(phase_space[1], phase_space[0], bins=bins,
                                normed=False, weights=None)
    elif xyrange == None:
        density_map = numpy.histogram2d(phase_space[1], phase_space[0], bins=bins, range=xyrange,
                                normed=False, weights=None)

    # Reverse the density map (probably not necessary as
    # it's being done because imshow has a funny origin):
    density = density_map[0][::-1]
    xedges = density_map[1]
    yedges = density_map[2]

    if plot:
        from matplotlib import pyplot
        extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
        imgplot = pyplot.imshow(density, extent=extent)
        imgplot.set_interpolation('nearest')  # makes image pixilated
        pyplot.title('Phase Plane Trajectory Density')
        pyplot.ylabel('dV/dt')
        pyplot.xlabel('Voltage (mV)')
        pyplot.colorbar()
        pyplot.show()

    return [density, xedges, yedges]

def spike_broadening(spike_width_list):
    """
    Returns the value of the width of the first AP over
    the mean value of the following APs.
    """

    first_spike = spike_width_list[0]
    mean_following_spikes = numpy.mean(spike_width_list[1:])
    broadening = first_spike / mean_following_spikes

    return broadening

def pptd_error(t_model, v_model, t_target, v_target, dvdt_threshold=None):
    """
    Returns error function value from comparison of two phase
    pptd maps as described by Van Geit 2007.
    """

    pptd_data = pptd(t_target, v_target, dvdt_threshold=dvdt_threshold)
    target_density_map = pptd_data[0]

    xedges = pptd_data[1]
    xmin = xedges[0]
    xmax = xedges[-1]
    yedges = pptd_data[1]
    ymin = yedges[0]
    ymax = yedges[-1]
    xyrng = [[xmin, xmax], [ymin, ymax]]

    model_density_map = pptd(t_model, v_model, xyrange=xyrng,
                           dvdt_threshold=dvdt_threshold)[0]

    # calculate number of data points for the model and target:
    N_target = sum(sum(target_density_map))
    N_model = sum(sum(model_density_map))

    # normalise each map:
    normalised_target_density_map = target_density_map / float(N_target)
    normalised_model_density_map = model_density_map / float(N_model)

    # calculate the differences and calculate the mod
    difference_matrix = normalised_target_density_map - normalised_model_density_map
    difference_matrix = abs(difference_matrix)

    # root each value:
    root_matrix = difference_matrix ** 0.5

    # sum each element:
    summed_matrix = sum(sum(root_matrix))

    # calculate the error:
    error = summed_matrix ** 2

    print 'pptd error:'
    print error

    return error

def minima_phases(t, y, delta=0):
    """
    Find the phases of minima.
    
    Minima are found by finding the minimum value between sets of two peaks.
    The phase of the minimum relative to the two peaks is then returned.
    i.e the fraction of time elapsed between the two peaks when the minimum
    occurs is returned.
    
    It is very important to make sure the correct delta is specified for
    peak discrimination, otherwise unexpected results may be returned.
     
    :param y: time-dependent variable (usually voltage)
    :param t: time-vector
    :param delta: the value by which a peak or trough has to exceed its
        neighbours to be considered "outside of the noise"
        
    :return: phase of minimum relative to peaks.
    
    """
    max_min_dictionary = max_min(y, t, delta)

    minima_num = max_min_dictionary['minima_number']
    maxima_times = max_min_dictionary['maxima_times']
    minima_times = max_min_dictionary['minima_times']
#     maxima_locations=max_min_dictionary['maxima_locations']

    minima_phases = []

    for i in range(minima_num):
        maximum_0_t = maxima_times[i]
        maximum_1_t = maxima_times[i + 1]
#         maximum_0_location=maxima_locations[i]
#         maximum_1_location=maxima_locations[i+1]
        minimum_time = minima_times[i]
        phase = (minimum_time - maximum_0_t) / (maximum_1_t - maximum_0_t)
        minima_phases.append(phase)

    phase_list = [minima_times, minima_phases]

    return phase_list


class TraceAnalysis(object):
    """
    Base class for analysis of electrophysiology data

    Constructor for TraceAnalysis base class takes the following arguments:
       
    :param v: time-dependent variable (usually voltage)
    :param t: time-array (1-to-1 correspondence with v-array)
    :param start_analysis: time in v,t where analysis is to start
    :param end_analysis: time in v,t where analysis is to end
    """

    def __nearest_index(self,
			array,
			target_value):

        """Finds index of first nearest value to target_value in array"""
        nparray = numpy.array(array)
        differences = numpy.abs(nparray - target_value)
        min_difference = differences.min()
        index = numpy.nonzero(differences == min_difference)[0][0]
        return index

    def __init__(self, v, t, start_analysis=0, end_analysis=None):

        self.v = v
        self.t = t

        start_index = self.__nearest_index(self.t, start_analysis)
        end_index = self.__nearest_index(self.t, end_analysis)

        if end_analysis != None:
            self.v = v[start_index:end_index]
            self.t = t[start_index:end_index]

    def plot_trace(self, save_fig=False, trace_name='voltage_trace.png', show_plot=True):
        """
        Plot the trace and save it if requested by user.
        """

        import matplotlib.pyplot as plt

        plt.plot(self.t, self.v)
        if save_fig:
            plt.savefig(trace_name)

        if show_plot:
            plt.show()

    def evaluate_fitness(self, target_dict={}, target_weights=None,
                         cost_function=normalised_cost_function):
        """
        Return the estimated fitness of the data, based on the cost function being used.
        
            :param target_dict: key-value pairs for targets
            :param target_weights: key-value pairs for target weights
            :param cost_function: cost function (callback) to assign individual targets sub-fitness.
        """

        # calculate max fitness value (TODO: there may be a more pythonic way to do this..)
        worst_cumulative_fitness = 0
        for target in target_dict.keys():
            print target
            if target_weights == None:
                target_weight = 1
            else:
                if target in target_weights.keys():
                    target_weight = target_weights[target]
                else:
                    target_weight = 1.0

            worst_cumulative_fitness += target_weight

        # if we have 1 or 0 peaks we won't conduct any analysis
        if self.analysable_data == False:
            print 'data is non-analysable'
            return worst_cumulative_fitness

        else:
            fitness = 0

            for target in target_dict.keys():

                target_value = target_dict[target]

                print 'examining target ' + target

                if target_weights == None:
                    target_weight = 1
                else:
                    if target in target_weights.keys():
                        target_weight = target_weights[target]
                    else:
                        target_weight = 1.0

                value = self.analysis_results[target]
                # let function pick Q automatically
                fitness += target_weight * cost_function(value, target_value)

            self.fitness = fitness
            return self.fitness


class IClampAnalysis(TraceAnalysis):
    """Analysis class for data from whole cell current injection experiments

    This is designed to work with simulations of spiking cells.

    :param v: time-dependent variable (usually voltage)
    :param t: time-vector
    :param analysis_var: dictionary containing parameters to be used
        in analysis such as delta for peak detection
    :param start_analysis: time t where analysis is to start
    :param end_analysis: time in t where analysis is to end

    """

    def __init__(self,
                 v,
                 t,
                 analysis_var,
                 start_analysis=0,
                 end_analysis=None,
                 target_data_path=None,
                 smooth_data=False,
                 show_smoothed_data=False,
                 smoothing_window_len=11):

        # call the parent constructor to prepare the v,t vectors:
        super(IClampAnalysis, self).__init__(v, t, start_analysis, end_analysis)

        if smooth_data == True:
            self.v = smooth(self.v, window_len=smoothing_window_len)

        if show_smoothed_data == True:
            from matplotlib import pyplot as plt
            plt.plot(self.t, self.v)
            plt.show()

        self.delta = analysis_var['peak_delta']
        self.baseline = analysis_var['baseline']
        self.dvdt_threshold = analysis_var['dvdt_threshold']

        self.target_data_path = target_data_path

        if "peak_threshold" in analysis_var.keys():
            peak_threshold = analysis_var["peak_threshold"]
        else:
            peak_threshold = None

        self.max_min_dictionary = max_min(self.v,
                                          self.t,
                                          self.delta,
                                          peak_threshold=peak_threshold)

        max_peak_no = self.max_min_dictionary['maxima_number']

        if max_peak_no < 3:
            self.analysable_data = False
        else:
            self.analysable_data = True

    def analyse(self):
        """If data is analysable analyses and puts all results into a dict"""

        if self.analysable_data:
            max_min_dictionary = self.max_min_dictionary
            analysis_results = {}

            analysis_results['average_minimum'] = numpy.average(max_min_dictionary['minima_values'])
            analysis_results['average_maximum'] = numpy.average(max_min_dictionary['maxima_values'])
            analysis_results['min_peak_no'] = max_min_dictionary['minima_number']
            analysis_results['max_peak_no'] = max_min_dictionary['maxima_number']
            analysis_results['mean_spike_frequency'] = mean_spike_frequency(max_min_dictionary['maxima_times'])
            analysis_results['interspike_time_covar'] = spike_covar(max_min_dictionary['maxima_times'])
            analysis_results['first_spike_time'] = max_min_dictionary['maxima_times'][0]
            trough_phases = minima_phases(self.t, self.v, delta=self.delta)
            analysis_results['trough_phase_adaptation'] = exp_fit(trough_phases[0], trough_phases[1])
            spike_width_list = spike_widths(self.v, self.t, self.baseline, self.delta)
            analysis_results['spike_width_adaptation'] = exp_fit(spike_width_list[0], spike_width_list[1])
            spike_frequency_list = spike_frequencies(max_min_dictionary['maxima_times'])
            analysis_results['peak_decay_exponent'] = three_spike_adaptation(max_min_dictionary['maxima_times'], max_min_dictionary['maxima_values'])
            analysis_results['trough_decay_exponent'] = three_spike_adaptation(max_min_dictionary['minima_times'], max_min_dictionary['minima_values'])
            analysis_results['spike_frequency_adaptation'] = exp_fit(spike_frequency_list[0], spike_frequency_list[1])
            analysis_results['spike_broadening'] = spike_broadening(spike_width_list[1])
            analysis_results['peak_linear_gradient'] = linear_fit(max_min_dictionary["maxima_times"], max_min_dictionary["maxima_values"])


            # this line here is because PPTD needs to be compared directly with experimental data:
            if self.target_data_path != None:
                t_experimental, v_experimental = load_csv_data(self.target_data_path)
                try:
                    analysis_results['pptd_error'] = pptd_error(self.t, self.v,
                                              t_experimental, v_experimental,
                                              dvdt_threshold=self.dvdt_threshold)
                except:
                    print 'WARNING PPTD failure'
                    analysis_results['pptd_error'] = 1

            self.analysis_results = analysis_results

        else:
            print 'data not suitable for analysis,<3 APs'
