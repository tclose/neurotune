from __future__ import absolute_import
from abc import ABCMeta  # Metaclass for abstract base classes
from collections import namedtuple
import numpy
import scipy.signal
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.optimize
import neo.io
from .__init__ import Objective
from ..simulation.__init__ import RecordingRequest


class PhasePlaneObjective(Objective):
    """
    Phase-plane histogram objective function based on the objective in Van Geit 2007 (Neurofitter)
    """

    __metaclass__ = ABCMeta  # Declare this class abstract to avoid accidental construction

    def __init__(self, reference_traces, time_start=500.0, time_stop=2000.0, record_variable=None,
                 exp_conditions=None, dvdt_scale=0.25, interp_order=3):
        """
        Creates a phase plane histogram from the reference traces and compares that with the 
        histograms from the simulated traces
        
        `reference_traces` -- traces (in Neo format) that are to be compared against 
                              [list(neo.AnalogSignal)]
        `time_stop`        -- the length of the recording [float]
        `record_variable`  -- the recording site [str]
        `exp_conditions`   -- the required experimental conditions (eg. initial voltage, current 
                              clamps, etc...) [neurotune.controllers.ExperimentalConditions]
        `dvdt_scale`       -- the scale used to compare the v and dV/dt traces. when calculating the
                              length of a interval between samples. Eg. a dvdt scale of 0.25 scales
                              the dV/dt axis down by 4 before calculating the sample lengths
        `interp_order`      -- the type of interpolation used to resample the traces 
                              (see scipy.interpolate.interp1d for list of options) [str]
        """
        super(PhasePlaneObjective, self).__init__(time_start, time_stop)
        # Save reference trace(s) as a list, converting if a single trace or loading from file
        # if a valid filename
        if isinstance(reference_traces, str):
            f = neo.io.PickleIO(reference_traces)
            seg = f.read_segment()
            self.reference_traces = seg.analogsignals
        elif isinstance(reference_traces, neo.AnalogSignal):
            self.reference_traces = [reference_traces]
        # Save members
        self.record_variable = record_variable
        self.exp_conditions = exp_conditions
        self.interp_order = interp_order
        self.dvdt_scale = dvdt_scale

    def get_recording_requests(self):
        """
        Gets all recording requests required by the objective function
        """
        return RecordingRequest(record_variable=self.record_variable, record_time=self.time_stop,
                                conditions=self.exp_conditions)

    def _calculate_v_and_dvdt(self, trace):
        """
        Trims the trace to the indices within the time_start and time_stop then calculates the 
        discrete time derivative and resamples the trace to the resampling interval if provided
        
        `trace`    -- voltage trace (in Neo format) [list(neo.AnalogSignal)]
        `resample` -- 
        `interp_order`  -- 
        
        returns trimmed voltage trace, dV and dV/dt in a tuple
        """
        # Calculate dv/dt via difference between trace samples. NB # the float() call is required to
        # remove the "python-quantities" units
        start_index = int(round(trace.sampling_rate * (self.time_start + float(trace.t_start))))
        stop_index = int(round(trace.sampling_rate * (self.time_stop + float(trace.t_start))))
        v = trace[start_index:stop_index]
        dv = numpy.diff(v)
        dt = numpy.diff(v.times)
        v = v[:-1]
        dvdt = dv / dt
        return v, dvdt

    def _get_interpolators(self, v, dvdt, interp_order=None):
        """
        Gets interpolators as returned from scipy.interpolate.interp1d for v and dV/dt as well as 
        the length of the trace in the phase plane
        
        `v`                -- voltage trace [numpy.array(float)]
        `dvdt`             -- dV/dt trace [numpy.array(float)]
        `interp_order`     -- the interpolation type used (see scipy.interpolate.interp1d) [str]
        return             -- a tuple containing scipy interpolators for v and dV/dt and the 
                              positions of the original samples along the interpolated path.
        """
        if interp_order is None:
            interp_order = self.interp_order
        dv = numpy.diff(v)
        d_dvdt = numpy.diff(dvdt)
        interval_lengths = numpy.sqrt(numpy.asarray(dv) ** 2 +
                                      (numpy.asarray(d_dvdt) * self.dvdt_scale) ** 2)
        # Calculate the "positions" of the samples in terms of the fraction of the length
        # of the v-dv/dt path
        s = numpy.concatenate(([0.0], numpy.cumsum(interval_lengths)))
        # Save the original (non-sparsified) value to be returned
        return (InterpolatedUnivariateSpline(s, v, k=interp_order),
                InterpolatedUnivariateSpline(s, dvdt, k=interp_order), s)

    def plot_d_dvdt(self, trace, show=True):
        """
        Used in debugging to plot a histogram from a given trace
        
        `trace` -- the trace to generate the histogram from [neo.AnalogSignal]
        `show`  -- whether to call the matplotlib 'show' function (depends on whether there are
                   subsequent plots to compare or not) [bool]
        """
        from matplotlib import pyplot as plt
        # Temporarily switch of resampling to get original positions of v dvdt plot
        orig_resample_length = self.resample_length
        self.resample_length = False
        orig_v, orig_dvdt = self._calculate_v_and_dvdt(trace)
        self.resample_length = orig_resample_length
        v, dvdt = self._calculate_v_and_dvdt(trace)
        if isinstance(show, str):
            import cPickle as pickle
            with open(show, 'w') as f:
                pickle.dump(((orig_v, orig_dvdt), (v, dvdt)), f)
        else:
            # Plot original positions and interpolated traces
            plt.figure()
            plt.plot(orig_v, orig_dvdt, 'x')
            plt.plot(v, dvdt)
            plt.xlabel('v')
            plt.ylabel('dv/dt')
            plt.title('v-dv/dt plot of trace')
            if show:
                plt.show()


class PhasePlaneHistObjective(PhasePlaneObjective):
    """
    Phase-plane histogram objective function based on the objective in Van Geit 2007 (Neurofitter)
    """

    _FRAC_TO_EXTEND_DEFAULT_BOUNDS = 0.5

    def __init__(self, reference_traces, num_bins=(150, 150), v_bounds=None, dvdt_bounds=None,
                 sample_to_bin_ratio=3.0, **kwargs):
        """
        Creates a phase plane histogram from the reference traces and compares that with the 
        histograms from the simulated traces
        
        `reference_traces` -- traces (in Neo format) that are to be compared against 
                              [list(neo.AnalogSignal)]
        `num_bins`         -- the number of bins to use for the histogram [tuple[2](int)]
        `v_bounds`         -- the bounds of voltages over which the histogram is generated for. If 
                              'None' then it is calculated from the bounds of the reference traces
                              [tuple[2](float)] 
        `dvdt_bounds`      -- the bounds of rates of change of voltage the histogram is generated 
                              for. If 'None' then it is calculated from the bounds of the reference
                              traces [tuple[2](float)]
        `sample_to_bin_ratio` -- the frequency for the reinterpolated samples as a fraction of bin
                              length. Can be a scalar or a tuple. No resampling is performed if it 
                              evaluates to False  
        """
        super(PhasePlaneHistObjective, self).__init__(reference_traces, **kwargs)
        self.num_bins = numpy.asarray(num_bins, dtype=int)
        self._set_bounds(v_bounds, dvdt_bounds, self.reference_traces)
        if sample_to_bin_ratio:
            resample = self.bin_size / sample_to_bin_ratio
            self.resample_length = numpy.sqrt(resample[0] ** 2 + resample[1] ** 2)
        else:
            self.resample_length = None
        # Generate the reference phase plane the simulated data will be compared against
        self.ref_phase_plane_hist = numpy.zeros(self.num_bins)
        for ref_trace in self.reference_traces:
            self.ref_phase_plane_hist += self._generate_phase_plane_hist(ref_trace)
        # Normalise the reference phase plane
        self.ref_phase_plane_hist /= len(self.reference_traces)

    @property
    def range(self):
        return numpy.array(((self.bounds[0][1] - self.bounds[0][0]),
                            (self.bounds[1][1] - self.bounds[1][0])))

    @property
    def bin_size(self):
        return self.range / self.num_bins

    def fitness(self, requests_dict):
        recordings = next(requests_dict.itervalues())
        phase_plane_hist = self._generate_phase_plane_hist(recordings)
        # Get the root-mean-square difference between the reference and simulated histograms
        diff = self.ref_phase_plane_hist - phase_plane_hist
        diff **= 2
        return diff.sum()

    def _set_bounds(self, v_bounds, dvdt_bounds, reference_traces):
        """
        Sets the bounds of the histogram. If v_bounds or dvdt_bounds is not provided (i.e. is None)
        then the bounds is taken to be the bounds between the maximum and minium values of the 
        reference trace extended in both directions by _FRAC_TO_EXTEND_DEFAULT_BOUNDS
        
        `v_bounds`         -- the bounds of voltages over which the histogram is generated for. If 
                              'None' then it is calculated from the bounds of the reference traces 
                              [tuple[2](float)]
        `dvdt_bounds`      -- the bounds of rates of change of voltage the histogram is generated for.
                              If 'None' then it is calculated from the bounds of the reference traces 
                              [tuple[2](float)]
        `reference_traces` -- traces (in Neo format) that are to be compared against 
                              [list(neo.AnalogSignal)]
        """
        # Get voltages and dV/dt values for all of the reference traces in a list of numpy.arrays
        # which will be converted into a single array for convenient maximum and minimum calculation
        v, dvdt = zip(*[self._calculate_v_and_dvdt(t) for t in reference_traces])
        # For both v and dV/dt bounds and see if any are None and therefore require a default value
        # to be calculated.
        self.bounds = []
        for bounds, trace in ((v_bounds, v), (dvdt_bounds, dvdt)):
            if bounds is None:
                # Calculate the bounds of the reference traces
                trace = numpy.array(trace)
                min_trace = numpy.min(trace)
                max_trace = numpy.max(trace)
                # Extend the bounds by the fraction in DEFAULT_RANGE_EXTEND
                range_extend = (max_trace - min_trace) * self._FRAC_TO_EXTEND_DEFAULT_BOUNDS
                bounds = (numpy.floor(min_trace - range_extend),
                          numpy.ceil(max_trace + range_extend))
            self.bounds.append(bounds)

    def _generate_phase_plane_hist(self, trace):
        """
        Generates the phase plane histogram see Neurofitter paper (Van Geit 2007)
        
        `trace` -- a voltage trace [neo.Anaologsignal]
        
        returns 2D histogram
        """
        v, dvdt = self._calculate_v_and_dvdt(trace)
        if self.resample_length:
            v, dvdt = self._resample_traces(v, dvdt, self.resample_length)
        return numpy.histogram2d(v, dvdt, bins=self.num_bins, range=self.bounds, normed=False)[0]


    def _resample_traces(self, v, dvdt, resample_length):
        """
        Resamples traces at intervals along their path of length one taking given the axes scaled
        by 'resample'
        
        `v`               -- voltage trace [numpy.array(float)]
        `dvdt`            -- dV/dt trace [numpy.array(float)]
        `resample_length` -- the new length between the samples 
        """
        v_interp, dvdt_interp, s = self._get_interpolators(v, dvdt)
        # Get a regularly spaced array of new positions along the phase-plane path to 
        # interpolate to
        new_s = numpy.arange(s[0], s[-1], resample_length)
        return v_interp(new_s), dvdt_interp(new_s)

    def plot_hist(self, trace_or_hist=None, min_max=None, diff=False, show=True):
        """
        Used in debugging to plot a histogram from a given trace
        
        `hist_or_trace`   -- a histogram to plot or a trace to generate the histogram from. If None
                             the reference trace is plotted
                             [numpy.array((n,n)) or neo.AnalogSignal]
        `min_max`         -- the minimum and maximum values the histogram bins will be capped at
        `diff`            -- plots the difference of the given histogram with the provided 
                             trace/histogram ('trace_or_hist' cannot be None in this case)
        `show`            -- whether to call the matplotlib 'show' function (depends on whether 
                             there are subsequent plots to compare or not) [bool]
        """
        from matplotlib import pyplot as plt
        if trace_or_hist is None:
            hist = self.ref_phase_plane_hist
        elif trace_or_hist.ndim == 2:
            hist = trace_or_hist
        else:
            hist = self._generate_phase_plane_hist(trace_or_hist)
        if diff:
            hist -= self.ref_phase_plane_hist
        kwargs = {}
        if min_max is not None:
            kwargs['vmin'], kwargs['vmax'] = min_max
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if isinstance(show, str):
            import cPickle as pickle
            with open(show, 'w') as f:
                pickle.dump(hist, f)
        else:
            plt.imshow(hist.T, interpolation='nearest', origin='lower', **kwargs)
            plt.xlabel('v')
            plt.ylabel('dv/dt')
            plt.xticks(numpy.linspace(0, self.num_bins[0] - 1, 11.0))
            plt.yticks(numpy.linspace(0, self.num_bins[1] - 1, 11.0))
            if diff:
                plt.title('v-dv/dt and reference histogram difference')
            elif trace_or_hist is None:    
                plt.title('Reference v-dv/dt histogram')
            else:
                plt.title('v-dv/dt histogram')
            plt.colorbar()
            ax.set_xticklabels([str(l) for l in numpy.arange(self.bounds[0][0],
                                                             self.bounds[0][1] + self.range[0] / 20.0,
                                                             self.range[0] / 10.0)])
            ax.set_yticklabels([str(l) for l in numpy.arange(self.bounds[1][0],
                                                             self.bounds[1][1] + self.range[1] / 20.0,
                                                             self.range[1] / 10.0)])
            if show:
                plt.show()


class ConvPhasePlaneHistObjective(PhasePlaneHistObjective):
    """
    Phase-plane histogram objective function based on the objective in Van Geit 2007 (Neurofitter)
    with the exception that the histograms are smoothed by a Gaussian kernel after they are generated
    """

    def __init__(self, reference_traces, num_bins=(150, 150), kernel_width=(5.25, 18.75),
                 num_stdevs=(5, 5), **kwargs):
        """
        Creates a phase plane histogram convolved with a Gaussian kernel from the reference traces 
        and compares that with a similarly convolved histogram of the simulated traces
        
        `reference_traces` -- traces (in Neo format) that are to be compared against 
                               [list(neo.AnalogSignal)]
        `num_bins`         -- the number of bins to use for the histogram [tuple(int)]
        `kernel_width`     -- the standard deviation of the Gaussian kernel used to convolve the 
                               histogram [tuple[2](float)]
        `num_stdevs`       -- the number of standard deviations the Gaussian kernel extends over
        """
        # Calculate the extent of the Gaussian kernel from the provided width and number of
        # standard deviations
        self.kernel_width = numpy.asarray(kernel_width)
        self.num_stdevs = numpy.asarray(num_stdevs)
        self.kernel_extent = self.kernel_width * self.num_stdevs
        # The creation of the kernel is delayed until it is required (in the
        # _generate_phase_plane_hist method) because it relies on the range of the histogram which
        # is set in the super().__init__ method
        self.kernel = None
        # Call the parent class __init__ method
        super(ConvPhasePlaneHistObjective, self).__init__(reference_traces, num_bins=num_bins,
                                                          **kwargs)

    def _generate_phase_plane_hist(self, trace):
        """
        Extends the vanilla phase plane histogram to allow it to be convolved with a Gaussian kernel
        
        `trace` -- a voltage trace [neo.Anaologsignal]
        """
        # Get the unconvolved histogram
        unconv_hist = super(ConvPhasePlaneHistObjective, self)._generate_phase_plane_hist(trace)
        if self.kernel is None:
            # Calculate the number of bins the kernel spans
            num_kernel_bins = (self.kernel_extent) / self.bin_size
            # Get the mesh of values over which the Gaussian kernel will be evaluated
            mesh = numpy.ogrid[-self.num_stdevs[0]:self.num_stdevs[0]:num_kernel_bins[0] * 1j,
                               - self.num_stdevs[1]:self.num_stdevs[1]:num_kernel_bins[1] * 1j]
            # Calculate the Gaussian kernel
            self.kernel = (numpy.exp(-mesh[0] ** 2) * numpy.exp(-mesh[1] ** 2) /
                           (2 * numpy.pi * self.kernel_width[0] * self.kernel_width[1]))
        # Convolve the histogram with the precalculated Gaussian kernel
        return scipy.signal.convolve2d(unconv_hist, self.kernel, mode='same')

    def plot_kernel(self, show=True):
        """
        Used in debugging to plot the kernel
        
        `show`  -- whether to call the matplotlib 'show' function (depends on whether there are
                   subsequent plots to compare or not) [bool]
        """
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(self.kernel.T, interpolation='nearest', origin='lower')
        plt.xlabel('v')
        plt.ylabel('dV/dt')
        plt.xticks(numpy.linspace(0, self.kernel.shape[0] - 1, 11.0))
        plt.yticks(numpy.linspace(0, self.kernel.shape[1] - 1, 11.0))
        ax.set_xticklabels([str(l) for l in numpy.arange(-self.kernel_extent[0],
                                                          self.kernel_extent[0] +
                                                          self.kernel_extent[0] / 10,
                                                          self.kernel_extent[0] / 5.0)])
        ax.set_yticklabels([str(l) for l in numpy.arange(-self.kernel_extent[1],
                                                          self.kernel_extent[1] +
                                                          self.kernel_extent[1] / 10,
                                                          self.kernel_extent[1] / 5.0)])
        if show:
            plt.show()


class PhasePlanePointwiseObjective(PhasePlaneObjective):

    def __init__(self, reference_traces, dvdt_thresholds, num_points, **kwargs):
        """
        Creates a phase plane histogram from the reference traces and compares that with the 
        histograms from the simulated traces
        
        `reference_traces` -- traces (in Neo format) that are to be compared against 
                              [list(neo.AnalogSignal)]
        `dvdt_thresholds`  -- the threshold above which the loop is considered to have ended
                              [tuple[2](float)]
        `num_points`       -- the number of sample points to interpolate between the loop start and
                              end points   
        """
        super(PhasePlanePointwiseObjective, self).__init__(reference_traces, **kwargs)
        self.thresh = dvdt_thresholds
        if self.thresh[0] < 0.0 or self.thresh[1] > 0.0:
            raise Exception("Start threshold must be above 0 and end threshold must be below 0 "
                            "(found {})".format(self.thresh))
        self.num_points = num_points
        self.reference_loops = []
        for t in self.reference_traces:
            self.reference_loops.extend(self._cut_out_loops(t))
        if len(self.reference_loops) == 0:
            raise Exception("No loops found in reference signal")

    def _cut_out_loops(self, trace):
        """
        Cuts outs loops (either spikes or sub-threshold oscillations) from the v-dV/dt trace based
        on the provided threshold values
        
        `trace`  -- the voltage trace [numpy.array(float)]                           
        """
        # Get the v and dvdt and their spline interpolators
        v, dvdt = self._calculate_v_and_dvdt(trace)
        v_spline, dvdt_spline, s_positions = self._get_interpolators(v, dvdt)
        # Find the indices where the v-dV/dt trace crosses the start and end thresholds respectively
        loop_starts = numpy.where((dvdt[1:] >= self.thresh[0]) & (dvdt[:-1] < self.thresh[0]))[0] + 1
        loop_ends = numpy.where((dvdt[1:] >= self.thresh[1]) & (dvdt[:-1] < self.thresh[1]))[0] + 1
        # Cut up the traces in between where the interpolated curve exactly crosses the start
        # and end thresholds
        loops = []
        def ensure_s_bound(lbound_index, thresh, direction):
            """
            A helper function to ensure that the start and end indices of the loop fall exactly 
            either side of the threshold and if not extend the search interval for the indices that do
            """
            try:
                while (direction * dvdt_spline(s_positions[lbound_index])) < direction * thresh:
                    lbound_index += direction
                return s_positions[lbound_index]
            except IndexError:
                raise Exception("Spline interpolation is not accurate enough to detect start of "
                                "loop, consider using a smaller simulation timestep or greater loop"
                                " threshold")
        for start_index, end_index in zip(loop_starts, loop_ends):
            start_s = scipy.optimize.brentq(lambda s: dvdt_spline(s) - self.thresh[0],
                                            ensure_s_bound(start_index - 1, self.thresh[0], -1),
                                            ensure_s_bound(start_index, self.thresh[0], 1))
            end_s = scipy.optimize.brentq(lambda s: dvdt_spline(s) - self.thresh[1],
                                            ensure_s_bound(end_index - 1, self.thresh[1], -1),
                                            ensure_s_bound(end_index, self.thresh[1], 1))
            # Over the loop length interpolate the splines at a fixed number of points
            s_range = numpy.linspace(start_s, end_s, self.num_points)
            loops.append(numpy.array((v_spline(s_range), dvdt_spline(s_range))))
        return loops

    def fitness(self, recordings):
        """
        Evaluates the fitness of the recordings by comparing all reference and recorded loops 
        (spikes or sub-threshold oscillations) and taking the sum of nearest matches from both
        reference-to-recorded and recorded-to-reference.
        
        `recordings`  -- a voltage trace [neo.AnalogSignal]
        """
        recorded_loops = self._cut_out_loops(recordings)
        # If the recording doesn't contain any loops make a dummy one which forms a straight line
        # between the min and max dvdt values on the mean voltage
        if len(recorded_loops) == 0:
            v, dvdt = self._calculate_v_and_dvdt(recordings)
            dummy_v = numpy.empty(self.num_points)
            dummy_v.fill(v.mean())
            dummy_dvdt = numpy.linspace(dvdt.min(), dvdt.max(), self.num_points)
            recorded_loops = [numpy.array((dummy_v, dummy_dvdt))]
        # Create matrix of sum-squared-differences between recorded to reference loops
        fit_mat = numpy.empty((len(recorded_loops), len(self.reference_loops)))
        for row_i, rec_loop in enumerate(recorded_loops):
            for col_i, ref_loop in enumerate(self.reference_loops):
                fit_mat[row_i, col_i] = numpy.sum((rec_loop - ref_loop) ** 2)
        # Get the minimum along every row and every colum and sum them together for the nearest
        # loop difference for every recorded loop to every reference loop and vice-versa
        fitness = ((numpy.sum(numpy.amin(fit_mat, axis=0) ** 2) +
                    numpy.sum(numpy.amin(fit_mat, axis=1) ** 2)) /
                   (fit_mat.shape[0] + fit_mat.shape[1]))
        return fitness
