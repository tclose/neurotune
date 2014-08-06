from __future__ import absolute_import
from abc import ABCMeta  # Metaclass for abstract base classes
import numpy
from numpy.linalg import norm
import scipy.signal
import neo.io
from . import Objective
from ..analysis import AnalysedSignal


class PhasePlaneObjective(Objective):
    """
    Phase-plane histogram objective function based on the objective in Van Geit
    2007 (Neurofitter)
    """

    # Declare this class abstract to avoid accidental construction
    __metaclass__ = ABCMeta

    def __init__(self, reference, time_start=None, time_stop=None,
                 record_variable=None, conditions=ExperimentalConditions(),
                 dvdt2v_scale=0.25, interp_order=3):
        """
        Creates a phase plane histogram from the reference traces and compares
        that with the histograms from the simulated traces

        `reference` -- traces (in Neo format) that are to be compared
                              against [list(neo.AnalogSignal)]
        `time_stop`        -- the length of the recording [float]
        `record_variable`  -- the recording site [str]
        `conditions`   -- the required experimental conditions (eg. initial
                              voltage, current clamps, etc...)
                              [neurotune.simulation.Conditions]
        `dvdt2v_scale`     -- the scale used to compare the v and dV/dt traces.
                              when calculating the length of a interval between
                              samples. Eg. a dvdt scale of 0.25 scales
                              the dV/dt axis down by 4 before calculating the
                              sample lengths
        `interp_order`     -- the type of interpolation used to resample the
                              traces (see scipy.interpolate.interp1d for list
                              of options) [str]
        """
        if time_start is None:
            time_start = (reference.t_stop - reference.t_start) / 4.0
        if time_stop is None:
            time_stop = reference.t_stop
        super(PhasePlaneObjective, self).__init__(time_start, time_stop,
                                                  conditions=conditions)
        # Save reference trace(s) as a list, converting if a single trace or
        # loading from file if a valid filename
        if isinstance(reference, str):
            self.reference_location = reference
            f = neo.io.PickleIO(reference)
            seg = f.read_segment()
            try:
                reference = AnalysedSignal(seg.analogsignals[0])
            except IndexError:
                raise Exception("No analog signals were loaded from file '{}'"
                                .format(reference))
        elif isinstance(reference, neo.AnalogSignal):
            reference = AnalysedSignal(reference)
        elif not isinstance(reference, AnalysedSignal):
            raise Exception("Unrecognised format for reference trace ({}), "
                            "must be either path-to-file, neo.AnalogSignal or "
                            "AnalysedSignal".format(type(reference)))
        if (time_start > reference.t_start or
            time_stop < reference.t_stop):
            reference = reference.slice(time_start, time_stop)
        self.reference = reference
        # Save members
        self.record_variable = record_variable
        self.interp_order = interp_order
        self.dvdt2v_scale = dvdt2v_scale


class PhasePlaneHistObjective(PhasePlaneObjective):
    """
    Phase-plane histogram objective function based on the objective in Van Geit
    2007 (Neurofitter)
    """

    BOUND_DEFAULT = 0.5

    def __init__(self, reference, num_bins=(150, 150),
                 v_bounds=(-100.0, 80.0), dvdt_bounds=(-300.0, 400.0),
                 resample_ratio=3.0, kernel_stdev=(10.0, 40.0),
                 kernel_cutoff=(3.5, 3.5), **kwargs):
        """
        Creates a phase plane histogram from the reference traces and compares
        that with the histograms from the simulated traces

        `reference`        -- traces (in Neo format) that are to be compared
                              against [list(neo.AnalogSignal)]
        `num_bins`         -- the number of bins to use for the histogram
                              [tuple[2](int)]
        `v_bounds`         -- the bounds of voltages over which the histogram
                              is generated for. If 'None' then it is calculated
                              from the bounds of the reference traces
                              [tuple[2](float)]
        `dvdt_bounds`      -- the bounds of rates of change of voltage the
                              histogram is generated for. If 'None' then it is
                              calculated from the bounds of the reference
                              traces [tuple[2](float)]
        `resample_ratio`   -- the frequency for the reinterpolated samples
                              as a fraction of bin length. Can be a scalar or a
                              tuple. No resampling is performed if it evaluates
                              to False
        `kernel_stdev`     -- the standard deviation of the Gaussian kernel
                              used to convolve the histogram. If None then
                              no convolution is performed [tuple[2](float)]
        `kernel_cutoff`    -- the number of standard deviations the Gaussian
                              kernel is truncated at
        """
        super(PhasePlaneHistObjective, self).__init__(reference,
                                                      **kwargs)
        self.num_bins = numpy.asarray(num_bins, dtype=int)
        self._set_bounds(v_bounds, dvdt_bounds)
        if resample_ratio:
            self.resample_length = norm(self.bin_size) / resample_ratio
        else:
            self.resample_length = None
        if kernel_stdev is not None:
            # Calculate the extent of the Gaussian kernel from the provided
            # width and number of standard deviations
            self.kernel_stdev = numpy.asarray(kernel_stdev)
            self.kernel_cutoff = numpy.asarray(kernel_cutoff)
            self.kernel_extent = self.kernel_stdev * kernel_cutoff
            # Calculate the number of bins the kernel spans
            nbins = 1j * (self.kernel_extent) / self.bin_size
            # Get the mesh of values over which the Gaussian kernel will be
            # evaluated
            mesh = numpy.ogrid[-kernel_cutoff[0]:kernel_cutoff[0]:nbins[0],
                               - kernel_cutoff[1]:kernel_cutoff[1]:nbins[1]]
            # Calculate the Gaussian kernel
            self.kernel = (numpy.exp(-mesh[0] ** 2) *
                           numpy.exp(-mesh[1] ** 2) /
                           (2 * numpy.pi * self.kernel_stdev[0] *
                            self.kernel_stdev[1]))
        # Generate the reference phase plane the simulated data will be
        # compared against
        self.ref_hist = self._generate_hist(self.reference)

    @property
    def range(self):
        return numpy.array(((self.bounds[0][1] - self.bounds[0][0]),
                            (self.bounds[1][1] - self.bounds[1][0])))

    @property
    def bin_size(self):
        return self.range / self.num_bins

    def fitness(self, analysis):
        signal = analysis.get_signal()
        assert((signal.t_stop - signal.t_start) == (self.reference.t_stop -
                                                    self.reference.t_start)), \
               "Attempting to compare traces of different lengths"
        phase_plane_hist = self._generate_hist(signal)
        # Get the root-mean-square difference between the reference and
        # simulated histograms
        diff = self.ref_hist - phase_plane_hist
        diff **= 2
        return diff.sum()

    def _set_bounds(self, v_bounds, dvdt_bounds):
        """
        Sets the bounds of the histogram. If v_bounds or dvdt_bounds is not
        provided (i.e. is None) then the bounds is taken to be the bounds
        between the maximum and minium values of the reference trace extended
        in both directions by BOUND_DEFAULT

        `v_bounds`         -- the bounds of voltages over which the histogram
                              is generated for. If 'None' then it is calculated
                              from the bounds of the reference traces
                              [tuple[2](float)]
        `dvdt_bounds`      -- the bounds of rates of change of voltage the
                              histogram is generated for. If 'None' then it is
                              calculated from the bounds of the reference
                              traces [tuple[2](float)]
        """
        # For both v and dV/dt bounds and see if any are None and therefore
        # require a default value to be calculated.
        self.bounds = []
        for bounds, trace in ((v_bounds, self.reference),
                              (dvdt_bounds, self.reference.dvdt)):
            if bounds is None:
                # Calculate the bounds of the reference traces
                trace = numpy.array(trace)
                min_trace = numpy.min(trace)
                max_trace = numpy.max(trace)
                # Extend the bounds by the fraction in DEFAULT_RANGE_EXTEND
                range_extend = (max_trace - min_trace) * self.BOUND_DEFAULT
                bounds = (numpy.floor(min_trace - range_extend),
                          numpy.ceil(max_trace + range_extend))
            self.bounds.append(bounds)

    def _generate_hist(self, trace):
        """
        Generates the phase plane histogram see Neurofitter paper (Van Geit
        2007). Optionally phase plane histogram is convolved
        with a Gaussian kernel

        `trace` -- a voltage trace [neo.Anaologsignal]

        returns 2D histogram
        """
        if self.resample_length:
            v, dvdt = trace.evenly_sampled_v_dvdt(self.resample_length,
                                                  self.dvdt2v_scale,
                                                  self.interp_order)
        else:
            v, dvdt = trace, trace.dvdt
        hist = numpy.histogram2d(v, dvdt, bins=self.num_bins,
                                 range=self.bounds, normed=False)[0]
        if self.kernel is not None:
            # Convolve the histogram with the precalculated Gaussian kernel
            hist = scipy.signal.convolve2d(hist, self.kernel, mode='same')
        return hist

    def plot_hist(self, trace_or_hist=None, min_max=None, diff=False,
                  show=True):
        """
        Used in debugging to plot a histogram from a given trace

        `hist_or_trace`   -- a histogram to plot or a trace to generate the
                             histogram from. If None the reference trace is
                             plotted [numpy.array((n,n)) or neo.AnalogSignal]
        `min_max`         -- the minimum and maximum values the histogram bins
                             will be capped at
        `diff`            -- plots the difference of the given histogram with
                             the provided trace/histogram ('trace_or_hist'
                             cannot be None in this case)
        `show`            -- whether to call the matplotlib 'show' function
                             (depends on whether there are subsequent plots to
                             compare or not) [bool]
        """
        from matplotlib import pyplot as plt
        if trace_or_hist is None:
            hist = self.ref_hist
        elif trace_or_hist.ndim == 2:
            hist = trace_or_hist
        else:
            hist = self._generate_hist(trace_or_hist)
        if diff:
            hist -= self.ref_hist
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
            plt.imshow(hist.T, interpolation='nearest', origin='lower',
                       **kwargs)
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
            ax.set_xticklabels([str(l)
                                for l in numpy.arange(self.bounds[0][0],
                                                      (self.bounds[0][1] +
                                                       self.range[0] / 20.0),
                                                      self.range[0] / 10.0)])
            ax.set_yticklabels([str(l)
                                for l in numpy.arange(self.bounds[1][0],
                                                      (self.bounds[1][1] +
                                                       self.range[1] / 20.0),
                                                       self.range[1] / 10.0)])
            if show:
                plt.show()

    def plot_kernel(self, show=True):
        """
        Used in debugging to plot the kernel

        `show`  -- whether to call the matplotlib 'show' function (depends on
                   whether there are subsequent plots to compare or not) [bool]
        """
        if self.kernel is None:
            raise Exception("Histogram does not use a convolution kernel")
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(self.kernel.T, interpolation='nearest', origin='lower')
        plt.xlabel('v')
        plt.ylabel('dV/dt')
        plt.xticks(numpy.linspace(0, self.kernel.shape[0] - 1, 11.0))
        plt.yticks(numpy.linspace(0, self.kernel.shape[1] - 1, 11.0))
        ax.set_xticklabels([str(l)
                            for l in numpy.arange(-self.kernel_extent[0],
                                                  self.kernel_extent[0] +
                                                  self.kernel_extent[0] / 10,
                                                  self.kernel_extent[0] / 5)])
        ax.set_yticklabels([str(l)
                            for l in numpy.arange(-self.kernel_extent[1],
                                                  self.kernel_extent[1] +
                                                  self.kernel_extent[1] / 10,
                                                  self.kernel_extent[1] / 5)])
        if show:
            plt.show()


class PhasePlanePointwiseObjective(PhasePlaneObjective):

    def __init__(self, reference, num_points=100, dvdt_thresholds=(10, -10),
                 no_spike_reference=(-100, 0.0), **kwargs):
        """
        Creates a phase plane histogram from the reference traces and compares
        that with the histograms from the simulated traces

        `reference`          -- traces (in Neo format) that are to be compared
                                against [list(neo.AnalogSignal)]
        `num_points`         -- the number of sample points to interpolate
                                between the loop start and end points
        `dvdt_thresholds`    -- the threshold above which the loop is
                                considered to have ended [tuple[2](float)]
        `no_spike_reference` -- the reference point which is used to compare
                                the reference spikes to when there are no
                                recorded spikes
        """
        super(PhasePlanePointwiseObjective, self).__init__(reference, **kwargs)
        self.thresh = dvdt_thresholds
        if self.thresh[0] < 0.0 or self.thresh[1] > 0.0:
            raise Exception("Start threshold must be above 0 and end threshold"
                            " must be below 0 (found {})".format(self.thresh))
        self.num_points = num_points
        self.no_spike_reference = no_spike_reference
        self.reference_loops = self.reference.spike_v_dvdt(
                                           self.num_points, self.dvdt2v_scale,
                                           self.interp_order, self.thresh[0],
                                           self.thresh[1])
        if len(self.reference_loops) == 0:
            self.reference_loops = [numpy.zeros((2, num_points))]
#            raise Exception("No loops found in reference signal")

    def fitness(self, analysis):
        """
        Evaluates the fitness of the recordings by comparing all reference and
        recorded loops (spikes or sub-threshold oscillations) and taking the
        sum of nearest matches from both reference-to-recorded and recorded-to-
        reference.

        `recordings`  -- a voltage trace [neo.AnalogSignal]
        """
        signal = analysis.get_signal()
        recorded_loops = signal.spike_v_dvdt(self.num_points,
                                             interp_order=self.interp_order,
                                             start_thresh=self.thresh[0],
                                             stop_thresh=self.thresh[1])
        # If the recording doesn't contain any loops make a dummy one centred
        # on the "no_spike_reference" point
        if len(recorded_loops) == 0:
            recorded_loops = [numpy.empty((2, self.num_points))]
            recorded_loops[0][0, :] = self.no_spike_reference[0]
            recorded_loops[0][1, :] = self.no_spike_reference[1]
        # Create matrix of sum-squared-differences between recorded to
        # reference loops
        fit_mat = numpy.zeros((len(recorded_loops), len(self.reference_loops)))
        for rec_loop, row in zip(recorded_loops, fit_mat):
            for i, ref_loop in enumerate(self.reference_loops):
                row[i] = numpy.sum((rec_loop - ref_loop) ** 2)
        # Get the minimum along every row and every colum and sum them together
        # for the nearest loop difference for every recorded loop to every
        # reference loop and vice-versa
        fitness = ((numpy.sum(numpy.amin(fit_mat, axis=0)) +
                    numpy.sum(numpy.amin(fit_mat, axis=1))) /
                   (fit_mat.shape[0] + fit_mat.shape[1]))
        return fitness
