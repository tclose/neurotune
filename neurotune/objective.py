from abc import ABCMeta  # Metaclass for abstract base classes
import numpy
import scipy.signal
import inspyred
import neo.io
from .simulation import RecordingRequest


class Objective(object):
    """
    Base Objective class
    """

    __metaclass__ = ABCMeta  # Declare this class abstract to avoid accidental construction

    def __init__(self, time_start=0, time_stop=2000.0):
        """
        `time_stop` -- the required length of the recording required to evaluate the objective
        """
        self.time_start = time_start
        self.time_stop = time_stop

    def fitness(self, recordings):
        """
        Evaluates the fitness function given the simulated data
        
        `recordings` -- a dictionary containing the simulated data to be assess, with the keys 
                            corresponding to the keys of the recording request dictionary returned 
                            by 'get_recording requests'
        """
        raise NotImplementedError("Derived Objective class '{}' does not implement fitness method"
                                  .format(self.__class__.__name__))

    def get_recording_requests(self):
        """
        Returns a RecordingRequest object or a dictionary of RecordingRequest objects with unique keys
        representing the recordings that are required from the simulation controller
        """
        return RecordingRequest(time_stop=self.time_stop)


class SpikeFrequencyObjective(Objective):
    """
    A simple objective based on the squared difference between the spike frequencies
    """

    def __init__(self, frequency, time_start, time_stop):
        """
        `frequency`  -- the desired spike frequency
        `time_stop` -- the length of time to run the simulation
        """
        super(SpikeFrequencyObjective, self).__init__(time_start, time_stop)
        self.frequency = frequency

    def fitness(self, recordings):
        recording_frequency = len(recordings[recordings < self.time_stop]) / self.time_stop
        return (self.frequency - recording_frequency) ** 2

    def get_recording_requests(self):
        """
        Returns a RecordingRequest object or a dictionary of RecordingRequest objects with unique keys
        representing the recordings that are required from the simulation controller
        """
        return RecordingRequest(time_stop=self.time_stop, record_variable='spikes')


class PhasePlaneObjective(Objective):
    """
    Phase-plane histogram objective function based on the objective in Van Geit 2007 (Neurofitter)
    """

    __metaclass__ = ABCMeta  # Declare this class abstract to avoid accidental construction

    def __init__(self, reference_traces, time_start=500.0, time_stop=2000.0, record_variable=None,
                 exp_conditions=None, resample=(0.375,  1.5), interp_type='cubic'):
        """
        Creates a phase plane histogram from the reference traces and compares that with the 
        histograms from the simulated traces
        
        `reference_traces` -- traces (in Neo format) that are to be compared against 
                              [list(neo.AnalogSignal)]
        `time_stop`        -- the length of the recording [float]
        `record_variable`  -- the recording site [str]
        `exp_conditions`   -- the required experimental conditions (eg. initial voltage, current 
                              clamps, etc...) [neurotune.controllers.ExperimentalConditions] 
        `resample`         -- the periods used to resample the traces on the v-dV/dt plot. Can be:
                              * False: no resampling)
                              * True: resampling to default sizes determinted by bin size and 
                                      _BIN_TO_SAMPLE_FREQ_RATIO_DEFAULT
                              * tuple(float)[2]: a pair of sample periods for v & dV/dt respectively
        `interp_type`    -- the type of interpolation used to resample the traces 
                              (see scipy.interpolate.interp1d for list of options) [str]
        """
        super(PhasePlaneObjective, self).__init__(time_start, time_stop)
        if isinstance(reference_traces, str):
            f = neo.io.PickleIO(reference_traces)
            seg = f.read_segment()
            self.reference_traces = seg.analogsignals
        elif isinstance(reference_traces, neo.AnalogSignal):
            self.reference_traces = [reference_traces]
        # Save the recording site and number of bins
        self.record_variable = record_variable
        self.exp_conditions = exp_conditions
        self.resample = numpy.asarray(resample, dtype=float) if resample is not False else False
        self.interp_type = interp_type

    def get_recording_requests(self):
        """
        Gets all recording requests required by the objective function
        """
        return RecordingRequest(record_variable=self.record_variable, record_time=self.time_stop,
                                conditions=self.exp_conditions)

    def _calculate_v_and_dvdt(self, trace, resample=None, interp_type=None):
        """
        Trims the trace to the indices within the time_start and time_stop then calculates the 
        discrete time derivative and resamples the trace to the resampling interval if provided
        
        `trace`    -- voltage trace (in Neo format) [list(neo.AnalogSignal)]
        `resample` -- 
        `interp_type`  -- 
        
        returns trimmed voltage trace, dV and dV/dt in a tuple
        """
        # Set default values for resample and interp type from class members
        if resample is None:
            resample = self.resample
            if interp_type is None:
                interp_type = self.interp_type
        # Calculate dv/dt via difference between trace samples. NB # the float() call is required to
        # remove the "python-quantities" units
        start_index = int(round(trace.sampling_rate * (self.time_start + float(trace.t_start))))
        stop_index = int(round(trace.sampling_rate * (self.time_stop + float(trace.t_start))))
        v = trace[start_index:stop_index]
        dv = numpy.diff(v)
        dt = numpy.diff(v.times)
        v = v[:-1]
        dvdt = dv / dt
        if resample is not False:
            v, dvdt = self._resample_traces(v, dvdt, resample, interp_type)
        return v, dvdt

    def _resample_traces(self, v, dvdt, resample, interp_type):
        """
        Resamples traces at intervals along their path of length one taking given the axes scaled
        by 'resample'
        
        `v`            -- voltage trace [numpy.array(float)]
        `dvdt`         -- dV/dt trace [numpy.array(float)]
        `resample`     --
        `interp_type`  -- 
        """
        # Set default values for resample and interp type from class members
        if resample is None:
            resample = self.resample
            if interp_type is None:
                interp_type = self.interp_type
        # Normalise the relative weights
        resample_norm = numpy.sqrt((resample * resample).sum())
        rescale = (0.5 * resample_norm) / resample
        # Get interpolators for v and dV/dt
        v_interp, dvdt_interp, s = self._get_interpolators(v, dvdt, rescale, interp_type)
        # Get a regularly spaced array of new positions along the phase-plane path to interpolate to
        new_s = numpy.arange(0, s[-1], resample_norm)
        return v_interp(new_s), dvdt_interp(new_s)

    def _get_interpolators(self, v, dvdt, relative_scale, interp_type, sparse_period=10.0):
        """
        Gets interpolators as returned from scipy.interpolate.interp1d for v and dV/dt as well as 
        the length of the trace in the phase plane
        
        `v`                -- voltage trace [numpy.array(float)]
        `dvdt`             -- dV/dt trace [numpy.array(float)]
        `relative_scale`   -- an arbitrary relative scale used to compare the v and dV/dt traces. 
                              For example a relative scale of (2.0, 1.0) would weight changes in the 
                              voltage dimension as travelling twice as far as changes in the dV/dt 
                              dimension of the phase plane [tuple[2](float)]
        `interp_type`      -- the interpolation type used (see scipy.interpolate.interp1d) [str]
        `sparse_period`    -- as performing computationally intensive interpolation on many samples
                              is a drag on performance, dense sections of the curve are first
                              decimated before the interpolation is performed. The 'sparse_period'
                              argument determines the sampling period that the dense sections 
                              (defined as any section with sampling denser than this period) are
                              resampled to.
        return             -- a tuple containing scipy interpolators for v and dV/dt and the 
                              positions of the original samples along the interpolated path
        """
        # In order to resample the traces, the length of each v-dV/dt path segment needs to be
        # calculated (as defined by the relative scale between them)
        dv = numpy.diff(v)
        d_dvdt = numpy.diff(dvdt)
        interval_lengths = numpy.sqrt((numpy.asarray(dv) * relative_scale[0]) ** 2 +
                                      (numpy.asarray(d_dvdt) * relative_scale[1]) ** 2)
        # Calculate the "positions" of the samples in terms of the fraction of the length
        # of the v-dv/dt path
        s = numpy.concatenate(([0.0], numpy.cumsum(interval_lengths)))
        # Save the original (non-sparsified) value to be returned
        original_s = s
        # If using a more computationally intensive interpolation technique, the v-dV/dt paths are
        # pre-processed to decimate the densely sampled sections of the path
        if interp_type in ('quadratic', 'cubic'):
            # Get a list of landmark samples that should be retained in the sparse sampling
            # i.e. samples with large intervals between them that occur during fast sections of
            # the phase plane (i.e. spikes)
            landmarks = numpy.empty(len(v) + 1)
            # Make the samples on both sides of large intervals "landmark" samples
            landmarks[:-2] = interval_lengths > sparse_period  # low edge of the large intervals
            landmarks[landmarks[:-2].nonzero()[0] + 1] = True  # high edge of the large intervals
            landmarks[0] = landmarks[-2:] = True  # Ensure the first and last samples are also included
            # TODO: Add points of inflexion to the landmark mask
            # Break the path up into chains of densely and sparsely sampled sections (i.e. fast and
            # slow parts of the voltage trace)
            end_landmark_samples = numpy.logical_and(landmarks[:-1] == 1, landmarks[1:] == 0)
            end_nonlandmark_samples = numpy.logical_and(landmarks[:-1] == 0, landmarks[1:] == 1)
            split_indices = numpy.logical_or(end_landmark_samples,
                                             end_nonlandmark_samples).nonzero()[0] + 1
            v_chains = numpy.split(v, split_indices)
            dvdt_chains = numpy.split(dvdt, split_indices)
            s_chains = numpy.split(s, split_indices)
            # Resample dense parts of the path and keep sparse parts
            sparse_v_list = []
            sparse_dvdt_list = []
            sparse_s_list = []
            is_landmark_chain = landmarks[0]
            for v_chain, dvdt_chain, s_chain in zip(v_chains, dvdt_chains, s_chains):
                # Check whether in landmark chain or not
                if is_landmark_chain:
                    # if landmark (typically already sparse) chain, append to sparse chain as is
                    sparse_v_list.append(v_chain)
                    sparse_dvdt_list.append(dvdt_chain)
                    sparse_s_list.append(s_chain)
                else:
                    # if non landmark chain, interpolate to a sparse 's' resolution and append to 
                    # sparse chain
                    new_s_chain = numpy.arange(s_chain[0], s_chain[-1] + sparse_period / 2.0,
                                               sparse_period)
                    sparse_v_list.append(numpy.interp(new_s_chain, s_chain, v_chain))
                    sparse_dvdt_list.append(numpy.interp(new_s_chain, s_chain, dvdt_chain))
                    sparse_s_list.append(new_s_chain)
                # Alternate to and from dense and sparse chains
                is_landmark_chain = not is_landmark_chain
            # Concatenate sparse chains into numpy arrays
            v = numpy.concatenate(sparse_v_list)
            dvdt = numpy.concatenate(sparse_dvdt_list)
            s = numpy.concatenate(sparse_s_list)
        # Get the Interpolators
        # FIXME: Switch to use more recent scipy.interpolate.UnivariateSpline class
        v_interp = scipy.interpolate.interp1d(s, v, kind=interp_type)
        dvdt_interp = scipy.interpolate.interp1d(s, dvdt, kind=interp_type)
        return v_interp, dvdt_interp, original_s

    def plot_d_dvdt(self, trace, show=True):
        """
        Used in debugging to plot a histogram from a given trace
        
        `trace` -- the trace to generate the histogram from [neo.AnalogSignal]
        `show`  -- whether to call the matplotlib 'show' function (depends on whether there are
                   subsequent plots to compare or not) [bool]
        """
        from matplotlib import pyplot as plt

        # Temporarily switch of resampling to get original positions of v dvdt plot
        orig_resample = self.resample
        self.resample = False
        orig_v, orig_dvdt = self._calculate_v_and_dvdt(trace)
        self.resample = orig_resample
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
            plt.ylabel('dV/dt')
            if show:
                plt.show()


class PhasePlaneHistObjective(PhasePlaneObjective):
    """
    Phase-plane histogram objective function based on the objective in Van Geit 2007 (Neurofitter)
    """

    _FRAC_TO_EXTEND_DEFAULT_BOUNDS = 0.1
    _BIN_TO_SAMPLE_FREQ_RATIO_DEFAULT = 3.0

    def __init__(self, reference_traces, num_bins=(100, 100), v_bounds=None, dvdt_bounds=None, 
                 **kwargs):
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
        """
        super(PhasePlaneHistObjective, self).__init__(reference_traces, **kwargs)
        self.num_bins = numpy.asarray(num_bins, dtype=int)
        self._set_bounds(v_bounds, dvdt_bounds, self.reference_traces)
        # Generate the reference phase plane the simulated data will be compared against
        self.ref_phase_plane_hist = numpy.zeros(self.num_bins)
        for ref_trace in self.reference_traces:
            self.ref_phase_plane_hist += self._generate_phase_plane_hist(ref_trace)
        # Normalise the reference phase plane
        self.ref_phase_plane_hist /= len(reference_traces)

    @property
    def range(self):
        return numpy.array(((self.bounds[0][1] - self.bounds[0][0]),
                            (self.bounds[1][1] - self.bounds[1][0])))

    @property
    def bin_size(self):
        return self.range / self.num_bins

    def fitness(self, recordings):
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
        # Loop through both v and dV/dt bounds and see if any are None and therefore require a
        # default value to be calculated.
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
        v, dv_dt = self._calculate_v_and_dvdt(trace)
        return numpy.histogram2d(v, dv_dt, bins=self.num_bins, range=self.bounds, normed=False)[0]

    def plot_hist(self, trace, range=None, show=True):
        """
        Used in debugging to plot a histogram from a given trace
        
        `trace` -- the trace to generate the histogram from [neo.AnalogSignal]
        `show`  -- whether to call the matplotlib 'show' function (depends on whether there are
                   subsequent plots to compare or not) [bool]
        """
        from matplotlib import pyplot as plt
        hist = self._generate_phase_plane_hist(trace)
        kwargs = {}
        if range is not None:
            kwargs['vmin'], kwargs['vmax'] = range
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if isinstance(show, str):
            import cPickle as pickle
            with open(show, 'w') as f:
                pickle.dump(hist, f)
        else:
            plt.imshow(hist.T, interpolation='nearest', origin='lower', **kwargs)
            plt.xlabel('v')
            plt.ylabel('dV/dt')
            plt.xticks(numpy.linspace(0, self.num_bins[0] - 1, 11.0))
            plt.yticks(numpy.linspace(0, self.num_bins[1] - 1, 11.0))
            ax.set_xticklabels([str(l) for l in numpy.arange(self.bounds[0][0],
                                                             self.bounds[0][1] + self.range[0] / 20.0,
                                                             self.range[0] / 10.0)])
            ax.set_yticklabels([str(l) for l in numpy.arange(self.bounds[1][0],
                                                             self.bounds[1][1] + self.range[1] / 20.0,
                                                             self.range[1] / 10.0)])
            if show:
                plt.show()


class PhasePlanePointToPointObjective(PhasePlaneObjective):
    
    
    def __init__(self, reference_traces, start_threshold, end_threshold, **kwargs):
        """
        Creates a phase plane histogram from the reference traces and compares that with the 
        histograms from the simulated traces
        
        `reference_traces` -- traces (in Neo format) that are to be compared against 
                              [list(neo.AnalogSignal)]
        """
        super(PhasePlaneHistObjective, self).__init__(reference_traces, **kwargs)
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.reference_loops = [self._cut_out_loops(t) for t in self.reference_traces]
        
    def _cut_out_loops(self, trace):
        """
        Cuts outs loops (either spikes or sub-threshold oscillations) from the v-dV/dt trace
        
        `trace`            -- the voltage trace [numpy.array(float)]
        `start_threshold`  -- the threshold above which the loop is considered to have started
                              [tuple[2](float)]
        `end_threshold`    -- the threshold above which the loop is considered to have ended
                              [tuple[2](float)]                              
        """
        v, dvdt = self._calculate_v_and_dvdt(trace)
        # Get the points at which the trace passes the start and end thresholds
        loop_started = numpy.logical_and(v >= self.start_threshold[0], 
                                         dvdt >= self.start_threshold[1])
        loop_ended = numpy.logical_and(v <= self.end_threshold[0], dvdt <= self.end_threshold[1])
        loop_starts = numpy.logical_and(loop_started[:-1], loop_started[1:].invert())
        loop_ends = numpy.logical_and(loop_ended[:-1], loop_ended[1:].invert())
        split_indices = numpy.logical_or(loop_starts, loop_ends).nonzero()[0] + 1
        v_chains = v.split(split_indices)
        dvdt_chains = dvdt.split(split_indices)
        # Check to see if first chain is in loop or not
        first_loop_index = int(not (v[0] >= self.start_threshold[0] and 
                                    dvdt[0] >=self.start_threshold[1]))
        # return only chains that are in loop
        return zip(v_chains, dvdt_chains)[first_loop_index:len(v_chains)+1:2]
    
    def fitness(self, recordings):
        v_dvdt_loops = self._cut_out_loops(recordings)
        

class ConvPhasePlaneHistObjective(PhasePlaneHistObjective):
    """
    Phase-plane histogram objective function based on the objective in Van Geit 2007 (Neurofitter)
    with the exception that the histograms are smoothed by a Gaussian kernel after they are generated
    """

    def __init__(self, reference_traces, num_bins=(100, 100), kernel_width=(5.25, 18.75),
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
        # The creation of the kernel is delayed until it is required
        # (in the _generate_phase_plane_hist method) because it relies on the range of the histogram
        # which is set in the super().__init__ method
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


class CombinedObjective(Objective):

    __metaclass__ = ABCMeta  # Declare this class abstract to avoid accidental construction

    def __init__(self, *objectives):
        """
        A list of objectives that are to be combined
        
        `objectives` -- a list of Objective objects [list(Objective)]
        """
        self.objectives = objectives

    def _iterate_recordings(self, recordings):
        """
        Yields a matching set of requested recordings with their objectives
        
        `recordings` -- the recordings returned from a neurotune.simulation.Simulation object
        """
        for objective in self.objectives:
            # Unzip the objective objects from the keys to pass them to the objective functions
            recordings = dict([(key[1], val)
                               for key, val in recordings.iteritems() if key[0] == objective])
            # Unwrap the dictionary from a single requested recording
            if len(recordings) == 1 and recordings.has_key(None):
                recordings = recordings.values()[0]
            yield objective, recordings

    def get_recording_requests(self):
        # Zip the recording requests keys with objective object in a tuple to guarantee unique
        # keys
        recordings_request = {}
        for objective in self.objectives:
            # Get the recording requests from the sub-objective function
            objective_rec_requests = objective.get_recording_requests()
            # Wrap single recording requests in a dictionary
            if isinstance(objective_rec_requests, RecordingRequest):
                objective_rec_requests = {None:objective_rec_requests}
            # Add the recording request to the collated dictionary
            recordings_request.upate([((objective, key), val)
                                      for key, val in objective_rec_requests.iteritems()])
        return recordings_request


class WeightedSumObjective(CombinedObjective):
    """
    A container class for multiple objectives, to be used with multiple objective optimisation
    algorithms
    """

    def __init__(self, *weighted_objectives):
        """
        `weighted_objectives` -- a list of weight-objective pairs [list((float, Objective))]
        """
        self.weights, self.objectives = zip(*weighted_objectives)

    def fitness(self, recordings):
        """
        Returns a inspyred.ec.emo.Pareto list of the fitness functions in the order the objectives
        were passed to the __init__ method
        """
        weighted_sum = 0.0
        for i, (objective, recordings) in enumerate(self._iterate_recordings(recordings)):
            weighted_sum += self.weight[i] * objective.fitness(recordings)
        return weighted_sum


class MultiObjective(CombinedObjective):
    """
    A container class for multiple objectives, to be used with multiple objective optimisation
    algorithms
    """

    def fitness(self, recordings):
        """
        Returns a inspyred.ec.emo.Pareto list of the fitness functions in the order the objectives
        were passed to the __init__ method
        """
        fitnesses = []
        for objective, recordings in self._iterate_recordings(recordings):
            fitnesses.append(objective.fitness(recordings))
        return inspyred.ec.emo.Pareto(fitnesses)

