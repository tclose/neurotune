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


class PhasePlaneHistObjective(Objective):
    """
    Phase-plane histogram objective function based on the objective in Van Geit 2007 (Neurofitter)
    """

    _FRAC_TO_EXTEND_DEFAULT_BOUNDS = 0.1
    _BIN_TO_SAMPLE_FREQ_RATIO_DEFAULT = 3.0

    def __init__(self, reference_traces, time_start=500.0, time_stop=2000.0, record_variable=None, 
                 exp_conditions=None, num_bins=(100, 100), v_bounds=None, 
                 dvdt_bounds=None, resample=True, resample_type='linear'):
        """
        Creates a phase plane histogram from the reference traces and compares that with the 
        histograms from the simulated traces
        
        `reference_traces` -- traces (in Neo format) that are to be compared against 
                              [list(neo.AnalogSignal)]
        `time_stop`        -- the length of the recording [float]
        `record_variable`  -- the recording site [str]
        `exp_conditions`   -- the required experimental conditions (eg. initial voltage, current 
                              clamps, etc...) [neurotune.controllers.ExperimentalConditions] 
        `num_bins`         -- the number of bins to use for the histogram [tuple[2](int)]
        `v_bounds`         -- the bounds of voltages over which the histogram is generated for. If 
                              'None' then it is calculated from the bounds of the reference traces
                              [tuple[2](float)] 
        `dvdt_bounds`      -- the bounds of rates of change of voltage the histogram is generated 
                              for. If 'None' then it is calculated from the bounds of the reference
                              traces [tuple[2](float)]
        `resample`         -- the periods used to resample the traces on the v-dV/dt plot. Can be:
                              * False: no resampling)
                              * True: resampling to default sizes determinted by bin size and 
                                      _BIN_TO_SAMPLE_FREQ_RATIO_DEFAULT
                              * tuple(float)[2]: a pair of sample periods for v & dV/dt respectively
        `resample_type`    -- the type of interpolation used to resample the traces 
                              (see scipy.interpolate.interp1d for list of options) [str]
        """
        super(PhasePlaneHistObjective, self).__init__(time_start, time_stop)
        if isinstance(reference_traces, str):
            f = neo.io.PickleIO(reference_traces)
            seg = f.read_segment()
            reference_traces = seg.analogsignals
        elif isinstance(reference_traces, neo.AnalogSignal):
            reference_traces = [reference_traces]
        # Save the recording site and number of bins
        self.record_variable = record_variable
        self.exp_conditions = exp_conditions
        self.num_bins = numpy.asarray(num_bins, dtype=int)
        self._set_bounds(v_bounds, dvdt_bounds, reference_traces)
        # Set resampling default
        if resample is True:
            self.resample = self.bin_size / self._BIN_TO_SAMPLE_FREQ_RATIO_DEFAULT
        else:
            self.resample = resample
        self.resample_type = resample_type
        # Generate the reference phase plane the simulated data will be compared against
        self.ref_phase_plane_hist = numpy.zeros(num_bins)
        for ref_trace in reference_traces:
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
        return numpy.sqrt(diff.sum())

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
        
        `trace` -- voltage trace (in Neo format) [list(neo.AnalogSignal)]
        
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
        dvdt = dv/dt
        if self.resample is not False:
            v, dvdt = self._resample_traces(v, dvdt)
        return v, dvdt
    
    def _resample_traces(self, v, dvdt):
        # In order to resample the traces, the length of each v-dV/dt path segment needs to be 
        # calculated 
        resample_norm = numpy.sqrt((self.resample * self.resample).sum())
        rescale = 2.0 * self.resample / resample_norm
        # Get the lengths of the intervals between v-dv/dt samples
        dv = numpy.diff(v)
        d_dvdt = numpy.diff(dvdt)
        interval_lengths = numpy.sqrt((numpy.asarray(dv) / rescale[0]) ** 2 + 
                                      (numpy.asarray(d_dvdt) / rescale[1]) ** 2)
        # Calculate the "positions" of the samples in terms of the fraction of the length
        # of the v-dv/dt path
        s = numpy.concatenate(([0.0], numpy.cumsum(interval_lengths)))
        # Get a regularly spaced array of new positions along the phase-plane path to 
        # interpolate the 
        new_s = numpy.arange(0, s[-1], resample_norm)
        # If using a basic resampling type there is no need to preprocess the v-dV/dt paths to  
        # improve performance, so a basic interpolation can be performed.
        if self.resample_type == 'linear':
            # Interpolate the samples onto an evenly spaced grid of "positions"
            v = numpy.interp(new_s, s, v)
            dvdt = numpy.interp(new_s, s, dvdt)
        # If using a more computationally intensive interpolation technique, the v-dV/dt paths are 
        # pre-processed to decimate the densely sampled sections of the path
        else:
            # Get a list of landmark samples that should be retained in the coarse sampling
            # i.e. samples either side of a 
            course_resample_norm = resample_norm * 10.0
            # Make the samples on both sides of large intervals "landmark" samples
            landmarks = numpy.empty(len(v)+1)
            landmarks[:-2] = interval_lengths > course_resample_norm
            landmarks[landmarks[:-2].nonzero()+1] = True
            landmarks[-2:] = True
            # Break the path up into chains of densely and sparsely sampled sections (i.e. fast and
            # slow parts of the voltage trace)
            end_dense_samples = numpy.logical_and(landmarks[:-1] == 0, landmarks[1:] == 1)
            end_sparse_samples = numpy.logical_and(landmarks[:-1] == 1, landmarks[1:] == 0)
            split_indices = numpy.logical_or(end_dense_samples, end_sparse_samples).nonzero()
            v_chains = numpy.split(v, split_indices)
            dvdt_chains = numpy.split(dvdt, split_indices)
            # Up to here!!
            v = scipy.interpolate.interp1d(s, v, kind=self.resample_type)(new_s)
            dvdt = scipy.interpolate.interp1d(s, dvdt, kind=self.resample_type)(new_s)
        return v, dvdt
        
        
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
        # This is a bit of a hack because the value for self.resample gets set after _set_bounds is
        # called because the default value relies on the bin size, which in turn relies on the bounds
        self.resample = False
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
        plt.imshow(hist.T, interpolation='nearest', origin='lower', **kwargs)
        plt.xlabel('v')
        plt.ylabel('dV/dt')
        plt.xticks(numpy.arange(0, self.num_bins[0], self.num_bins[0] / 10.0))
        plt.yticks(numpy.arange(0, self.num_bins[1], self.num_bins[1] / 10.0))
        ax.set_xticklabels([str(l) for l in numpy.arange(self.bounds[0][0], self.bounds[0][1], 
                                                         self.range[0] / 10.0)])
        ax.set_yticklabels([str(l) for l in numpy.arange(self.bounds[1][0], self.bounds[1][1], 
                                                         self.range[1] / 10.0)])
        if show:
            plt.show()


class ConvPhasePlaneHistObjective(PhasePlaneHistObjective):
    """
    Phase-plane histogram objective function based on the objective in Van Geit 2007 (Neurofitter)
    with the exception that the histograms are smoothed by a Gaussian kernel after they are generated
    """

    def __init__(self, reference_traces, num_bins=(100, 100), kernel_width=(5, 12.5), 
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
        self.num_stdevs =  numpy.asarray(num_stdevs)
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
                               -self.num_stdevs[1]:self.num_stdevs[1]:num_kernel_bins[1] * 1j]
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
        plt.xticks(numpy.linspace(0, self.kernel.shape[0]-1, 11.0))
        plt.yticks(numpy.linspace(0, self.kernel.shape[1]-1, 11.0))
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

