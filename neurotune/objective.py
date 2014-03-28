from abc import ABCMeta  # Metaclass for abstract base classes
import numpy
import scipy
import inspyred
import neo.io
from .simulation import RecordingRequest
from matplotlib import pyplot as plt


class Objective(object):

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

    _FRAC_TO_EXTEND_DEFAULT_BOUNDS = 0.25

    def __init__(self, reference_traces, time_start=500.0, time_stop=2000.0, record_variable=None, 
                 resample=False, exp_conditions=None, num_bins=(50, 50), v_bounds=None, 
                 dvdt_bounds=None):
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
        `v_bounds`         -- the range of voltages over which the histogram is generated for. If 
                              'None' then it is calculated from the range of the reference traces
                              [tuple[2](float)] 
        `dvdt_bounds`      -- the range of rates of change of voltage the histogram is generated 
                              for. If 'None' then it is calculated from the range of the reference
                              traces [tuple[2](float)]
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
        self.resample = resample
        self.exp_conditions = exp_conditions
        self.num_bins = num_bins
        self._set_range(v_bounds, dvdt_bounds, reference_traces)
        # Generate the reference phase plane the simulated data will be compared against
        self.ref_phase_plane_hist = numpy.zeros(num_bins)
        for ref_trace in reference_traces:
            self.ref_phase_plane_hist += self._generate_phase_plane_hist(ref_trace)
        # Normalise the reference phase plane
        self.ref_phase_plane_hist /= len(reference_traces)
        
    def _set_range(self, v_bounds, dvdt_bounds, reference_traces):
        """
        Sets the range of the histogram. If v_bounds or dvdt_bounds is not provided (i.e. is None)
        then the range is taken to be the range between the maximum and minium values of the 
        reference trace extended in both directions by _FRAC_TO_EXTEND_DEFAULT_BOUNDS
        
        `v_bounds`         -- the range of voltages over which the histogram is generated for. If 
                              'None' then it is calculated from the range of the reference traces 
                              [tuple[2](float)]
        `dvdt_bounds`      -- the range of rates of change of voltage the histogram is generated for.
                              If 'None' then it is calculated from the range of the reference traces 
                              [tuple[2](float)]
        `reference_traces` -- traces (in Neo format) that are to be compared against 
                              [list(neo.AnalogSignal)]
        """
        v, _, dvdt = zip(*[self._calculate_v_dv_dvdt(t) for t in reference_traces])
        self.range = []
        for bounds, trace in ((v_bounds, v), (dvdt_bounds, dvdt)):
            if bounds is None:
                # Calculate the range of the reference traces
                trace = numpy.array(trace)
                min_trace = numpy.min(trace)
                max_trace = numpy.max(trace)
                # Extend the range by the fraction in DEFAULT_RANGE_EXTEND
                range_extend = (max_trace - min_trace) * self._FRAC_TO_EXTEND_DEFAULT_BOUNDS
                bounds = (min_trace - range_extend, max_trace + range_extend)
            self.range.append(bounds)

    def fitness(self, recordings):
        phase_plane_hist = self._generate_phase_plane_hist(recordings)
        # Get the root-mean-square difference between the reference and simulated histograms
        plt.figure(0)
        plt.imshow(self.ref_phase_plane_hist, interpolation='nearest')
        plt.figure(1)
        plt.imshow(phase_plane_hist, interpolation='nearest')
        plt.show()
        diff = self.ref_phase_plane_hist - phase_plane_hist
        diff **= 2
        return numpy.sqrt(diff.sum())

    def get_recording_requests(self):
        """
        Gets all recording requests required by the objective function
        """
        return RecordingRequest(record_variable=self.record_variable, record_time=self.time_stop,
                                conditions=self.exp_conditions)

    
    def _calculate_v_dv_dvdt(self, trace):
        """
        Trims the trace to the indices within the time_start and time_stop then calculates the 
        discrete time derivative
        
        `trace` -- voltage trace (in Neo format) [list(neo.AnalogSignal)]
        
        returns trimmed voltage trace, dV and dV/dt in a tuple
        """
        # Calculate dv/dt via difference between trace samples
        start_index = int(round(trace.sampling_period * (self.time_start + float(trace.t_start))))
        stop_index = int(round(trace.sampling_period * (self.time_stop + float(trace.t_start))))
        v = trace[start_index:stop_index]
        dv = numpy.diff(v)
        dt = numpy.diff(v.times)
        v = trace[:-1]
        return v, dv, dv / dt

    def _generate_phase_plane_hist(self, trace):
        """
        Generates the phase plane histogram see Neurofitter paper (Van Geit 2007)
        
        `trace` -- a voltage trace [neo.Anaologsignal]
        
        returns 2D histogram
        """
        v, dv, dv_dt = self._calculate_v_dv_dvdt(trace)
        if self.resample:
            # Get the lengths of the intervals between v-dv/dt samples
            d_dv_dt = numpy.diff(dv_dt)
            interval_lengths = numpy.sqrt(dv[:-1] ** 2 + d_dv_dt ** 2)
            # Calculate the "positions" of the samples in terms of the fraction of the length
            # of the v-dv/dt path
            s = numpy.concatenate(([0.0], numpy.ufunc.accumulate(interval_lengths)))
            # Interpolate the samples onto an evenly spaced grid of "positions"
            new_s = numpy.arange(0, s[-1], self.resample)
            v = scipy.interp(new_s, s, v)
            dv_dt = scipy.interp(new_s, s, dv_dt)
        return numpy.histogram2d(v, dv_dt, bins=self.num_bins, range=self.range, normed=True)[0]


class ConvPhasePlaneHistObjective(PhasePlaneHistObjective):

    def __init__(self, reference_traces, num_bins=(100, 100), kernel_stdev=(5, 5), 
                 kernel_width=(3.5, 3.5), **kwargs):
        """
        Creates a phase plane histogram convolved with a Gaussian kernel from the reference traces 
        and compares that with a similarly convolved histogram of the simulated traces
        
        `reference_traces` -- traces (in Neo format) that are to be compared against 
                              [list(neo.AnalogSignal)]
        `num_bins`         -- the number of bins to use for the histogram [tuple(int)]
        `kernel_stdev`     -- the standard deviation of the Gaussian kernel used to convolve the 
                              histogram [tuple[2](float)]
        `kernel_width`     -- the number of standard deviations the Gaussian kernel extends over
        """
        # Call the parent class __init__ method
        super(ConvPhasePlaneHistObjective, self).__init__(reference_traces, num_bins=num_bins, 
                                                          **kwargs)
        # Pre-calculate the Gaussian kernel
        kernel_extent = numpy.asarray(kernel_width) / (2 * numpy.asarray(kernel_stdev) ** 2)
        # Get the range of x-values for the y = 1/(stdev * sqrt(2*pi)) * exp(-x^2/(2*stdev^2))
        kernel_grid = numpy.ogrid[-kernel_extent[0]:kernel_extent[0]:kernel_width[0] * 1j,
                                   - kernel_extent[1]:kernel_extent[1]:kernel_width[1] * 1j]
        # Calculate the Gaussian kernel
        self.kernel = numpy.exp(-kernel_grid[0] ** 2) * numpy.exp(-kernel_grid[1] ** 2)
        self.kernel /= 2 * numpy.pi * kernel_stdev[0] * kernel_stdev[1]
        # Convolve the Gaussian kernel with the reference phase plane histogram
        self.ref_phase_plane_hist = scipy.signal.convolve2d(self.ref_phase_plane_hist, self.kernel)

    def _generate_phase_plane_hist(self, trace):
        """
        Extends the vanilla phase plane histogram to allow it to be convolved with a Gaussian kernel
        
        `trace` -- a voltage trace [neo.Anaologsignal]
        """
        # Get the unconvolved histogram
        unconv_hist = super(ConvPhasePlaneHistObjective, self)._generate_phase_plane_hist(trace)
        # Convolve the histogram with the precalculated Gaussian kernel
        return scipy.signal.convolve2d(unconv_hist, self.kernel)


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

