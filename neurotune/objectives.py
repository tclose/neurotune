from abc import ABCMeta # Metaclass for abstract base classes
import numpy
import scipy.signal
import inspyred
from .controllers import RecordingRequest


class _Objective(object):
    
    __metaclass__ = ABCMeta # Declare this class abstract to avoid accidental construction
    
    def fitness(self, simulated_data):
        """
        Evaluates the fitness function given the simulated data
        
        `simulated_data` -- a dictionary containing the simulated data to be assess, with the keys 
                            corresponding to the keys of the recording request dictionary returned 
                            by 'get_recording requests'
        """
        raise NotImplementedError("Derived Objective class '{}' does not implement fitness method"
                                  .format(self.__class__.__name__))
    
    def get_recording_requests(self):
        """
        Returns a dictionary of neurotune.controllers.RecordingRequest objects with unique keys
        representing the recordings that are required from the simulation controller
        """
        raise NotImplementedError("Derived Objective class '{}' does not implement "
                                  "get_recording_requests property".format(self.__class__.__name__))


class PhasePlaneHistObjective(_Objective):
    
    V_RANGE_DEFAULT=(-90, 60) # Default range of voltages in the histogram
    DVDT_RANGE_DEFAULT=(-0.5, 0.5) # Default range of dV/dt values in the histogram
    RECORDING_KEY='volt_trace'
    
    def __init__(self, reference_traces, record_time=2000.0, record_site=None, exp_conditions=None,
                 num_bins=(10, 10), v_range=V_RANGE_DEFAULT, dvdt_range=DVDT_RANGE_DEFAULT):
        """
        Creates a phase plane histogram from the reference traces and compares that with the 
        histograms from the simulated traces
        
        `reference_traces` -- traces (in Neo format) that are to be compared against 
                              [list(neo.AnalogSignal)]
        `record_time`      -- the length of the recording [float]
        `record_site`      -- the recording site [str]
        `exp_conditions`   -- the required experimental conditions (eg. initial voltage, current 
                              clamps, etc...) [neurotune.controllers.ExperimentalConditions] 
        `num_bins`         -- the number of bins to use for the histogram [tuple[2](int)]
        `v_range`          -- the range of voltages over which the histogram is generated for 
                              [tuple[2](float)] 
        `dvdt_range`       -- the range of rates of change of voltage the histogram is generated 
                              for [tuple[2](float)]
        """
        # Allow flexibility to provide reference traces as a list or a single trace
        if not isinstance(reference_traces, list):
            reference_traces = [reference_traces]
        # Save the recording site and number of bins
        self.record_site = record_site
        self.record_time = record_time
        self.exp_conditions = exp_conditions
        self.num_bins = num_bins
        self.range = (v_range, dvdt_range)
        # Generate the reference phase plane the simulated data will be compared against
        self.ref_phase_plane_hist = numpy.zeros(num_bins)
        for ref_trace in reference_traces:
            self.ref_phase_plane_hist += self._generate_phase_plane_hist(self, ref_trace, num_bins)
        # Normalise the reference phase plane
        self.ref_phase_plane_hist /= len(reference_traces)
        
    def get_recording_requests(self):
        return {self.RECORDING_KEY: RecordingRequest(record_site=self.record_site, 
                                                     record_time=self.record_time, 
                                                     conditions=self.exp_conditions)}

    def fitness(self, simulated_data):
        trace = simulated_data[self.RECORDING_KEY]
        phase_plane_hist = self._generate_phase_plane_hist(trace)
        # Get the root-mean-square difference between the reference and simulated histograms
        diff = self.ref_phase_plane_hist - phase_plane_hist
        diff **= 2
        return numpy.sqrt(diff.sum())
        
    def _generate_phase_plane_hist(self, trace):
        """
        Generates the phase plane histogram see Neurofitter paper (Van Geit 2007)
        
        `trace` -- a voltage trace [neo.Anaologsignal]
        """
        # Calculate dv/dt via difference between trace samples
        dv=numpy.diff(trace)
        dt=numpy.diff(trace.times)
        return numpy.histogram2d(trace[:-1], dv/dt, bins=self.num_bins, range=self.range, 
                                 normed=False, weights=None)
        
        
class ConvPhasePlaneHistObjective(PhasePlaneHistObjective):
    
    def __init__(self, reference_traces, record_site, num_bins=(100, 100), 
                 v_range=PhasePlaneHistObjective.V_RANGE_DEFAULT, 
                 dvdt_range=PhasePlaneHistObjective.DVDT_RANGE_DEFAULT, 
                 kernel_stdev=(5, 5), kernel_width=(3.5, 3.5)):
        """
        Creates a phase plane histogram convolved with a Gaussian kernel from the reference traces 
        and compares that with a similarly convolved histogram of the simulated traces
        
        `reference_traces` -- traces (in Neo format) that are to be compared against 
                              [list(neo.AnalogSignal)]
        `record_site`      -- the recording site [str]
        `num_bins`         -- the number of bins to use for the histogram [tuple(int)]
        `v_range`          -- the range of voltages over which the histogram is generated for 
                              [tuple[2](float)]
        `dvdt_range`       -- the range of rates of change of voltage the histogram is generated 
                              for [tuple[2](float)]
        `kernel_stdev`     -- the standard deviation of the Gaussian kernel used to convolve the 
                              histogram [tuple[2](float)]
        `kernel_width`     -- the number of standard deviations the Gaussian kernel extends over
        """
        # Call the parent class __init__ method
        super(ConvPhasePlaneHistObjective, self).__init__(reference_traces=reference_traces, 
                                                          record_site=record_site, 
                                                          num_bins=num_bins, v_range=v_range, 
                                                          dvdt_range=dvdt_range)
        # Pre-calculate the Gaussian kernel
        kernel_extent = numpy.asarray(kernel_width) / (2 * numpy.asarray(kernel_stdev) ** 2)
        # Get the range of x-values for the y = 1/(stdev * sqrt(2*pi)) * exp(-x^2/(2*stdev^2))
        kernel_grid  = numpy.ogrid[-kernel_extent[0]:kernel_extent[0]:kernel_width[0] * 1j,
                                   -kernel_extent[1]:kernel_extent[1]:kernel_width[1] * 1j]
        # Calculate the Gaussian kernel
        self.kernel = numpy.exp(-kernel_grid[0]**2) * numpy.exp(-kernel_grid[1]**2) 
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


class MultiObjective(_Objective):
    """
    A container class for multiple objectives, to be used with multiple objective optimisation
    algorithms
    """
    
    def __init__(self, *objectives):
        self.objectives = objectives
    
    def fitness(self, simulated_data):
        """
        Returns a inspyred.ec.emo.Pareto list of the fitness functions in the order the objectives
        were passed to the __init__ method
        """
        fitnesses = []
        for o in self.objectives:
            # Unzip the objective objects from the keys and pass them to the objective functions
            # to determine their fitness
            objective_data = dict([(key[1], val) 
                                   for key, val in simulated_data.iteritems() if key[0] == o])
            fitnesses.append(o.fitness(objective_data))
        return inspyred.ec.emo.Pareto(fitnesses)
    
    def get_recording_requests(self):
        # Zip the recording requests keys with objective object in a tuple to guarantee unique 
        # keys
        recordings_request = {}
        for o in self.objectives:
            recordings_request.upate([((o, key), val) 
                                      for key, val in o.get_recording_requests().iteritems()])
        return recordings_request
