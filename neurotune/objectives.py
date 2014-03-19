from abc import ABCMeta # Metaclass for abstract base classes
import numpy
import scipy.signal
import inspyred
from .controllers import ExperimentalConditions, RecordingRequest


class _Objective(object):
    
    __metaclass__ = ABCMeta # Declare this class abstract to avoid accidental construction
    
    def fitness(self, simulated_data):
        raise NotImplementedError("Derived Objective class '{}' does not implement fitness method"
                                  .format(self.__class__.__name__))
    
    @property
    def request_recordings(self):
        raise NotImplementedError("Derived Objective class '{}' does not implement request_recordings "
                                  "property".format(self.__class__.__name__))


class PhasePlaneHistObjective(_Objective):
    
    V_RANGE_DEFAULT=(-90, 60) # Default range of voltages in the histogram
    DVDT_RANGE_DEFAULT=(-0.5, 0.5) # Default range of dV/dt values in the histogram
    
    def __init__(self, reference_traces, record_site='soma', record_time=2000.0, num_bins=(10, 10), 
                 v_range=V_RANGE_DEFAULT, dvdt_range=DVDT_RANGE_DEFAULT):
        """
        Creates a phase plane histogram from the reference traces and compares that with the 
        histograms from the simulated traces
        
        `reference_traces` -- traces (in Neo format) that are to be compared against [list(neo.AnalogSignal)]
        `record_site`   -- the recording site [str]
        `num_bins`         -- the number of bins to use for the histogram [tuple[2](int)]
        `v_range`          -- the range of voltages over which the histogram is generated for [tuple[2](float)] 
        `dvdt_range`       -- the range of rates of change of voltage the histogram is generated for [tuple[2](float)]
        """
        # Allow flexibility to provide reference traces as a list or a single trace
        if not isinstance(reference_traces, list):
            reference_traces = [reference_traces]
        # Save the recording site and number of bins
        self.record_site = record_site
        self.record_time = record_time
        self.num_bins = num_bins
        self.range = (v_range, dvdt_range)
        # Generate the reference phase plane the simulated data will be compared against
        self.ref_phase_plane_hist = numpy.zeros(num_bins)
        for ref_trace in reference_traces:
            self.ref_phase_plane_hist += self._generate_phase_plane_hist(self, ref_trace, num_bins)
        # Normalise the reference phase plane
        self.ref_phase_plane_hist /= len(reference_traces)
        
    def request_recordings(self):
        return [RecordingRequest(key=self.__class__.__name__, record_site=self.record_site, 
                                 record_time=self.record_time)]

    def fitness(self, simulated_data):
        trace = simulated_data[self.__class__.__name__]
        phase_plane_hist = self._generate_phase_plane_hist(trace)
        # Get the root-mean-square difference between the reference and simulated histograms
        diff = self.ref_phase_plane_hist - phase_plane_hist
        diff **= 2
        return numpy.sqrt(diff.sum())
        
    def _generate_phase_plane_hist(self, trace):
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
        
        `reference_traces` -- traces (in Neo format) that are to be compared against [list(neo.AnalogSignal)]
        `record_site`   -- the recording site [str]
        `num_bins`         -- the number of bins to use for the histogram [tuple(int)]
        `v_range`          -- the range of voltages over which the histogram is generated for [tuple[2](float)]
        `dvdt_range`       -- the range of rates of change of voltage the histogram is generated for [tuple[2](float)]
        `kernel_stdev`     -- the standard deviation of the Gaussian kernel used to convolve the histogram [tuple[2](float)]
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
        return inspyred.ec.emo.Pareto([o.fitness(simulated_data) for o in self.objectives])
    
    @property
    def request_recordings(self):
        # Combine the required recording sites for each objective into a single set
        recordings_to_request = []
        for objective in self.objectives:
            recordings_to_request.extend(objective.request_recordings())
        return recordings_to_request
