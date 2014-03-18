from abc import ABCMeta # Metaclass for abstract base classes
import numpy
import inspyred


class _Objective(object):
    
    __metaclass__ = ABCMeta # Declare this class abstract to avoid accidental construction
    
    def fitness(self, simulated_data):
        raise NotImplementedError("Derived class does not implement fitness function")


class PhasePlaneObjective(_Objective):
    
    def __init__(self, reference_traces, recording_site, num_bins=(10, 10), 
                 v_range=(-90, 60), dvdt_range=(-0.5, 0.5)):
        """
        Creates a phase plane histogram from the reference traces and compares that with the 
        simulated traces
        
        `reference_traces` -- traces (in Neo format) that are to be compared against [list(neo.AnalogSignal)]
        `recording_site`   -- the recording site [str]
        `num_bins`         -- the number of bins to use for the histogram [tuple(int)] 
        """
        # Allow flexibility to provide reference traces as a list or a single trace
        if not isinstance(reference_traces, list):
            reference_traces = [reference_traces]
        # Save the recording site and number of bins
        self.recording_sites = [recording_site]
        self.num_bins = num_bins
        self.range = (v_range, dvdt_range)
        # Generate the reference phase plane the simulated data will be compared against
        self.ref_phase_plane_hist = numpy.zeros(num_bins)
        for ref_trace in reference_traces:
            self.ref_phase_plane_hist += self._generate_phase_plane_hist(self, ref_trace, num_bins)
        # Normalise the reference phase plane
        self.ref_phase_plane_hist /= len(reference_traces)

    def fitness(self, simulated_data):
        trace = simulated_data[self.recording_sites[0]]
        phase_plane_hist = self._generate_phase_plane_hist(trace)
        diff = self.ref_phase_plane_hist - phase_plane_hist
        diff **= 2
        return numpy.sqrt(diff.sum())
        
    def _generate_phase_plane_hist(self, trace):
        dv=numpy.diff(trace)
        dt=numpy.diff(trace.times)
        return numpy.histogram2d(trace[:-1], dv/dt, bins=self.num_bins, range=self.range,
                                 normed=False, weights=None)


class MultiObjective(_Objective):
    """
    A container class for multiple objectives, to be used with multiple objective optimisation
    algorithms
    """
    
    def __init__(self, *objectives):
        self.objectives = objectives
    
    def fitness(self, simulated_data):
        return inspyred.ec.emo.Pareto([o.fitness(simulated_data) for o in self.objectives])
    
    @property
    def recording_sites(self):
        req_traces = set()
        for objective in self.objectives:
            for rt in objective.recording_sites:
                req_traces.add(rt)
        return req_traces
