from __future__ import absolute_import
import numpy
import quantities as pq
import neo.core
from ..analysis import AnalysedSignal
from . import Objective
from ..simulation import RecordingRequest, ExperimentalConditions


class SpikeFrequencyObjective(Objective):
    """
    A simple objective based on the squared difference between the spike
    frequencies
    """

    def __init__(self, frequency, time_start=500.0 * pq.ms,
                 time_stop=2000.0 * pq.ms):
        """
        `frequency`  -- the desired spike frequency [quantities.Quantity]
        `time_start` -- the time from which to start calculating the frequency
        `time_stop`  -- the length of time to run the simulation
        """
        super(SpikeFrequencyObjective, self).__init__(time_start, time_stop)
        if isinstance(frequency, neo.core.AnalogSignal):
            self.frequency = AnalysedSignal(frequency).spike_frequency()
        else:
            self.frequency = pq.Quantity(frequency, units='Hz')

    def fitness(self, analysis):
        """
        Calculates the sum squared difference between the reference freqency
        and the spike frequency of the recorded trace

        `analysis` -- The analysis object containing all recordings and
                      analysis of them [analysis.Analysis]
        """
        signal = analysis.get_signal()
        frequency = signal.spike_frequency()
        return float((self.frequency - frequency) ** 2)


class SpikeAmplitudeObjective(Objective):

    def __init__(self, amplitude, time_start=500.0 * pq.ms,
                 time_stop=2000.0 * pq.ms):
        super(SpikeAmplitudeObjective, self).__init__(time_start, time_stop)
        self.amplitude = AnalysedSignal(amplitude).spike_amplitude()

    def fitness(self, analysis):
        signal = analysis.get_signal()
        amplitude = signal.spike_amplitude()
        return float((self.amplitude - amplitude) ** 2)        
#        pass

class SpikeTimesObjective(Objective):
    """
    The sum of squared time differences between all simulated spikes and the
    nearest spike in the reference set and vice versa.
    """

    def __init__(self, reference, time_start=500.0 * pq.ms,
                 time_stop=2000.0 * pq.ms, time_buffer=250 * pq.ms):
        """
        `reference`  -- reference signal or spike train
                        [neo.AnalogSignal or neo.SpikeTrain]
        `time_start` -- time from which to start including spikes [float]
        `time_stop`  -- length of time to run the simulation [float]
        `buffer`     -- time buffer either side of the "inner window"
                        in which spikes in the inner window will be compared
                        with but which won't be compared with the inner spikes
        """
        super(SpikeTimesObjective, self).__init__(time_start, time_stop)
        if time_stop - time_start - time_buffer * 2 <= 0:
            raise Exception("Buffer time ({}) exceeds half of spike train "
                            "time ({}) and therefore the inner window is "
                            "empty".format(buffer, (time_stop - time_start)))
        if isinstance(reference, neo.core.SpikeTrain):
            self.ref_spikes = reference
        elif isinstance(reference, neo.core.AnalogSignal):
            self.ref_spikes = AnalysedSignal(reference).spikes()
        else:
            raise Exception("Spikes must be a neo.core.SpikeTrain object not "
                            "{}".format(type(reference)))
        self.time_buffer = time_buffer
        self.ref_inner = reference[numpy.where(
                              (self.ref_spikes >= (time_start + time_buffer)) &
                              (self.ref_spikes <= (time_stop - time_buffer)))]
        if not len(self.ref_inner):
            raise Exception("Inner window does not contain any spikes")

    def fitness(self, analysis):
        """
        Calculates the sum squared difference between each spike in the
        signal and the closest spike in the reference spike train, plus the
        vice-versa case

        `analysis` -- The analysis object containing all recordings and
                      analysis of them [analysis.Analysis]
        """
        spikes = analysis.get_signal().spikes()
        inner = spikes[numpy.where(
                             (spikes >= (self.time_start + self.time_buffer)) &
                             (spikes <= (self.time_stop - self.time_buffer)))]
        # If no spikes were generated create a dummy spike that is guaranteed
        # to be further away from a reference spike than any within the time
        # window
        if len(spikes) == 0:
            spike_t = self.time_stop + self.time_start
            spikes = neo.SpikeTrain([spike_t], spike_t, units=spike_t.units)
        fitness = 0.0
        for spike in inner:
            fitness += float(numpy.square(self.ref_spikes - spike).min())
        for ref_spike in self.ref_inner:
            fitness += float(numpy.square(spikes - ref_spike).min())
        return fitness


class MinCurrentToSpikeObjective(Objective):

    def __init__(self, reference=None, time_start=500 * pq.ms,  # @UnusedVariable @IgnorePep8
                 time_stop=2500.0 * pq.ms, num_steps=20,
                 max_current=20 * pq.pA):
        super(MinCurrentToSpikeObjective, self).__init__(time_start, time_stop)
        self.max_current = max_current
        self.num_steps = num_steps

    def get_recording_requests(self):
        """
        Returns a RecordingRequest object or a dictionary of RecordingRequest
        objects with unique keys representing the recordings that are required
        from the simulation controller
        """
        # Generate a linear ramp in current values from 0 to self.max_current
        # starting from time_start + wait_period until time_stop and inject it
        # into the soma
        steps = numpy.linspace(0.0, float(pq.Quantity(self.max_current, 'nA')),
                               self.num_steps)
        # Alternate between step and a zero current block
        steps = numpy.vstack((steps, numpy.zeros(self.num_steps))).T.ravel()
        # Prepend 0 nA current to steps
        pulse_length = ((self.time_stop - self.time_start) /
                        (self.num_steps * 2.0))
        initial_zeros = numpy.zeros(int(round(self.time_start / pulse_length)))
        steps = numpy.hstack((initial_zeros, steps))
        current = neo.AnalogSignal(steps, sampling_period=pulse_length,
                                   units='nA')
        conds = ExperimentalConditions(injected_currents={'soma': current})
        return {None: RecordingRequest(time_start=self.time_start,
                                       time_stop=self.time_stop,
                                       conditions=conds,
                                       record_variable=None)}

    def fitness(self, analysis):
        spikes = analysis.get_signal().spikes(threshold='v', start=-20.0,
                                              stop=-20.0)
        if len(spikes):
            fitness = ((spikes[0] - self.time_start) /
                       (self.time_stop - self.time_start))
            if fitness < 0.0:
                fitness = 0.0
        else:
            fitness = 1.0
        return fitness
