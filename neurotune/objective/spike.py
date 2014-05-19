from __future__ import absolute_import
import numpy
from . import Objective
from ..simulation import RecordingRequest


class SpikeFrequencyObjective(Objective):
    """
    A simple objective based on the squared difference between the spike
    frequencies
    """

    def __init__(self, frequency, time_start=0.0, time_stop=2000.0):
        """
        `frequency`  -- the desired spike frequency [quantities.Quantity]
        `time_start` -- the time from which to start calculating the frequency
        `time_stop`  -- the length of time to run the simulation
        """
        super(SpikeFrequencyObjective, self).__init__(time_stop,
                                                      record_sites=['spikes'])
        self.time_start = time_start
        self.frequency = frequency

    def fitness(self, signal):
        signal_frequency = (len(signal[signal >= self.time_start &
                                       signal <= self.time_stop])
                            / self.time_stop)
        return (self.frequency - signal_frequency) ** 2


class SpikeTimesObjective(Objective):
    """
    A simple objective based on the squared difference between the spike
    frequencies
    """

    def __init__(self, spikes, time_stop=2000.0):
        """
        `spikes`    -- the reference spike train [neo.SpikeTrain]
        `time_stop` -- the length of time to run the simulation
        """
        super(SpikeTimesObjective, self).__init__(time_stop,
                                                  record_sites=['spikes'])
        self.reference_spikes = spikes

    def fitness(self, signal):
        """
        Calculates the sum squared difference between each spike in the
        signal and the closest spike in the reference spike train, plus the
        vice-versa case

        `signal` -- the recorded signal
        """
        spikes = signal[signal < self.time_stop].spikes
        fitness = 0.0
        for spike in spikes:
            fitness += numpy.square(self.reference_spikes - spike).min()
        for ref_spike in self.reference_spikes:
            fitness += numpy.square(spikes - ref_spike).min()
        return fitness
