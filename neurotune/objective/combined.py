from __future__ import absolute_import
from abc import ABCMeta  # Metaclass for abstract base classes
import inspyred
from .__init__ import Objective
from ..simulation.__init__ import RecordingRequest


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

