from __future__ import absolute_import
from .__init__ import Objective


class MultiObjective(Objective):
    """
    A container class for multiple objectives, to be used with multiple
    objective optimisation algorithms
    """

    def __init__(self, *objectives):
        """
        A list of objectives that are to be combined

        `objectives` -- a list of Objective objects [list(Objective)]
        """
        self.objectives = list(objectives)

    def append(self, objective):
        self.objectives.append(objective)

    def __getitem__(self, i):
        return self.objectives[i]

    def __setitem__(self, i, val):
        self.objectives[i] = val

    def fitness(self, recordings):
        """
        Returns a inspyred.ec.emo.Pareto list of the fitness functions in the
        order the objectives were passed to the __init__ method
        """
        fitnesses = []
        for objective, objective_recordings in \
                                          self._iterate_recordings(recordings):
            fitnesses.append(objective.fitness(objective_recordings))
        return fitnesses

    def _iterate_recordings(self, recordings):
        """
        Yields a matching set of requested recordings with their objectives

        `recordings` -- the recordings returned from a
                        neurotune.simulation.Simulation object
        """
        for objective in self.objectives:
            # Unzip the objective objects from the keys to pass them to the
            # objective functions
            rec = dict([(key[1], val)
                        for key, val in recordings.iteritems()
                        if key[0] == objective])
            # Unwrap the dictionary from a single requested recording
            if len(rec) == 1 and None in rec:
                rec = rec.values()[0]
            yield objective, rec

    def get_recording_requests(self):
        # Zip the recording requests keys with objective object in a tuple to
        # guarantee unique keys
        recordings_request = {}
        for objective in self.objectives:
            # Get the recording requests from the sub-objective function
            objective_rec_requests = objective.get_recording_requests()
            # Add the recording request to the collated dictionary
            recordings_request.update([((objective, key), val)
                                      for key, val in
                                      objective_rec_requests.iteritems()])
        return recordings_request


class WeightedSumObjective(MultiObjective):
    """
    Combines multiple objectives into a single weighted sum between objectives.
    Useful for optimization algorithms that require a single objective
    """

    def __init__(self, *weighted_objectives):
        """
        `weighted_objectives` -- a list of weight-objective pairs
                                 [list((float, Objective))]
        """
        self.weights, self.objectives = zip(*weighted_objectives)

    def fitness(self, recordings):
        """
        Returns a inspyred.ec.emo.Pareto list of the fitness functions in the
        order the objectives were passed to the __init__ method
        """
        weighted_sum = 0.0
        for i, (objective, recordings) in \
                               enumerate(self._iterate_recordings(recordings)):
            weighted_sum += self.weight[i] * objective.fitness(recordings)
        return weighted_sum
