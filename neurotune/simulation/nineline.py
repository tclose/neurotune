from __future__ import absolute_import
import quantities as pq
import neo.core
from nineline.cells import DummyNinemlModel
from nineml.extensions.biophysical_cells import ComponentClass as BiophysNineml
from nineline.cells.neuron import NineCellMetaClass, \
                                  simulation_controller as nineline_controller
from . import Simulation


class NineLineSimulation(Simulation):
    "A simulation class for 9ml descriptions"

    supported_conditions = ['injected_currents', 'voltage_clamps']

    def __init__(self, celltype, model=None, build_mode='lazy'):
        """
        `cell_9ml`    -- A 9ml file [str]
        """
        # Generate the NineLine class from the nineml file and initialise a
        # single cell from it
        if (isinstance(celltype, BiophysNineml) or isinstance(celltype, str) or
            isinstance(celltype, DummyNinemlModel)):
            self.celltype = NineCellMetaClass(celltype, build_mode=build_mode)
        else:
            self.celltype = celltype
        self.default_seg = self.celltype(model=model).source_section.name
        self._model = model
        self.genome_keys = []
        self.log_scales = []

    def __reduce__(self):
        return self.__class__, (self.celltype.nineml_model,
                                self._model, 'lazy')

    def set_tune_parameters(self, tune_parameters):
        super(NineLineSimulation, self).set_tune_parameters(tune_parameters)
        self.genome_keys = []
        self.log_scales = []
        for param in tune_parameters:
            if '.' in param.name:
                key = param.name
            else:
                key = self.default_seg + '.' + param.name
            self.genome_keys.append(key)
            self.log_scales.append(param.log_scale)

    def prepare_simulations(self):
        """
        Prepare all simulations (eg. create cells and set recorders if
        possible)
        """
        # Parse all recording sites into a tuple containing the variable name,
        # segment name and component names
        for setup in self._simulation_setups:
            for i, rec in enumerate(setup.record_variables):
                if rec is None:
                    # Records the voltage in the default segment by default
                    var = 'v'
                    segname = self.default_seg
                    component = None
                else:
                    parts = rec.split('.')
                    if len(parts) == 1:
                        var = parts[0]
                        segname = self.default_seg
                        component = None
                    elif len(parts) == 2:
                        segname, var = parts
                        component = None
                    else:
                        segname, component, var = parts
                setup.record_variables[i] = (var, segname, component)
        # Check to see if there are multiple setups, because if there aren't
        # the cell can be initialised (they can't in general if there are
        # multiple as there is only ever one instance of NEURON running)
        if len(self._simulation_setups) == 1:
            self._prepare(self._simulation_setups[0])

    def run(self, candidate, setup):
        """
        Run a simulation given a requested experimental setup required to
        assess the candidate

        `candidate` -- a list of parameters [list(float)]
        `setup`     -- a simulation setup [Setup]

        returns neo.Segment containing the measured analog signals

        """
        # If there aren't multiple simulation setups the same setup can be
        # reused with just the recorders being reset
        if len(self._simulation_setups) != 1:
            self._prepare(setup)
        else:
            nineline_controller.reset()
        if candidate is not None:  # Used to generate reference data
            self._set_candidate_params(candidate)
        # Convert requested record time to ms
        record_time = float(pq.Quantity(setup.record_time, units='ms'))
        # Run simulation
        nineline_controller.run(record_time)
        # Return neo Segment object with all recordings
        seg = neo.core.Segment()
        recordings = self.cell.get_recording(*zip(*setup.record_variables))
        seg.analogsignals.extend(recordings)
        return seg

    def _prepare(self, setup):
        """
        Initialises cell and sets recording sites. Record sites are delimited
        by '.'s into segment names, component names and variable names.
        Sitenames without '.'s are interpreted as properties of the default
        segment and site-names with only one '.' are interpreted as (segment
        name - property) pairs. Therefore in order to record from component
        states you must also provide the segment name to disambiguate it from
        the segment name - property case.

        `setup` -- A set of simulation setup instructions [Simulation.Setup]
        """
        # Initialise cell
        self.cell = self.celltype(model=self._model)
        for rec in setup.record_variables:
            self.cell.record(*rec)
        if 'injected_currents' in setup.conditions:
            for loc, current in setup.conditions['injected_currents'].items():
                getattr(self.cell, loc).inject_current(current)
        if 'voltage_clamps' in setup.conditions:
            for loc, voltages in setup.conditions['voltage_clamps'].items():
                getattr(self.cell, loc).voltage_clamp(voltages)
        if 'synaptic_spikes' in setup.conditions:
            for loc, syn, spkes in setup.conditions['synaptic_spikes'].items():
                getattr(self.cell, loc).synaptic_stimulation(spkes, syn)

    def _set_candidate_params(self, candidate):
        """
        Set the parameters of the candidate

        `candidate` -- a list of parameters [list(float)]
        """
        assert len(candidate) == len(self.genome_keys), \
                                 "length of candidate and genome keys do " \
                                 "not match"
        for key, val, log_scale in zip(self.genome_keys, candidate,
                                       self.log_scales):
            if log_scale:
                val = 10 ** val
            setattr(self.cell, key, val)
