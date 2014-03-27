import os.path
import numpy
import quantities as pq
import neo
import neuron
import neurotune as nt

neuron.load_mechanisms(os.path.join(os.path.dirname(__file__), 'masas_io_nmodl'))

class MasasIOSimulation(nt.simulation.SimpleCustomSimulation):
    
    class SingleIO(object):
        def __init__(self, g_l, g_ca, e_l):
            soma = neuron.h.Section()
            soma.diam=25
            soma.L=25
            soma.nseg = 1
            soma.insert('leak')
            soma.insert('stoca')
    ##        soma.insert('iona')
    ##        soma.insert('iokdr')
            for seg in soma:
                seg.gbar_leak = g_l
                seg.el_leak = e_l
                seg.gbar_stoca = g_ca
    ##            seg.gbar_iona = g_na
    ##            seg.gbar_iokdr = g_k
            self.soma = soma #put soma on the class
            vec1, vec2 = self.record()
            self.runsim()
            self.recordedt, self.recordedV = self.getRecordedVm(vec1,vec2)
    
        def record(self): #record t and Vm from hoc to vector
            vec = {}
            for var in 'v_soma', 'i_stoca','t':
                vec[var] = neuron.h.Vector()
            vec['v_soma'].record(self.soma(0.5)._ref_v) #record Vm of soma
            vec['t'].record(neuron.h._ref_t) #record time
            return vec['v_soma'], vec['t']
            
        def getRecordedVm(self,vec1,vec2) : # traslate hoc data to array
            rec = {}
            rec['t'] = numpy.array(vec2)
            rec['vm'] = numpy.array(vec1)
            tsize = len(vec2)
        ##    print 'Vm recorded'
            return rec['t'], rec['vm']
    
        def runsim(self):    # run simulation in neuron simlator
            #h.v_init = -55
            #h.tstop = 3000.0
            #h.init()
            neuron.h.finitialize(-55)
            neuron.run(3000)
            
    def simulate(self, candidate):
        io_cell = self.SingleIO(*candidate)
        return io_cell.recordedV, io_cell.recordedt
    
    
class MasasIOObjective(

genome_centroid = (0.3, 0.85, -60)

simulation = MasasIOSimulation()
v, _ = simulation.simulate(genome_centroid)
reference_trace = neo.AnalogSignal(v, sampling_rate=1.0 / (neuron.h.dt * pq.ms), units='ms')

tuner = nt.Tuner([nt.Parameter('g_l', 'uS', 0.1, 0.5),
                  nt.Parameter('g_ca', 'uS', 0.2, 1.5),
                  nt.Parameter('e_l', 'mV', -70, -50)], 
                 nt.objective.PhasePlaneHistObjective(reference_trace, record_time=2000.0), 
                 nt.algorithm.EDAAlgorithm(), 
                 simulation)
pop, ea = tuner.tune(num_candidates=10, max_iterations=100)