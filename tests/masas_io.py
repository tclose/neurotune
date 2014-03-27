import os.path
import csv
import numpy
import quantities as pq
import neo
import neuron
import neurotune as nt

neuron.load_mechanisms(os.path.join(os.path.dirname(__file__), 'masas_io'))

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
    
#     
# class MasasIOObjective(nt.objective.Objective):
#     
#     def __init__(self):
#         super(MasasIOObjective, self).__init__()
#         with open(os.path.join(os.path.dirname(__file__), 'masas_io', 'dens.csv'), 'rb') as f:
#             reader = csv.reader(f, delimiter=',')
#             x = list(reader)
#         self.reference_densities = numpy.array(x).astype('float')
#     
#     def _derivative(self, recordings):    # derivative dV/dt(approximated to v(t+1)-v(t-1)/2dt)                                 #it should be matched with simulation
#         vmemb = recordings    # recordedV is from simulation
#         start = 500    # start point of recording (ms)
#         end = 2500    # end point of recording (ms)
#         Ndata = float((end - start) * recordings.sampling_rate)
#         start = int(start * recordings.sampling_rate)    # make integer for loop
#         end = int(end * recordings.sampling_rate)    # make integer for loop
#         recdVdt = numpy.zeros(end - start)
#         recdV = numpy.zeros(end - start)
#         for i in range(start, end):    # make dV/dt matrix
#             tpre = vmemb[i + 1]
#             tpost = vmemb[i - 1]
#             recdVdt[i - start] = (tpre - tpost) / 2 * recordings.sampling_rate * 1000    # 1000 means V to mV
#             recdV[i - start] = vmemb[i]
#             # print 'total number of data V is', len(recdV)
#             # print 'total number of data dV/dt is', len(recdVdt)
#         return recdV, recdVdt, Ndata
#     
#     def _makedens(self, recdV, recdVdt):    # maek density matrix based on derivative()
#         ax = 0        # needed for loop
#         ay = 0    # needed for loop
#         minV = -65    # minimum membrane voltage for matrix
#         maxV = -45    # maximum membrane voltage for matrix
#         Vstep = 0.5    # step of voltage
#         mindVdt = -0.25    # minimum voltage derivative for matrix
#         maxdVdt = 0.25    # maximum voltage derivative for matrix
#         dVdtstep = 0.0125    # step of voltage derivative
#         rangeV = numpy.arange(minV, maxV, Vstep)
#         rangedVdt = numpy.arange(mindVdt, maxdVdt, dVdtstep)
#         dens = numpy.zeros((len(rangeV), len(rangedVdt)))
#         for j in rangeV:
#             a = numpy.where((recdV >= j) & (recdV < j + Vstep))    # range of colums
#             for h in rangedVdt:
#                 b = numpy.where((recdVdt[a] >= h) & (recdVdt[a] < h + dVdtstep))    # range of rows
#                 dens[ax, ay] = len(b[0])
#                 ay = ay + 1
#             ax = ax + 1
#             ay = 0
#         return dens, rangeV, rangedVdt
#     
#     def fitness(self, recordings):
#         trace = recordings.values()[0]
#         recdV, recdVdt, NdataA = self._derivative(trace)    # return derivative
#         densA, _, _ = self._makedens(recdV, recdVdt)    # put derivative, return density matrix
#         NdensA = numpy.sum(densA)
#         OutOfRangeA = NdataA - NdensA    # plots which are out of the density matrix
#         error = numpy.fabs(self.reference_densities - densA)
#         return numpy.sum(error) + OutOfRangeA    # value of fitness

genome_centroid = (0.3, 0.85, -60)        
simulation = MasasIOSimulation()
v, _ = simulation.simulate(genome_centroid)
reference_trace = neo.AnalogSignal(v, sampling_rate=1.0 / (neuron.h.dt * pq.ms), units='ms')
objective = nt.objective.PhasePlaneHistObjective(reference_trace, record_time=2000.0)    
#objective = MasasIOObjective()

tuner = nt.Tuner([nt.Parameter('g_l', 'uS', 0.1, 0.5),
                  nt.Parameter('g_ca', 'uS', 0.2, 1.5),
                  nt.Parameter('e_l', 'mV', -70, -50)], 
                 objective, 
                 nt.algorithm.EDAAlgorithm(), 
                 simulation)
pop, ea = tuner.tune(num_candidates=10, max_iterations=100)
