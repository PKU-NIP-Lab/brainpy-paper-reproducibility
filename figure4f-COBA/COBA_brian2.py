from brian2 import *

defaultclock.dt = 0.1 * ms
# clear_cache('cython')
# set_device('cpp_standalone', directory='brian2_COBA')



def run_(scale=1., res_dict=None, tstop=1000.):
  start_scope()
  device.reinit()
  device.activate()

  # prefs.codegen.target = "cython"

  taum = 20 * ms
  taue = 5 * ms
  taui = 10 * ms
  Vt = -50 * mV
  Vr = -60 * mV
  El = -60 * mV
  Erev_exc = 0. * mV
  Erev_inh = -80. * mV
  I = 20. * mvolt
  num_exc = int(3200 * scale)
  num_inh = int(800 * scale)

  eqs = '''
  dv/dt  = (ge*(Erev_exc-v)+gi*(Erev_inh-v)-(v-El) + I)*(1./taum) : volt (unless refractory)
  dge/dt = -ge/taue : 1 
  dgi/dt = -gi/taui : 1 
  '''
  P = NeuronGroup(num_exc + num_inh,
                  eqs,
                  threshold='v>Vt',
                  reset='v = Vr',
                  refractory=5 * ms,
                  method='euler')

  we = 0.6 / scale  # excitatory synaptic weight (voltage)
  wi = 6.7 / scale  # inhibitory synaptic weight
  Ce = Synapses(P[:3200], P, on_pre='ge += we')
  Ci = Synapses(P[3200:], P, on_pre='gi += wi')

  P.v = (np.random.randn(num_exc + num_inh) * 5. - 55.) * mvolt
  Ce.connect(p=0.02)
  Ci.connect(p=0.02)

  t1 = time.time()
  run(tstop * ms, report='text')
  t2 = time.time()
  print('Done in', t2 - t1)

  if res_dict is not None:
    res_dict['brain2'].append({'num_neuron': num_exc + num_inh,
                               'sim_len': tstop,
                               'num_thread': 1,
                               'sim_time': t2 - t1,
                               'dt': 0.1})


run_(scale=10, tstop=5000.)


if __name__ == '__main__':
  import json

  speed_res = {'brain2': []}
  for scale in [1, 1, 2, 4, 6, 8, 10]:
    for stim in [5. * 1e3]:
      run_(scale=scale, res_dict=speed_res, tstop=stim)

  with open('speed_results/brian2-2.json', 'w') as f:
    json.dump(speed_res, f, indent=2)
