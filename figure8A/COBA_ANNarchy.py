# -*- coding: utf-8 -*-

from ANNarchy import *
import sys
import time

# ANNarchy-4.6.10

# num_thread = int(sys.argv[1])
dt = 0.1
# setup(dt=dt, num_threads=1)
setup(dt=dt)


# setup(dt=dt, paradigm="cuda")


def run(scale, duration, res_dict=None, ):
  num_neu = int(4000 * scale)

  NI = int(num_neu / 5)
  NE = num_neu - NI

  clear()

  COBA = Neuron(
    parameters="""
            El = -60.0  : population
            Vr = -60.0  : population
            Erev_exc = 0.0  : population
            Erev_inh = -80.0  : population
            Vt = -50.0   : population
            tau = 20.0   : population
            tau_exc = 5.0   : population
            tau_inh = 10.0  : population
            I = 20.0 : population
        """,
    equations="""
            tau * dv/dt = (El - v) + g_exc * (Erev_exc - v) + g_inh * (Erev_inh - v ) + I
            tau_exc * dg_exc/dt = - g_exc 
            tau_inh * dg_inh/dt = - g_inh 
        """,
    spike="""
            v > Vt
        """,
    reset="""
            v = Vr
        """,
    refractory=5.0
  )

  # ###########################################
  # Population
  # ###########################################
  P = Population(geometry=NE + NI, neuron=COBA)
  Pe = P[:NE]
  Pi = P[NE:]
  P.v = Normal(-55.0, 5.0)

  # ###########################################
  # Projections
  # ###########################################
  we = 6. / scale  # excitatory synaptic weight (voltage)
  wi = 67. / scale  # inhibitory synaptic weight

  Ce = Projection(pre=Pe, post=P, target='exc')
  Ci = Projection(pre=Pi, post=P, target='inh')
  Ce.connect_fixed_probability(weights=we, probability=0.02)
  Ci.connect_fixed_probability(weights=wi, probability=0.02)

  t0 = time.time()
  compile()
  t1 = time.time()
  simulate(duration)
  t2 = time.time()
  print(f'Network size {num_neu}, ANNarchy compilation + simulation used time {t2 - t0} s.')
  print(f'Network size {num_neu}, ANNarchy simulation used time {time.time() - t1} s.')
  if res_dict is not None:
    res_dict['annarchy'].append({'num_neuron': NE + NI,
                                 'sim_len': duration,
                                 'sim_time': t2 - t0,
                                 'sim_time_without_compile': t2 - t1,
                                 'dt': 0.1})


if __name__ == '__main__':
  import json

  speed_res = {f'annarchy': []}
  for scale in [1, 2, 4, 6, 8, 10]:
    for stim in [5. * 1e3]:
      run(scale=scale, res_dict=speed_res, duration=stim)

  with open(f'speed_results/annarchy-v2.json', 'w') as f:
    json.dump(speed_res, f, indent=2)
