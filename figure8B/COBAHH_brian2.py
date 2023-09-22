import sys

from brian2 import *
import time
import json

run_on = sys.argv[1]

if len(sys.argv) > 2:
  prefs.devices.cpp_standalone.openmp_threads = int(sys.argv[2])

if run_on == 'cpp_standalone':
  set_device('cpp_standalone', directory='brian2_cpp')

elif run_on == 'genn':
  import brian2genn
  set_device("genn")

elif run_on == 'cuda_standalone':
  import brian2cuda
  set_device("cuda_standalone", directory='brian2_cuda')

else:
  raise ValueError

defaultclock.dt = 0.1 * ms

Cm = 200
gl = 10.
g_na = 20 * 1000
g_kd = 6. * 1000

time_unit = 1 * ms
El = -60.
EK = -90.
ENa = 50.
VT = -63.
# Time constants
taue = 5 * ms
taui = 10 * ms
# Reversal potentials
Ee = 0
Ei = -80

# excitatory synaptic weight
we = 6
# inhibitory synaptic weight
wi = 67

# The model
eqs = Equations('''
    dv/dt = (gl*(El-v) + ge*(Ee-v) + gi*(Ei-v)-
             g_na*(m*m*m)*h*(v-ENa)-
             g_kd*(n*n*n*n)*(v-EK))/Cm/time_unit : 1
    dm/dt = (alpha_m*(1-m)-beta_m*m)/time_unit : 1
    dn/dt = (alpha_n*(1-n)-beta_n*n)/time_unit : 1
    dh/dt = (alpha_h*(1-h)-beta_h*h)/time_unit : 1
    dge/dt = -ge/taue : 1
    dgi/dt = -gi/taui : 1
    alpha_m = 0.32*(13.-v+VT)/(exp((13.-v+VT)/4.)-1.) : 1
    beta_m = 0.28*(v-VT-40.)/(exp((v-VT-40.)/5.)-1.) : 1
    alpha_h = 0.128*exp((17.-v+VT)/18.) : 1
    beta_h = 4./(1+exp((40.-v+VT)/5.)) : 1
    alpha_n = 0.032*(15-v+VT)/(exp((15.-v+VT)/5.)-1.) : 1
    beta_n = .5*exp((10.-v+VT)/40.) : 1
''')


def simulate(scale, duration, monitor=False):
  start_scope()
  device.reinit()
  device.activate()

  num = int(4000 * scale)
  P = NeuronGroup(num, model=eqs, threshold='v>-20', refractory='v>-20',
                  method='exponential_euler')
  Pe = P[:int(3200 * scale)]
  Pi = P[int(3200 * scale):]
  Ce = Synapses(Pe, P, on_pre='ge += we')
  Ci = Synapses(Pi, P, on_pre='gi += wi')
  Ce.connect(p=80 / num)
  Ci.connect(p=80 / num)

  # Initialization
  P.v = 'El + (randn() * 5 - 5)'

  if monitor:
    mon = SpikeMonitor(P)

  # Record a few traces
  t0 = time.time()
  run(duration * ms)
  t1 = time.time()
  if monitor:
    rate = len(mon.i) / num / duration * 1e3
    print(f'size = {num}, '
          f'execution time = {device._last_run_time} s, '
          f'running time = {t1 - t0} s, '
          f'rate = {len(mon.i) / num / duration * 1e3} Hz')
  else:
    rate = -1e5
    print(f'size = {num}, execution time = {device._last_run_time} s, running time = {t1 - t0} s')
  print()

  return {'num': num,
          'exe_time': device._last_run_time,
          'run_time': t1 - t0,
          'fr': rate}


def check_firing_rate():
  simulate(scale=1, duration=1e3, monitor=True)
  simulate(scale=2, duration=1e3, monitor=True)
  simulate(scale=4, duration=1e3, monitor=True)
  simulate(scale=6, duration=1e3, monitor=True)
  simulate(scale=8, duration=1e3, monitor=True)
  simulate(scale=10, duration=1e3, monitor=True)


def benchmark(duration=1000.):
  final_results = dict()
  for scale in [1, 2, 4, 6, 8, 10, 20, 30, 40, 50]:
    for _ in range(10):
      r = simulate(scale=scale, duration=duration, monitor=False)
      if r['num'] not in final_results:
        final_results[r['num']] = {'exetime': [], 'runtime': [], 'firing_rate': []}
      final_results[r['num']]['exetime'].append(r['exe_time'])
      final_results[r['num']]['runtime'].append(r['run_time'])
      final_results[r['num']]['firing_rate'].append(r['fr'])
  name_ = run_on
  if len(sys.argv) > 2:
    name_ = run_on + f'-thread{prefs.devices.cpp_standalone.openmp_threads}'
  with open(f'speed_results/brian2-COBAHH-{name_}.json', 'w') as fout:
    json.dump(final_results, fout, indent=2)


def visualize_spike_raster(scale, duration):
  start_scope()
  device.reinit()
  device.activate()

  num = int(4000 * scale)
  P = NeuronGroup(num, model=eqs, threshold='v>-20', refractory='v>-20',
                  method='exponential_euler')
  Pe = P[:int(3200 * scale)]
  Pi = P[int(3200 * scale):]
  Ce = Synapses(Pe, P, on_pre='ge += we')
  Ci = Synapses(Pi, P, on_pre='gi += wi')
  Ce.connect(p=80 / num)
  Ci.connect(p=80 / num)

  # Initialization
  P.v = 'El + (randn() * 5 - 5)'

  mon = SpikeMonitor(P)

  # Record a few traces
  run(duration * ms)


  import brainpy as bp
  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.)
  ax = fig.add_subplot(gs[0])
  plt.plot(mon.t / ms, mon.i, '.k', markersize=2,)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.title(f'Brian2 {run_on}')
  plt.savefig(f'COBAHH-brian2-{run_on}.pdf')



if __name__ == '__main__':
  # check_firing_rate()
  # benchmark(duration=5e3)

  visualize_spike_raster(1., 100.)

