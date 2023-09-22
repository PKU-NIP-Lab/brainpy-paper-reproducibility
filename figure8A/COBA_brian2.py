from brian2 import *
import json


run_on = sys.argv[1]
# run_on = 'cpp_standalone'

if run_on == 'cpp_standalone':
  set_device('cpp_standalone')

elif run_on == 'genn':
  import brian2genn
  set_device("genn")

elif run_on == 'cuda_standalone':
  import brian2cuda
  set_device("cuda_standalone")

else:
  raise ValueError

if len(sys.argv) > 2:
  prefs.devices.cpp_standalone.openmp_threads = int(sys.argv[2])

defaultclock.dt = 0.1 * ms


def run_(scale=1., duration=1000., monitor=False):
  start_scope()
  device.reinit()
  device.activate()

  taum = 20 * ms
  taue = 5 * ms
  taui = 10 * ms
  Vt = -50 * mV
  Vr = -60 * mV
  El = -60 * mV
  Erev_exc = 0. * mV
  Erev_inh = -80. * mV
  I = 20. * mvolt

  we = 0.6  # excitatory synaptic weight (voltage)
  wi = 6.7  # inhibitory synaptic weight

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

  if monitor:
    mon = SpikeMonitor(P)

  Ce = Synapses(P[:num_exc], P, on_pre='ge += we')
  Ci = Synapses(P[num_exc:], P, on_pre='gi += wi')

  P.v = (np.random.randn(num_exc + num_inh) * 5. - 55.) * mvolt
  Ce.connect(p=80 / (num_exc + num_inh))
  Ci.connect(p=80 / (num_exc + num_inh))

  t1 = time.time()
  run(duration * ms)
  t2 = time.time()
  if monitor:
    rate = len(mon.i) / (num_exc + num_inh) / duration * 1e3
    print(f'size = {num_exc + num_inh}, '
          f'execution time = {device._last_run_time} s, '
          f'running time = {t2 - t1} s, '
          f'rate = {rate} Hz')
  else:
    rate = None
    print(f'size = {num_exc + num_inh}, '
          f'execution time = {device._last_run_time} s, '
          f'running time = {t2 - t1} s')
  print()
  print()

  return {'num': num_exc + num_inh,
          'exe_time': device._last_run_time,
          'run_time': t2 - t1,
          'fr': rate}


def benchmark(duration=1000.):
  final_results = dict()
  for scale in [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]:
    for _ in range(10):
      r = run_(scale=scale, duration=duration, monitor=False)
      if r['num'] not in final_results:
        final_results[r['num']] = {'exetime': [], 'runtime': [], 'firing_rate': []}
      final_results[r['num']]['exetime'].append(r['exe_time'])
      final_results[r['num']]['runtime'].append(r['run_time'])
      final_results[r['num']]['firing_rate'].append(r['fr'])
  if len(sys.argv) > 2:
    name_ = run_on + f'-thread{prefs.devices.cpp_standalone.openmp_threads}'
  else:
    name_ = run_on
  # with open(f'speed_results/brian2-COBA-{run_on}-thread{prefs.devices.cpp_standalone.openmp_threads}.json', 'w') as fout:
  with open(f'speed_results/brian2-COBA-{name_}.json', 'w') as fout:
    json.dump(final_results, fout, indent=2)


def check_firing_rate():
  run_(scale=1, duration=1e3, monitor=True)
  run_(scale=2, duration=1e3, monitor=True)
  run_(scale=4, duration=1e3, monitor=True)
  run_(scale=6, duration=1e3, monitor=True)
  run_(scale=8, duration=1e3, monitor=True)
  run_(scale=10, duration=1e3, monitor=True)


def visualize_spike_raster(duration=100.):
  start_scope()
  device.reinit()
  device.activate()

  taum = 20 * ms
  taue = 5 * ms
  taui = 10 * ms
  Vt = -50 * mV
  Vr = -60 * mV
  El = -60 * mV
  Erev_exc = 0. * mV
  Erev_inh = -80. * mV
  I = 20. * mvolt

  we = 0.6  # excitatory synaptic weight (voltage)
  wi = 6.7  # inhibitory synaptic weight

  num_exc = int(3200)
  num_inh = int(800)

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

  mon = SpikeMonitor(P)

  Ce = Synapses(P[:num_exc], P, on_pre='ge += we')
  Ci = Synapses(P[num_exc:], P, on_pre='gi += wi')

  P.v = (np.random.randn(num_exc + num_inh) * 5. - 55.) * mvolt
  Ce.connect(p=80 / (num_exc + num_inh))
  Ci.connect(p=80 / (num_exc + num_inh))

  run(duration * ms)

  import brainpy as bp
  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.)
  ax = fig.add_subplot(gs[0])
  plt.plot(mon.t / ms, mon.i, '.k', markersize=2,)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.title(f'Brian2 {run_on}')
  plt.savefig(f'COBA-brian2-{run_on}.pdf')



if __name__ == '__main__':
  benchmark(duration=5. * 1e3)
  # check_firing_rate()
  # visualize_spike_raster()

