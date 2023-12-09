from brian2 import *
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--backend', type=str, default='cpp_standalone',
                    choices=['cpp_standalone', 'genn', 'cuda_standalone'])
parser.add_argument('--dtype', type=str, default='f64', choices=['f64', 'f32'])
parser.add_argument('--threads', type=int, default=1)
args = parser.parse_args()


if args.backend == 'cpp_standalone':
  set_device('cpp_standalone')

elif args.backend == 'genn':
  import brian2genn
  set_device("genn")

elif args.backend == 'cuda_standalone':
  import brian2cuda
  set_device("cuda_standalone")

else:
  raise ValueError

if args.threads > 1:
  prefs.devices.cpp_standalone.openmp_threads = args.threads
if args.dtype == 'f32':
  prefs.core.default_float_dtype = float32
elif args.dtype == 'f64':
  prefs.core.default_float_dtype = float64
else:
  raise ValueError


defaultclock.dt = 0.1 * ms


def run_(scale=1., duration=1000., monitor=False):
  start_scope()
  device.reinit()
  device.activate(directory=None)

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
                  method='exponential_euler')
  Ce = Synapses(P[:num_exc], P, on_pre='ge += we')
  Ci = Synapses(P[num_exc:], P, on_pre='gi += wi')
  P.v = (np.random.randn(num_exc + num_inh) * 5. - 55.) * mvolt
  Ce.connect(p=80 / (num_exc + num_inh))
  Ci.connect(p=80 / (num_exc + num_inh))
  mon = SpikeMonitor(P)

  t1 = time.time()
  run(duration * ms)
  t2 = time.time()

  rate = len(mon.i) / (num_exc + num_inh) / duration * 1e3
  print(f'size = {num_exc + num_inh}, '
        f'execution time = {device._last_run_time} s, '
        f'running time = {t2 - t1} s, '
        f'rate = {rate} Hz')

  return {'num': num_exc + num_inh,
          'exe_time': device._last_run_time,
          'run_time': t2 - t1,
          'fr': rate}


def benchmark(duration=1000.):
  final_results = dict()
  fn = f'speed_results/brian2-COBA-{args.backend}-th{args.threads}-{args.dtype}.json'
  if args.backend == 'cpp_standalone':
    scales = [1, 2, 4, 6, 8, 10, 20]
  else:
    scales = [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]

  for scale in scales:
    for _ in range(4):
      r = run_(scale=scale, duration=duration, monitor=False)
      if r['num'] not in final_results:
        final_results[r['num']] = {'exetime': [], 'runtime': [], 'firing_rate': []}
      final_results[r['num']]['exetime'].append(r['exe_time'])
      final_results[r['num']]['runtime'].append(r['run_time'])
      final_results[r['num']]['firing_rate'].append(r['fr'])
    with open(fn, 'w') as fout:
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
  plt.title(f'Brian2 {args.backend}')
  plt.savefig(f'COBA-brian2-{args.backend}.pdf')


if __name__ == '__main__':
  benchmark(duration=5. * 1e3)
  # check_firing_rate()
  # visualize_spike_raster()

