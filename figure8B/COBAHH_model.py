from brian2 import *
import os
import time
import sys

run_on = sys.argv[1]  # Take device as command line arg

prefs.codegen.cpp.headers += ['<chrono>']

SETUP_TIMER = '''
std::chrono::steady_clock::time_point _benchmark_start, _benchmark_now;
_benchmark_start = std::chrono::steady_clock::now();
std::ofstream _benchmark_file;
_benchmark_file.open("{fname}");
'''

TIME_DIFF = '''
_benchmark_now = std::chrono::steady_clock::now();
_benchmark_file << "{name}" << " "
                << std::chrono::duration_cast<std::chrono::microseconds>(
                       _benchmark_now - _benchmark_start
                   ).count()
                << std::endl;
'''

CLOSE_TIMER = '''
_benchmark_file.close();
'''


def insert_benchmark_point(name, slot="main"):
  device.insert_code(slot, TIME_DIFF.format(name=name))


if run_on == 'cpp_standalone':
  set_device('cpp_standalone')

elif run_on == 'genn':
  import brian2genn
  set_device("genn")

elif run_on == 'cuda':
  import brian2cuda
  set_device("cuda_standalone")

else:
  raise ValueError

defaultclock.dt = 0.1 * ms

area = 0.02
Cm = 200
gl = 10.
g_na = 20 * 1000
g_kd = 6. * 1000

time_unit = 1 * ms
El = -60
EK = -90
ENa = 50
VT = -63
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
    beta_h = 4./(1.+exp((40.-v+VT)/5.)) : 1
    alpha_n = 0.032*(15.-v+VT)/(exp((15.-v+VT)/5.)-1.) : 1
    beta_n = .5*exp((10.-v+VT)/40.) : 1
''')


def simulate(scale, duration):
  start_scope()
  device.reinit()
  device.activate()
  # Inject code for benchmarking
  benchmark_fname = os.path.join('/tmp', 'benchmark_points.txt')
  device.insert_code("before_start", SETUP_TIMER.format(fname=benchmark_fname))

  num = int(4000 * scale)
  P = NeuronGroup(num,
                  model=eqs,
                  threshold='v>-20',
                  refractory=0 * ms,
                  method='exponential_euler')
  Pe = P[:int(3200 * scale)]
  Pi = P[int(3200 * scale):]
  Ce = Synapses(Pe, P, on_pre='ge += we')
  Ci = Synapses(Pi, P, on_pre='gi += wi')

  insert_benchmark_point("before_synapse_construction")
  Ce.connect(p=80 / num)
  Ci.connect(p=80 / num)
  insert_benchmark_point("after_synapse_construction")
  # Initialization
  P.v = 'El + (randn() * 5 - 5)'
  P.ge = '(randn() * 1.5 + 4) * 10.'
  P.gi = '(randn() * 12 + 20) * 10.'
  insert_benchmark_point("after_initialisation")

  insert_benchmark_point("before_run", slot="before_network_run")
  insert_benchmark_point("after_run", slot="after_network_run")
  insert_benchmark_point("end_of_main", slot="after_end")
  device.insert_code("after_end", CLOSE_TIMER)

  t0 = time.time()
  run(duration * ms, report='text', profile=True)
  t1 = time.time()
  total_time = t1 - t0

  print(f'size = {num}')
  with open(benchmark_fname) as bf:
    benchmark_details = {l.split(" ")[0]: float(l.split(" ")[1]) for l in bf.readlines() if l.strip()}
  print("Total time: ", total_time)
  print("  Compilation/code generation: ",
        total_time - (benchmark_details["end_of_main"] / 1e6))
  print("  Synapse construction       : ",
        (benchmark_details["after_synapse_construction"] - benchmark_details["before_synapse_construction"]) / 1e6)
  print("  Variable initialisation    : ",
        (benchmark_details["after_initialisation"] - benchmark_details["after_synapse_construction"]) / 1e6)
  print("  Simulation run             : ",
        device._last_run_time)
  print("  Simulation run (check)     : ",
        (benchmark_details["after_run"] - benchmark_details["before_run"]) / 1e6)
  profiling_summary(show=5)


if __name__ == '__main__':
  simulate(scale=1., duration=5. * 1e3)
