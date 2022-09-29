# coding: utf-8

"""
Balanced network of excitatory and inhibitory neurons.

An implementation of benchmarks 1 and 2 from

    Brette et al. (2007) Journal of Computational Neuroscience 23: 349-398

The network is based on the CUBA and COBA models of Vogels & Abbott
(J. Neurosci, 2005).  The model consists of a network of excitatory and
inhibitory neurons, connected via current-based "exponential"
synapses (instantaneous rise, exponential decay).


Usage: python VAbenchmarks.py [-h] [--plot-figure] [--use-views] [--use-assembly]
                              [--use-csa] [--debug DEBUG]
                              simulator benchmark

positional arguments:
  simulator       neuron, nest, brian or another backend simulator
  benchmark       either CUBA or COBA

optional arguments:
  -h, --help      show this help message and exit
  --plot-figure   plot the simulation results to a file
  --use-views     use population views in creating the network
  --use-assembly  use assemblies in creating the network
  --use-csa       use the Connection Set Algebra to define the connectivity
  --debug DEBUG   print debugging information


Andrew Davison, UNIC, CNRS
August 2006

pyNN == 0.9.5
NEURON = 7.8.1
neo == 0.7.1

"""

import json

from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility import get_simulator, Timer, ProgressBar, init_logging

# === Configure the simulator ================================================

sim, options = get_simulator(
  ("benchmark", "either CUBA or COBA"),
  ("--plot-figure", "plot the simulation results to a file", {"action": "store_true"}),
  ("--use-views", "use population views in creating the network", {"action": "store_true"}),
  ("--use-assembly", "use assemblies in creating the network", {"action": "store_true"}),
  ("--use-csa", "use the Connection Set Algebra to define the connectivity", {"action": "store_true"}),
  ("--debug", "print debugging information"))

if options.use_csa:
  import csa

if options.debug:
  init_logging(None, debug=True)

# === Define parameters ========================================================

threads = 1
rngseed = 98765
parallel_safe = True

r_ei = 4.0  # number of excitatory cells:number of inhibitory cells
pconn = 0.02  # connection probability
stim_dur = 50.  # (ms) duration of random stimulation
rate = 100.  # (Hz) frequency of the random stimulation

dt = 0.1  # (ms) simulation timestep

delay = 0.2

# Cell parameters
area = 20000.  # (µm²)
tau_m = 20.  # (ms)
cm = 1.  # (µF/cm²)
g_leak = 5e-5  # (S/cm²)
if options.benchmark == "COBA":
  E_leak = -60.  # (mV)
elif options.benchmark == "CUBA":
  E_leak = -49.  # (mV)
v_thresh = -50.  # (mV)
v_reset = -60.  # (mV)
t_refrac = 5.  # (ms) (clamped at v_reset)
v_mean = -60.  # (mV) 'mean' membrane potential, for calculating CUBA weights
tau_exc = 5.  # (ms)
tau_inh = 10.  # (ms)

# Synapse parameters
if options.benchmark == "COBA":
  Gexc = 4.  # (nS)
  Ginh = 51.  # (nS)
elif options.benchmark == "CUBA":
  Gexc = 0.27  # (nS) #Those weights should be similar to the COBA weights
  Ginh = 4.5  # (nS) # but the delpolarising drift should be taken into account
Erev_exc = 0.  # (mV)
Erev_inh = -80.  # (mV)

# === Calculate derived parameters =============================================

area = area * 1e-8  # convert to cm²
cm = cm * area * 1000  # convert to nF
Rm = 1e-6 / (g_leak * area)  # membrane resistance in MΩ
assert tau_m == cm * Rm  # just to check

if options.benchmark == "COBA":
  celltype = sim.IF_cond_exp
  w_exc_raw = Gexc * 1e-3  # We convert conductances to uS
  w_inh_raw = Ginh * 1e-3
elif options.benchmark == "CUBA":
  celltype = sim.IF_curr_exp
  w_exc_raw = 1e-3 * Gexc * (Erev_exc - v_mean)  # (nA) weight of excitatory synapses
  w_inh_raw = 1e-3 * Ginh * (Erev_inh - v_mean)  # (nA)
  assert w_exc_raw > 0;
  assert w_inh_raw < 0


def run(scale=4, res_dict=None, num_thread=1, tstop=1000.):
  n = int(4000 * scale)  # number of cells
  n_exc = int(round((n * r_ei / (1 + r_ei))))  # number of excitatory cells
  n_inh = n - n_exc  # number of inhibitory cells
  w_inh = w_inh_raw / scale
  w_exc = w_exc_raw / scale

  extra = {'threads': num_thread, 'label': 'VA'}
  sim.setup(timestep=dt, min_delay=delay, max_delay=1.0, **extra)
  num_process = sim.num_processes()
  cell_params = {'tau_m': tau_m, 'tau_syn_E': tau_exc, 'tau_syn_I': tau_inh,
                 'v_rest': E_leak, 'v_reset': v_reset, 'v_thresh': v_thresh,
                 'cm': cm, 'tau_refrac': t_refrac}

  if (options.benchmark == "COBA"):
    cell_params['e_rev_E'] = Erev_exc
    cell_params['e_rev_I'] = Erev_inh

  timer = Timer()
  timer.start()
  if options.use_views:
    # create a single population of neurons, and then use population views to define
    # excitatory and inhibitory sub-populations
    all_cells = sim.Population(n_exc + n_inh, celltype(**cell_params), label="All Cells")
    exc_cells = all_cells[:n_exc]
    exc_cells.label = "Excitatory cells"
    inh_cells = all_cells[n_exc:]
    inh_cells.label = "Inhibitory cells"
  else:
    # create separate populations for excitatory and inhibitory neurons
    exc_cells = sim.Population(n_exc, celltype(**cell_params), label="Excitatory_Cells")
    inh_cells = sim.Population(n_inh, celltype(**cell_params), label="Inhibitory_Cells")
    if options.use_assembly:
      # group the populations into an assembly
      all_cells = exc_cells + inh_cells

  if options.benchmark == "COBA":
    ext_stim = sim.Population(20, sim.SpikeSourcePoisson(rate=rate, duration=stim_dur), label="expoisson")
    rconn = 0.01
    ext_conn = sim.FixedProbabilityConnector(rconn)
    ext_syn = sim.StaticSynapse(weight=0.1)

  rng = NumpyRNG(seed=rngseed, parallel_safe=parallel_safe)
  uniformDistr = RandomDistribution('uniform', low=v_reset, high=v_thresh, rng=rng)
  if options.use_views:
    all_cells.initialize(v=uniformDistr)
  else:
    exc_cells.initialize(v=uniformDistr)
    inh_cells.initialize(v=uniformDistr)

  progress_bar = ProgressBar(width=20)
  if options.use_csa:
    connector = sim.CSAConnector(csa.cset(csa.random(pconn)))
  else:
    connector = sim.FixedProbabilityConnector(pconn, rng=rng, callback=progress_bar)
  exc_syn = sim.StaticSynapse(weight=w_exc, delay=delay)
  inh_syn = sim.StaticSynapse(weight=w_inh, delay=delay)

  connections = {}
  if options.use_views or options.use_assembly:
    connections['exc'] = sim.Projection(exc_cells, all_cells, connector, exc_syn, receptor_type='excitatory')
    connections['inh'] = sim.Projection(inh_cells, all_cells, connector, inh_syn, receptor_type='inhibitory')
    if (options.benchmark == "COBA"):
      connections['ext'] = sim.Projection(ext_stim, all_cells, ext_conn, ext_syn, receptor_type='excitatory')
  else:
    connections['e2e'] = sim.Projection(exc_cells, exc_cells, connector, exc_syn, receptor_type='excitatory')
    connections['e2i'] = sim.Projection(exc_cells, inh_cells, connector, exc_syn, receptor_type='excitatory')
    connections['i2e'] = sim.Projection(inh_cells, exc_cells, connector, inh_syn, receptor_type='inhibitory')
    connections['i2i'] = sim.Projection(inh_cells, inh_cells, connector, inh_syn, receptor_type='inhibitory')
    if (options.benchmark == "COBA"):
      connections['ext2e'] = sim.Projection(ext_stim, exc_cells, ext_conn, ext_syn, receptor_type='excitatory')
      connections['ext2i'] = sim.Projection(ext_stim, inh_cells, ext_conn, ext_syn, receptor_type='excitatory')
  buildCPUTime = timer.diff()

  # === Run simulation ===========================================================
  sim.run(tstop)
  simCPUTime = timer.diff()

  if options.use_views or options.use_assembly:
    connections = "%d e→e,i  %d i→e,i" % (connections['exc'].size(), connections['inh'].size())
  else:
    connections = u"%d e→e  %d e→i  %d i→e  %d i→i" % (
      connections['e2e'].size(), connections['e2i'].size(),
      connections['i2e'].size(), connections['i2i'].size())

  print("\n--- Vogels-Abbott Network Simulation ---")
  print("Nodes                  : %d" % num_process)
  print("Simulation type        : %s" % options.benchmark)
  print("Number of Neurons      : %d" % n)
  print("Number of Synapses     : %s" % connections)
  print("Excitatory conductance : %g nS" % Gexc)
  print("Inhibitory conductance : %g nS" % Ginh)
  print("Build time             : %g s" % buildCPUTime)
  print("Simulation time        : %g s" % simCPUTime)

  # === Finished with simulator ==================================================
  sim.end()
  print('\n\n')

  if res_dict is not None:
    res_dict['neuron'].append({'num_neuron': n,
                               'sim_len': tstop,
                               'num_thread': 1,
                               'sim_time': simCPUTime,
                               'build_time': buildCPUTime,
                               'dt': 0.1})



if __name__ == '__main__':
  for scale in [1, 2, 4]:
    for stim in [5. * 1e3]:
      run(scale=scale, num_thread=1, tstop=stim)


if __name__ == '__main__1':
  speed_res = {'neuron': []}
  for scale in [1, 2, 4, 6, 8, 10]:
    for nthread in [1, 2, 4]:
      for stim in [5. * 1e3]:
        run(scale=scale, res_dict=speed_res, num_thread=nthread, tstop=stim)

  with open('speed_results/neuron.json', 'w') as f:
    json.dump(speed_res, f, indent=2)
