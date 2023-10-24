# coding: utf-8

"""
Balanced network of excitatory and inhibitory neurons.

Usage: python COBA_pynn.py simulator benchmark

positional arguments:
  simulator       neuron, nest, brian or another backend simulator

"""

import json
import matplotlib.pyplot as plt
from pyNN.random import RandomDistribution
from pyNN.utility import get_simulator, Timer, ProgressBar
from quantities import ms

# === Configure the simulator ================================================
sim, options = get_simulator(("--threads", "number of threads", {'type': int}),)

# === Define parameters ========================================================

# other parameters
dt = 0.1  # (ms) simulation timestep
pconn = 0.02  # connection probability
r_ei = 4.0  # number of excitatory cells:number of inhibitory cells


def run(scale=4., num_thread=1, duration=1000., monitor=False):
  n = int(4000 * scale)  # number of cells
  n_exc = int(round((n * r_ei / (1 + r_ei))))  # number of excitatory cells
  n_inh = n - n_exc  # number of inhibitory cells

  sim.setup(timestep=dt, threads=num_thread, label='VA')
  num_process = sim.num_processes()

  we = 6. / 1e3
  wi = 67. / 1e3
  cell_params = {'gbar_Na': 20., 'gbar_K': 6., 'g_leak': 0.01,
                 'cm': 0.2, 'v_offset': -63.,
                 'e_rev_Na': 50.,  'e_rev_K': -90., 'e_rev_leak': -60.,
                 'e_rev_E': 0., 'e_rev_I': -80., 'i_offset': 0.,
                 'tau_syn_E': 5., 'tau_syn_I': 10.}

  timer = Timer()
  timer.start()
  # create a single population of neurons, and then use population views to define
  # excitatory and inhibitory sub-populations
  all_cells = sim.Population(n_exc + n_inh, sim.HH_cond_exp(**cell_params), label="All Cells")
  all_cells.record('spikes')

  # initialize the cells
  all_cells.initialize(v=RandomDistribution('normal', mu=-55., sigma=5.))

  # synapses
  connector = sim.FixedProbabilityConnector(pconn / scale, callback=ProgressBar(width=100))
  exc_syn = sim.StaticSynapse(weight=we)
  inh_syn = sim.StaticSynapse(weight=wi)
  exc_conn = sim.Projection(all_cells[:n_exc], all_cells, connector, exc_syn, receptor_type='excitatory')
  inh_conn = sim.Projection(all_cells[n_exc:], all_cells, connector, inh_syn, receptor_type='inhibitory')
  buildCPUTime = timer.diff()

  # === Run simulation ===========================================================
  sim.run(duration)
  simCPUTime = timer.diff()

  connections = "%d e→e,i  %d i→e,i" % (exc_conn.size(), inh_conn.size())
  data = all_cells.get_data().segments[0]
  fr = len(data.spiketrains.multiplexed[0]) / n / duration * 1e3

  print("\n--- Vogels-Abbott Network Simulation ---")
  print("Simulator              : %s" % options.simulator)
  print("Nodes                  : %d" % num_process)
  print("Number of Neurons      : %d" % n)
  print("Number of Synapses     : %s" % connections)
  print("Excitatory conductance : %g nS" % we)
  print("Inhibitory conductance : %g nS" % wi)
  print("Build time             : %g s" % buildCPUTime)
  print("Simulation time        : %g s" % simCPUTime)
  print("Firing rate            : %g Hz" % fr)

  if monitor:
    import brainpy as bp
    data = all_cells.get_data().segments[0]
    indices, times = data.spiketrains.multiplexed
    fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.)
    ax = fig.add_subplot(gs[0])
    plt.plot(times, indices, '.k', markersize=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title(f'{options.simulator.upper()}')
    plt.savefig(f'COBA-{options.simulator}-{options.threads}.pdf')

  # === Finished with simulator ==================================================
  sim.end()
  print('\n\n')
  return n, simCPUTime, buildCPUTime, fr


def benchmark(duration=1000.):
  final_results = dict()
  for scale in [1, 2, 4, 6, 8, 10, 20]:
  # for scale in [1, ]:
    for _ in range(2):
      num, t_exe, t_py, fr = run(scale=scale, num_thread=options.threads, duration=duration, monitor=False)
      if num not in final_results:
          final_results[num] = {'exetime': [], 'runtime': [], 'firing_rate': []}
      final_results[num]['exetime'].append(t_exe)
      final_results[num]['runtime'].append(t_py)
      final_results[num]['firing_rate'].append(fr)

    with open(f'speed_results/{options.simulator}-COBAHH-cpu-th{options.threads}.json', 'w') as fout:
      json.dump(final_results, fout, indent=2)


run(scale=1., num_thread=1, duration=1e2, monitor=True)
# benchmark(5e3)

