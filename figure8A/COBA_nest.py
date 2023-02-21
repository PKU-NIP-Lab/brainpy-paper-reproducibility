import time

import nest
import numpy


def run(scale=4, nb_threads=1, simtime=1000.0, res_dict=None):
  nest.ResetKernel()
  numpy.random.seed(98765)
  nest.SetKernelStatus({"resolution": 0.1})
  nest.SetKernelStatus({"local_num_threads": int(nb_threads)})

  NE = int(3200 * scale)  # number of exc. neurons
  NI = int(800 * scale)  # number of inh. neurons

  nest.SetDefaults("iaf_cond_exp",
                   {"C_m": 200.,
                    "g_L": 10.,
                    "tau_syn_ex": 5.,
                    "tau_syn_in": 10.,
                    "E_ex": 0.,
                    "E_in": -80.,
                    "t_ref": 5.,
                    "E_L": -60.,
                    "V_th": -50.,
                    "I_e": 200.,
                    "V_reset": -60.,
                    "V_m": -60.})

  nodes_ex = nest.Create("iaf_cond_exp", NE)
  nodes_in = nest.Create("iaf_cond_exp", NI)
  nodes = nodes_ex + nodes_in

  v = -55.0 + 5.0 * numpy.random.normal(size=NE + NI)
  for i, node in enumerate(nodes):
    nest.SetStatus(node, {"V_m": v[i]})

  # Create the synapses
  w_exc = 6. / scale
  w_inh = -67. / scale
  nest.SetDefaults("static_synapse", {"delay": 0.1})
  nest.CopyModel("static_synapse", "excitatory", {"weight": w_exc})
  nest.CopyModel("static_synapse", "inhibitory", {"weight": w_inh})

  nest.Connect(nodes_ex, nodes, {'rule': 'pairwise_bernoulli', 'p': 0.02}, syn_spec="excitatory")
  nest.Connect(nodes_in, nodes, {'rule': 'pairwise_bernoulli', 'p': 0.02}, syn_spec="inhibitory")

  tstart = time.time()
  nest.Simulate(simtime)
  tend = time.time()
  print('Done in', tend - tstart)

  if res_dict is not None:
    res_dict['nest'].append({'num_neuron': NE + NI,
                             'sim_len': simtime,
                             'num_thread': nb_threads,
                             'sim_time': tend - tstart,
                             'dt': 0.1})


if __name__ == '__main__':
  import json

  speed_res = {'nest': []}
  for scale in [1, 2, 4, 6, 8, 10]:
    for nthread in [1, 2, 4]:
      for stim in [5. * 1e3]:
        run(scale=scale, res_dict=speed_res, nb_threads=nthread, simtime=stim)

  with open('speed_results/nest.json', 'w') as f:
    json.dump(speed_res, f, indent=2)
