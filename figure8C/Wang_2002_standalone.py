"""
Decision network as in:

Wang, X.-J.
Probabilistic decision making by slow reverberation in cortical circuits.
Neuron, 2002, 36, 955-968.

Author Klaus Wimmer (kwimmer@crm.cat) with minor adjustments by Marcel Stimberg


Also see: https://brian2.readthedocs.io/en/latest/examples/frompapers.Wang_2002.html
"""

# brainpy-tower2: 1324239

from brian2 import *
import json
import time

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

# -----------------------------------------------------------------------------------------------
# set up the simulation
# -----------------------------------------------------------------------------------------------

def simulate_a_trial(scale=4.0, monitor=False):
  start_scope()
  device.reinit()
  device.activate(directory=None)

  # stimulus and simulation parameters
  coh = 25.6  # coherence of random dots
  sigma = 10.0 * Hz  # standard deviation of stimulus input
  mu0 = 40.0 * Hz  # stimulus input at zero coherence
  mu1 = 40.0 * Hz  # selective stimulus input at highest coherence
  stim_interval = 50.0 * ms  # stimulus changes every 50 ms
  stim_on = 100 * ms  # stimulus onset
  stim_off = 1100 * ms  # stimulus offset
  runtime = 1600 * ms  # total simulation time

  # external noise inputs
  N_ext = 1000  # number of external Poisson neurons
  rate_ext_E = 2400 * Hz / N_ext  # external Poisson rate for excitatory population
  rate_ext_I = 2400 * Hz / N_ext  # external Poisson rate for inhibitory population

  # network parameters
  N = int(2000 * scale)  # number of neurons
  f_inh = 0.2  # fraction of inhibitory neurons
  NE = int(N * (1.0 - f_inh))  # number of excitatory neurons (1600)
  NI = int(N * f_inh)  # number of inhibitory neurons (400)
  fE = 0.15  # coding fraction
  subN = int(fE * NE)  # number of neurons in decision pools (240)

  # neuron parameters
  El = -70.0 * mV  # resting potential
  Vt = -50.0 * mV  # firing threshold
  Vr = -55.0 * mV  # reset potential
  CmE = 0.5 * nF  # membrane capacitance for pyramidal cells (excitatory neurons)
  CmI = 0.2 * nF  # membrane capacitance for interneurons (inhibitory neurons)
  gLeakE = 25.0 * nS  # membrane leak conductance of excitatory neurons
  gLeakI = 20.0 * nS  # membrane leak conductance of inhibitory neurons
  refE = 2.0 * ms  # refractory periodof excitatory neurons
  refI = 1.0 * ms  # refractory period of inhibitory neurons

  # synapse parameters
  V_E = 0. * mV  # reversal potential for excitatory synapses
  V_I = -70. * mV  # reversal potential for inhibitory synapses
  tau_AMPA = 2.0 * ms  # AMPA synapse decay
  tau_NMDA_rise = 2.0 * ms  # NMDA synapse rise
  tau_NMDA_decay = 100.0 * ms  # NMDA synapse decay
  tau_GABA = 5.0 * ms  # GABA synapse decay
  alpha = 0.5 * kHz  # saturation of NMDA channels at high presynaptic firing rates
  C = 1 * mmole  # extracellular magnesium concentration

  # synaptic conductance
  gextE = 2.1 * nS  # external -> excitatory neurons (AMPA)
  gextI = 1.62 * nS  # external -> inhibitory neurons (AMPA)
  gEEA = 0.05 * nS / NE * 1600  # excitatory -> excitatory neurons (AMPA)
  gEIA = 0.04 * nS / NE * 1600  # excitatory -> inhibitory neurons (AMPA)
  gEEN = 0.165 * nS / NE * 1600  # excitatory -> excitatory neurons (NMDA)
  gEIN = 0.13 * nS / NE * 1600  # excitatory -> inhibitory neurons (NMDA)
  gIE = 1.3 * nS / NI * 400  # inhibitory -> excitatory neurons (GABA)
  gII = 1.0 * nS / NI * 400  # inhibitory -> inhibitory neurons (GABA)

  # synaptic footprints
  Jp = 1.7  # relative synaptic strength inside a selective population (1.0: no potentiation))
  Jm = 1.0 - fE * (Jp - 1.0) / (1.0 - fE)

  # neuron equations
  # note the "(unless refractory)" statement serves to clamp the membrane voltage during the refractory period;
  # otherwise the membrane potential continues to be integrated but no spikes are emitted
  eqsE = """
     label : integer (constant)  # label for decision encoding populations
     dV/dt = (- gLeakE * (V - El) - I_AMPA - I_NMDA - I_GABA - I_AMPA_ext + I_input) / CmE : volt (unless refractory)
  
     I_AMPA = s_AMPA * (V - V_E) : amp
     ds_AMPA / dt = - s_AMPA / tau_AMPA : siemens
  
     I_NMDA = gEEN * s_NMDA_tot * (V - V_E) / ( 1 + exp(-0.062 * V/mvolt) * (C/mmole / 3.57) ) : amp
     s_NMDA_tot : 1
  
     I_GABA = s_GABA * (V - V_I) : amp
     ds_GABA / dt = - s_GABA / tau_GABA : siemens
  
     I_AMPA_ext = s_AMPA_ext * (V - V_E) : amp
     ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA : siemens
  
     I_input : amp
  
     ds_NMDA / dt = - s_NMDA / tau_NMDA_decay + alpha * x * (1 - s_NMDA) : 1
     dx / dt = - x / tau_NMDA_rise : 1
  """

  eqsI = """
     dV/dt = (- gLeakI * (V - El) - I_AMPA - I_NMDA - I_GABA - I_AMPA_ext) / CmI : volt (unless refractory)
  
     I_AMPA = s_AMPA * (V - V_E) : amp
     ds_AMPA / dt = - s_AMPA / tau_AMPA : siemens
  
     I_NMDA = gEIN * s_NMDA_tot * (V - V_E) / ( 1 + exp(-0.062 * V/mvolt) * (C/mmole / 3.57) ): amp
     s_NMDA_tot : 1
  
     I_GABA = s_GABA * (V - V_I) : amp
     ds_GABA / dt = - s_GABA / tau_GABA : siemens
  
     I_AMPA_ext = s_AMPA_ext * (V - V_E) : amp
     ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA : siemens
  """

  # neuron populations
  popE = NeuronGroup(NE, model=eqsE, threshold='V > Vt', reset='V = Vr', refractory=refE,
                     method='exponential_euler', name='popE')
  popI = NeuronGroup(NI, model=eqsI, threshold='V > Vt', reset='V = Vr', refractory=refI,
                     method='exponential_euler', name='popI')
  popE1 = popE[:subN]
  popE2 = popE[subN:2 * subN]
  popE3 = popE[2 * subN:]
  popE1.label = 0
  popE2.label = 1
  popE3.label = 2

  # connections involving AMPA synapses
  # excitatory -> excitatory connection through AMPAR
  C_EE_AMPA = Synapses(popE, popE, 'w : siemens', on_pre='s_AMPA += w', delay=0.5 * ms, method='euler',
                       name='C_EE_AMPA')
  C_EE_AMPA.connect()
  C_EE_AMPA.w[:] = gEEA
  C_EE_AMPA.w["label_pre == label_post and label_pre < 2"] = gEEA * Jp
  C_EE_AMPA.w["label_pre != label_post and label_post < 2"] = gEEA * Jm

  # p_gEEA = gEEA * Jp
  # m_gEEA = gEEA * Jm
  # C_EE_on_pre = ('s_AMPA += int(label_post == 2) * gEEA + '
  #                '          int(label_pre == label_post and label_pre < 2) * p_gEEA +'
  #                '          int(label_pre != label_post and label_post < 2) * m_gEEA')
  # C_EE_AMPA = Synapses(popE, popE, on_pre=C_EE_on_pre, delay=0.5 * ms, method='exponential_euler', name='C_EE_AMPA')
  # C_EE_AMPA.connect()

  # excitatory -> inhibitory connection through AMPAR
  C_EI_AMPA = Synapses(popE, popI, on_pre='s_AMPA += gEIA', delay=0.5 * ms,
                       method='exponential_euler', name='C_EI_AMPA')
  C_EI_AMPA.connect()

  # external inputs (fixed background firing rates)
  extinputE = PoissonInput(popE, 's_AMPA_ext', N_ext, rate_ext_E, gextE)
  extinputI = PoissonInput(popI, 's_AMPA_ext', N_ext, rate_ext_I, gextI)

  # connections involving NMDA synapses
  C_EE_NMDA = Synapses(popE, popE, on_pre='x_pre += 1', delay=0.5 * ms, method='exponential_euler', name='C_EE_NMDA')
  C_EE_NMDA.connect(j='i')

  # Dummy population to store the summed activity of the three populations
  NMDA_sum_group = NeuronGroup(3, 's : 1', name='NMDA_sum_group')

  # Sum the activity according to the subpopulation labels
  NMDA_sum = Synapses(popE, NMDA_sum_group, 's_post = s_NMDA_pre : 1 (summed)', name='NMDA_sum')
  NMDA_sum.connect(j='label_pre')

  # Propagate the summed activity to the NMDA synapses
  NMDA_set_total_E = Synapses(NMDA_sum_group, popE,
                              '''w : 1 (constant)
                                 s_NMDA_tot_post = w*s_pre : 1 (summed)''', name='NMDA_set_total_E')
  NMDA_set_total_E.connect()
  # Weights to target populations
  # Population 3 receives input from all populations (no selectivity) with weight 1
  # Population 1 and 2 receive input from themselves with Jp and from the other population with Jm, together with input
  # from population 3 with weight 1
  NMDA_set_total_E.w = 1
  NMDA_set_total_E.w["i == label_post and label_post < 2"] = Jp
  NMDA_set_total_E.w["i != label_post and label_post < 2"] = Jm

  # # Propagate the summed activity to the NMDA synapses
  # C_EE_on_pre = ('s_NMDA_tot_post = (int(label_post == 2) + '
  #                'int(i == label_post and label_post < 2)*Jp +'
  #                'int(i != label_post and label_post < 2)*Jm) * s_pre : 1 (summed)')
  # NMDA_set_total_E = Synapses(NMDA_sum_group, popE, C_EE_on_pre, name='NMDA_set_total_E')
  # NMDA_set_total_E.connect()

  # The inihibitory population receives input from all excitatory populations with weight 1
  NMDA_set_total_I = Synapses(NMDA_sum_group, popI,
                              '''s_NMDA_tot_post = s_pre : 1 (summed)''', name='NMDA_set_total_I')
  NMDA_set_total_I.connect()

  # connections involving GABA synapses

  # inhibitory-excitatory connections
  C_IE = Synapses(popI, popE, on_pre='s_GABA += gIE', delay=0.5 * ms, method='exponential_euler', name='C_IE')
  C_IE.connect()

  # inhibitory-inhibitory connections
  C_II = Synapses(popI, popI, on_pre='s_GABA += gII', delay=0.5 * ms, method='exponential_euler', name='C_II')
  C_II.connect()

  # set initial conditions
  popE.s_NMDA_tot = tau_NMDA_decay * 10 * Hz * 0.2
  popI.s_NMDA_tot = tau_NMDA_decay * 10 * Hz * 0.2
  popE.V = Vt - 2 * mV
  popI.V = Vt - 2 * mV

  # stimulus input (updated every 50ms)
  stiminputE1 = PoissonGroup(subN, rates=0 * Hz, name='stiminputE1')
  stiminputE2 = PoissonGroup(subN, rates=0 * Hz, name='stiminputE2')
  stiminputE1.run_regularly("rates = int(t > stim_on and t < stim_off) * (mu0 + coh / 100.0 * mu1 + sigma*randn())",
                            dt=stim_interval)
  stiminputE2.run_regularly("rates = int(t > stim_on and t < stim_off) * (mu0 - coh / 100.0 * mu1 + sigma*randn())",
                            dt=stim_interval)

  C_stimE1 = Synapses(stiminputE1, popE1, on_pre='s_AMPA_ext += gextE', name='C_stimE1')
  C_stimE1.connect(j='i')
  C_stimE2 = Synapses(stiminputE2, popE2, on_pre='s_AMPA_ext += gextE', name='C_stimE2')
  C_stimE2.connect(j='i')

  # -----------------------------------------------------------------------------------------------
  # run the simulation
  # -----------------------------------------------------------------------------------------------

  if monitor:
    # record spikes of excitatory neurons in the decision encoding populations
    SME1 = SpikeMonitor(popE1, record=True)
    SME2 = SpikeMonitor(popE2, record=True)

    # record population activity
    R1 = PopulationRateMonitor(popE1)
    R2 = PopulationRateMonitor(popE2)

    # record input
    E1 = StateMonitor(stiminputE1, 'rates', record=0, dt=1 * ms)
    E2 = StateMonitor(stiminputE2, 'rates', record=0, dt=1 * ms)

    # run the simulation, switching on/off the stimuli
    t0 = time.time()
    run(runtime, profile=True)
    print(time.time() - t0)
    print(device._last_run_time)
    print(profiling_summary())

    # show results
    fig, axs = plt.subplots(4, 1, sharex=True, layout='constrained',
                            gridspec_kw={'height_ratios': [2, 2, 2, 1]})
    axs[0].plot(SME1.t / ms, SME1.i, '.', markersize=2, color='darkred')
    axs[0].set(ylabel='population 1', ylim=(0, subN))

    axs[1].plot(SME2.t / ms, SME2.i, '.', markersize=2, color='darkblue')
    axs[1].set(ylabel='population 2', ylim=(0, subN))

    axs[2].plot(R1.t / ms, R1.smooth_rate(window='flat', width=100 * ms) / Hz, color='darkred')
    axs[2].plot(R2.t / ms, R2.smooth_rate(window='flat', width=100 * ms) / Hz, color='darkblue')
    axs[2].set(ylabel='Firing rate (Hz)')

    axs[3].plot(E1.t / ms, E1.rates[0] / Hz, color='darkred')
    axs[3].plot(E2.t / ms, E2.rates[0] / Hz, color='darkblue')
    axs[3].set(ylabel='Input (Hz)', xlabel='Time (ms)')

    fig.align_ylabels(axs)
    plt.show()

  else:
    # run the simulation, switching on/off the stimuli
    t0 = time.time()
    run(runtime)
    t1 = time.time()
    print(f'Network size {N}, python runtime {t1 - t0} s, device time {device._last_run_time} s')
    return N, device._last_run_time, t1 - t0


def benchmark():
  if args.backend == 'cpp_standalone':
    scales = [1, 4, 8, 10, 20, 40, 60, 80, 100]
    n_time = 1
  elif args.backend == 'cuda_standalone':
    scales = [1, 4, 8, 10, 20, 40, 60, 80, 100, 200, 400, 800, 1000]
    n_time = 1
  else:
    raise ValueError

  # for scale in [1, 2, 4, 6, 8, 10]:
  # for scale in [1, ]:
  # for scale in [1, 4, 8, 10, 20, 40, 60, 80, 100]:
  final_results = dict()
  for scale in scales:
    for _ in range(n_time):
      r = simulate_a_trial(scale=scale, monitor=False)
      if r[0] not in final_results:
        final_results[r[0]] = {'exetime': [], 'runtime': []}
      final_results[r[0]]['exetime'].append(r[1])
      final_results[r[0]]['runtime'].append(r[2])
    with open(f'speed_results/brian2-DMv2-{args.backend}-th{args.threads}-{args.dtype}.json', 'w') as fout:
      json.dump(final_results, fout, indent=2)


if __name__ == '__main__':
  benchmark()
  # simulate_a_trial(scale=40, monitor=True)

