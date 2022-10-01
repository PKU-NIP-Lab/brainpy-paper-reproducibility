# -*- coding: utf-8 -*-

import time
from random import sample

import brian2 as b2
import matplotlib.pyplot as plt
import numpy
import numpy.random as rnd
from brian2 import NeuronGroup, Synapses, PoissonInput, PoissonGroup, network_operation
from brian2.monitors import StateMonitor, SpikeMonitor, PopulationRateMonitor

try:
  b2.clear_cache('cython')
except:
  pass

b2.defaultclock.dt = 0.10 * b2.ms
method = 'euler'


def sim_decision_making_network(
    scale=1., t_stimulus_start=100 * b2.ms,
    t_stimulus_duration=9999 * b2.ms, coherence_level=0.,
    stimulus_update_interval=30 * b2.ms, mu0_mean_stimulus_Hz=160.,
    stimulus_std_Hz=20., N_extern=1000, firing_rate_extern=9.8 * b2.Hz,
    w_pos=1.90, f_Subpop_size=0.25,  # .15 in publication [1]
    max_sim_time=1000. * b2.ms, res_dict=None,
    monitored_subset_size=512,
    file=None):
  N_Excit = int(1600 * scale)
  N_Inhib = int(400 * scale)
  weight_scaling_factor = 1.27659574 / scale

  print("simulating {} neurons. Start: {}".format(N_Excit + N_Inhib, time.ctime()))
  t_stimulus_end = t_stimulus_start + t_stimulus_duration

  N_Group_A = int(N_Excit * f_Subpop_size)  # size of the excitatory subpopulation sensitive to stimulus A
  N_Group_B = N_Group_A  # size of the excitatory subpopulation sensitive to stimulus B
  N_Group_Z = N_Excit - N_Group_A - N_Group_B  # (1-2f)Ne excitatory neurons do not respond to either stimulus.

  Cm_excit = 0.5 * b2.nF  # membrane capacitance of excitatory neurons
  G_leak_excit = 25.0 * b2.nS  # leak conductance
  E_leak_excit = -70.0 * b2.mV  # reversal potential
  v_spike_thr_excit = -50.0 * b2.mV  # spike condition
  v_reset_excit = -60.0 * b2.mV  # reset voltage after spike
  t_abs_refract_excit = 2. * b2.ms  # absolute refractory period

  # specify the inhibitory interneurons:
  # N_Inhib = 200
  Cm_inhib = 0.2 * b2.nF
  G_leak_inhib = 20.0 * b2.nS
  E_leak_inhib = -70.0 * b2.mV
  v_spike_thr_inhib = -50.0 * b2.mV
  v_reset_inhib = -60.0 * b2.mV
  t_abs_refract_inhib = 1.0 * b2.ms

  # specify the AMPA synapses
  E_AMPA = 0.0 * b2.mV
  tau_AMPA = 2.5 * b2.ms

  # specify the GABA synapses
  E_GABA = -70.0 * b2.mV
  tau_GABA = 5.0 * b2.ms

  # specify the NMDA synapses
  E_NMDA = 0.0 * b2.mV
  tau_NMDA_s = 100.0 * b2.ms
  tau_NMDA_x = 2. * b2.ms
  alpha_NMDA = 0.5 * b2.kHz

  # projections from the external population
  g_AMPA_extern2inhib = 1.62 * b2.nS
  g_AMPA_extern2excit = 2.1 * b2.nS

  # projectsions from the inhibitory populations
  g_GABA_inhib2inhib = weight_scaling_factor * 1.25 * b2.nS
  g_GABA_inhib2excit = weight_scaling_factor * 1.60 * b2.nS

  # projections from the excitatory population
  g_AMPA_excit2excit = weight_scaling_factor * 0.012 * b2.nS
  g_AMPA_excit2inhib = weight_scaling_factor * 0.015 * b2.nS
  g_NMDA_excit2excit = weight_scaling_factor * 0.040 * b2.nS
  g_NMDA_excit2inhib = weight_scaling_factor * 0.045 * b2.nS  # stronger projection to inhib.

  # weights and "adjusted" weights.
  w_neg = 1. - f_Subpop_size * (w_pos - 1.) / (1. - f_Subpop_size)
  # We use the same postsyn AMPA and NMDA conductances. Adjust the weights coming from different sources:
  w_ext2inhib = g_AMPA_extern2inhib / g_AMPA_excit2inhib
  w_ext2excit = g_AMPA_extern2excit / g_AMPA_excit2excit
  # other weights are 1
  # print("w_neg={}, w_ext2inhib={}, w_ext2excit={}".format(w_neg, w_ext2inhib, w_ext2excit))

  # Define the inhibitory population
  # dynamics:
  inhib_lif_dynamics = """
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_inhib * (v-E_leak_inhib)
        - g_AMPA_excit2inhib * s_AMPA * (v-E_AMPA)
        - g_GABA_inhib2inhib * s_GABA * (v-E_GABA)
        - g_NMDA_excit2inhib * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
        )/Cm_inhib : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
    """

  inhib_pop = NeuronGroup(N_Inhib,
                          model=inhib_lif_dynamics,
                          threshold="v>v_spike_thr_inhib",
                          reset="v=v_reset_inhib",
                          refractory=t_abs_refract_inhib,
                          method=method)
  # initialize with random voltages:
  inhib_pop.v = rnd.uniform(v_spike_thr_inhib / b2.mV - 4.,
                            high=v_spike_thr_inhib / b2.mV - 1.,
                            size=N_Inhib) * b2.mV

  # Specify the excitatory population:
  # dynamics:
  excit_lif_dynamics = """
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_excit * (v-E_leak_excit)
        - g_AMPA_excit2excit * s_AMPA * (v-E_AMPA)
        - g_GABA_inhib2excit * s_GABA * (v-E_GABA)
        - g_NMDA_excit2excit * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-0.062*v/volt)/3.57)
        )/Cm_excit : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_NMDA/dt = -s_NMDA/tau_NMDA_s + alpha_NMDA * x * (1-s_NMDA) : 1
        dx/dt = -x/tau_NMDA_x : 1
    """

  # define the three excitatory subpopulations.
  # A: subpop receiving stimulus A
  excit_pop_A = NeuronGroup(N_Group_A,
                            model=excit_lif_dynamics,
                            threshold="v>v_spike_thr_excit",
                            reset="v=v_reset_excit",
                            refractory=t_abs_refract_excit,
                            method=method)
  excit_pop_A.v = rnd.uniform(E_leak_excit / b2.mV,
                              high=E_leak_excit / b2.mV + 5.,
                              size=excit_pop_A.N) * b2.mV

  # B: subpop receiving stimulus B
  excit_pop_B = NeuronGroup(N_Group_B,
                            model=excit_lif_dynamics,
                            threshold="v>v_spike_thr_excit",
                            reset="v=v_reset_excit",
                            refractory=t_abs_refract_excit,
                            method=method)
  excit_pop_B.v = rnd.uniform(E_leak_excit / b2.mV,
                              high=E_leak_excit / b2.mV + 5.,
                              size=excit_pop_B.N) * b2.mV
  # Z: non-sensitive
  excit_pop_Z = NeuronGroup(N_Group_Z,
                            model=excit_lif_dynamics,
                            threshold="v>v_spike_thr_excit",
                            reset="v=v_reset_excit",
                            refractory=t_abs_refract_excit,
                            method=method)
  excit_pop_Z.v = rnd.uniform(v_reset_excit / b2.mV,
                              high=v_spike_thr_excit / b2.mV - 1.,
                              size=excit_pop_Z.N) * b2.mV

  # now define the connections:
  # projections FROM EXTERNAL POISSON GROUP: ####################################################
  poisson2Inhib = PoissonInput(target=inhib_pop,
                               target_var="s_AMPA",
                               N=N_extern,
                               rate=firing_rate_extern,
                               weight=w_ext2inhib)
  poisson2A = PoissonInput(target=excit_pop_A,
                           target_var="s_AMPA",
                           N=N_extern,
                           rate=firing_rate_extern,
                           weight=w_ext2excit)

  poisson2B = PoissonInput(target=excit_pop_B,
                           target_var="s_AMPA",
                           N=N_extern,
                           rate=firing_rate_extern,
                           weight=w_ext2excit)
  poisson2Z = PoissonInput(target=excit_pop_Z,
                           target_var="s_AMPA",
                           N=N_extern,
                           rate=firing_rate_extern,
                           weight=w_ext2excit)

  ###############################################################################################

  # GABA projections FROM INHIBITORY population: ################################################
  syn_inhib2inhib = Synapses(inhib_pop,
                             target=inhib_pop,
                             on_pre="s_GABA += 1.0",
                             delay=0.5 * b2.ms)
  syn_inhib2inhib.connect(p=1.)
  syn_inhib2A = Synapses(inhib_pop,
                         target=excit_pop_A,
                         on_pre="s_GABA += 1.0",
                         delay=0.5 * b2.ms)
  syn_inhib2A.connect(p=1.)
  syn_inhib2B = Synapses(inhib_pop,
                         target=excit_pop_B,
                         on_pre="s_GABA += 1.0",
                         delay=0.5 * b2.ms)
  syn_inhib2B.connect(p=1.)
  syn_inhib2Z = Synapses(inhib_pop,
                         target=excit_pop_Z,
                         on_pre="s_GABA += 1.0",
                         delay=0.5 * b2.ms)
  syn_inhib2Z.connect(p=1.)
  ###############################################################################################

  # AMPA projections FROM EXCITATORY A: #########################################################
  syn_AMPA_A2A = Synapses(excit_pop_A,
                          target=excit_pop_A,
                          on_pre="s_AMPA += w_pos",
                          delay=0.5 * b2.ms)
  syn_AMPA_A2A.connect(p=1.)
  syn_AMPA_A2B = Synapses(excit_pop_A,
                          target=excit_pop_B,
                          on_pre="s_AMPA += w_neg",
                          delay=0.5 * b2.ms)
  syn_AMPA_A2B.connect(p=1.)
  syn_AMPA_A2Z = Synapses(excit_pop_A,
                          target=excit_pop_Z,
                          on_pre="s_AMPA += 1.0",
                          delay=0.5 * b2.ms)
  syn_AMPA_A2Z.connect(p=1.)
  syn_AMPA_A2inhib = Synapses(excit_pop_A,
                              target=inhib_pop,
                              on_pre="s_AMPA += 1.0",
                              delay=0.5 * b2.ms)
  syn_AMPA_A2inhib.connect(p=1.)
  ###############################################################################################

  # AMPA projections FROM EXCITATORY B: #########################################################
  syn_AMPA_B2A = Synapses(excit_pop_B,
                          target=excit_pop_A,
                          on_pre="s_AMPA += w_neg",
                          delay=0.5 * b2.ms)
  syn_AMPA_B2A.connect(p=1.)
  syn_AMPA_B2B = Synapses(excit_pop_B,
                          target=excit_pop_B,
                          on_pre="s_AMPA += w_pos",
                          delay=0.5 * b2.ms)
  syn_AMPA_B2B.connect(p=1.)
  syn_AMPA_B2Z = Synapses(excit_pop_B,
                          target=excit_pop_Z,
                          on_pre="s_AMPA += 1.0",
                          delay=0.5 * b2.ms)
  syn_AMPA_B2Z.connect(p=1.)
  syn_AMPA_B2inhib = Synapses(excit_pop_B,
                              target=inhib_pop,
                              on_pre="s_AMPA += 1.0",
                              delay=0.5 * b2.ms)
  syn_AMPA_B2inhib.connect(p=1.)
  ###############################################################################################

  # AMPA projections FROM EXCITATORY Z: #########################################################
  syn_AMPA_Z2A = Synapses(excit_pop_Z,
                          target=excit_pop_A,
                          on_pre="s_AMPA += 1.0",
                          delay=0.5 * b2.ms)
  syn_AMPA_Z2A.connect(p=1.)
  syn_AMPA_Z2B = Synapses(excit_pop_Z,
                          target=excit_pop_B,
                          on_pre="s_AMPA += 1.0",
                          delay=0.5 * b2.ms)
  syn_AMPA_Z2B.connect(p=1.)
  syn_AMPA_Z2Z = Synapses(excit_pop_Z,
                          target=excit_pop_Z,
                          on_pre="s_AMPA += 1.0",
                          delay=0.5 * b2.ms)
  syn_AMPA_Z2Z.connect(p=1.)
  syn_AMPA_Z2inhib = Synapses(excit_pop_Z,
                              target=inhib_pop,
                              on_pre="s_AMPA += 1.0",
                              delay=0.5 * b2.ms)
  syn_AMPA_Z2inhib.connect(p=1.)

  ###############################################################################################

  # NMDA projections FROM EXCITATORY to INHIB, A,B,Z
  @network_operation()
  def update_nmda_sum():
    sum_sNMDA_A = sum(excit_pop_A.s_NMDA)
    sum_sNMDA_B = sum(excit_pop_B.s_NMDA)
    sum_sNMDA_Z = sum(excit_pop_Z.s_NMDA)
    # note the _ at the end of s_NMDA_total_ disables unit checking
    inhib_pop.s_NMDA_total_ = (1.0 * sum_sNMDA_A + 1.0 * sum_sNMDA_B + 1.0 * sum_sNMDA_Z)
    excit_pop_A.s_NMDA_total_ = (w_pos * sum_sNMDA_A + w_neg * sum_sNMDA_B + w_neg * sum_sNMDA_Z)
    excit_pop_B.s_NMDA_total_ = (w_neg * sum_sNMDA_A + w_pos * sum_sNMDA_B + w_neg * sum_sNMDA_Z)
    excit_pop_Z.s_NMDA_total_ = (1.0 * sum_sNMDA_A + 1.0 * sum_sNMDA_B + 1.0 * sum_sNMDA_Z)

  # set a self-recurrent synapse to introduce a delay when updating the intermediate
  # gating variable x
  syn_x_A2A = Synapses(excit_pop_A, excit_pop_A, on_pre="x += 1.", delay=0.5 * b2.ms)
  syn_x_A2A.connect(j="i")
  syn_x_B2B = Synapses(excit_pop_B, excit_pop_B, on_pre="x += 1.", delay=0.5 * b2.ms)
  syn_x_B2B.connect(j="i")
  syn_x_Z2Z = Synapses(excit_pop_Z, excit_pop_Z, on_pre="x += 1.", delay=0.5 * b2.ms)
  syn_x_Z2Z.connect(j="i")
  ###############################################################################################

  # Define the stimulus: two PoissonInput with time time-dependent mean.
  poissonStimulus2A = PoissonGroup(N_Group_A, 0. * b2.Hz)
  syn_Stim2A = Synapses(poissonStimulus2A, excit_pop_A, on_pre="s_AMPA+=w_ext2excit")
  syn_Stim2A.connect(j="i")
  poissonStimulus2B = PoissonGroup(N_Group_B, 0. * b2.Hz)
  syn_Stim2B = Synapses(poissonStimulus2B, excit_pop_B, on_pre="s_AMPA+=w_ext2excit")
  syn_Stim2B.connect(j="i")

  @network_operation(dt=stimulus_update_interval)
  def update_poisson_stimulus(t):
    if t >= t_stimulus_start and t < t_stimulus_end:
      offset_A = mu0_mean_stimulus_Hz * (0.5 + 0.5 * coherence_level)
      offset_B = mu0_mean_stimulus_Hz * (0.5 - 0.5 * coherence_level)

      rate_A = numpy.random.normal(offset_A, stimulus_std_Hz)
      rate_A = (max(0, rate_A)) * b2.Hz  # avoid negative rate
      rate_B = numpy.random.normal(offset_B, stimulus_std_Hz)
      rate_B = (max(0, rate_B)) * b2.Hz

      poissonStimulus2A.rates = rate_A
      poissonStimulus2B.rates = rate_B
      # print("stim on. rate_A= {}, rate_B = {}".format(rate_A, rate_B))
    else:
      # print("stim off")
      poissonStimulus2A.rates = 0.
      poissonStimulus2B.rates = 0.

  ###############################################################################################

  def get_monitors(pop, monitored_subset_size):
    """
    Internal helper.
    Args:
        pop:
        monitored_subset_size:
    Returns:
    """
    monitored_subset_size = min(monitored_subset_size, pop.N)
    idx_monitored_neurons = sample(range(pop.N), monitored_subset_size)
    rate_monitor = PopulationRateMonitor(pop)
    # record parameter: record=idx_monitored_neurons is not supported???
    spike_monitor = SpikeMonitor(pop, record=idx_monitored_neurons)
    voltage_monitor = StateMonitor(pop, "v", record=idx_monitored_neurons)
    return rate_monitor, spike_monitor, voltage_monitor, idx_monitored_neurons

  # collect data of a subset of neurons:
  rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib = \
    get_monitors(inhib_pop, monitored_subset_size)

  rate_monitor_A, spike_monitor_A, voltage_monitor_A, idx_monitored_neurons_A = \
    get_monitors(excit_pop_A, monitored_subset_size)

  rate_monitor_B, spike_monitor_B, voltage_monitor_B, idx_monitored_neurons_B = \
    get_monitors(excit_pop_B, monitored_subset_size)

  rate_monitor_Z, spike_monitor_Z, voltage_monitor_Z, idx_monitored_neurons_Z = \
    get_monitors(excit_pop_Z, monitored_subset_size)

  t0 = time.time()
  b2.run(max_sim_time)
  t = time.time() - t0
  print(t)

  print("sim end: {}".format(time.ctime()))

  ret_vals = dict()

  ret_vals["rate_monitor_A"] = rate_monitor_A
  ret_vals["spike_monitor_A"] = spike_monitor_A
  ret_vals["voltage_monitor_A"] = voltage_monitor_A
  ret_vals["idx_monitored_neurons_A"] = idx_monitored_neurons_A

  ret_vals["rate_monitor_B"] = rate_monitor_B
  ret_vals["spike_monitor_B"] = spike_monitor_B
  ret_vals["voltage_monitor_B"] = voltage_monitor_B
  ret_vals["idx_monitored_neurons_B"] = idx_monitored_neurons_B

  ret_vals["rate_monitor_Z"] = rate_monitor_Z
  ret_vals["spike_monitor_Z"] = spike_monitor_Z
  ret_vals["voltage_monitor_Z"] = voltage_monitor_Z
  ret_vals["idx_monitored_neurons_Z"] = idx_monitored_neurons_Z

  ret_vals["rate_monitor_inhib"] = rate_monitor_inhib
  ret_vals["spike_monitor_inhib"] = spike_monitor_inhib
  ret_vals["voltage_monitor_inhib"] = voltage_monitor_inhib
  ret_vals["idx_monitored_neurons_inhib"] = idx_monitored_neurons_inhib

  if res_dict is not None:
    res_dict['brian2'].append({'num_neuron': N_Excit + N_Inhib,
                               'sim_len': max_sim_time / b2.ms,
                               'num_thread': 1,
                               'sim_time': t,
                               'dt': 0.1})
  if file is not None:
    file.write(f'scale={scale}, num={N_Excit + N_Inhib}, time={t}\n')
  print(f'Used time: {t} s')


  return ret_vals


def plot_network_activity(
    rate_monitor, spike_monitor, voltage_monitor=None, spike_train_idx_list=None,
    t_min=None, t_max=None, N_highlighted_spiketrains=3, avg_window_width=1.0 * b2.ms,
    sup_title=None, figure_size=(10, 4)):
  """
  Visualizes the results of a network simulation: spike-train, population activity and voltage-traces.
  Args:
      rate_monitor (PopulationRateMonitor): rate of the population
      spike_monitor (SpikeMonitor): spike trains of individual neurons
      voltage_monitor (StateMonitor): optional. voltage traces of some (same as in spike_train_idx_list) neurons
      spike_train_idx_list (list): optional. A list of neuron indices whose spike-train is plotted.
          If no list is provided, all (up to 500) spike-trains in the spike_monitor are plotted. If None, the
          the list in voltage_monitor.record is used.
      t_min (Quantity): optional. lower bound of the plotted time interval.
          if t_min is None, it is set to the larger of [0ms, (t_max - 100ms)]
      t_max (Quantity): optional. upper bound of the plotted time interval.
          if t_max is None, it is set to the timestamp of the last spike in
      N_highlighted_spiketrains (int): optional. Number of spike trains visually highlighted, defaults to 3
          If N_highlighted_spiketrains==0 and voltage_monitor is not None, then all voltage traces of
          the voltage_monitor are plotted. Otherwise N_highlighted_spiketrains voltage traces are plotted.
      avg_window_width (Quantity): optional. Before plotting the population rate (PopulationRateMonitor), the rate
          is smoothed using a window of width = avg_window_width. Defaults is 1.0ms
      sup_title (String): figure suptitle. Default is None.
      figure_size (tuple): (width,height) tuple passed to pyplot's figsize parameter.
  Returns:
      Figure: The whole figure
      Axes: Top panel, Raster plot
      Axes: Middle panel, population activity
      Axes: Bottom panel, voltage traces. None if no voltage monitor is provided.
  """

  assert isinstance(rate_monitor, b2.PopulationRateMonitor), \
    "rate_monitor  is not of type PopulationRateMonitor"
  assert isinstance(spike_monitor, b2.SpikeMonitor), \
    "spike_monitor is not of type SpikeMonitor"
  assert (voltage_monitor is None) or (isinstance(voltage_monitor, b2.StateMonitor)), \
    "voltage_monitor is not of type StateMonitor"
  assert (spike_train_idx_list is None) or (isinstance(spike_train_idx_list, list)), \
    "spike_train_idx_list is not of type list"

  all_spike_trains = spike_monitor.spike_trains()
  if spike_train_idx_list is None:
    if voltage_monitor is not None:
      # if no index list is provided use the one from the voltage monitor
      spike_train_idx_list = numpy.sort(voltage_monitor.record)
    else:
      # no index list AND no voltage monitor: plot all spike trains
      spike_train_idx_list = numpy.sort(all_spike_trains.keys())
    if len(spike_train_idx_list) > 5000:
      # avoid slow plotting of a large set
      print("Warning: raster plot with more than 5000 neurons truncated!")
      spike_train_idx_list = spike_train_idx_list[:5000]

  # get a reasonable default interval
  if t_max is None:
    t_max = max(rate_monitor.t / b2.ms)
  else:
    t_max = t_max / b2.ms
  if t_min is None:
    t_min = max(0., t_max - 100.)  # if none, plot at most the last 100ms
  else:
    t_min = t_min / b2.ms

  fig = None
  ax_raster = None
  ax_rate = None
  ax_voltage = None
  if voltage_monitor is None:
    fig, (ax_raster, ax_rate) = plt.subplots(2, 1, sharex=True, figsize=figure_size)
  else:
    fig, (ax_raster, ax_rate, ax_voltage) = plt.subplots(3, 1, sharex=True, figsize=figure_size)

  # nested helpers to plot the parts, note that they use parameters defined outside.
  def get_spike_train_ts_indices(spike_train):
    """
    Helper. Extracts the spikes within the time window from the spike train
    """
    ts = spike_train / b2.ms
    # spike_within_time_window = (ts >= t_min) & (ts <= t_max)
    # idx_spikes = numpy.where(spike_within_time_window)
    idx_spikes = (ts >= t_min) & (ts <= t_max)
    ts_spikes = ts[idx_spikes]
    return idx_spikes, ts_spikes

  def plot_raster():
    """
    Helper. Plots the spike trains of the spikes in spike_train_idx_list
    """
    neuron_counter = 0
    for neuron_index in spike_train_idx_list:
      idx_spikes, ts_spikes = get_spike_train_ts_indices(all_spike_trains[neuron_index])
      ax_raster.scatter(ts_spikes, neuron_counter * numpy.ones(ts_spikes.shape),
                        marker=".", c="k", s=15, lw=0)
      neuron_counter += 1
    ax_raster.set_ylim([0, neuron_counter])

  def highlight_raster(neuron_idxs):
    """
    Helper. Highlights three spike trains
    """
    for i in range(len(neuron_idxs)):
      color = "r" if i == 0 else "k"
      raster_plot_index = neuron_idxs[i]
      population_index = spike_train_idx_list[raster_plot_index]
      idx_spikes, ts_spikes = get_spike_train_ts_indices(all_spike_trains[population_index])
      ax_raster.axhline(y=raster_plot_index, linewidth=.5, linestyle="-", color=[.9, .9, .9])
      ax_raster.scatter(
        ts_spikes, raster_plot_index * numpy.ones(ts_spikes.shape),
        marker=".", c=color, s=144, lw=0)
    ax_raster.set_ylabel("neuron #")
    ax_raster.set_title("Raster Plot", fontsize=10)

  def plot_population_activity(window_width=0.5 * b2.ms):
    """
    Helper. Plots the population rate and a mean
    """
    ts = rate_monitor.t / b2.ms
    idx_rate = (ts >= t_min) & (ts <= t_max)
    # ax_rate.plot(ts[idx_rate],rate_monitor.rate[idx_rate]/b2.Hz, ".k", markersize=2)
    smoothed_rates = rate_monitor.smooth_rate(window="flat", width=window_width) / b2.Hz
    ax_rate.plot(ts[idx_rate], smoothed_rates[idx_rate])
    ax_rate.set_ylabel("A(t) [Hz]")
    ax_rate.set_title("Population Activity", fontsize=10)

  def plot_voltage_traces(voltage_traces_i):
    """
    Helper. Plots three voltage traces
    """
    ts = voltage_monitor.t / b2.ms
    idx_voltage = (ts >= t_min) & (ts <= t_max)
    for i in range(len(voltage_traces_i)):
      color = "r" if i == 0 else ".7"
      raster_plot_index = voltage_traces_i[i]
      population_index = spike_train_idx_list[raster_plot_index]
      ax_voltage.plot(
        ts[idx_voltage], voltage_monitor[population_index].v[idx_voltage] / b2.mV,
        c=color, lw=1.)
      ax_voltage.set_ylabel("V(t) [mV]")
      ax_voltage.set_title("Voltage Traces", fontsize=10)

  plot_raster()
  plot_population_activity(avg_window_width)
  nr_neurons = len(spike_train_idx_list)
  highlighted_neurons_i = []  # default to an empty list.
  if N_highlighted_spiketrains > 0:
    fract = numpy.linspace(0, 1, N_highlighted_spiketrains + 2)[1:-1]
    highlighted_neurons_i = [int(nr_neurons * v) for v in fract]
    highlight_raster(highlighted_neurons_i)

  if voltage_monitor is not None:
    if N_highlighted_spiketrains == 0:
      traces_i = range(nr_neurons)
    else:
      traces_i = highlighted_neurons_i
    plot_voltage_traces(traces_i)

  plt.xlabel("t [ms]")

  if sup_title is not None:
    plt.suptitle(sup_title)

  return fig, ax_raster, ax_rate, ax_voltage


def getting_started(scale=1.):
  results = sim_decision_making_network(scale=scale,
                                        coherence_level=0.8,
                                        w_pos=2.0,
                                        mu0_mean_stimulus_Hz=500 * b2.Hz,
                                        t_stimulus_start=100. * b2.ms,
                                        t_stimulus_duration=1000 * b2.ms,
                                        max_sim_time=1600. * b2.ms)
  plot_network_activity(results["rate_monitor_A"],
                        results["spike_monitor_A"],
                        results["voltage_monitor_A"],
                        t_min=0. * b2.ms,
                        avg_window_width=20. * b2.ms,
                        sup_title="Left")
  plot_network_activity(results["rate_monitor_B"],
                        results["spike_monitor_B"],
                        results["voltage_monitor_B"],
                        t_min=0. * b2.ms,
                        avg_window_width=20. * b2.ms,
                        sup_title="Right")

  plt.show()


if __name__ == '__main__':

  import sys

  if len(sys.argv) == 1:
    platform = 'cpu'
  else:
    if sys.argv[1] == 'gpu':
      platform = 'gpu'
    else:
      raise ValueError

  getting_started(0.2)

  # speed_res = {'brian2': []}
  # with open(f'speed_results/brian2.txt', 'w') as fout:
  #   for scale in np.array(list(range(1, 3, 1))) / 4:
  #     sim_decision_making_network(scale=scale,
  #                                 res_dict=speed_res,
  #                                 coherence_level=0.256,
  #                                 w_pos=2.0,
  #                                 mu0_mean_stimulus_Hz=500 * b2.Hz,
  #                                 t_stimulus_start=100. * b2.ms,
  #                                 t_stimulus_duration=1000 * b2.ms,
  #                                 max_sim_time=1600. * b2.ms,
  #                                 file=fout)
  #
  # with open('speed_results/brian2.json', 'w') as f:
  #   json.dump(speed_res, f, indent=2)
