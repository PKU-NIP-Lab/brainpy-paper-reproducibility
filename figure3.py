# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap

from figure2 import HH, Network, System, area_names, num_exc, num_inh, conn_data, delay_data
from src.decision_making_network import DecisionMakingNet


def a1_a2_visualize_channel_and_neuron():
  model = HH(1, V_initializer=bp.init.OneInit(-60.))
  runner = bp.DSRunner(model, monitors=['V', 'IK.p', 'INa.p', 'INa.q'])
  runner.predict(500.)

  # sns.set_theme(font_scale=1.5)
  fig, gs = bp.visualize.get_figure(2, 1, 2.25, 6.)
  fig.add_subplot(gs[0, 0])
  plt.plot(runner.mon['ts'], runner.mon['IK.p'], label='IK.p')
  plt.plot(runner.mon['ts'], runner.mon['INa.p'], label='INa.p')
  plt.plot(runner.mon['ts'], runner.mon['INa.q'], label='INa.q')
  plt.title('Channel model')
  plt.xticks([])
  plt.xlim(0., 500.)
  lg = plt.legend(loc='right')
  lg.get_frame().set_alpha(0.5)
  fig.add_subplot(gs[1, 0])
  plt.plot(runner.mon['ts'], runner.mon['V'][:, 0], label='V')
  plt.title('Neuron model')
  plt.xticks([])
  plt.xlabel('Time [ms]')
  plt.xlim(0., 500.)
  lg = plt.legend(loc='right')
  lg.get_frame().set_alpha(0.5)
  plt.show()


def a3_visualize_network(seed=20873, gEE=0.03, gEI=0.03, gIE=.335, gII=0.335, ):
  bm.random.seed(seed)
  model = Network(num_exc, num_inh, gEE=gEE, gEI=gEI, gIE=gIE, gII=gII)
  runner = bp.DSRunner(model, monitors={'exc.spike': model.E.spike})
  runner.run(200.)

  plt.rcParams.update({"font.size": 15})
  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
  fig.add_subplot(gs[0, 0])
  indices, times = bp.measure.raster_plot(runner.mon['exc.spike'], runner.mon['ts'])
  plt.plot(times, indices, '.', markersize=1)
  plt.xticks([])
  plt.yticks([])
  plt.ylabel('Neuron index')
  plt.xlabel('Time [ms]')
  plt.xlim(50., 200.)
  ax = plt.gca()
  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')
  plt.ylim(0, num_exc)
  plt.title('Network Model')
  plt.show()


def a4_visualize_system():
  seed = 2546234

  gc = 1.
  gEE = 0.0060
  gEI = 0.0060
  gIE = 0.26800
  gII = 0.26800

  bm.random.seed(seed)
  model = System(conn=gc * bm.asarray(conn_data),
                 delay=bm.asarray(delay_data),
                 gEE=gEE, gEI=gEI,
                 gIE=gIE, gII=gII)
  inputs, duration = bp.inputs.section_input([0., 1., 0.],
                                             [400., 100., 300.],
                                             return_length=True)
  runner = bp.DSRunner(
    model,
    monitors={
      'exc.spike': lambda tdi: bm.concatenate([area.E.spike for area in model.areas]),
    },
    inputs=[model.areas[0].E.input, inputs, 'iter'],
  )
  runner.run(duration)

  plt.rcParams.update({"font.size": 15})
  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.)
  fig.add_subplot(gs[0, 0])
  indices, times = bp.measure.raster_plot(runner.mon['exc.spike'], runner.mon['ts'])
  plt.plot(times, indices, '.', markersize=1)
  plt.yticks(np.arange(len(area_names)) * num_exc + num_exc / 2, area_names)
  plt.ylim(0, len(area_names) * num_exc)
  plt.xlim(375., 750.)
  plt.plot([375., 750.], (np.arange(1, len(area_names)) * num_exc).repeat(2).reshape(-1, 2).T,
           color='k', linestyle='--', linewidth=1)
  plt.title('System Model')
  plt.xlabel('Time [ms]')
  plt.xticks([])
  ax = plt.gca()
  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')
  plt.show()


def a5_decision_making_batch_simulation():
  bm.random.seed(32554)
  len_pre_stimulus = 500.
  len_stimulus = 1000.
  len_delay = 500.
  total_period = len_pre_stimulus + len_stimulus + len_delay
  num_exc = 1600 * 0.15

  # running a single simulation
  def single_run(coh):
    net = DecisionMakingNet(coherence=coh,
                            stimulus_period=len_stimulus,
                            pre_stimulus_period=len_pre_stimulus)
    runner = bp.DSRunner(
      net,
      monitors=['A.spike', 'B.spike', 'IA.freq', 'IB.freq'],
      numpy_mon_after_run=False,
    )
    runner.run(total_period)
    return runner.mon

  # running a batch of simulations
  batch_run = vmap(single_run)

  # batch running with multiple inputs
  coherence = bm.asarray([-100., -20., 20., 100.])
  mon = batch_run(coherence)

  # visualization
  plt.rcParams.update({"font.size": 15})
  coherence = coherence.to_numpy()
  mon.to_numpy()

  # visualize raster plot
  fig, gs = bp.visualize.get_figure(coherence.size, 1, 4.5 / coherence.size, 6)
  for i in range(coherence.size):
    ax = fig.add_subplot(gs[i, 0])
    elements = np.where(mon['A.spike'][i] > 0.)
    index, time = elements[1], mon['ts'][i, elements[0]]
    ax.plot(time, index, '.', markersize=1)
    elements = np.where(mon['B.spike'][i] > 0.)
    index, time = elements[1], mon['ts'][i, elements[0]]
    ax.plot(time, index, '.', markersize=1)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0., total_period - 10.)
    ax.set_ylim(0., num_exc)
    ax.text(50, num_exc - 100, f"coherence={int(coherence[i]):d}%", fontsize=10)
    ax.set_ylabel('Raster')
    if i == 0:
      ax.set_title('Rater Plot of Batch Simulation Results')
    if i == coherence.size - 1:
      ax.set_xlabel('Time [ms]')
  plt.show()

  # visualize firing rates
  fig, gs = bp.visualize.get_figure(coherence.size, 1, 4.5 / coherence.size, 6)
  for i in range(coherence.size):
    ax = fig.add_subplot(gs[i, 0])
    rateA = bp.measure.firing_rate(mon['A.spike'][i], width=10.)
    rateB = bp.measure.firing_rate(mon['B.spike'][i], width=10.)
    ax.plot(mon['ts'][i], rateA)
    ax.plot(mon['ts'][i], rateB)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0., total_period - 10.)
    ax.text(50, 40, f"coherence={int(coherence[i]):d}%", fontsize=10)
    ax.set_ylabel('Rate')
    if i == 0:
      ax.set_title('Firing Rate of Batch Simulation Results')
    if i == coherence.size - 1:
      ax.set_xlabel('Time [ms]')
  plt.show()


if __name__ == '__main__':
  a1_a2_visualize_channel_and_neuron()
  a3_visualize_network()
  a4_visualize_system()
  a5_decision_making_batch_simulation()
