# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap

from src.decision_making_network import DecisionMakingNet


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
  a5_decision_making_batch_simulation()
