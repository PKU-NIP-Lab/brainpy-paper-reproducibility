# -*- coding: utf-8 -*-

import json

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jax.experimental.sparse import BCOO

taum = 20
taue = 5
taui = 10
Vt = -50
Vr = -60
El = -60
Erev_exc = 0.
Erev_inh = -80.
Ib = 20.
ref = 5.0


class LIF(bp.dyn.NeuGroup):
  def __init__(self, size, **kwargs):
    super().__init__(size=size, **kwargs)

    # parameters
    self.V_rest = Vr
    self.V_reset = El
    self.V_th = Vt
    self.tau = taum
    self.tau_ref = ref

    # variables
    self.V = bm.Variable(bm.zeros(self.num))
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

  def update(self, tdi):
    refractory = (bp.share['dt'] - self.t_last_spike) <= self.tau_ref
    V = self.V + (-self.V + self.V_rest + self.input) / self.tau * bp.share['dt']
    V = bm.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike.value = bm.where(spike, bp.share['dt'], self.t_last_spike)
    self.V.value = bm.where(spike, self.V_reset, V)
    self.spike.value = spike
    self.input[:] = Ib


class ExpDense(bp.dyn.TwoEndConn):
  def __init__(self, pre, post, conn, g_max, tau, E):
    super().__init__(pre, post, conn)

    # parameters
    self.tau = tau
    self.E = E
    self.g_max = self.conn.require('conn_mat') * g_max

    # variables
    self.g = bm.Variable(bm.zeros((self.pre.num, self.post.num)))

    # functions
    self.integral = bp.odeint(lambda g, t: -g / self.tau)

  def update(self, tdi):
    post_vs = bm.expand_dims(self.pre.spike, 1) * self.g_max
    self.g.value = self.integral(self.g.value, bp.share['t'], bp.share['dt']) + post_vs
    self.post.input += bm.sum(self.g, axis=0) * (self.E - self.post.V)


class ExpSparse(bp.dyn.TwoEndConn):
  def __init__(self, pre, post, conn, g_max, tau, E):
    super().__init__(pre, post, conn)

    # parameters
    self.tau = tau
    self.E = E
    conn_mat = self.conn.require('conn_mat')
    self.conn_mat = BCOO.fromdense(conn_mat.value)
    self.g_max = g_max

    # variables
    self.g = bm.Variable(bm.zeros((self.post.num,)))

    # functions
    self.integral = bp.odeint(lambda g, t: -g / self.tau)

  def update(self, tdi):
    post_vs = self.pre.spike @ self.conn_mat
    self.g.value = self.integral(self.g.value, bp.share['t'], bp.share['dt'])
    self.g.value += post_vs * self.g_max
    self.post.input += self.g.value * (self.E - self.post.V)


class ExpEventSparse(bp.dyn.TwoEndConn):
  def __init__(self, pre, post, conn, g_max, tau, E):
    super().__init__(pre, post, conn)

    # parameters
    self.tau = tau
    self.E = E
    self.g_max = g_max
    self.pre2post = self.conn.requires('pre2post')

    # variables
    self.g = bm.Variable(bm.zeros(post.num))  # variables

    # functions
    self.integral = bp.odeint(lambda g, t: -g / self.tau)

  def update(self, tdi):
    syn_vs = bm.pre2post_event_sum(self.pre.spike, self.pre2post, self.post.num, self.g_max)
    self.g.value = self.integral(self.g.value, bp.share['t'], bp.share['dt']) + syn_vs
    self.post.input += self.g * (self.E - self.post.V)


class CobaSparse(bp.dyn.Network):
  def __init__(self, scale):
    super().__init__()

    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)
    we = 0.6 / scale  # excitatory synaptic weight (voltage)
    wi = 6.7 / scale  # inhibitory synaptic weight

    self.E = LIF(num_exc)
    self.I = LIF(num_inh)
    self.E.V[:] = bm.random.randn(self.E.num) * 5. - 55.
    self.I.V[:] = bm.random.randn(self.I.num) * 5. - 55.

    # # synapses
    self.E2E = ExpSparse(self.E, self.E, bp.conn.FixedProb(0.02), E=Erev_exc, g_max=we, tau=taue)
    self.E2I = ExpSparse(self.E, self.I, bp.conn.FixedProb(0.02), E=Erev_exc, g_max=we, tau=taue)
    self.I2E = ExpSparse(self.I, self.E, bp.conn.FixedProb(0.02), E=Erev_inh, g_max=wi, tau=taui)
    self.I2I = ExpSparse(self.I, self.I, bp.conn.FixedProb(0.02), E=Erev_inh, g_max=wi, tau=taui)


class CobaEventSparse(bp.dyn.Network):
  def __init__(self, scale):
    super().__init__()

    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)
    we = 0.6 / scale  # excitatory synaptic weight (voltage)
    wi = 6.7 / scale  # inhibitory synaptic weight

    self.E = LIF(num_exc)
    self.I = LIF(num_inh)
    self.E.V[:] = bm.random.randn(self.E.num) * 5. - 55.
    self.I.V[:] = bm.random.randn(self.I.num) * 5. - 55.

    # # synapses
    self.E2E = ExpEventSparse(self.E, self.E, bp.conn.FixedProb(0.02), E=Erev_exc, g_max=we, tau=taue)
    self.E2I = ExpEventSparse(self.E, self.I, bp.conn.FixedProb(0.02), E=Erev_exc, g_max=we, tau=taue)
    self.I2E = ExpEventSparse(self.I, self.E, bp.conn.FixedProb(0.02), E=Erev_inh, g_max=wi, tau=taui)
    self.I2I = ExpEventSparse(self.I, self.I, bp.conn.FixedProb(0.02), E=Erev_inh, g_max=wi, tau=taui)


def compare_with_or_without_event_op(duration=1e3, check=False, n_run=20,
                                     res_file=None, platform='cpu'):
  bm.set_platform(platform)

  setting = dict(progress_bar=False)
  if check:
    setting = dict(progress_bar=True, monitors=['E.spike'])
  results = dict()
  for scale in [0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]:
    for _ in range(n_run):
      bm.random.seed()
      net = COBA_JIT_Comparison(scale)
      runner = bp.DSRunner(net, **setting)
      t = runner.run(duration, eval_time=True)
      print(f'scale = {scale}, dense + jit, running time = {t[0]} s')
      if check:
        bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], show=True)
      bm.clear_buffer_memory()
      if scale not in results:
        results[scale] = []
      results[scale].append(t[0])
  if res_file is not None:
    with open(res_file, 'w') as file:
      json.dump(results, file)


def get_linux_result():
  results = dict()
  results['num'] = np.asarray([4000 * s for s in (0.1, 0.2, 0.4, 0.8, 1.0, 2.0,
                                                  3.0, 4.0, 5., 6., 7., 8., 9.)])

  # cpu
  event_sparse_jit_cpu = [0.17807912826538086, 0.19947028160095215, 0.2758767604827881, 0.33082103729248047,
                          0.3412942886352539, 0.5303840637207031, 0.8610129356384277, 1.1178460121154785,
                          1.181762456893921, 1.302236795425415, 1.5212359428405762, 1.9606742858886719,
                          2.0316593647003174]
  dense_jit_cpu = [1.3729751110076904, 6.097690582275391, 18.534690141677856, 69.70568895339966,
                   91.58101773262024, 218.5230414867401, 436.8243465423584, 811.8477251529694,
                   1168.808358669281, 1618.178232908249, 2259.479111433029, 3396.037414073944,
                   3618.585748195648, ]
  dense_cpu = [26.399487733840942, 43.48150682449341, 122.88581395149231, 324.1572985649109,
               618.4884805679321, ]

  results['event_sparse_jit_cpu'] = np.asarray(event_sparse_jit_cpu)
  results['dense_jit_cpu'] = np.asarray(dense_jit_cpu)
  results['dense_cpu'] = np.asarray(dense_cpu)

  print(results['dense_cpu'] / results['dense_jit_cpu'][:5])
  # print(results['dense_jit_cpu'] / results['event_sparse_jit_cpu'])

  # gpu
  event_sparse_jit_gpu = [0.5612215995788574, 0.5763955116271973, 0.5795598030090332, 0.550288200378418,
                          0.5799703598022461, 0.6764547824859619, 0.7912883758544922, 0.987189769744873,
                          1.1811728477478027, 1.4392459392547607, 1.704448938369751, 2.120877265930176,
                          2.4642958641052246, ]
  dense_jit_gpu = [0.5780010223388672, 0.7102963924407959, 1.4245562553405762, 3.670361280441284,
                   5.310930252075195, 17.00484275817871, 39.14670395851135, 68.99373984336853,
                   105.47189927101135, 148.22078132629395, 199.0734441280365, 241.07212591171265,
                   328.23902130126953, ]
  dense_gpu = [64.4805896282196, 65.79613924026489, 63.659353494644165, 64.5964105129242,
               66.47876954078674, 105.78456974029541, 205.21649408340454, 311.4041268825531,
               450.2032811641693, 628.1145832538605, 832.2800834178925, 1048.7322623729706]

  results['event_sparse_jit_gpu'] = np.asarray(event_sparse_jit_gpu)
  results['dense_jit_gpu'] = np.asarray(dense_jit_gpu)
  results['dense_gpu'] = np.asarray(dense_gpu)

  # print(results['dense_gpu'] / results['dense_jit_gpu'][:5])

  return results


def visualize_results(num=20, device='cpu', data=True, fig_save_name=None):
  results = get_linux_result()
  # sns.set(font_scale=1.5)
  # sns.set_style("white")
  plt.rcParams.update({"font.size": 15})
  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.)
  ax = fig.add_subplot(gs[0, 0])

  if device == 'cpu':
    if data:
      plt.semilogy(results['num'][:num], results['event_sparse_jit_cpu'][:num],
                   linestyle="--", marker='v', label='With dedicated OP',
                   linewidth=3, markersize=10)
      plt.semilogy(results['num'][:num], results['dense_jit_cpu'][:num],
                   linestyle="--", marker='D', label='Without dedicated OP',
                   linewidth=3, markersize=10)
    else:
      plt.semilogy(results['num'][:num],
                   results['dense_jit_cpu'][:num] / results['event_sparse_jit_cpu'][:num],
                   marker='x', linestyle="--", label='Acceleration ratio',
                   linewidth=3, markersize=10, color='r', )

  elif device == 'gpu':
    if data:
      plt.semilogy(results['num'][:num], results['event_sparse_jit_gpu'][:num],
                   linestyle="--", marker='v', label='With dedicated OP',
                   linewidth=3, markersize=10)
      plt.semilogy(results['num'][:num], results['dense_jit_gpu'][:num],
                   linestyle="--", marker='D', label='Without dedicated OP',
                   linewidth=3, markersize=10)
    else:
      plt.semilogy(results['num'][:num],
                   results['dense_jit_gpu'][:num] / results['event_sparse_jit_gpu'][:num],
                   marker='x', linestyle="--", label='Acceleration ratio',
                   linewidth=3, markersize=10, color='r')

  else:
    raise ValueError

  # lg = ax.legend(fontsize=12, loc='center right')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  lg = ax.legend(fontsize=12, loc='best')
  lg.get_frame().set_alpha(0.3)
  plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
  # ax.set_title(f'Dedicated operator on {device.upper()}')
  ax.set_xlabel('Number of neurons')
  ax.set_ylabel('Simulation time [s]')
  # plt.savefig(r'D:\weiyun\WCM\Sync\Projects\2020.05 - BrainPy\文章'
  #             r'\2022.02 Full Paper\图\speed\event-op-accelerate-coba-1000-times.png',
  #             dpi=1000, transparent=True)
  if fig_save_name:
    plt.savefig(fig_save_name, dpi=1000, transparent=True)
  plt.show()


def visualize_results2(num=20, data=True, fig_save_name=None):
  results = get_linux_result()
  # sns.set(font_scale=1.5)
  # sns.set_style("white")
  plt.rcParams.update({"font.size": 15})
  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.)
  ax = fig.add_subplot(gs[0, 0])

  # CPU
  if data:
    plt.semilogy(results['num'][:num], results['event_sparse_jit_cpu'][:num],
                 linestyle="--", marker='v', label='CPU, With Event OP',
                 linewidth=3, markersize=10)
    plt.semilogy(results['num'][:num], results['dense_jit_cpu'][:num],
                 linestyle="--", marker='D', label='CPU, Without Event OP',
                 linewidth=3, markersize=10)
  else:
    plt.semilogy(results['num'][:num],
                 results['dense_jit_cpu'][:num] / results['event_sparse_jit_cpu'][:num],
                 marker='x', linestyle="--", label='CPU Acceleration Ratio',
                 linewidth=3, markersize=10, color='r', )

  # GPU
  if data:
    plt.semilogy(results['num'][:num], results['event_sparse_jit_gpu'][:num],
                 linestyle="--", marker='v', label='GPU, With Event OP',
                 linewidth=3, markersize=10)
    plt.semilogy(results['num'][:num], results['dense_jit_gpu'][:num],
                 linestyle="--", marker='D', label='GPU, Without Event OP',
                 linewidth=3, markersize=10)
  else:
    plt.semilogy(results['num'][:num],
                 results['dense_jit_gpu'][:num] / results['event_sparse_jit_gpu'][:num],
                 marker='x', linestyle="--", label='GPU Acceleration Ratio',
                 linewidth=3, markersize=10, color='r')

  # lg = ax.legend(fontsize=12, loc='center right')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  lg = ax.legend(fontsize=12, loc='best')
  lg.get_frame().set_alpha(0.3)
  plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
  # ax.set_title(f'Dedicated operator on {device.upper()}')
  ax.set_xlabel('Number of neurons')
  ax.set_ylabel('Simulation Time [s]')
  if fig_save_name:
    plt.savefig(fig_save_name, dpi=1000, transparent=True)
  plt.show()


if __name__ == '__main__':
  pass
  # compare_with_or_without_jit(res_file='results/coba-dense-jit=False.json', platform='cpu')
  # visualize_results2(data=True, fig_save_name='results/coba-with-event-op.png')
  visualize_results(device='gpu', data=True,
                    fig_save_name='results/coba-with-event-op-gpu.pdf')
  visualize_results(device='cpu', data=True,
                    fig_save_name='results/coba-with-event-op-cpu.pdf')
  # visualize_results(device='cpu', data=False,
  #                   fig_save_name='results/coba-with-event-op-ratio-cpu.pdf')
  # visualize_results(device='gpu', data=False,
  #                   fig_save_name='results/coba-with-event-op-ratio-gpu.pdf')
