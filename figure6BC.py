# -*- coding: utf-8 -*-
import json

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
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


class LIF(bp.dyn.NeuDyn):
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

  def update(self):
    refractory = (bp.share['dt'] - self.t_last_spike) <= self.tau_ref
    V = self.V + (-self.V + self.V_rest + self.input) / self.tau * bp.share['dt']
    V = bm.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike.value = bm.where(spike, bp.share['dt'], self.t_last_spike)
    self.V.value = bm.where(spike, self.V_reset, V)
    self.spike.value = spike
    self.input[:] = Ib


class ExpDense(bp.synapses.TwoEndConn):
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

  def update(self):
    post_vs = bm.expand_dims(self.pre.spike, 1) * self.g_max
    self.g.value = self.integral(self.g.value, bp.share['t'], bp.share['dt']) + post_vs
    self.post.input += bm.sum(self.g, axis=0) * (self.E - self.post.V)


class ExpSparse(bp.synapses.TwoEndConn):
  def __init__(self, pre, post, conn, g_max, tau, E):
    super().__init__(pre, post, conn)

    # parameters
    self.tau = tau
    self.E = E
    conn_mat = self.conn.require('conn_mat')
    self.conn_mat = BCOO.fromdense(conn_mat.value)
    self.g_max = g_max

    # variables
    self.g = bm.Variable(bm.zeros((self.pre.num, self.post.num)))

    # functions
    self.integral = bp.odeint(lambda g, t: -g / self.tau)

  def update(self):
    post_vs = bm.expand_dims(self.pre.spike, 1) * self.conn_mat * self.g_max
    self.g.value = self.integral(self.g.value, bp.share['t'], bp.share['dt'])
    self.g.value += bm.as_jax(post_vs).todense()
    self.post.input += bm.sum(self.g, axis=0) * (self.E - self.post.V)


class COBA_JIT_Comparison(bp.DynSysGroup):
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
    self.E2E = ExpDense(self.E, self.E, bp.conn.FixedProb(0.02), E=Erev_exc, g_max=we, tau=taue)
    self.E2I = ExpDense(self.E, self.I, bp.conn.FixedProb(0.02), E=Erev_exc, g_max=we, tau=taue)
    self.I2E = ExpDense(self.I, self.E, bp.conn.FixedProb(0.02), E=Erev_inh, g_max=wi, tau=taui)
    self.I2I = ExpDense(self.I, self.I, bp.conn.FixedProb(0.02), E=Erev_inh, g_max=wi, tau=taui)


def compare_with_or_without_jit(duration=1e3, check=False, n_run=2, jit=False, res_file=None, platform='cpu'):
  bm.set_platform(platform)
  setting = dict(progress_bar=False)
  if check:
    setting = dict(progress_bar=True, monitors=['E.spike'])
  results = dict()
  for scale in [0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]:
    for _ in range(n_run):
      bm.random.seed()
      net = COBA_JIT_Comparison(scale)
      runner = bp.DSRunner(net, jit=jit, **setting)
      t = runner.run(duration, eval_time=True)
      print(f'scale = {scale}, dense + jit, running time = {t[0]} s')
      if check:
        bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], show=True)
      bm.clear_buffer_memory(platform)
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
  dense_jit_cpu = [1.3729751110076904, 6.097690582275391, 18.534690141677856, 69.70568895339966,
                   91.58101773262024, 218.5230414867401, 436.8243465423584, 811.8477251529694,
                   1168.808358669281, 1618.178232908249, 2259.479111433029, 3396.037414073944,
                   3618.585748195648, ]
  dense_cpu = [26.399487733840942, 43.48150682449341, 122.88581395149231, 324.1572985649109,
               618.4884805679321, ]
  dense_jit_cpu = [1.397280216217041, 7.878061532974243, 20.594850301742554, 73.83361840248108,
                   101.459481716156, 227.6046645641327, 468.59902596473694]
  dense_cpu = [341.69391717, 383.56625386, 495.81000615, 900.64367979, 1376.53286872,
               4645.87830868, 8886.88013333]

  results['dense_jit_cpu'] = np.asarray(dense_jit_cpu)
  results['dense_cpu'] = np.asarray(dense_cpu)

  # print(results['dense_cpu'] / results['dense_jit_cpu'][:5])
  # print(results['dense_jit_cpu'] / results['event_sparse_jit_cpu'])

  # gpu
  dense_jit_gpu = [0.5780010223388672, 0.7102963924407959, 1.4245562553405762, 3.670361280441284,
                   5.310930252075195, 17.00484275817871, 39.14670395851135, 68.99373984336853,
                   105.47189927101135, 148.22078132629395, 199.0734441280365, 241.07212591171265,
                   328.23902130126953, ]
  dense_gpu = [64.4805896282196, 65.79613924026489, 63.659353494644165, 64.5964105129242,
               66.47876954078674, 105.78456974029541, 205.21649408340454, 311.4041268825531,
               450.2032811641693, 628.1145832538605, 832.2800834178925, 1048.7322623729706]

  results['dense_jit_gpu'] = np.asarray(dense_jit_gpu)
  results['dense_gpu'] = np.asarray(dense_gpu)

  # print(results['dense_gpu'] / results['dense_jit_gpu'][:5])

  return results


def visualize_coba_with_or_without_jit(num=5, device='cpu'):
  results = get_linux_result()

  plt.rcParams.update({"font.size": 15})
  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.)
  ax = fig.add_subplot(gs[0, 0])
  if device == 'cpu':
    plt.semilogy(results['num'][:num], results['dense_jit_cpu'][:num],
                 linestyle="--", marker='v', label='With JIT',
                 linewidth=3, markersize=10)
    plt.semilogy(results['num'][:num], results['dense_cpu'][:num],
                 linestyle="--", marker='D', label='Without JIT',
                 linewidth=3, markersize=10)
  elif device == 'gpu':
    plt.semilogy(results['num'][:num], results['dense_jit_gpu'][:num],
                 linestyle="--", marker='v', label='With JIT',
                 linewidth=3, markersize=10)
    plt.semilogy(results['num'][:num], results['dense_gpu'][:num],
                 linestyle="--", marker='D', label='Without JIT',
                 linewidth=3, markersize=10)
  else:
    raise ValueError
  lg = plt.legend(fontsize=12, loc='best')
  lg.get_frame().set_alpha(0.3)
  # ax.set_title(f'Reducing overhead in COBA with JIT ({device})')
  plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
  plt.xlabel('Number of neurons')
  plt.ylabel('Simulation Time [s]')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.savefig(f'results/jit-reduce-coba-overhead-{device}.pdf', dpi=1000, transparent=True)
  plt.show()


if __name__ == '__main__':
  pass
  # bm.set_platform('cpu')
  # compare_with_or_without_jit(res_file='results/coba-dense-jit=False.json', jit=True, platform='cpu')
  # visualize_coba_with_or_without_jit(12, device='gpu')
  visualize_coba_with_or_without_jit(7, device='cpu')
  # visualize_coba_with_or_without_event_op(device='cpu')
