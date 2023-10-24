# -*- coding: utf-8 -*-

import time

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

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

  def update(self):
    _t = bp.share['t']
    _dt = bp.share['dt']
    refractory = (_t - self.t_last_spike) <= self.tau_ref
    V = self.V + (-self.V + self.V_rest + self.input) / self.tau * _dt
    V = bm.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)
    self.V.value = bm.where(spike, self.V_reset, V)
    self.spike.value = spike
    self.input[:] = Ib


class _DenseSynapse(bp.synapses.TwoEndConn):
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


class _EventSparseSynapse(bp.synapses.TwoEndConn):
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

  def update(self):
    spike = self.pre.spike
    syn_vs = bm.pre2post_event_sum(spike, self.pre2post, self.post.num, self.g_max)
    self.g.value = self.integral(self.g.value, bp.share['t'], bp.share['dt']) + syn_vs
    self.post.input += self.g * (self.E - self.post.V)


class COBA_Dense(bp.Network):
  def __init__(self, scale):
    super().__init__()

    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)
    we = 0.6  # excitatory synaptic weight (voltage)
    wi = 6.7  # inhibitory synaptic weight

    self.E = LIF(num_exc)
    self.I = LIF(num_inh)
    self.E.V[:] = bm.random.randn(self.E.num) * 5. - 55.
    self.I.V[:] = bm.random.randn(self.I.num) * 5. - 55.

    # synapses
    p = 80 / (num_inh + num_exc)
    self.E2E = _DenseSynapse(self.E, self.E, bp.conn.FixedProb(p), E=Erev_exc, g_max=we, tau=taue)
    self.E2I = _DenseSynapse(self.E, self.I, bp.conn.FixedProb(p), E=Erev_exc, g_max=we, tau=taue)
    self.I2E = _DenseSynapse(self.I, self.E, bp.conn.FixedProb(p), E=Erev_inh, g_max=wi, tau=taui)
    self.I2I = _DenseSynapse(self.I, self.I, bp.conn.FixedProb(p), E=Erev_inh, g_max=wi, tau=taui)

  @bm.cls_jit
  def neu_step(self, i):
    bp.share.save(t=i * bm.dt, i=i)
    self.E.update()
    self.I.update()

  @bm.cls_jit
  def net_step(self, i):
    bp.share.save(t=i * bm.dt, i=i)
    self.E2E.update()
    self.E2I.update()
    self.I2E.update()
    self.I2I.update()
    self.E.update()
    self.I.update()


class Coba_Sparse(bp.DynSysGroup):
  def __init__(self, scale):
    super().__init__()

    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)
    we = 0.6  # excitatory synaptic weight (voltage)
    wi = 6.7  # inhibitory synaptic weight

    self.E = LIF(num_exc)
    self.I = LIF(num_inh)
    self.E.V[:] = bm.random.randn(self.E.num) * 5. - 55.
    self.I.V[:] = bm.random.randn(self.I.num) * 5. - 55.

    # # synapses
    p = 80 / (num_inh + num_exc)
    self.E2E = _EventSparseSynapse(self.E, self.E, bp.conn.FixedProb(p), E=Erev_exc, g_max=we, tau=taue)
    self.E2I = _EventSparseSynapse(self.E, self.I, bp.conn.FixedProb(p), E=Erev_exc, g_max=we, tau=taue)
    self.I2E = _EventSparseSynapse(self.I, self.E, bp.conn.FixedProb(p), E=Erev_inh, g_max=wi, tau=taui)
    self.I2I = _EventSparseSynapse(self.I, self.I, bp.conn.FixedProb(p), E=Erev_inh, g_max=wi, tau=taui)

  @bm.cls_jit
  def neu_step(self, i):
    bp.share.save(t=i * bm.dt, i=i)
    self.E.update()
    self.I.update()

  @bm.cls_jit
  def net_step(self, i):
    bp.share.save(t=i * bm.dt, i=i)
    self.E2E.update()
    self.E2I.update()
    self.I2E.update()
    self.I2I.update()
    self.E.update()
    self.I.update()


def _init_net(scale, op):
  if op == 'dense':
    net = COBA_Dense(scale)
  elif op == 'event':
    net = Coba_Sparse(scale)
  else:
    raise ValueError
  return net


def separate_syn_net_sim_time(op='dense', platform='cpu', n_run=20, n_scale=100):
  results = {'num': [], 'network': [], 'neuron': []}
  bm.set_platform(platform)

  indices = np.arange(5000) + 1
  # scales = np.asarray([0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
  scales = np.asarray([3.0])

  for scale in scales[:n_scale]:
    for ni in range(n_run):
      bm.random.seed()

      net = _init_net(scale, op)
      net.neu_step(0)
      t0 = time.time()
      for i in indices:
        net.neu_step(i)
      t_neu = time.time() - t0

      net = _init_net(scale, op)
      net.net_step(0)
      t0 = time.time()
      for i in indices:
        net.net_step(i)
      t_net = time.time() - t0

      if ni == 0:
        results['num'].append(net.E.num + net.I.num)
      results['neuron'].append(t_neu)
      results['network'].append(t_net)

      print(f'scale = {scale}, neuron time = {t_neu}, network time = {t_net}')
      bm.clear_buffer_memory(platform)

  results['num'] = np.asarray(results['num'])
  results['neuron'] = np.asarray(results['neuron']).reshape(-1, n_run)
  results['network'] = np.asarray(results['network']).reshape(-1, n_run)

  np.savez(f'results/syn-vs-net-ratio-{op}-{platform}-v2.npz', **results)

  plt.plot(results['num'],
           (results['network'].mean(1) - results['neuron'].mean(1)) / results['network'].mean(1),
           linestyle="--",
           marker='v',
           linewidth=2)
  plt.show()


if __name__ == '__main__':
  separate_syn_net_sim_time(op='dense', platform='gpu')
