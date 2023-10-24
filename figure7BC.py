# -*- coding: utf-8 -*-

import json

import brainpy as bp
import brainpy.math as bm
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
    self.conn_mat = BCOO.fromdense(bm.as_jax(conn_mat))
    self.g_max = g_max

    # variables
    self.g = bm.Variable(bm.zeros((self.post.num,)))

    # functions
    self.integral = bp.odeint(lambda g, t: -g / self.tau)

  def update(self):
    post_vs = self.pre.spike @ self.conn_mat
    self.g.value = self.integral(self.g.value, bp.share['t'], bp.share['dt'])
    self.g.value += post_vs * self.g_max
    self.post.input += self.g.value * (self.E - self.post.V)


class ExpEventSparse(bp.synapses.TwoEndConn):
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
    syn_vs = bm.pre2post_event_sum(self.pre.spike, self.pre2post, self.post.num, self.g_max)
    self.g.value = self.integral(self.g.value, bp.share['t'], bp.share['dt']) + syn_vs
    self.post.input += self.g * (self.E - self.post.V)


class CobaDense(bp.DynSysGroup):
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


class CobaSparse(bp.DynSysGroup):
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


class CobaEventSparse(bp.DynSysGroup):
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


def compare_with_or_without_event_op(duration=1e3, check=False, n_run=20, res_file=None, platform='cpu',
                                     type_='event'):
  bm.set_platform(platform)

  if type_ == 'event':
    cls = CobaEventSparse
  elif type_ == 'sparse':
    cls = CobaSparse
  elif type_ == 'dense':
    cls = CobaDense
  else:
    raise TypeError

  setting = dict(progress_bar=False)
  if check:
    setting = dict(progress_bar=True, monitors=['E.spike'])

  results = dict()
  for scale in [0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]:
    for _ in range(n_run):
      bm.random.seed()
      net = cls(scale)
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


if __name__ == '__main__':
  compare_with_or_without_event_op(res_file='results/coba-event-op.json', platform='cpu', type_='event')
  compare_with_or_without_event_op(res_file='results/coba-sparse-op.json', platform='cpu', type_='sparse')
  compare_with_or_without_event_op(res_file='results/coba-dense-op.json', platform='cpu', type_='dense')
