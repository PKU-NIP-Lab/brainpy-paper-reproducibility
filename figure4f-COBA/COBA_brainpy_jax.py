# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm

assert bp.__version__ > '2.2.0'

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
    super(LIF, self).__init__(size=size, **kwargs)

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
    refractory = (tdi.t - self.t_last_spike) <= self.tau_ref
    V = self.V + (-self.V + self.V_rest + self.input) / self.tau * tdi.dt
    V = bm.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike.value = bm.where(spike, tdi.t, self.t_last_spike)
    self.V.value = bm.where(spike, self.V_reset, V)
    self.spike.value = spike
    self.input[:] = Ib


class ExpCOBA(bp.dyn.TwoEndConn):
  def __init__(self, pre, post, conn, E, w, tau, **kwargs):
    super(ExpCOBA, self).__init__(pre, post, conn=conn, **kwargs)

    # parameters
    self.E = E
    self.tau = tau
    self.w = w
    self.pre2post = self.conn.requires('pre2post')

    # variables
    self.g = bm.Variable(bm.zeros(post.num))  # variables

  def update(self, tdi):
    syn_vs = bm.pre2post_event_sum(self.pre.spike, self.pre2post, self.post.num, self.w)
    self.g.value = self.g - self.g / self.tau * tdi.dt + syn_vs
    self.post.input += self.g * (self.E - self.post.V)


def run1(scale=10, duration=1e3):
  num_exc = int(3200 * scale)
  num_inh = int(800 * scale)
  we = 0.6 / scale  # excitatory synaptic weight (voltage)
  wi = 6.7 / scale  # inhibitory synaptic weight

  E = LIF(num_exc)
  I = LIF(num_inh)
  E.V[:] = bm.random.randn(E.num) * 5. - 55.
  I.V[:] = bm.random.randn(I.num) * 5. - 55.

  # # synapses
  E2E = ExpCOBA(E, E, bp.conn.FixedProb(0.02), E=Erev_exc, w=we, tau=taue)
  E2I = ExpCOBA(E, I, bp.conn.FixedProb(0.02), E=Erev_exc, w=we, tau=taue)
  I2E = ExpCOBA(I, E, bp.conn.FixedProb(0.02), E=Erev_inh, w=wi, tau=taui)
  I2I = ExpCOBA(I, I, bp.conn.FixedProb(0.02), E=Erev_inh, w=wi, tau=taui)

  # running
  net = bp.Network(E2E, E2I, I2I, I2E, E=E, I=I)
  runner = bp.dyn.DSRunner(net)
  t = runner.run(duration)
  print(f'size = {num_exc + num_inh}, running time = {t} s')

  # runner = bp.dyn.DSRunner(net, monitors=['E.spike'])
  # t = runner.run(duration)
  # bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], show=True)


if __name__ == '__main__':
  run1(1, duration=2e2)
  # for s in [1, 2, 4, 6, 8, 10]:
  #   run1(s, duration=5e3)
