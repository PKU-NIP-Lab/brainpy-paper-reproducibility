# -*- coding: utf-8 -*-
import time

import brainpy as bp
import brainpy.math as bm
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
we = 0.6  # excitatory synaptic weight (voltage)
wi = 6.7  # inhibitory synaptic weight


class LIF(bp.dyn.NeuDyn):
  def __init__(self, size, V_init: callable, **kwargs):
    super(LIF, self).__init__(size=size, **kwargs)

    # parameters
    self.V_rest = Vr
    self.V_reset = El
    self.V_th = Vt
    self.tau = taum
    self.tau_ref = ref

    # variables
    self.V = bp.init.variable_(V_init, self.num)
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

  def update(self, inp):
    inp = self.sum_inputs(self.V.value, init=inp)  # sum all projection inputs
    refractory = (bp.share['t'] - self.t_last_spike) <= self.tau_ref
    V = self.V + (-self.V + self.V_rest + inp) / self.tau * bp.share['dt']
    V = bm.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike.value = bm.where(spike, bp.share['t'], self.t_last_spike)
    self.V.value = bm.where(spike, self.V_reset, V)
    self.spike.value = spike
    return spike


class Exponential(bp.Projection):
  def __init__(self, num_pre, post, prob, g_max, tau, E):
    super().__init__()

    self.proj = bp.dyn.ProjAlignPostMg1(
      comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(prob, pre=num_pre, post=post.num), g_max),
      syn=bp.dyn.Expon.desc(post.num, tau=tau),
      out=bp.dyn.COBA.desc(E=E),
      post=post
    )

  def update(self, spk):
    self.proj.update(spk)


class COBA(bp.DynSysGroup):
  def __init__(self, scale):
    super().__init__()
    self.num_exc = int(3200 * scale)
    self.num_inh = int(800 * scale)
    self.N = LIF(self.num_exc + self.num_inh, V_init=bp.init.Normal(-55., 5.))
    self.E = Exponential(self.num_exc, self.N, prob=80. / self.N.num, E=Erev_exc, g_max=we, tau=taue)
    self.I = Exponential(self.num_inh, self.N, prob=80. / self.N.num, E=Erev_inh, g_max=wi, tau=taui)

  def update(self, inp):
    self.E(self.N.spike[:self.num_exc])
    self.I(self.N.spike[self.num_exc:])
    self.N(inp)
    return self.N.spike.value


def run1(scale=10, duration=1e3):
  net = COBA(scale=scale)
  indices = np.arange(int(duration/ bm.get_dt()))

  t0 = time.time()
  bm.for_loop(lambda i: net.step_run(i, Ib), indices, progress_bar=False)
  t1 = time.time()

  # running
  print(f'size = {net.N.num}, running time = {t1 - t0} s')

  # runner = bp.dyn.DSRunner(net, monitors=['E.spike'])
  # t = runner.run(duration)
  # bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], show=True)


if __name__ == '__main__':
  # run1(1, duration=2e2)
  for s in [1, 2, 4, 6, 8, 10]:
    run1(s, duration=5e3)
