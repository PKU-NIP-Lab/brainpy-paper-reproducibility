# -*- coding: utf-8 -*-

import time
import brainpy as bp
import brainpy.math as bm
import numpy as np

bm.set_platform('cpu')

Cm = 200  # Membrane Capacitance [pF]
gl = 10.  # Leak Conductance   [nS]
g_Na = 20. * 1000
g_Kd = 6. * 1000  # K Conductance      [nS]
El = -60.  # Resting Potential [mV]
ENa = 50.  # reversal potential (Sodium) [mV]
EK = -90.  # reversal potential (Potassium) [mV]
VT = -63.
V_th = -20.
taue = 5.  # Excitatory synaptic time constant [ms]
taui = 10.  # Inhibitory synaptic time constant [ms]
Ee = 0.  # Excitatory reversal potential (mV)
Ei = -80.  # Inhibitory reversal potential (Potassium) [mV]
we = 6.  # excitatory synaptic conductance [nS]
wi = 67.  # inhibitory synaptic conductance [nS]


class HH(bp.dyn.NeuDyn):
  def __init__(self, size, method='exp_auto'):
    super(HH, self).__init__(size)

    # variables
    self.V = bm.Variable(El + (bm.random.randn(self.num) * 5 - 5))
    self.m = bm.Variable(bm.zeros(self.num))
    self.n = bm.Variable(bm.zeros(self.num))
    self.h = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))

    # functions
    self.integral = bp.odeint(bp.JointEq([self.dV, self.dm, self.dh, self.dn]), method=method)

  def dV(self, V, t, m, h, n, Isyn):
    gna = g_Na * (m * m * m) * h
    gkd = g_Kd * (n * n * n * n)
    dVdt = (-gl * (V - El) - gna * (V - ENa) - gkd * (V - EK) + Isyn) / Cm
    return dVdt

  def dm(self, m, t, V, ):
    m_alpha = 0.32 * (13 - V + VT) / (bm.exp((13 - V + VT) / 4) - 1.)
    m_beta = 0.28 * (V - VT - 40) / (bm.exp((V - VT - 40) / 5) - 1)
    dmdt = (m_alpha * (1 - m) - m_beta * m)
    return dmdt

  def dh(self, h, t, V):
    h_alpha = 0.128 * bm.exp((17 - V + VT) / 18)
    h_beta = 4. / (1 + bm.exp(-(V - VT - 40) / 5))
    dhdt = (h_alpha * (1 - h) - h_beta * h)
    return dhdt

  def dn(self, n, t, V):
    c = 15 - V + VT
    n_alpha = 0.032 * c / (bm.exp(c / 5) - 1.)
    n_beta = .5 * bm.exp((10 - V + VT) / 40)
    dndt = (n_alpha * (1 - n) - n_beta * n)
    return dndt

  def update(self, inp=0.):
    inp = self.sum_inputs(self.V.value, init=inp)  # sum projection inputs
    V, m, h, n = self.integral(self.V, self.m, self.h, self.n, bp.share['t'],
                               Isyn=inp, dt=bp.share['dt'])
    self.spike.value = bm.logical_and(self.V < V_th, V >= V_th)
    self.m.value = m
    self.h.value = h
    self.n.value = n
    self.V.value = V
    return self.spike.value


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


class COBA_HH_Net(bp.DynSysGroup):
  def __init__(self, scale=1., method='exp_auto'):
    super(COBA_HH_Net, self).__init__()
    self.num_exc = int(3200 * scale)
    self.num_inh = int(800 * scale)
    self.num = self.num_exc + self.num_inh

    self.N = HH(self.num, method=method)
    self.E = Exponential(self.num_exc, self.N, prob=80 / self.num, g_max=we, tau=taue, E=Ee)
    self.I = Exponential(self.num_inh, self.N, prob=80 / self.num, g_max=wi, tau=taui, E=Ei)

  def update(self, *args, **kwargs):
    self.E(self.N.spike[:self.num_exc])
    self.I(self.N.spike[self.num_exc:])
    self.N()


def run(scale, duration, res_dict=None):
  net = COBA_HH_Net(scale=scale)
  indices = np.arange(int(duration / bm.get_dt()))

  t0 = time.time()
  bm.for_loop(net.step_run, indices, progress_bar=False)
  t1 = time.time()
  print(f'Time {t1 - t0} s')
  # if res_dict is not None:
  #   res_dict['brainpy'].append({'num_neuron': runner.target.num,
  #                               'sim_len': duration,
  #                               'num_thread': 1,
  #                               'sim_time': t,
  #                               'dt': runner.dt})


if __name__ == '__main__':
  import json

  run(scale=4, res_dict=None, duration=1e4)

  # speed_res = {'brainpy': []}
  # for scale in [1, 2, 4, 6, 8, 10, 15, 20, 30]:
  # # for scale in [15, 20, 30]:
  #   for stim in [10. * 1e3]:
  #     run(scale=scale, res_dict=speed_res, duration=stim)
  #
  # with open('speed_results/brainpy-2.json', 'w') as f:
  #   json.dump(speed_res, f, indent=2)
