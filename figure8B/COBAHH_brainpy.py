# -*- coding: utf-8 -*-

import argparse
import json
import time

import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt


bm.set_dt(0.1)

s = 1e-2
Cm = 200 * s  # Membrane Capacitance [pF]
gl = 10. * s  # Leak Conductance   [nS]
g_Na = 20. * 1000 * s
g_Kd = 6. * 1000 * s  # K Conductance      [nS]
El = -60.  # Resting Potential [mV]
ENa = 50.  # reversal potential (Sodium) [mV]
EK = -90.  # reversal potential (Potassium) [mV]
VT = -63.
V_th = -20.
taue = 5.  # Excitatory synaptic time constant [ms]
taui = 10.  # Inhibitory synaptic time constant [ms]
Ee = 0.  # Excitatory reversal potential (mV)
Ei = -80.  # Inhibitory reversal potential (Potassium) [mV]
we = 6. * s  # excitatory synaptic conductance [nS]
wi = 67. * s  # inhibitory synaptic conductance [nS]


class HH(bp.dyn.NeuDyn):
  def __init__(self, size, method='exp_auto'):
    super(HH, self).__init__(size)

    # variables
    self.V = bm.Variable(El + bm.random.randn(self.num) * 5 - 5.)
    self.m = bm.Variable(bm.zeros(self.num))
    self.n = bm.Variable(bm.zeros(self.num))
    self.h = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))

    # functions
    self.integral = bp.odeint(bp.JointEq(self.dV, self.dm, self.dh, self.dn), method=method)

  def dV(self, V, t, m, h, n, Isyn):
    Isyn = self.sum_current_inputs(self.V.value, init=Isyn)  # sum projection inputs
    gna = g_Na * (m * m * m) * h
    n2 = n * n
    gkd = g_Kd * (n2 * n2)
    dVdt = (-gl * (V - El) - gna * (V - ENa) - gkd * (V - EK) + Isyn) / Cm
    return dVdt

  def dm(self, m, t, V, ):
    if bm.float_ == bm.float64:
      m_alpha = 0.32 * (13 - V + VT) / (bm.exp((13 - V + VT) / 4) - 1.)
      m_beta = 0.28 * (V - VT - 40) / (bm.exp((V - VT - 40) / 5) - 1)
    else:
      m_alpha = 1.28 / bm.exprel((13 - V + VT) / 4)
      m_beta = 1.4 / bm.exprel((V - VT - 40) / 5)
    dmdt = (m_alpha * (1 - m) - m_beta * m)
    return dmdt

  def dh(self, h, t, V):
    h_alpha = 0.128 * bm.exp((17 - V + VT) / 18)
    h_beta = 4. / (1 + bm.exp(-(V - VT - 40) / 5))
    dhdt = (h_alpha * (1 - h) - h_beta * h)
    return dhdt

  def dn(self, n, t, V):
    if bm.float_ == bm.float64:
      c = 15 - V + VT
      n_alpha = 0.032 * c / (bm.exp(c / 5) - 1.)
    else:
      n_alpha = 0.16 / bm.exprel((15 - V + VT) / 5.)
    n_beta = 0.5 * bm.exp((10 - V + VT) / 40)
    dndt = (n_alpha * (1 - n) - n_beta * n)
    return dndt

  def update(self, inp=0.):
    V, m, h, n = self.integral(self.V, self.m, self.h, self.n, bp.share['t'], Isyn=inp, dt=bp.share['dt'])
    self.spike.value = bm.logical_and(self.V < V_th, V >= V_th)
    self.m.value = m
    self.h.value = h
    self.n.value = n
    self.V.value = V
    return self.spike.value


class Exponential(bp.Projection):
  def __init__(self, num_pre, post, prob, g_max, tau, E):
    super().__init__()

    self.proj = bp.dyn.HalfProjAlignPostMg(
      comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(prob, pre=num_pre, post=post.num, allow_multi_conn=True), g_max),
      # comm=bp.dnn.EventJitFPHomoLinear(num_pre, post.num, prob, g_max),
      syn=bp.dyn.Expon.desc(post.num, tau=tau),
      out=bp.dyn.COBA.desc(E=E),
      post=post
    )

  def update(self, spk):
    self.proj.update(spk)


class COBA_HH_Net(bp.DynSysGroup):
  def __init__(self, scale=1., method='exp_auto', monitor=False):
    super(COBA_HH_Net, self).__init__()
    self.monitor = monitor
    self.num_exc = int(3200 * scale)
    self.num_inh = int(800 * scale)
    self.num = self.num_exc + self.num_inh

    self.N = HH(self.num, method=method)
    self.E = Exponential(self.num_exc, self.N, prob=80 / self.num, g_max=we, tau=taue, E=Ee)
    self.I = Exponential(self.num_inh, self.N, prob=80 / self.num, g_max=wi, tau=taui, E=Ei)

  def update(self):
    self.E(self.N.spike[:self.num_exc])
    self.I(self.N.spike[self.num_exc:])
    self.N()
    if self.monitor:
      return self.N.spike.value


def run_a_simulation(scale=10, duration=1e3, platform='cpu', x64=True, monitor=False):
  bm.set_platform(platform)
  bm.random.seed()
  if x64:
    bm.enable_x64()
  else:
    bm.disable_x64()

  net = COBA_HH_Net(scale=scale, monitor=True)
  indices = np.arange(int(duration / bm.get_dt()))

  t0 = time.time()
  # if the network size is big, please turn on "progress_bar"
  # otherwise, the XLA may compute wrongly
  r = bm.for_loop(net.step_run, indices, progress_bar=False)
  t1 = time.time()

  if monitor:
    fig, gs = bp.visualize.get_figure(1, 1, 15, 30)
    r = bm.as_numpy(r)
    bp.visualize.raster_plot(indices, r, show=False)
    # plt.savefig(f'speed_results/COBAHH-scale={scale}.png')
    plt.show()

  # running
  if r is None:
    rate = 0.
  else:
    rate = bm.as_numpy(r).sum() / net.N.num / duration * 1e3

  print(f'scale={scale}, size={net.num}, time = {t1 - t0} s, firing rate = {rate} Hz')
  bm.disable_x64()
  bm.clear_buffer_memory(platform)
  return {'num': net.N.num,
          'exe_time': t1 - t0,
          'run_time': t1 - t0,
          'fr': rate}


def check_firing_rate(x64=True, platform='cpu'):
  for scale in [1, 2, 4, 6, 8, 10, 20, 30, 40, 50, 80, 100]:
  # for scale in [20, 30, 40, 50]:
    run_a_simulation(scale=scale, duration=2e3, platform=platform, x64=x64, monitor=False)


def check_nan(x64=True, platform='cpu', duration=2e3, n_time=4):
  bm.set_platform(platform)
  if x64:
    bm.enable_x64()
  else:
    bm.disable_x64()
  indices = np.arange(int(duration / bm.get_dt()))

  for scale in [1, 2, 4, 6, 8, 10, 20, 30, 40]:
    all_nan_nums = []
    for _ in range(n_time):
      net = COBA_HH_Net(scale=scale)
      bm.for_loop(net.step_run, indices, progress_bar=True)
      num_nan = np.count_nonzero(np.isnan(np.asarray(net.N.V.value)))
      bm.clear_buffer_memory(platform)
      all_nan_nums.append(num_nan)
    print(f'scale={scale}, size={net.num}, nans = {all_nan_nums}, mean = {np.mean(all_nan_nums)}')


def benchmark(duration=1000., platform='cpu', x64=True):
  postfix = 'x64' if x64 else 'x32'
  fn = f'speed_results/brainpy-COBAHH-{platform}-{postfix}.json'

  if platform == 'cpu':
    scales = [1, 2, 4, 6, 8, 10, 20]
  else:
    scales = [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]
  final_results = dict()
  for scale in scales:
    for _ in range(4):
      r = run_a_simulation(scale=scale, duration=duration, platform=platform, x64=x64, monitor=False)
      if r['num'] not in final_results:
        final_results[r['num']] = {'exetime': [], 'runtime': [], 'firing_rate': []}
      final_results[r['num']]['exetime'].append(r['exe_time'])
      final_results[r['num']]['runtime'].append(r['run_time'])
      final_results[r['num']]['firing_rate'].append(r['fr'])
    with open(fn, 'w') as fout:
      json.dump(final_results, fout, indent=2)


def visualize_spike_raster(duration=100., x64=True, platform='cpu'):
  bm.set_platform(platform)
  if x64:
    bm.enable_x64()
  name = 'x64' if x64 else 'x32'
  net = COBA_HH_Net(scale=1., monitor=True)
  indices = np.arange(int(duration / bm.get_dt()))
  r = bm.for_loop(net.step_run, indices, progress_bar=True)

  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.)
  ax = fig.add_subplot(gs[0])
  bp.visualize.raster_plot(indices * bm.get_dt(), r, ax=ax)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.title(f'BrainPy {platform.upper()} {name}')
  # plt.savefig(f'COBAHH-brainpy-{platform}-{name}.pdf')
  plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-platform', default='gpu', help='platform')
  parser.add_argument('-x64', action='store_true')
  args = parser.parse_args()

  # visualize_spike_raster(duration=100., platform=args.platform, x64=args.x64)
  # benchmark(duration=5. * 1e3, platform=args.platform, x64=args.x64)
  # check_nan(duration=5. * 1e3, platform=args.platform, x64=args.x64)
  check_firing_rate(x64=args.x64, platform=args.platform)
