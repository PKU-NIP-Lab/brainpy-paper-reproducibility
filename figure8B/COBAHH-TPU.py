import time

import brainpy as bp
import brainpy.math as bm
import numpy as np
import jax

# bm.set_host_device_count(4)
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
  def __init__(self, size, method='exp_auto', sharding=None):
    super(HH, self).__init__(size, sharding=sharding)

    # variables
    self.V = self.init_variable(lambda s: El + bm.random.randn(*s) * 5 - 5., None)
    self.m = self.init_variable(bm.zeros, None)
    self.n = self.init_variable(bm.zeros, None)
    self.h = self.init_variable(bm.zeros, None)
    self.spike = self.init_variable(lambda s: bm.zeros(s, dtype=bool), None)

    # functions
    self.integral = bp.odeint(bp.JointEq(self.dV, self.dm, self.dh, self.dn), method=method)

  def dV(self, V, t, m, h, n, Isyn):
    Isyn = self.sum_inputs(self.V.value, init=Isyn)  # sum projection inputs
    gna = g_Na * (m * m * m) * h
    gkd = g_Kd * (n * n * n * n)
    dVdt = (-gl * (V - El) - gna * (V - ENa) - gkd * (V - EK) + Isyn) / Cm
    return dVdt

  def dm(self, m, t, V):
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
    V, m, h, n = self.integral(self.V, self.m, self.h, self.n, bp.share['t'],
                               Isyn=inp, dt=bp.share['dt'])
    self.spike.value = bm.logical_and(self.V < V_th, V >= V_th)
    self.m.value = m
    self.h.value = h
    self.n.value = n
    self.V.value = V
    return self.spike.value


class RandomLinear(bp.dnn.Layer):
  def __init__(self, num_pre, num_post, prob, weight):
    super().__init__()
    self.weight = weight
    self.prob = prob
    self.num_pre = num_pre
    self.num_post = num_post
    self.keys = bm.random.DEFAULT.split_keys(num_post)
    print('Using random linear')

  def update(self, x):
    def _f(key):
      return bm.inner(x, bm.random.random(self.num_pre, key=key) < self.prob)

    return bm.for_loop(_f, self.keys) * self.weight


class MaskedLinear(bp.dnn.Layer):
  def __init__(self, num_pre, num_post, prob, weight, sharding=None):
    super().__init__()
    print('Using masked linear')
    self.weight = weight
    f = bm.jit(
        lambda key: jax.random.bernoulli(key, prob, (num_pre, num_post)),
        out_shardings=bm.sharding.get_sharding(sharding),
    )
    self.mask = f(bm.random.split_key())

  def update(self, x):
    return (x @ self.mask) * self.weight


class IndLinear(bp.dnn.Layer):
  def __init__(self, num_pre, num_post, prob, weight, sharding=None):
    super().__init__()
    self.weight = weight
    f = bm.jit(
      lambda key: jax.random.randint(key, (int(num_pre * prob), num_post), 0, num_pre, ),
      out_shardings=bm.sharding.get_sharding(sharding),
    )
    self.indices = f(bm.random.split_key())

  def update(self, x):
    x = bm.asarray(x, dtype=float)
    r = jax.vmap(lambda ids: bm.sum(x[ids]), in_axes=1)(self.indices)
    return r * self.weight


class Exponential(bp.Projection):
  def __init__(self, num_pre, post, prob, g_max, tau, E):
    super().__init__()
    self.proj = bp.dyn.ProjAlignPostMg1(
      # comm=RandomLinear(num_pre, post.num, prob, g_max),
      # comm=bp.dnn.MaskedLinear(bp.conn.FixedProb(prob, pre=num_pre, post=post.num), g_max,
      #                          sharding=[None, bm.sharding.NEU_AXIS]),
      comm=IndLinear(num_pre, post.num, prob, g_max, sharding=[None, bm.sharding.NEU_AXIS]),
      syn=bp.dyn.Expon.desc(post.num, tau=tau, sharding=[bm.sharding.NEU_AXIS]),
      out=bp.dyn.COBA.desc(E=E),
      post=post
    )

  def update(self, spk):
    spk = bm.asarray(spk, dtype=float)
    self.proj.update(spk)


class COBA_HH_Net(bp.DynSysGroup):
  def __init__(self, scale=1., method='exp_auto', monitor=False):
    super(COBA_HH_Net, self).__init__()
    self.monitor = monitor
    self.num_exc = int(3200 * scale)
    self.num_inh = int(800 * scale)
    self.num = self.num_exc + self.num_inh

    self.N = HH(self.num, method=method, sharding=[bm.sharding.NEU_AXIS])
    self.E = Exponential(self.num_exc, self.N, prob=80 / self.num, g_max=we, tau=taue, E=Ee)
    self.I = Exponential(self.num_inh, self.N, prob=80 / self.num, g_max=wi, tau=taui, E=Ei)

  def update(self):
    self.E(self.N.spike[:self.num_exc])
    self.I(self.N.spike[self.num_exc:])
    self.N()
    if self.monitor:
      return self.N.spike.value


def run_a_simulation2(scale=10, duration=1e3, platform='cpu', x64=True, monitor=False):
  bm.set_platform(platform)
  bm.random.seed()
  if x64:
    bm.enable_x64()

  net = COBA_HH_Net(scale=scale, monitor=monitor)

  @bm.jit
  def run(indices):
    return bm.for_loop(net.step_run, indices, progress_bar=False)

  indices = np.arange(int(duration / bm.get_dt()))
  t0 = time.time()
  r = jax.block_until_ready(run(indices))
  t1 = time.time()
  print(f'first run time = {t1 - t0} s')

  indices = np.arange(int(duration / bm.get_dt()), int(duration / bm.get_dt()) * 2)
  t2 = time.time()
  r = jax.block_until_ready(run(indices))
  t3 = time.time()
  jax.debug.visualize_array_sharding(r)
  print(f'second run time = {t3 - t2} s')

  # running
  if monitor:
    r = jax.device_put(r, jax.devices('cpu')[0])
    r = bm.as_numpy(r)
    print(f'scale={scale}, size={net.num}, first run time = {t1 - t0} s, second run time = {t3 - t2} s, '
          f'firing rate = {r.sum() / net.num / duration * 1e3} Hz')
  else:
    print(f'scale={scale}, size={net.num}, first run time = {t1 - t0} s, second run time = {t3 - t2} s')
  bm.disable_x64()
  bm.clear_buffer_memory(platform)
  return net.N.num, t1 - t0, t3 - t2


with bm.sharding.device_mesh(jax.devices(), [bm.sharding.NEU_AXIS]):
  run_a_simulation2(scale=1, duration=5e3, platform='cpu', x64=False, monitor=True)

