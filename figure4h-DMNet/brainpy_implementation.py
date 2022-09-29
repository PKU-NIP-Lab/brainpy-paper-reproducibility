# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm

bp.check.turn_off()


class LIF(bp.dyn.NeuGroup):
  def __init__(self, size, V_L=-70., V_reset=-55., V_th=-50.,
               Cm=0.5, gL=0.025, t_refractory=2.):
    super(LIF, self).__init__(size=size)

    # parameters
    self.V_L = V_L
    self.V_reset = V_reset
    self.V_th = V_th
    self.Cm = Cm
    self.gL = gL
    self.t_refractory = t_refractory

    # variables
    self.V = bm.Variable(bm.ones(self.num) * V_L)
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

    # functions
    self.integral = bp.odeint(lambda V, t: (- self.gL * (V - self.V_L) + self.input) / self.Cm)

  def update(self, _t, _dt):
    ref = (_t - self.t_last_spike) <= self.t_refractory
    V = self.integral(self.V, _t, dt=_dt)
    V = bm.where(ref, self.V, V)
    spike = (V >= self.V_th)
    self.V.value = bm.where(spike, self.V_reset, V)
    self.spike.value = spike
    self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)
    self.refractory.value = bm.logical_or(spike, ref)
    self.input[:] = 0.


class PoissonStim(bp.dyn.NeuGroup):
  def __init__(self, size, freq_mean, freq_var, t_interval,
               pre_stimulus_period=100.,
               stimulus_period=1000.,
               **kwargs):
    super(PoissonStim, self).__init__(size=size, **kwargs)

    # parameters
    self.freq_mean = freq_mean
    self.freq_var = freq_var
    self.t_interval = t_interval
    self.pre_stimulus_period = pre_stimulus_period
    self.stimulus_period = stimulus_period
    self.dt = bm.get_dt() / 1000.

    # variables
    self.freq = bm.Variable(bm.zeros(1))
    self.freq_t_last_change = bm.Variable(bm.ones(1) * -1e7)
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.rng = bm.random.RandomState()

  def update(self, _t, _dt):
    in_interval = bm.logical_and(self.pre_stimulus_period < _t,
                                 _t < self.pre_stimulus_period + self.stimulus_period)
    prev_freq = bm.where(in_interval, self.freq[0], 0.)
    in_interval = bm.logical_and(in_interval, (_t - self.freq_t_last_change[0]) >= self.t_interval)
    self.freq[0] = bm.where(in_interval, self.rng.normal(self.freq_mean, self.freq_var), prev_freq)
    self.freq_t_last_change[0] = bm.where(in_interval, _t, self.freq_t_last_change[0])
    self.spike.value = self.rng.random(self.num) < self.freq[0] * self.dt


class DecisionMaking(bp.dyn.Network):
  def __init__(self, scale=1., mu0=40., coherence=25.6, dt=0.1,
               pre_stimulus_period=100., stimulus_period=1000.):
    super(DecisionMaking, self).__init__()

    f = 0.15
    num_exc = int(1600 * scale)
    num_inh = int(400 * scale)
    num_A = int(f * num_exc)
    num_B = int(f * num_exc)
    num_N = num_exc - num_A - num_B
    print(f'Total network size: {num_exc + num_inh}')
    self.num = num_exc + num_inh

    poisson_freq = 2400.  # Hz
    w_pos = 1.7
    w_neg = 1. - f * (w_pos - 1.) / (1. - f)
    g_ext2E_AMPA = 2.1  # nS
    g_ext2I_AMPA = 1.62  # nS
    g_E2E_AMPA = 0.05 / scale / 1e3  # nS
    g_E2E_NMDA = 0.165 / scale / 1e3  # nS
    g_E2I_AMPA = 0.04 / scale / 1e3  # nS
    g_E2I_NMDA = 0.13 / scale / 1e3  # nS
    g_I2E_GABAa = 1.3 / scale / 1e3  # nS
    g_I2I_GABAa = 1.0 / scale / 1e3  # nS

    ampa_par = dict(delay_step=int(0.5 / dt), E=0., tau=2.0)
    gaba_par = dict(delay_step=int(0.5 / dt), E=-70., tau=5.0)
    nmda_par = dict(delay_step=int(0.5 / dt), tau_decay=100, tau_rise=2., E=0., cc_Mg=1., a=0.5)

    # E neurons/pyramid neurons
    A = LIF(num_A, Cm=500. / 1e3, gL=25. / 1e3, t_refractory=2.)
    B = LIF(num_B, Cm=500. / 1e3, gL=25. / 1e3, t_refractory=2.)
    N = LIF(num_N, Cm=500. / 1e3, gL=25. / 1e3, t_refractory=2.)

    # I neurons/interneurons
    I = LIF(num_inh, Cm=200. / 1e3, gL=20. / 1e3, t_refractory=1.)

    # poisson stimulus
    IA = PoissonStim(num_A, freq_var=10., t_interval=50.,
                     freq_mean=mu0 + mu0 / 100. * coherence,
                     pre_stimulus_period=pre_stimulus_period,
                     stimulus_period=stimulus_period)
    IB = PoissonStim(num_B, freq_var=10., t_interval=50.,
                     freq_mean=mu0 - mu0 / 100. * coherence,
                     pre_stimulus_period=pre_stimulus_period,
                     stimulus_period=stimulus_period)

    # noise neurons
    self.noise_A = bp.dyn.PoissonGroup(num_A, freqs=poisson_freq)
    self.noise_B = bp.dyn.PoissonGroup(num_B, freqs=poisson_freq)
    self.noise_N = bp.dyn.PoissonGroup(num_N, freqs=poisson_freq)
    self.noise_I = bp.dyn.PoissonGroup(num_inh, freqs=poisson_freq)

    # define external inputs
    self.IA2A = bp.dyn.ExpCOBA(IA, A, bp.conn.One2One(), g_max=g_ext2E_AMPA, **ampa_par)
    self.IB2B = bp.dyn.ExpCOBA(IB, B, bp.conn.One2One(), g_max=g_ext2E_AMPA, **ampa_par)

    # define E2E conn
    self.A2A_AMPA = bp.dyn.ExpCOBA(A, A, bp.conn.All2All(), g_max=g_E2E_AMPA * w_pos, **ampa_par)
    self.A2A_NMDA = bp.dyn.NMDA(A, A, bp.conn.All2All(), g_max=g_E2E_NMDA * w_pos, **nmda_par)

    self.A2B_AMPA = bp.dyn.ExpCOBA(A, B, bp.conn.All2All(), g_max=g_E2E_AMPA * w_neg, **ampa_par)
    self.A2B_NMDA = bp.dyn.NMDA(A, B, bp.conn.All2All(), g_max=g_E2E_NMDA * w_neg, **nmda_par)

    self.A2N_AMPA = bp.dyn.ExpCOBA(A, N, bp.conn.All2All(), g_max=g_E2E_AMPA, **ampa_par)
    self.A2N_NMDA = bp.dyn.NMDA(A, N, bp.conn.All2All(), g_max=g_E2E_NMDA, **nmda_par)

    self.B2A_AMPA = bp.dyn.ExpCOBA(B, A, bp.conn.All2All(), g_max=g_E2E_AMPA * w_neg)
    self.B2A_NMDA = bp.dyn.NMDA(B, A, bp.conn.All2All(), g_max=g_E2E_NMDA * w_neg, **nmda_par)

    self.B2B_AMPA = bp.dyn.ExpCOBA(B, B, bp.conn.All2All(), g_max=g_E2E_AMPA * w_pos, **ampa_par)
    self.B2B_NMDA = bp.dyn.NMDA(B, B, bp.conn.All2All(), g_max=g_E2E_NMDA * w_pos, **nmda_par)

    self.B2N_AMPA = bp.dyn.ExpCOBA(B, N, bp.conn.All2All(), g_max=g_E2E_AMPA, **ampa_par)
    self.B2N_NMDA = bp.dyn.NMDA(B, N, bp.conn.All2All(), g_max=g_E2E_NMDA, **nmda_par)

    self.N2A_AMPA = bp.dyn.ExpCOBA(N, A, bp.conn.All2All(), g_max=g_E2E_AMPA * w_neg, **ampa_par)
    self.N2A_NMDA = bp.dyn.NMDA(N, A, bp.conn.All2All(), g_max=g_E2E_NMDA * w_neg, **nmda_par)

    self.N2B_AMPA = bp.dyn.ExpCOBA(N, B, bp.conn.All2All(), g_max=g_E2E_AMPA * w_neg, **ampa_par)
    self.N2B_NMDA = bp.dyn.NMDA(N, B, bp.conn.All2All(), g_max=g_E2E_NMDA * w_neg, **nmda_par)

    self.N2N_AMPA = bp.dyn.ExpCOBA(N, N, bp.conn.All2All(), g_max=g_E2E_AMPA, **ampa_par)
    self.N2N_NMDA = bp.dyn.NMDA(N, N, bp.conn.All2All(), g_max=g_E2E_NMDA, **nmda_par)

    # define E2I conn
    self.A2I_AMPA = bp.dyn.ExpCOBA(A, I, bp.conn.All2All(), g_max=g_E2I_AMPA, **ampa_par)
    self.A2I_NMDA = bp.dyn.NMDA(A, I, bp.conn.All2All(), g_max=g_E2I_NMDA, **nmda_par)

    self.B2I_AMPA = bp.dyn.ExpCOBA(B, I, bp.conn.All2All(), g_max=g_E2I_AMPA, **ampa_par)
    self.B2I_NMDA = bp.dyn.NMDA(B, I, bp.conn.All2All(), g_max=g_E2I_NMDA, **nmda_par)

    self.N2I_AMPA = bp.dyn.ExpCOBA(N, I, bp.conn.All2All(), g_max=g_E2I_AMPA, **ampa_par)
    self.N2I_NMDA = bp.dyn.NMDA(N, I, bp.conn.All2All(), g_max=g_E2I_NMDA, **nmda_par)

    # define I2E conn
    self.I2A = bp.dyn.ExpCOBA(I, A, bp.conn.All2All(), g_max=g_I2E_GABAa, **gaba_par)
    self.I2B = bp.dyn.ExpCOBA(I, B, bp.conn.All2All(), g_max=g_I2E_GABAa, **gaba_par)
    self.I2N = bp.dyn.ExpCOBA(I, N, bp.conn.All2All(), g_max=g_I2E_GABAa, **gaba_par)

    # define I2I conn
    self.I2I = bp.dyn.ExpCOBA(I, I, bp.conn.All2All(), g_max=g_I2I_GABAa, **gaba_par)

    # define external projections
    self.noise2A = bp.dyn.ExpCOBA(self.noise_A, A, bp.conn.One2One(), g_max=g_ext2E_AMPA, **ampa_par)
    self.noise2B = bp.dyn.ExpCOBA(self.noise_B, B, bp.conn.One2One(), g_max=g_ext2E_AMPA, **ampa_par)
    self.noise2N = bp.dyn.ExpCOBA(self.noise_N, N, bp.conn.One2One(), g_max=g_ext2E_AMPA, **ampa_par)
    self.noise2I = bp.dyn.ExpCOBA(self.noise_I, I, bp.conn.One2One(), g_max=g_ext2I_AMPA, **ampa_par)

    # nodes
    self.A = A
    self.B = B
    self.N = N
    self.I = I
    self.IA = IA
    self.IB = IB

  def update(self, _t, _dt):
    nodes = self.nodes(level=1, include_self=False)
    nodes = nodes.subset(bp.dyn.DynamicalSystem).unique()
    for node in nodes.values():
      node.update(_t, _dt)


def build_and_run(scale, file=None,
                  pre_stimulus_period=500.,
                  stimulus_period=1000.,
                  delay_period=500.,
                  progress_bar=False):
  net = DecisionMaking(scale=scale,
                       pre_stimulus_period=pre_stimulus_period,
                       stimulus_period=stimulus_period)

  runner = bp.DSRunner(net,
                       # monitors=['A.spike', 'B.spike', 'IA.freq', 'IB.freq'],
                       dyn_vars=net.vars(),
                       progress_bar=progress_bar)
  total_period = pre_stimulus_period + stimulus_period + delay_period
  t = runner(total_period)

  if file is not None:
    file.write(f'scale={scale}, num={net.num}, time={t}\n')
    file.flush()
  print(f'Used time: {t} s')


if __name__ == '__main__':
  import sys

  if len(sys.argv) == 1:
    platform = 'cpu'
    bp.math.set_platform('cpu')
  else:
    if sys.argv[1] == 'gpu':
      platform = 'gpu'
      bp.math.set_platform('gpu')
    else:
      raise ValueError
  name = f'brainpy-v2-{platform}'

  # for scale in [a / 4 for a in range(1, 41, 2)]:
  #   build_and_run(scale=scale)

  build_and_run(scale=(1e5 / 2e3), file=None, progress_bar=True)

  # with open(f'speed_results/{name}.txt', 'w') as fout:
  #   for size in [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 2e6]:
  #     scale = size / 2e3
  #     build_and_run(scale=scale, file=fout)
