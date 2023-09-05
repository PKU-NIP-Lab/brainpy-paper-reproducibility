# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt


class PoissonStim(bp.NeuGroup):
  def __init__(self, size, freq_mean, freq_var, t_interval,
               pre_stimulus_period=500., stimulus_period=1000.,
               delay_period=500.):
    super().__init__(size=size)

    # parameters
    self.freq_mean = freq_mean
    self.freq_var = freq_var
    self.t_interval = t_interval
    self.pre_stimulus_period = pre_stimulus_period
    self.stimulus_period = stimulus_period
    self.delay_period = delay_period

    self.reset_state(self.mode)

  def reset_state(self, batch_size=None):
    self.freq = bp.init.variable(bm.zeros, batch_size, 1)
    self.freq_t_last_change = bp.init.variable(lambda s: bm.ones(s) * -1e7, batch_size, 1)
    self.spike = bp.init.variable(lambda s: bm.zeros(s, dtype=bool), batch_size, self.varshape)

  def update(self):
    t = bp.share['t']
    in_interval = bm.logical_and(self.pre_stimulus_period < t,
                                 t < self.pre_stimulus_period + self.stimulus_period)
    in_interval = bm.ones_like(self.freq, dtype=bool) * in_interval
    prev_freq = bm.where(in_interval, self.freq, 0.)
    in_interval = bm.logical_and(in_interval, (t - self.freq_t_last_change) >= self.t_interval)
    self.freq.value = bm.where(in_interval, bm.random.normal(self.freq_mean, self.freq_var, self.freq.shape), prev_freq)
    self.freq_t_last_change.value = bm.where(in_interval, t, self.freq_t_last_change)
    self.spike.value = bm.random.rand_like(self.spike) < self.freq * bp.share['dt'] / 1000.
    return self.spike

  def return_info(self):
    return self.spike


class Exponential(bp.Projection):
  def __init__(self, pre, post, conn, delay, g_max, tau, E):
    super().__init__()
    if isinstance(conn, bp.conn.All2All):
      comm = bp.dnn.AllToAll(pre.num, post.num, g_max)
    elif isinstance(conn, bp.conn.One2One):
      comm = bp.dnn.OneToOne(pre.num, g_max)
    else:
      raise ValueError
    syn = bp.dyn.Expon.desc(post.num, tau=tau)
    out = bp.dyn.COBA.desc(E=E)
    self.proj = bp.dyn.ProjAlignPostMg2(
      pre=pre, delay=delay, comm=comm,
      syn=syn, out=out, post=post
    )

  def update(self, *args, **kwargs):
    self.proj()


class NMDA(bp.Projection):
  def __init__(self, pre, post, conn, delay, g_max):
    super().__init__()
    if isinstance(conn, bp.conn.All2All):
      comm = bp.dnn.AllToAll(pre.num, post.num, g_max)
    elif isinstance(conn, bp.conn.One2One):
      comm = bp.dnn.OneToOne(pre.num, g_max)
    else:
      raise ValueError
    syn = bp.dyn.NMDA.desc(pre.num, a=0.5, tau_decay=100., tau_rise=2.)
    out = bp.dyn.MgBlock(E=0., cc_Mg=1.0)
    self.proj = bp.dyn.ProjAlignPreMg2(
      pre=pre, delay=delay, syn=syn,
      comm=comm, out=out, post=post
    )

  def update(self, *args, **kwargs):
    self.proj()


class DecisionMakingNet(bp.DynSysGroup):
  def __init__(self, scale=1., mu0=40., coherence=25.6, f=0.15,
               pre_stimulus_period=100., stimulus_period=1000., delay_period=500.):
    super().__init__()

    num_exc = int(1600 * scale)
    num_I = int(400 * scale)
    num_A = int(f * num_exc)
    num_B = int(f * num_exc)
    num_N = num_exc - num_A - num_B
    print(f'Total network size: {num_exc + num_I}')

    poisson_freq = 2400.  # Hz
    w_pos = 1.7
    w_neg = 1. - f * (w_pos - 1.) / (1. - f)
    g_ext2E_AMPA = 2.1  # nS
    g_ext2I_AMPA = 1.62  # nS
    g_E2E_AMPA = 0.05 / scale  # nS
    g_E2I_AMPA = 0.04 / scale  # nS
    g_E2E_NMDA = 0.165 / scale  # nS
    g_E2I_NMDA = 0.13 / scale  # nS
    g_I2E_GABAa = 1.3 / scale  # nS
    g_I2I_GABAa = 1.0 / scale  # nS

    neu_par = dict(V_rest=-70., V_reset=-55., V_th=-50., V_initializer=bp.init.OneInit(-70.))
    stim_par = dict(freq_var=10., t_interval=50., pre_stimulus_period=pre_stimulus_period,
                    stimulus_period=stimulus_period, delay_period=delay_period)

    # E neurons/pyramid neurons
    A = bp.dyn.LifRef(num_A, tau=20., R=0.04, tau_ref=2., **neu_par)
    B = bp.dyn.LifRef(num_B, tau=20., R=0.04, tau_ref=2., **neu_par)
    N = bp.dyn.LifRef(num_N, tau=20., R=0.04, tau_ref=2., **neu_par)

    # I neurons/interneurons
    I = bp.dyn.LifRef(num_I, tau=10., R=0.05, tau_ref=1., **neu_par)

    # poisson stimulus
    IA = PoissonStim(num_A, freq_mean=mu0 + mu0 / 100. * coherence, **stim_par)
    IB = PoissonStim(num_B, freq_mean=mu0 - mu0 / 100. * coherence, **stim_par)

    # noise neurons
    self.noise_B = bp.dyn.PoissonGroup(num_B, freqs=poisson_freq)
    self.noise_A = bp.dyn.PoissonGroup(num_A, freqs=poisson_freq)
    self.noise_N = bp.dyn.PoissonGroup(num_N, freqs=poisson_freq)
    self.noise_I = bp.dyn.PoissonGroup(num_I, freqs=poisson_freq)

    # define external inputs
    self.IA2A = Exponential(IA, A, bp.conn.One2One(), delay=None, g_max=g_ext2E_AMPA, tau=2., E=0.)
    self.IB2B = Exponential(IB, B, bp.conn.One2One(), delay=None, g_max=g_ext2E_AMPA, tau=2., E=0.)

    # define E->E/I conn
    self.N2B_AMPA = Exponential(N, B, bp.conn.All2All(), delay=0.5, g_max=g_E2E_AMPA * w_neg, tau=2., E=0.)
    self.N2A_AMPA = Exponential(N, A, bp.conn.All2All(), delay=0.5, g_max=g_E2E_AMPA * w_neg, tau=2., E=0.)
    self.N2N_AMPA = Exponential(N, N, bp.conn.All2All(), delay=0.5, g_max=g_E2E_AMPA, tau=2., E=0.)
    self.N2I_AMPA = Exponential(N, I, bp.conn.All2All(), delay=0.5, g_max=g_E2I_AMPA, tau=2., E=0.)
    self.N2B_NMDA = NMDA(N, B, bp.conn.All2All(), delay=0.5, g_max=g_E2E_NMDA * w_neg)
    self.N2A_NMDA = NMDA(N, A, bp.conn.All2All(), delay=0.5, g_max=g_E2E_NMDA * w_neg)
    self.N2N_NMDA = NMDA(N, N, bp.conn.All2All(), delay=0.5, g_max=g_E2E_NMDA)
    self.N2I_NMDA = NMDA(N, I, bp.conn.All2All(), delay=0.5, g_max=g_E2I_NMDA)

    self.B2B_AMPA = Exponential(B, B, bp.conn.All2All(), delay=0.5, g_max=g_E2E_AMPA * w_pos, tau=2., E=0.)
    self.B2A_AMPA = Exponential(B, A, bp.conn.All2All(), delay=0.5, g_max=g_E2E_AMPA * w_neg, tau=2., E=0.)
    self.B2N_AMPA = Exponential(B, N, bp.conn.All2All(), delay=0.5, g_max=g_E2E_AMPA, tau=2., E=0.)
    self.B2I_AMPA = Exponential(B, I, bp.conn.All2All(), delay=0.5, g_max=g_E2I_AMPA, tau=2., E=0.)
    self.B2B_NMDA = NMDA(B, B, bp.conn.All2All(), delay=0.5, g_max=g_E2E_NMDA * w_pos)
    self.B2A_NMDA = NMDA(B, A, bp.conn.All2All(), delay=0.5, g_max=g_E2E_NMDA * w_neg)
    self.B2N_NMDA = NMDA(B, N, bp.conn.All2All(), delay=0.5, g_max=g_E2E_NMDA)
    self.B2I_NMDA = NMDA(B, I, bp.conn.All2All(), delay=0.5, g_max=g_E2I_NMDA)

    self.A2B_AMPA = Exponential(A, B, bp.conn.All2All(), delay=0.5, g_max=g_E2E_AMPA * w_neg, tau=2., E=0.)
    self.A2A_AMPA = Exponential(A, A, bp.conn.All2All(), delay=0.5, g_max=g_E2E_AMPA * w_pos, tau=2., E=0.)
    self.A2N_AMPA = Exponential(A, N, bp.conn.All2All(), delay=0.5, g_max=g_E2E_AMPA, tau=2., E=0.)
    self.A2I_AMPA = Exponential(A, I, bp.conn.All2All(), delay=0.5, g_max=g_E2I_AMPA, tau=2., E=0.)
    self.A2B_NMDA = NMDA(A, B, bp.conn.All2All(), delay=0.5, g_max=g_E2E_NMDA * w_neg)
    self.A2A_NMDA = NMDA(A, A, bp.conn.All2All(), delay=0.5, g_max=g_E2E_NMDA * w_pos)
    self.A2N_NMDA = NMDA(A, N, bp.conn.All2All(), delay=0.5, g_max=g_E2E_NMDA)
    self.A2I_NMDA = NMDA(A, I, bp.conn.All2All(), delay=0.5, g_max=g_E2I_NMDA)

    # define I->E/I conn
    self.I2B = Exponential(I, B, bp.conn.All2All(), delay=0.5, g_max=g_I2E_GABAa, tau=5., E=-70.)
    self.I2A = Exponential(I, A, bp.conn.All2All(), delay=0.5, g_max=g_I2E_GABAa, tau=5., E=-70.)
    self.I2N = Exponential(I, N, bp.conn.All2All(), delay=0.5, g_max=g_I2E_GABAa, tau=5., E=-70.)
    self.I2I = Exponential(I, I, bp.conn.All2All(), delay=0.5, g_max=g_I2I_GABAa, tau=5., E=-70.)

    # define external projections
    self.noise2B = Exponential(self.noise_B, B, bp.conn.One2One(), delay=None, g_max=g_ext2E_AMPA, tau=2., E=0.)
    self.noise2A = Exponential(self.noise_A, A, bp.conn.One2One(), delay=None, g_max=g_ext2E_AMPA, tau=2., E=0.)
    self.noise2N = Exponential(self.noise_N, N, bp.conn.One2One(), delay=None, g_max=g_ext2E_AMPA, tau=2., E=0.)
    self.noise2I = Exponential(self.noise_I, I, bp.conn.One2One(), delay=None, g_max=g_ext2I_AMPA, tau=2., E=0.)

    # nodes
    self.B = B
    self.A = A
    self.N = N
    self.I = I
    self.IA = IA
    self.IB = IB


def visualize_results(axes, mon, t_start=0., title=None, pre_stimulus_period=500.,
                      stimulus_period=1000., delay_period=500.):
  total_period = pre_stimulus_period + stimulus_period + delay_period
  ax = axes[0]
  bp.visualize.raster_plot(mon['ts'], mon['A.spike'], markersize=1, ax=ax)
  if title:
    ax.set_title(title)
  ax.set_ylabel("Group A")
  ax.set_xlim(t_start, total_period + 1)
  ax.axvline(pre_stimulus_period, linestyle='dashed')
  ax.axvline(pre_stimulus_period + stimulus_period, linestyle='dashed')
  ax.axvline(pre_stimulus_period + stimulus_period + delay_period, linestyle='dashed')

  ax = axes[1]
  bp.visualize.raster_plot(mon['ts'], mon['B.spike'], markersize=1, ax=ax)
  ax.set_ylabel("Group B")
  ax.set_xlim(t_start, total_period + 1)
  ax.axvline(pre_stimulus_period, linestyle='dashed')
  ax.axvline(pre_stimulus_period + stimulus_period, linestyle='dashed')
  ax.axvline(pre_stimulus_period + stimulus_period + delay_period, linestyle='dashed')

  ax = axes[2]
  rateA = bp.measure.firing_rate(mon['A.spike'], width=10.)
  rateB = bp.measure.firing_rate(mon['B.spike'], width=10.)
  ax.plot(mon['ts'], rateA, label="Group A")
  ax.plot(mon['ts'], rateB, label="Group B")
  ax.set_ylabel('Population activity [Hz]')
  ax.set_xlim(t_start, total_period + 1)
  ax.axvline(pre_stimulus_period, linestyle='dashed')
  ax.axvline(pre_stimulus_period + stimulus_period, linestyle='dashed')
  ax.axvline(pre_stimulus_period + stimulus_period + delay_period, linestyle='dashed')
  ax.legend()

  ax = axes[3]
  ax.plot(mon['ts'], mon['IA.freq'], label="group A")
  ax.plot(mon['ts'], mon['IB.freq'], label="group B")
  ax.set_ylabel("Input activity [Hz]")
  ax.set_xlim(t_start, total_period + 1)
  ax.axvline(pre_stimulus_period, linestyle='dashed')
  ax.axvline(pre_stimulus_period + stimulus_period, linestyle='dashed')
  ax.axvline(pre_stimulus_period + stimulus_period + delay_period, linestyle='dashed')
  ax.legend()
  ax.set_xlabel("Time [ms]")


def single_run():
  pre_stimulus_period = 100.
  stimulus_period = 1000.
  delay_period = 500.
  total_period = pre_stimulus_period + stimulus_period + delay_period

  net = DecisionMakingNet(coherence=-80., mu0=50.)
  runner = bp.DSRunner(net, monitors=['A.spike', 'B.spike', 'IA.freq', 'IB.freq', 'IA.spike'])
  runner.run(total_period)

  fig, gs = bp.visualize.get_figure(4, 1, 3, 10)
  axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]
  visualize_results(axes,
                    mon=runner.mon, pre_stimulus_period=pre_stimulus_period,
                    stimulus_period=stimulus_period, delay_period=delay_period)
  plt.show()


if __name__ == '__main__':
  # build_and_run(scale=(1e5/2e3), file=None, progress_bar=True)

  single_run()
