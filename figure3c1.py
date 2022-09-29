# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

bp.math.enable_x64()

# parameters
gamma = 0.641  # Saturation factor for gating variable
tau = 0.06  # Synaptic time constant [sec]
a = 270.
b = 108.
d = 0.154

JE = 0.3725  # self-coupling strength [nA]
JI = -0.1137  # cross-coupling strength [nA]
JAext = 0.00117  # Stimulus input strength [nA]

mu = 20.  # Stimulus firing rate [spikes/sec]
coh = 0.5  # Stimulus coherence [%]
Ib1 = 0.3297
Ib2 = 0.3297


@bp.odeint
def int_s1(s1, t, s2, coh=0.5, mu=20.):
  I1 = JE * s1 + JI * s2 + Ib1 + JAext * mu * (1. + coh)
  r1 = (a * I1 - b) / (1. - bm.exp(-d * (a * I1 - b)))
  return - s1 / tau + (1. - s1) * gamma * r1


@bp.odeint
def int_s2(s2, t, s1, coh=0.5, mu=20.):
  I2 = JE * s2 + JI * s1 + Ib2 + JAext * mu * (1. - coh)
  r2 = (a * I2 - b) / (1. - bm.exp(-d * (a * I2 - b)))
  return - s2 / tau + (1. - s2) * gamma * r2


def phase_plane_analysis():
  plt.rcParams.update({"font.size": 15})

  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
  fig.add_subplot(gs[0, 0])

  # phase plane analysis
  analyzer = bp.analysis.PhasePlane2D(
    model=[int_s1, int_s2],
    target_vars={'s1': [0, 1], 's2': [0, 1]},
    resolutions=0.001,
  )
  analyzer.plot_vector_field()
  analyzer.plot_nullcline(coords=dict(s2='s2-s1'),
                          x_style={'fmt': '-'},
                          y_style={'fmt': '-'})
  analyzer.plot_fixed_point()
  plt.legend().get_frame().set_facecolor('none')
  ax = plt.gca()
  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')
  plt.title('Phase Plane Analysis')
  analyzer.show_figure()


def bifurcation_analysis():
  # codimension 1 bifurcation
  analyzer = bp.analysis.Bifurcation2D(
    model=[int_s1, int_s2],
    target_vars={'s1': [0., 1.], 's2': [0., 1.]},
    target_pars={'coh': [0., 1.]},
    pars_update={'mu': 40.},
    resolutions={'coh': 0.005},
  )
  analyzer.plot_bifurcation(num_rank=50)
  analyzer.show_figure()


if __name__ == '__main__':
  phase_plane_analysis()
  bifurcation_analysis()

