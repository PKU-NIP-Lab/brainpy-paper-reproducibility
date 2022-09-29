# -*- coding: utf-8 -*-


import brainpy as bp
import matplotlib.pyplot as plt

bp.math.enable_x64()


@bp.odeint
def int_V(V, t, w, Iext=0.):
  return V - V * V * V / 3 - w + Iext


@bp.odeint
def int_w(w, t, V, a=0.7, b=0.8, tau=12.5):
  return (V + a - b * w) / tau


def bifurcation_1d():
  plt.rcParams.update({"font.size": 15})

  plt.figure('V', figsize=(6, 4.5))
  plt.figure('w', figsize=(6, 4.5))
  analyzer = bp.analysis.Bifurcation2D(
    model=[int_V, int_w],
    target_vars={'V': [-3, 3], 'w': [-3., 3.]},
    target_pars={'Iext': [0., 2.]},
    resolutions={'Iext': 0.005},
  )
  analyzer.plot_bifurcation(num_rank=10)
  analyzer.plot_limit_cycle_by_sim()
  analyzer.show_figure()


def bifurcation_2d():
  plt.rcParams.update({"font.size": 15})

  analyzer = bp.analysis.Bifurcation2D(
    model=[int_V, int_w],
    target_vars=dict(V=[-3, 3], w=[-3., 3.]),
    target_pars=dict(a=[0.5, 1.], Iext=[0., 2.]),
    resolutions={'a': 0.005, 'Iext': 0.005},
    options={bp.analysis.C.y_by_x_in_fy: (lambda V, a=0.7, b=0.8: (V + a) / b)}
  )
  analyzer.plot_bifurcation()
  analyzer.show_figure()


if __name__ == '__main__':
  bifurcation_1d()
  bifurcation_2d()
