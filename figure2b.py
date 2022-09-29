# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm
import numpy as np
from brainpy.connect import FixedProb
from brainpy.dyn.channels import INa_TM1991, IL
from brainpy.dyn.synapses import Exponential
from brainpy.dyn.synouts import COBA

comp_method = 'sparse'
area_names = ['V1', 'V2', 'V4', 'TEO', 'TEpd']
data = np.load('./data/visual_conn.npz')
conn_data = data['conn']
delay_data = (data['delay'] / bm.get_dt()).astype(int)
num_exc = 3200
num_inh = 800


class IK(bp.dyn.Channel):
  def __init__(self, size, E=-90., g_max=10., phi=1., V_sh=-50., method='exp_euler'):
    super(IK, self).__init__(size)
    self.g_max, self.E, self.V_sh, self.phi = g_max, E, V_sh, phi
    self.p = bm.Variable(bm.zeros(size))
    self.integral = bp.odeint(self.dp, method=method)

  def dp(self, p, t, V):
    tmp = V - self.V_sh - 15.
    alpha = 0.032 * tmp / (1. - bm.exp(-tmp / 5.))
    beta = 0.5 * bm.exp(-(V - self.V_sh - 10.) / 40.)
    return self.phi * (alpha * (1. - p) - beta * p)

  def update(self, tdi, V):
    self.p.value = self.integral(self.p, tdi.t, V, dt=tdi.dt)

  def current(self, V):
    return self.g_max * self.p ** 4 * (self.E - V)


class HH(bp.dyn.CondNeuGroup):
  def __init__(self, size, V_initializer=bp.init.Uniform(-70, -50.), method='exp_auto'):
    super(HH, self).__init__(size, V_initializer=V_initializer)
    self.IK = IK(size, g_max=30., V_sh=-63., method=method)
    self.INa = INa_TM1991(size, g_max=100., V_sh=-63., method=method)
    self.IL = IL(size, E=-60., g_max=0.05)


class Network(bp.dyn.Network):
  def __init__(self, num_E, num_I, gEE=0.03, gEI=0.03, gIE=0.335, gII=0.335):
    super(Network, self).__init__()
    self.E, self.I = HH(num_E), HH(num_I)
    self.E2E = Exponential(self.E, self.E, FixedProb(0.02),
                           g_max=gEE,
                           tau=5,
                           output=COBA(E=0.),
                           comp_method=comp_method)
    self.E2I = Exponential(self.E, self.I, FixedProb(0.02),
                           g_max=gEI,
                           tau=5.,
                           output=COBA(E=0.),
                           comp_method=comp_method)
    self.I2E = Exponential(self.I, self.E, FixedProb(0.02),
                           g_max=gIE,
                           tau=10.,
                           output=COBA(E=-80),
                           comp_method=comp_method)
    self.I2I = Exponential(self.I, self.I, FixedProb(0.02),
                           g_max=gII,
                           tau=10.,
                           output=COBA(E=-80.),
                           comp_method=comp_method)


class Projection(bp.dyn.DynamicalSystem):
  def __init__(self, pre, post, delay, conn, gEE=0.03, gEI=0.03, tau=5.):
    super(Projection, self).__init__()
    self.E2E = Exponential(pre.E, post.E, bp.conn.FixedProb(0.02),
                           delay_step=delay,
                           g_max=gEE * conn,
                           tau=tau,
                           output=COBA(0.),
                           comp_method=comp_method)
    self.E2I = Exponential(pre.E, post.I, bp.conn.FixedProb(0.02),
                           delay_step=delay,
                           g_max=gEI * conn,
                           tau=tau,
                           output=COBA(0.),
                           comp_method=comp_method)

  def update(self, tdi):
    self.E2E.update(tdi)
    self.E2I.update(tdi)


class System(bp.dyn.System):
  def __init__(self, conn, delay, gEE=0.03, gEI=0.03, gIE=0.335, gII=0.335):
    super(System, self).__init__()

    num_area = conn.shape[0]
    self.areas = [Network(num_exc, num_inh, gEE=gEE, gEI=gEI, gII=gII, gIE=gIE)
                  for _ in range(num_area)]
    self.projections = []
    for i in range(num_area):
      for j in range(num_area):
        if i != j:
          proj = Projection(self.areas[j],
                            self.areas[i],
                            delay=delay[i, j],
                            conn=conn[i, j],
                            gEE=gEE,
                            gEI=gEI)
          self.projections.append(proj)
    self.register_implicit_nodes(self.projections, self.areas)

