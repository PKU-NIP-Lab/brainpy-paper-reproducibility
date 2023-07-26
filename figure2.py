# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm
from brainpy.connect import FixedProb


class IK(bp.dyn.IonChannel):
  master_type = bp.dyn.HHTypedNeuron

  def __init__(self, size, E=-90., g_max=10., phi=1., V_sh=-50., method='exp_auto'):
    super().__init__(size)
    self.g_max, self.E, self.V_sh, self.phi = g_max, E, V_sh, phi
    self.p = bm.Variable(bm.zeros(size))
    self.integral = bp.odeint(self.dp, method=method)

  def dp(self, p, t, V):
    tmp = V - self.V_sh - 15.
    alpha = 0.032 * tmp / (1. - bm.exp(-tmp / 5.))
    beta = 0.5 * bm.exp(-(V - self.V_sh - 10.) / 40.)
    return self.phi * (alpha * (1. - p) - beta * p)

  def update(self, V):
    self.p.value = self.integral(self.p, bp.share['t'], V)

  def current(self, V):
    return self.g_max * self.p ** 4 * (self.E - V)


class HH(bp.dyn.CondNeuGroupLTC):
  def __init__(self, size, V_initializer=bp.init.Uniform(-70, -50.), method='exp_auto'):
    super().__init__(size, V_initializer=V_initializer)
    self.IK = IK(size, g_max=30., V_sh=-63., method=method)
    self.INa = bp.dyn.INa_TM1991(size, g_max=100., V_sh=-63., method=method)
    self.IL = bp.dyn.IL(size, E=-60., g_max=0.05)


class EINet(bp.Network, bp.mixin.ReceiveInputProj):
  def __init__(self, num_E, num_I, gEE=0.03, gEI=0.03, gIE=0.335, gII=0.335):
    super().__init__()
    self.E, self.I = HH(num_E), HH(num_I)
    self.E2E = bp.dyn.ProjAlignPostMg2(
      pre=self.E,
      delay=None,
      comm=bp.dnn.EventCSRLinear(FixedProb(0.02, pre=num_E, post=num_E), gEE),
      syn=bp.dyn.Expon.desc(num_E, tau=5.),
      out=bp.dyn.COBA.desc(E=0.),
      post=self.E
    )
    self.E2I = bp.dyn.ProjAlignPostMg2(
      self.E,
      None,
      bp.dnn.EventCSRLinear(FixedProb(0.02, pre=num_E, post=num_I), gEI),
      bp.dyn.Expon.desc(num_I, tau=5.),
      bp.dyn.COBA.desc(E=0.),
      self.I
    )
    self.I2E = bp.dyn.ProjAlignPostMg2(
      self.I,
      None,
      bp.dnn.EventCSRLinear(FixedProb(0.02, pre=num_I, post=num_E), gIE),
      bp.dyn.Expon.desc(num_E, tau=10.),
      bp.dyn.COBA.desc(E=-80.),
      self.E
    )
    self.I2I = bp.dyn.ProjAlignPostMg2(
      self.I,
      None,
      bp.dnn.EventCSRLinear(FixedProb(0.02, pre=num_I, post=num_I), gII),
      bp.dyn.Expon.desc(num_I, tau=10.),
      bp.dyn.COBA.desc(E=-80.),
      self.I
    )



class Projection(bp.DynSysGroup):
  def __init__(self, pre: EINet, post: EINet, delay, conn, gEE=0.03, gEI=0.03, tau=5.):
    super().__init__()
    self.E2E = bp.dyn.ProjAlignPostMg2(
      pre=pre.E,
      delay=delay * bm.get_dt(),
      comm=bp.dnn.EventCSRLinear(FixedProb(0.02, pre=pre.E.num, post=post.E.num), gEE * conn),
      syn=bp.dyn.Expon.desc(post.E.num, tau=tau),
      out=bp.dyn.COBA.desc(E=0.),
      post=post.E,
    )
    self.E2I = bp.dyn.ProjAlignPostMg2(
      pre=pre.E,
      delay=delay * bm.get_dt(),
      comm=bp.dnn.EventCSRLinear(FixedProb(0.02, pre=pre.E.num, post=post.I.num), gEI * conn),
      syn=bp.dyn.Expon.desc(post.I.num, tau=tau),
      out=bp.dyn.COBA.desc(E=-80.),
      post=post.I,
    )


class VisualSystem(bp.DynSysGroup):
  def __init__(self, ne, ni, conn, delay, gEE=0.03, gEI=0.03, gIE=0.335, gII=0.335):
    super().__init__()
    num_area = conn.shape[0]

    areas = [EINet(ne, ni, gEE=gEE, gEI=gEI, gII=gII, gIE=gIE)
             for _ in range(num_area)]
    self.areas = bm.node_list(areas)
    self.projections = bm.node_list()
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
