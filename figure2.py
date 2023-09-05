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


class ExponSyn(bp.Projection):
    def __init__(self, pre, post, delay, prob, g_max, tau, E):
        super().__init__()
        self.proj = bp.dyn.ProjAlignPostMg2(
            pre,
            delay,
            bp.dnn.EventCSRLinear(FixedProb(prob, pre=pre.num, post=post.num), g_max),
            bp.dyn.Expon.desc(post.num, tau=tau),
            bp.dyn.COBA.desc(E=E),
            post,
        )


class EINet(bp.DynSysGroup):
    def __init__(self, num_E, num_I, gEE=0.03, gEI=0.03, gIE=0.335, gII=0.335):
        super().__init__()
        self.cur_inputs = bm.node_dict()
        self.E, self.I = HH(num_E), HH(num_I)
        self.E2E = ExponSyn(self.E, self.E, None, prob=0.02, g_max=gEE, tau=5., E=0.)
        self.E2I = ExponSyn(self.E, self.I, None, prob=0.02, g_max=gEI, tau=5., E=0.)
        self.I2E = ExponSyn(self.I, self.E, None, prob=0.02, g_max=gIE, tau=10., E=-80.)
        self.I2I = ExponSyn(self.I, self.I, None, prob=0.02, g_max=gII, tau=10., E=-80.)


class AreaConn(bp.Projection):
    def __init__(self, pre: EINet, post: EINet, delay, conn, gEE=0.03, gEI=0.03, tau=5.):
        super().__init__()
        self.E2E = ExponSyn(pre.E, post.E, delay, prob=conn, g_max=gEE, tau=tau, E=0.)
        self.E2I = ExponSyn(pre.E, post.I, delay, prob=conn, g_max=gEI, tau=tau, E=0.)


class VisualSystem(bp.DynSysGroup):
    def __init__(self, ne, ni, conn, delay, gEE=0.03, gEI=0.03, gIE=0.335, gII=0.335):
        super().__init__()
        num_area = conn.shape[0]

        # brain areas
        self.areas = bm.node_list()
        for _ in range(num_area):
            self.areas.append(EINet(ne, ni, gEE=gEE, gEI=gEI, gII=gII, gIE=gIE))

        # projections
        self.projections = bm.node_list()
        for i in range(num_area):
            for j in range(num_area):
                if i != j:
                    proj = AreaConn(self.areas[j],
                                    self.areas[i],
                                    delay=delay[i, j],
                                    conn=conn[i, j],
                                    gEE=gEE,
                                    gEI=gEI)
                    self.projections.append(proj)
