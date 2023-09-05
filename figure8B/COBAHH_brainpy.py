# -*- coding: utf-8 -*-

import argparse
import json
import time

import brainpy as bp
import brainpy.math as bm
import numpy as np

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
        Isyn = self.sum_inputs(self.V.value, init=Isyn)  # sum projection inputs
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
            comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(prob, pre=num_pre, post=post.num, allow_multi_conn=True), g_max),
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

    net = COBA_HH_Net(scale=scale, monitor=monitor)
    indices = np.arange(int(duration / bm.get_dt()))

    t0 = time.time()
    r = bm.for_loop(net.step_run, indices, progress_bar=False)
    t1 = time.time()

    # running
    if monitor:
        print(f'scale={scale}, size={net.num}, time = {t1 - t0} s, '
              f'firing rate = {r.sum() / net.num / duration * 1e3} Hz')
    else:
        print(f'scale={scale}, size={net.num}, time = {t1 - t0} s')
    bm.disable_x64()
    bm.clear_buffer_memory(platform)
    return net.N.num, t1 - t0, t1 - t0


def check_firing_rate(x64=True, platform='cpu'):
    for scale in [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]:
        run_a_simulation(scale=scale, duration=5e3, platform=platform, x64=x64, monitor=True)


def benchmark(duration=1000., platform='cpu', x64=True):
    final_results = dict()
    # for scale in [1, 2, 4, 6, 8, 10]:
    # for scale in [20, 40, 60, 80, 100]:
    for scale in [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]:
        for _ in range(10):
            r = run_a_simulation(scale=scale, duration=duration, platform=platform, x64=x64)
            if r[0] not in final_results:
                final_results[r[0]] = {'exetime': [], 'runtime': []}
            final_results[r[0]]['exetime'].append(r[1])
            final_results[r[0]]['runtime'].append(r[2])

    postfix = 'x64' if x64 else 'x32'
    with open(f'speed_results/brainpy-COBAHH-{platform}-{postfix}.json', 'w') as fout:
        json.dump(final_results, fout, indent=2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-platform', default='cpu', help='platform')
    parser.add_argument('-x64', action='store_true')
    args = parser.parse_args()

    # benchmark(duration=5. * 1e3, platform=args.platform, x64=args.x64)
    check_firing_rate(x64=args.x64, platform=args.platform)
