# -*- coding: utf-8 -*-
import sys
import time

import json
import brainpy as bp
import brainpy.math as bm
import numpy as np

taum = 20
taue = 5
taui = 10
Vt = -50
Vr = -60
El = -60
Erev_exc = 0.
Erev_inh = -80.
Ib = 20.
ref = 5.0
we = 0.6
wi = 6.7


class LIF(bp.dyn.NeuDyn):
    def __init__(self, size, V_init: callable, **kwargs):
        super(LIF, self).__init__(size=size, **kwargs)

        # parameters
        self.V_rest = Vr
        self.V_reset = El
        self.V_th = Vt
        self.tau = taum
        self.tau_ref = ref

        # variables
        self.V = bp.init.variable_(V_init, self.num)
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
        self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

    def update(self, inp):
        inp = self.sum_inputs(self.V.value, init=inp)  # sum all projection inputs
        refractory = (bp.share['t'] - self.t_last_spike) <= self.tau_ref
        V = self.V + (-self.V + self.V_rest + inp) / self.tau * bp.share['dt']
        V = bm.where(refractory, self.V, V)
        spike = self.V_th <= V
        self.t_last_spike.value = bm.where(spike, bp.share['t'], self.t_last_spike)
        self.V.value = bm.where(spike, self.V_reset, V)
        self.spike.value = spike
        return spike


class Exponential(bp.Projection):
    def __init__(self, num_pre, post, prob, g_max, tau, E):
        super().__init__()
        self.proj = bp.dyn.ProjAlignPostMg1(
            comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(prob, pre=num_pre, post=post.num), g_max),
            syn=bp.dyn.Expon.desc(post.num, tau=tau),
            out=bp.dyn.COBA.desc(E=E),
            post=post
        )

    def update(self, spk):
        self.proj.update(spk)


class COBA(bp.DynSysGroup):
    def __init__(self, scale, monitor=False):
        super().__init__()
        self.monitor = monitor
        self.num_exc = int(3200 * scale)
        self.num_inh = int(800 * scale)
        self.N = LIF(self.num_exc + self.num_inh, V_init=bp.init.Normal(-55., 5.))
        self.E = Exponential(self.num_exc, self.N, prob=80. / self.N.num, E=Erev_exc, g_max=we, tau=taue)
        self.I = Exponential(self.num_inh, self.N, prob=80. / self.N.num, E=Erev_inh, g_max=wi, tau=taui)

    def update(self, inp=Ib):
        self.E(self.N.spike[:self.num_exc])
        self.I(self.N.spike[self.num_exc:])
        self.N(inp)
        if self.monitor:
            return self.N.spike.value


def run_a_simulation(scale=10, duration=1e3, platform='cpu', x64=True, monitor=False):
    bm.set_platform(platform)
    bm.random.seed()
    if x64:
        bm.enable_x64()

    net = COBA(scale=scale, monitor=monitor)
    indices = np.arange(int(duration / bm.get_dt()))
    t0 = time.time()
    r = bm.for_loop(net.step_run, indices, progress_bar=False)
    t1 = time.time()

    # running
    if monitor:
        print(f'scale={scale}, size={net.N.num}, time = {t1 - t0} s, '
              f'firing rate = {r.sum() / net.N.num / duration * 1e3} Hz')
    else:
        print(f'scale={scale}, size={net.N.num}, time = {t1 - t0} s')
    bm.clear_buffer_memory(platform)
    bm.disable_x64()

    return net.N.num, t1 - t0, t1 - t0


def check_firing_rate():
    run_a_simulation(scale=1, duration=1e3, platform='cpu', x64=False, monitor=True)
    run_a_simulation(scale=2, duration=1e3, platform='cpu', x64=False, monitor=True)
    run_a_simulation(scale=4, duration=1e3, platform='cpu', x64=False, monitor=True)
    run_a_simulation(scale=6, duration=1e3, platform='cpu', x64=False, monitor=True)
    run_a_simulation(scale=8, duration=1e3, platform='cpu', x64=False, monitor=True)
    run_a_simulation(scale=10, duration=1e3, platform='cpu', x64=False, monitor=True)


def benchmark(duration=1000., platform='cpu', x64=True):
    final_results = dict()
    # for scale in [1, 2, 4, 6, 8, 10]:
    for scale in [20, 40, 60, 80, 100]:
        for _ in range(10):
            r = run_a_simulation(scale=scale, duration=duration, platform=platform, x64=x64)
            if r[0] not in final_results:
                final_results[r[0]] = {'exetime': [], 'runtime': []}
            final_results[r[0]]['exetime'].append(r[1])
            final_results[r[0]]['runtime'].append(r[2])

    postfix = 'x64' if x64 else 'x32'
    with open(f'speed_results/brainpy-COBA-{platform}-{postfix}.json', 'w') as fout:
        json.dump(final_results, fout, indent=2)


if __name__ == '__main__':
    x64 = False if sys.argv[1] == '0' else True
    print(sys.argv[1], x64)
    benchmark(duration=5. * 1e3, platform='gpu', x64=x64)
    # check_firing_rate()

