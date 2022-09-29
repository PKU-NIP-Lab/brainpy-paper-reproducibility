# -*- coding: utf-8 -*-

import time

import brainpy as bp
import torch

num_thread = 4
torch.set_num_threads(num_thread)

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


class LIF(object):
  def __init__(self, size):
    # parameters
    self.V_rest = Vr
    self.V_reset = El
    self.V_th = Vt
    self.tau = taum
    self.tau_ref = ref
    self.num = bp.tools.size2num(size)

    # variables
    self.V = torch.zeros(self.num)
    self.input = torch.zeros(self.num)
    self.spike = torch.zeros(self.num, dtype=torch.bool)
    self.t_last_spike = torch.ones(self.num) * -1e7

  def update(self, _t, _dt):
    refractory = (_t - self.t_last_spike) <= self.tau_ref
    V = self.V + (-self.V + self.V_rest + self.input) / self.tau * _dt
    V = torch.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike = torch.where(spike, _t, self.t_last_spike)
    self.V = torch.where(spike, torch.ones_like(V) * self.V_reset, V)
    self.spike = spike
    self.input[:] = Ib


class ExpCOBA(object):
  def __init__(self, pre, post, E, w, tau, prob):
    # parameters
    self.pre = pre
    self.post = post
    self.E = E
    self.tau = tau
    self.w = w
    self.conn = (torch.rand(pre.num, post.num) < prob).double()

    # variables
    self.g = torch.zeros(post.num)

  def update(self, _t, _dt):
    post_vs = (self.pre.spike.double() @ self.conn) * self.w
    self.g = self.g - self.g / self.tau * _dt + post_vs
    self.post.input += self.g * (self.E - self.post.V)


def run_net(scale, duration=1e3, res_dict=None):
  num_exc = int(3200 * scale)
  num_inh = int(800 * scale)
  we = 0.6 / scale  # excitatory synaptic weight (voltage)
  wi = 6.7 / scale  # inhibitory synaptic weight

  E = LIF(num_exc)
  I = LIF(num_inh)
  E.V[:] = torch.randn(E.num) * 5. - 55.
  I.V[:] = torch.randn(I.num) * 5. - 55.

  # # synapses
  E2E = ExpCOBA(E, E, E=Erev_exc, w=we, tau=taue, prob=0.02)
  E2I = ExpCOBA(E, I, E=Erev_exc, w=we, tau=taue, prob=0.02)
  I2E = ExpCOBA(I, E, E=Erev_inh, w=wi, tau=taui, prob=0.02)
  I2I = ExpCOBA(I, I, E=Erev_inh, w=wi, tau=taui, prob=0.02)

  # running
  dt = bp.math.get_dt()
  # spikes = []
  times = torch.arange(0, duration, dt)

  t0 = time.time()
  for t in times:
    E.update(t, dt)
    I.update(t, dt)
    E2E.update(t, dt)
    E2I.update(t, dt)
    I2E.update(t, dt)
    I2I.update(t, dt)
    # spikes.append(E.spike)
  t = time.time() - t0
  print(num_exc + num_inh, t)

  if res_dict is not None:
    res_dict['pytorch'].append({'num_neuron': num_exc + num_inh,
                                'num_thread': num_thread,
                                'sim_len': duration,
                                'sim_time': t,
                                'dt': 0.1})

  # spikes = torch.stack(spikes)
  # spikes = np.asarray(spikes)
  # bp.visualize.raster_plot(times.numpy(), spikes, show=True)


if __name__ == '__main__':
  import json

  speed_res = {'pytorch': []}
  for scale in [1, 2, 4, 6, 8, 10]:
    for stim in [5. * 1e3]:
      run_net(scale=scale, res_dict=speed_res, duration=stim)

  with open('speed_results/pytorch.json', 'w') as f:
    json.dump(speed_res, f, indent=2)
