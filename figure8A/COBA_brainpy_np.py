import brainpy as bp

assert '1.1.0' < bp.__version__ < '2.0.0'

bp.math.use_backend('numpy')

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


class LIF(bp.NeuGroup):
  target_backend = 'numpy'

  def __init__(self, size, **kwargs):
    super(LIF, self).__init__(size=size, **kwargs)

    self.V = bp.math.Variable(bp.math.ones(size) * Vr)
    self.Isyn = bp.math.Variable(bp.math.zeros(size))
    self.t_spike = bp.math.Variable(-1e7 * bp.math.ones(size))
    self.spike = bp.math.Variable(bp.math.zeros(size, dtype=bool))

    self.integral = bp.odeint(self.derivative)

  def derivative(self, V, t, Isyn):
    return (Isyn + (El - V) + Ib) / taum

  def update(self, _t, _dt):
    for i in range(self.num):
      self.spike[i] = 0.
      if (_t - self.t_spike[i]) > ref:
        V = self.integral(self.V[i], _t, self.Isyn[i])
        self.spike[i] = 0.
        if V >= Vt:
          self.V[i] = Vr
          self.spike[i] = 1.
          self.t_spike[i] = _t
        else:
          self.V[i] = V
    self.Isyn[:] = 0.


# %%
class SynVec(bp.TwoEndConn):
  target_backend = 'numpy'

  def __init__(self, pre, post, conn, E, w, tau, **kwargs):
    super(SynVec, self).__init__(pre, post, conn=conn, **kwargs)

    # parameters
    self.E = E
    self.w = w
    self.tau = tau

    self.pre2post = self.conn.requires('pre2post')  # connections
    self.g = bp.math.Variable(bp.math.zeros(post.num))  # variables

    self.integral = bp.odeint(self.derivative)

  def derivative(self, g, t):
    dg = - g / self.tau
    return dg

  def update(self, _t, _dt):
    self.g[:] = self.integral(self.g, _t)
    for pre_id in range(self.pre.num):
      if self.pre.spike[pre_id]:
        post_ids = self.pre2post[pre_id]
        self.g[post_ids] += self.w
    self.post.Isyn += self.g * (self.E - self.post.V)


class SynMat(bp.TwoEndConn):
  target_backend = 'numpy'

  def __init__(self, pre, post, conn, E, w, tau, **kwargs):
    super(SynMat, self).__init__(pre, post, conn=conn, **kwargs)

    # parameters
    self.E = E
    self.w = w
    self.tau = tau

    self.conn_mat = self.conn.requires('conn_mat')  # connections
    self.g = bp.math.Variable(bp.math.zeros(post.num))  # variables

    self.integral = bp.odeint(self.derivative)

  def derivative(self, g, t):
    dg = - g / self.tau
    return dg

  def update(self, _t, _dt):
    self.g[:] = self.integral(self.g, _t)
    for pre_id in range(self.pre.num):
      if self.pre.spike[pre_id]:
        for post_id in range(self.post.num):
          if self.conn_mat[pre_id, post_id]:
            self.g[post_id] += self.w
    self.post.Isyn += self.g * (self.E - self.post.V)


def run(SYnModel, scale=10, duration=1000., res_dict=None):
  num_exc = int(3200 * scale)
  num_inh = int(800 * scale)
  we = 0.6 / scale  # excitatory synaptic weight (voltage)
  wi = 6.7 / scale  # inhibitory synaptic weight

  E = LIF(num_exc)
  I = LIF(num_inh)
  E.V[:] = bp.math.random.randn(num_exc) * 5. - 55.
  I.V[:] = bp.math.random.randn(num_inh) * 5. - 55.

  E2E = SYnModel(pre=E, post=E, conn=bp.connect.FixedProb(prob=0.02), E=Erev_exc, w=we, tau=taue)
  E2I = SYnModel(pre=E, post=I, conn=bp.connect.FixedProb(prob=0.02), E=Erev_exc, w=we, tau=taue)
  I2E = SYnModel(pre=I, post=E, conn=bp.connect.FixedProb(prob=0.02), E=Erev_inh, w=wi, tau=taui)
  I2I = SYnModel(pre=I, post=I, conn=bp.connect.FixedProb(prob=0.02), E=Erev_inh, w=wi, tau=taui)
  net = bp.Network(E, I, E2E, E2I, I2I, I2E)
  net = bp.math.jit(net)

  t = net.run(duration, report=0.1)
  print(SYnModel, t)
  if res_dict is not None:
    res_dict['brainpy-np'].append({'num_neuron': num_exc + num_inh,
                                   'sim_len': duration,
                                   'num_thread': 1,
                                   'sim_time': t,
                                   'dt': 0.1})


if __name__ == '__main__':
  run(SynMat, scale=4., duration=5e3)
  run(SynVec, scale=4., duration=5e3)


if __name__ == '__main__1':
  import json

  speed_res = {'brainpy-np': []}
  for scale in [1, 2, 4, 6, 8, 10]:
    for stim in [5. * 1e3]:
      run(scale=scale, res_dict=speed_res, duration=stim)

  with open('speed_results/brainpy-np.json', 'w') as f:
    json.dump(speed_res, f, indent=2)
