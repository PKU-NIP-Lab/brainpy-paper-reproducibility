import brainpy as bp
import brainpy.math as bm
import numpy as np
from jax.core import ShapedArray
import numba


def abs_eval(data, indices, indptr, vector, shape):
  return [ShapedArray((shape[1],), data.dtype)]


@numba.njit(fastmath=True)
def sparse_op(outs, ins):
  res_val = outs[0]
  res_val.fill(0)
  values, col_indices, row_ptr, vector, shape = ins
  values = values[()]

  for row_i in range(shape[0]):
    if vector[row_i]:
      for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
        res_val[col_indices[j]] += values


event_op = bm.XLACustomOp(eval_shape=abs_eval, con_compute=sparse_op)


class EventMV(bp.DynamicalSystem):
  def __init__(self, n_in, n_out, g_max, prob):
    super().__init__()
    self.n_in = n_in
    self.n_out = n_out
    self.g_max = g_max
    self.prob = prob
    conn = bp.conn.FixedProb(prob=prob, pre=n_in, post=n_out)
    self.indices, self.inptr = conn.require('csr')

  def update(self, spike):
    r = event_op(self.g_max, self.indices, self.inptr, spike, shape=(self.n_in, self.n_out))
    return r[0]


class Exponential(bp.Projection):
  def __init__(self, num_pre, post, prob, g_max, tau, E):
    super().__init__()
    self.proj = bp.dyn.ProjAlignPostMg1(
      comm=EventMV(num_pre, post.num, g_max, prob),
      syn=bp.dyn.Expon.desc(post.num, tau=tau),
      out=bp.dyn.COBA.desc(E=E),
      post=post
    )


class COBA(bp.DynSysGroup):
  def __init__(self, scale=1.0):
    super().__init__()
    self.num_exc = int(3200 * scale)
    self.num_inh = int(800 * scale)
    self.N = bp.dyn.LifRef(self.num_exc + self.num_inh, V_rest=-60.,
                           V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                           V_initializer=bp.init.Normal(-55., 2.), method='exp_auto')
    self.E = Exponential(self.num_exc, self.N, prob=80. / self.N.num, E=0., g_max=0.6, tau=5.)
    self.I = Exponential(self.num_inh, self.N, prob=80. / self.N.num, E=-80., g_max=6.7, tau=10.)

  def update(self, inp=20.):
    self.E(self.N.spike[:self.num_exc])
    self.I(self.N.spike[self.num_exc:])
    self.N(inp)
    return self.N.spike.value


bm.set_platform('cpu')
net = COBA(scale=1.)
indices = np.arange(int(100. / bm.get_dt()))
sps = bm.for_loop(net.step_run, indices, progress_bar=True)
bp.visualize.raster_plot(indices * bm.get_dt(), sps, show=True)


