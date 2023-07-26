import brainpy as bp
import brainpy.math as bm
import jax


neu_pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                V_initializer=bp.init.Normal(-55., 2.))


class EICOBA_PostAlign(bp.DynamicalSystem):
  def __init__(self, num_exc, num_inh, inp=20.):
    super().__init__()
    self.inp = inp

    self.E = bp.dyn.LifRefLTC(num_exc, **neu_pars)
    self.I = bp.dyn.LifRefLTC(num_inh, **neu_pars)

    self.E2E = bp.dyn.ProjAlignPostMg2(
      pre=self.E,
      delay=None,
      comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(0.02, pre=self.E.num, post=self.E.num), 0.6),
      syn=bp.dyn.Expon.desc(self.E.varshape, tau=5.),
      out=bp.dyn.COBA.desc(E=0.),
      post=self.E,
    )
    self.E2I = bp.dyn.ProjAlignPostMg2(
      pre=self.E,
      delay=None,
      comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(0.02, pre=self.E.num, post=self.I.num), 0.6),
      syn=bp.dyn.Expon.desc(self.I.varshape, tau=5.),
      out=bp.dyn.COBA.desc(E=0.),
      post=self.I,
    )
    self.I2E = bp.dyn.ProjAlignPostMg2(
      pre=self.I,
      delay=None,
      comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(0.02, pre=self.I.num, post=self.E.num), 6.7),
      syn=bp.dyn.Expon.desc(self.E.varshape, tau=10.),
      out=bp.dyn.COBA.desc(E=-80.),
      post=self.E,
    )
    self.I2I = bp.dyn.ProjAlignPostMg2(
      pre=self.I,
      delay=None,
      comm=bp.dnn.EventCSRLinear(bp.conn.FixedProb(0.02, pre=self.I.num, post=self.I.num), 6.7),
      syn=bp.dyn.Expon.desc(self.I.varshape, tau=10.),
      out=bp.dyn.COBA.desc(E=-80.),
      post=self.I,
    )

  def update(self):
    self.E2I()
    self.I2I()
    self.I2E()
    self.E2E()
    self.E(self.inp)
    self.I(self.inp)


def single_run(inp):
  net = EICOBA_PostAlign(3200, 800, inp=inp)
  runner = bp.DSRunner(net,
                       monitors={'E.spike': net.E.spike},
                       numpy_mon_after_run=False)
  runner.run(100.)
  return runner.mon


def batch_run_using_jax_vmap():
  inputs = bp.math.asarray([10., 15., 20., 25.])
  batches = jax.vmap(single_run)(inputs)
  for i, inp in enumerate(inputs):
    bp.visualize.raster_plot(batches['ts'][i], batches['E.spike'][i], title=str(inp), show=True)


def batch_run_using_jax_pmap():
  bm.set_host_device_count(4)
  inputs = bp.math.asarray([10., 15., 20., 25.])
  batches = jax.pmap(single_run)(inputs)
  for i, inp in enumerate(inputs):
    bp.visualize.raster_plot(batches['ts'][i], batches['E.spike'][i], title=str(inp), show=True)


def intrinsic_batch_run():
  inputs = bm.expand_dims(bm.asarray([10., 15., 20., 25.]), axis=1)
  with bm.environment(mode=bm.BatchingMode(4)):
    net = EICOBA_PostAlign(3200, 800, inp=inputs)
  runner = bp.DSRunner(net, monitors={'E.spike': net.E.spike}, numpy_mon_after_run=False)
  runner.run(100.)
  for i, inp in enumerate(inputs):
    bp.visualize.raster_plot(runner.mon['ts'], runner.mon['E.spike'][i], title=str(inp), show=True)


if __name__ == '__main__':
  pass
  # batch_run_using_jax_vmap()
  batch_run_using_jax_pmap()
  # intrinsic_batch_run()

