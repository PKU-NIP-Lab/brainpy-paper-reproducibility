# -*- coding: utf-8 -*-


import os
import time

import brainpy as bp
import brainpy.math as bm
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Limit jax multithreading
# https://github.com/google/jax/issues/1539
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREAD"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

taum = 20
taue = 5
taui = 10
Vt = -50
Vr = -60
El = -60
Ib = 20.
ref = 5.0


class LIF(bp.BrainPyObject):
  def __init__(self, size, **kwargs):
    super().__init__(**kwargs)

    # parameters
    self.num = size
    self.V_rest = Vr
    self.V_reset = El
    self.V_th = Vt
    self.tau = taum
    self.tau_ref = ref

    # variables
    self.V = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

    self.data = dict(V=self.V.value,
                     spike=self.spike.value,
                     t_last_spike=self.t_last_spike.value)

  def update(self, _t, _dt, inp=0.):
    refractory = (_t - self.t_last_spike) <= self.tau_ref
    V = self.V + (-self.V + self.V_rest + inp) / self.tau * _dt
    V = bm.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)
    self.V.value = bm.where(spike, self.V_reset, V)
    self.spike.value = spike

  @staticmethod
  @jax.jit
  def update2(data, t, dt, inp=0.):
    refractory = (t - data['t_last_spike']) <= ref
    V = data['V'] + (-data['V'] + Vr + inp) / taum * dt
    V = jnp.where(refractory, data['V'], V)
    spike = Vt <= V
    data['t_last_spike'] = jnp.where(spike, t, data['t_last_spike'])
    data['V'] = jnp.where(spike, El, V)
    data['spike'] = spike
    return data

  def flops(self):
    # refractory = (_t - self.t_last_spike) <= self.tau_ref
    n = self.num * 2

    # V = self.V + (-self.V + self.V_rest + self.input) / self.tau * _dt
    n += self.num * 6

    # V = bm.where(refractory, self.V, V)
    n += self.num

    # spike = self.V_th <= V
    n += self.num

    # self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)
    n += self.num

    # self.V.value = bm.where(spike, self.V_reset, V)
    n += self.num

    return n


class MatrixDot(bp.BrainPyObject):
  def __init__(self, num):
    super(MatrixDot, self).__init__()
    self.num = num
    self.w = bm.Variable(bm.random.random((num, num)))
    self.x = bm.Variable(bm.random.random((num,)))

  def __call__(self, *args, **kwargs):
    return self.w.value @ self.x.value

  def flops(self):
    return self.num * self.num + self.num * (self.num - 1)


def visualize_results(resfile=None, res=None, save_filename=None):
  if resfile is None:
    assert res is not None
  else:
    res = np.load(resfile)

  results = dict()
  results['FLOPs'] = np.asarray(res['FLOPs'])
  results['LIF'] = np.asarray(res['LIF'])
  results['Dot'] = np.asarray(res['Dot'])
  results['LIF_with_JIT'] = np.asarray(res['LIF_with_JIT'])
  print(results['LIF_with_JIT'] / results['Dot'])

  # sns.set(font_scale=1.5)
  # sns.set_style("white")
  plt.rcParams.update({"font.size": 15})
  fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.)
  ax = fig.add_subplot(gs[0, 0])
  plt.plot(results['FLOPs'], results['LIF'], linestyle="--", marker='v',
           label='LIF', linewidth=3, markersize=10)
  plt.plot(results['FLOPs'], results['Dot'], linestyle="--", marker='D',
           label='Dot', linewidth=3, markersize=10)
  plt.plot(results['FLOPs'], results['LIF_with_JIT'], linestyle="--", marker='o',
           label="LIF with JIT", linewidth=3, markersize=10)
  ax.set_ylabel('Time [s]')
  # ax.set_title('Reducing overhead with JIT (CPU)')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.set_xlabel('FLOPs')
  lg = plt.legend(fontsize=11, loc='best')
  lg.get_frame().set_alpha(0.3)
  if save_filename:
    plt.savefig(save_filename, dpi=1000, transparent=True)
  plt.show()


def show_comparison(resfile=None, platform='cpu', num_run=20):
  compare_results = {'FLOPs': [],
                     'LIF': [],
                     'LIF_with_JIT': [],
                     'Dot': []}

  bm.set_platform(platform)

  # for i in np.asarray(np.arange(1000, 21001, 2000), dtype=np.int64):
  # for i in np.asarray(np.arange(1000, 41001, 4000), dtype=np.int64):
  for i in np.asarray(np.arange(1000, 36001, 4000), dtype=np.int64):
    # for i in np.asarray(np.arange(10000, 40001, 5000), dtype=np.int64):
    print()
    print(i)
    dot = MatrixDot(i)
    flops = dot.flops()
    num_lif = int(flops / 12)
    print('LIF number: ', num_lif)
    lif = LIF(num_lif)

    print('LIF: ', flops)
    print('Dot: ', dot.flops())
    compare_results['FLOPs'].append(flops)

    # LIF without JIT
    t_lif = []
    jax.block_until_ready(lif.update(0., 0.1))
    for _ in range(num_run):
      t0 = time.time()
      jax.block_until_ready(lif.update(0., 0.1))
      t_lif.append(time.time() - t0)
    t_lif = np.mean(t_lif)

    # LIF with JIT
    jax.block_until_ready(lif.update2(lif.data, 0.1, 0.1))  # compile
    t_lif_jit = []
    for _ in range(num_run):
      t0 = time.time()
      jax.block_until_ready(lif.update2(lif.data, 0.1, 0.1))
      t_lif_jit.append(time.time() - t0)
    t_lif_jit = np.mean(t_lif_jit)

    # Matrix-vector multiplication
    jax.block_until_ready(dot())  # compile
    t_dot = []
    for _ in range(num_run):
      t0 = time.time()
      jax.block_until_ready(dot())
      t_dot.append(time.time() - t0)
    t_dot = np.mean(t_dot)

    print(f'Time of LIF: {t_lif:.10f}', )
    print(f'Time of LIF (jit): {t_lif_jit:.30f}', )
    print(f'Time of dot: {t_dot:.10f}', )

    compare_results['LIF'].append(t_lif)
    compare_results['Dot'].append(t_dot)
    compare_results['LIF_with_JIT'].append(t_lif_jit)

  if resfile:
    np.savez(
      resfile,
      FLOPs=np.asarray(compare_results['FLOPs']),
      LIF=np.asarray(compare_results['LIF']),
      LIF_with_JIT=np.asarray(compare_results['LIF_with_JIT']),
      Dot=np.asarray(compare_results['Dot']),
    )
  save_filename = os.path.splitext(resfile)[0] + '.png' if resfile else None
  visualize_results(res=compare_results, save_filename=save_filename)



if __name__ == '__main__':
  pass

  cpu_file = 'results/speed_comparison_under_same_FLOPs-cpu.npz'
  gpu_file = 'results/speed_comparison_under_same_FLOPs-gpu.npz'

  show_comparison(resfile=cpu_file, platform='cpu')
  show_comparison(resfile=gpu_file, platform='gpu')

