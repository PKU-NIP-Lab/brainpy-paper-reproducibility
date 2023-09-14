# -*- coding: utf-8 -*-

import json
import os

import brainpy as bp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def read_fn_v1(fn, xs, filter=None):
  with open(os.path.join('speed_results/', fn), 'r') as fin:
    rs = json.load(fin)
  if filter is None:
    times = [(a['num_neuron'], a['sim_time']) for a in tuple(rs.values())[0]]
  else:
    times = [(a['num_neuron'], a['sim_time']) for a in tuple(rs.values())[0] if filter(a)]
  times = dict(times)
  return [(times[x] if x in times else np.nan) for x in xs]


def read_fn_v2(fn, xs, filter=None):
  with open(os.path.join('speed_results/', fn), 'r') as fin:
    rs = json.load(fin)
  return [(np.mean(rs[str(x)]['exetime'])  if (str(x) in rs) else np.nan) for x in xs]


xs = [4000 * i for i in [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]]

platform = 'gpu'

if platform == 'cpu':
  files = ['neuron', 'nest', 'brian2cuda', 'genn', 'brian2', 'brainpy-gpu', 'brainpy-cpu']
  files = ['neuron', 'nest', 'brian2', 'brainpy-cpu', 'TPUv3x8']
elif platform == 'gpu':
  files = ['brian2cuda', 'genn', 'brainpy-gpu',]
else:
  raise ValueError
sns.set(font_scale=1.5)
sns.set_style("white")
fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.)
ax = fig.add_subplot(gs[0, 0])

if 'neuron' in files:
  res = read_fn_v1('neruon-v2.json', xs=xs)
  plt.semilogy(xs, res, linestyle="--", marker='P', label='NEURON', linewidth=3, markersize=10)

if 'nest' in files:
  res = read_fn_v1('nest.json', filter=lambda a: a['num_thread'] == 1, xs=xs)
  plt.semilogy(xs, res, linestyle="--", marker='s', label='NEST', linewidth=3, markersize=10)

if 'brainpy-cpu' in files:
  res = read_fn_v2('brainpy-COBA-wsl2-cpu-x32.json', xs=xs)
  plt.semilogy(xs, res, linestyle="--", marker='D', label='BrainPy CPU x32', linewidth=3, markersize=10)
  res = read_fn_v2('brainpy-COBA-wsl2-cpu-x64.json', xs=xs)
  plt.semilogy(xs, res, linestyle="--", marker='D', label='BrainPy CPU x64', linewidth=3, markersize=10)

if 'brainpy-gpu' in files:
  res = read_fn_v2('brainpy-COBA-A6000-gpu-x32.json', xs=xs)
  res = read_fn_v2('brainpy-COBA-gpu-x32.json', xs=xs)
  # plt.semilogy(xs, res, linestyle="--", marker='D', label='BrainPy GPU x32', linewidth=3, markersize=10)
  plt.plot(xs, res, linestyle="--", marker='D', label='BrainPy GPU x32', linewidth=3, markersize=10)
  res = read_fn_v2('brainpy-COBA-A6000-gpu-x64.json', xs=xs)
  res = read_fn_v2('brainpy-COBA-gpu-x64.json', xs=xs)
  # plt.semilogy(xs, res, linestyle="--", marker='D', label='BrainPy GPU x64', linewidth=3, markersize=10)
  plt.plot(xs, res, linestyle="--", marker='D', label='BrainPy GPU x64', linewidth=3, markersize=10)

if 'brian2' in files:
  res = read_fn_v2('brian2-COBA-wsl-cpp_standalone.json', xs=xs)
  plt.semilogy(xs, res, linestyle="--", marker='v', label='Brian2', linewidth=3, markersize=10)

if 'genn' in files:
  # res = read_fn_v2('brian2-COBA-A6000-genn.json', xs=xs)
  res = read_fn_v2('brian2-COBA-genn.json', xs=xs)
  # plt.semilogy(xs, res, linestyle="--", marker='x', label='GeNN', linewidth=3, markersize=10)
  plt.plot(xs, res, linestyle="--", marker='x', label='GeNN', linewidth=3, markersize=10)

if 'brian2cuda' in files:
  # res = read_fn_v2('brian2-COBA-A6000-cuda_standalone.json', xs=xs)
  res = read_fn_v2('brian2-COBA-cuda_standalone.json', xs=xs)
  # plt.semilogy(xs, res, linestyle="--", marker='*', label='Brian2CUDA', linewidth=3, markersize=10)
  plt.plot(xs, res, linestyle="--", marker='*', label='Brian2CUDA', linewidth=3, markersize=10)

if 'TPUv3x8' in files:
  res = read_fn_v2('brainpy-COBA-TPUx8-x32.json', xs=xs)
  # plt.semilogy(xs, res, linestyle="--", marker='*', label='Brian2CUDA', linewidth=3, markersize=10)
  plt.plot(xs, res, linestyle="--", marker='D', label='TPU v3', linewidth=3, markersize=10)


# plt.xticks(xs)
# plt.ylim(-1., 5.)

# pytorch = read_fn('pytorch.json', xs=xs)
# plt.semilogy(xs, pytorch, linestyle="--", marker='x', label='PyTorch', linewidth=2)

# annarchy = read_fn('annarchy-1.json', xs=xs)
# plt.semilogy(xs, annarchy, linestyle="--", marker='*', label='ANNarchy', linewidth=2)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('Number of Neurons')
plt.ylabel('Simulation Time [s]')
# lg = plt.legend(fontsize=12, loc='upper right')
lg = plt.legend(fontsize=12, loc='best')
lg.get_frame().set_alpha(0.5)
plt.title(f'COBA {platform.upper()}')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.savefig(f'COBA-speed-{platform}.pdf')
plt.show()
