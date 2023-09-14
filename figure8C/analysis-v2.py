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


def read_fn_v2(fn, xs):
  with open(fn, 'r') as fin:
    rs = json.load(fin)
  return [(np.mean(rs[str(x)]['exetime']) if str(x) in rs else np.nan) for x in xs]


platform = 'gpu'

xs = [2000 * i for i in [1, 4, 8, 10, 20, 40, 60, 80, 100, 200, 400, 800, 1000]]
xs = [2000 * i for i in [1,  40, 60, 80, 100, 200, 400, 800, 1000]]
if platform == 'cpu':
  files = ['neuron', 'nest', 'brian2cuda', 'genn', 'brian2', 'brainpy-gpu', 'brainpy-cpu']
  files = ['neuron', 'nest', 'brian2', 'brain2-th12', 'brainpy-cpu', 'TPUv3x8']
  files = ['brainpy-cpu', ]
elif platform == 'gpu':
  files = ['brian2cuda', 'genn', 'brainpy-gpu']
  files = ['brainpy-gpu', 'brainpy-tpu', ]
else:
  raise ValueError

sns.set(font_scale=1.5)
sns.set_style("white")
fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6.)
ax = fig.add_subplot(gs[0, 0])

prefix = 'speed_results_mon'

if 'neuron' in files:
  res = read_fn_v1('NEURON.json', xs=xs)
  plt.semilogy(xs, res, linestyle="--", marker='P', label='NEURON', linewidth=3, markersize=10)

if 'nest' in files:
  res = read_fn_v1('NEST.json', filter=lambda a: a['num_thread'] == 1, xs=xs)
  plt.semilogy(xs, res, linestyle="--", marker='s', label='NEST', linewidth=3, markersize=10)

if 'brainpy-cpu' in files:
  res = read_fn_v2(f'{prefix}/brainpy-DM-cpu-x32.json', xs=xs)
  plt.semilogy(xs, res, linestyle="--", marker='D', label='BrainPy CPU x32', linewidth=3, markersize=10)
  res = read_fn_v2(f'{prefix}/brainpy-DM-cpu-x64.json', xs=xs)
  plt.semilogy(xs, res, linestyle="--", marker='D', label='BrainPy CPU x64', linewidth=3, markersize=10)

if 'brainpy-gpu' in files:
  res = read_fn_v2(f'{prefix}/brainpy-DM-gpu-x32.json', xs=xs)
  plt.plot(xs, res, linestyle="--", marker='D', label='BrainPy GPU x32', linewidth=3, markersize=10)
  # plt.semilogy(xs, res, linestyle="--", marker='D', label='BrainPy GPU x32', linewidth=3, markersize=10)

  res = read_fn_v2(f'{prefix}/brainpy-DM-gpu-x64.json', xs=xs)
  plt.plot(xs, res, linestyle="--", marker='D', label='BrainPy GPU x64', linewidth=3, markersize=10)
  # plt.semilogy(xs, res, linestyle="--", marker='D', label='BrainPy GPU x64', linewidth=3, markersize=10)

if 'brain2-th12' in files:
  res = read_fn_v2('speed_results/brian2-COBAHH-cpp_standalone-thread12.json', xs=xs)
  plt.semilogy(xs, res, linestyle="--", marker='v', label='Brian2 12 threads', linewidth=3, markersize=10)

if 'brian2' in files:
  res = read_fn_v2('speed_results/brian2-COBAHH-cpp_standalone.json', xs=xs)
  plt.semilogy(xs, res, linestyle="--", marker='v', label='Brian2', linewidth=3, markersize=10)

if 'genn' in files:
  res = read_fn_v2(f'{prefix}/brian2-COBAHH-genn.json', xs=xs)
  # plt.semilogy(xs, res, linestyle="--", marker='x', label='GeNN', linewidth=3, markersize=10)
  plt.plot(xs, res, linestyle="--", marker='x', label='GeNN', linewidth=3, markersize=10)

if 'brian2cuda' in files:
  res = read_fn_v2(f'{prefix}/brian2-COBAHH-cuda_standalone.json', xs=xs)
  # plt.semilogy(xs, res, linestyle="--", marker='*', label='Brian2CUDA', linewidth=3, markersize=10)
  plt.plot(xs, res, linestyle="--", marker='*', label='Brian2CUDA', linewidth=3, markersize=10)


if 'brainpy-tpu' in files:
  res = read_fn_v2('speed_results_mon/brainpy-DM-tpu-x32.json', xs=xs)
  # plt.semilogy(xs, res, linestyle="--", marker='*', label='Brian2CUDA', linewidth=3, markersize=10)
  plt.plot(xs, res, linestyle="--", marker='D', label='TPU v3 x32', linewidth=3, markersize=10)

# plt.xticks(xs)
# plt.ylim(-1., 11.)

# pytorch = read_fn('pytorch.json', xs=xs)
# plt.semilogy(xs, pytorch, linestyle="--", marker='x', label='PyTorch', linewidth=2)

# annarchy = read_fn('annarchy-1.json', xs=xs)
# plt.semilogy(xs, annarchy, linestyle="--", marker='*', label='ANNarchy', linewidth=2)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('Number of Neurons')
plt.ylabel('Simulation Time [s]')
lg = plt.legend(fontsize=12, loc='best')
# lg = plt.legend(fontsize=12, loc='upper right')
lg.get_frame().set_alpha(0.3)
if platform == 'gpu':
  plt.title(f'DMNet GPU & TPU')
else:
  plt.title(f'DMNet {platform.upper()}')
# if platform == 'cpu':
#   plt.xlim(-1, 8.2e4)
# elif platform == 'gpu':
#   pass
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.savefig(f'DM-speed-{platform}.pdf')
plt.show()

