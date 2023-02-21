# -*- coding: utf-8 -*-

import seaborn as sns
import json

import brainpy as bp
import os

import matplotlib.pyplot as plt
import numpy as np


def read_fn(fn, xs, filter=None):
  with open(os.path.join('speed_results/', fn), 'r') as fin:
    rs = json.load(fin)
  if filter is None:
    times = [(a['num_neuron'], a['sim_time']) for a in tuple(rs.values())[0]]
  else:
    times = [(a['num_neuron'], a['sim_time']) for a in tuple(rs.values())[0] if filter(a)]
  times = dict(times)
  return [(times[x] if x in times else np.nan) for x in xs]


xs = [4000 * i for i in [1, 2, 4, 6, 8, 10, 15, 20, 30]]

brainpy_cpu = read_fn('brainpy-cpu.json', xs=xs)
brian2 = read_fn('brian2.json', xs=xs)
nest = read_fn('NEST.json', xs=xs)
neuron = read_fn('NEURON.json', xs=xs)

sns.set(font_scale=1.5)
sns.set_style("white")
fig, gs = bp.visualize.get_figure(1, 1, 6., 6.)
ax = fig.add_subplot(gs[0, 0])
plt.semilogy(xs, neuron, linestyle="--", marker='P', label='NEURON', linewidth=3, markersize=10)
plt.semilogy(xs, nest, linestyle="--", marker='s', label='NEST', linewidth=3, markersize=10)
plt.semilogy(xs, brian2, linestyle="--", marker='v', label='Brian2', linewidth=3, markersize=10)
plt.semilogy(xs, brainpy_cpu, linestyle="--", marker='D', label='BrainPy', linewidth=3, markersize=10)
# plt.xticks(xs)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('Number of neurons')
plt.ylabel('Simulation time [s]')
lg = plt.legend(fontsize=12, loc='upper right')
lg.get_frame().set_alpha(0.3)
plt.title('COBAHH Network')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
# plt.tight_layout()
plt.show()
