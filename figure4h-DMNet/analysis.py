# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import seaborn as sns
import brainpy as bp


def read_fn(fn):
  with open(os.path.join('speed_results/', fn), 'r') as fin:
    rs = fin.read()
  times = []
  for line in rs.strip().split('\n'):
    line = line.strip()
    if line:
      sps = line.split(',')
      times.append(tuple([int(sps[1].split('=')[1]), float(sps[2].split('=')[1])]))
  times = dict(times)
  return list(times.keys()), list(times.values())
  # return np.asarray([(times[x] if x in times else np.nan) for x in xs])


# Fig3-DM-network-speed

sns.set(font_scale=1.5)
sns.set_style("white")
fig, gs = bp.visualize.get_figure(1, 1, 6., 6.)
ax = fig.add_subplot(gs[0, 0])

xs, brian2_cpu = read_fn('brian2-cpu-I7-6700K-2022-4-13.txt')
plt.semilogy(xs, brian2_cpu, linestyle="--",  marker="o", label='Brian2', linewidth=3, markersize=10)

xs, brainpy_cpu = read_fn('brainpy-v2-cpu-I7-6700K-2022-4-18.txt')
plt.semilogy(xs, brainpy_cpu, linestyle="--",  marker='v', label='BrainPy (cpu)', linewidth=3, markersize=10)

xs, brainpy_tpu_v2 = read_fn('brainpy-v2-tpu-v2-colab-2022-4-18.txt')
plt.semilogy(xs, brainpy_tpu_v2, linestyle="--",  marker='s', label='BrainPy (tpu v2)', linewidth=3, markersize=10)

xs, brainpy_gpu = read_fn('brainpy-v2-gpu-RTX-A6000-2022-4-18.txt')
plt.semilogy(xs, brainpy_gpu, linestyle="--",  marker='D', label='BrainPy (gpu)', linewidth=3, markersize=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title('Decision Making Network')
plt.xlabel('Number of neurons')
plt.ylabel('Simulation time [s]')
lg = plt.legend(fontsize=12)
lg.get_frame().set_alpha(0.5)
# plt.title('Decision Making Network')
plt.show()

