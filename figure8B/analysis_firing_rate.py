import json

import brainpy as bp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


xs = [4000 * i for i in [1, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]]


def read_fn_v2(name, fn):
    data = []
    with open(fn, 'r') as fin:
        rs = json.load(fin)
        for x in xs:
            if str(x) in rs:
                for et, fr in zip(rs[str(x)]['exetime'], rs[str(x)]['firing_rate']):
                    data.append([x, name, et, fr])
    return data


all_data = []
all_data.extend(read_fn_v2('Brian2CUDA', 'speed_results_mon/brian2-COBAHH-cuda_standalone.json'))
all_data.extend(read_fn_v2('GeNN', 'speed_results_mon/brian2-COBAHH-genn.json'))
all_data.extend(read_fn_v2('BrainPy GPU x32', 'speed_results_mon/brainpy-COBAHH-gpu-x32.json'))
all_data.extend(read_fn_v2('BrainPy GPU x64', 'speed_results_mon/brainpy-COBAHH-gpu-x64.json'))
all_data.extend(read_fn_v2('BrainPy TPU x32', 'speed_results_mon/brainpy-COBAHH-TPUx8-x32-v2.json'))
df = pd.DataFrame(all_data, columns=['size', 'simulator', 'exe_time', 'firing_rate'])

fig, gs = bp.visualize.get_figure(1, 1, 5, 10.)
ax = fig.add_subplot(gs[0, 0])
g = sns.barplot(data=df, x="size", y="firing_rate", hue="simulator", errorbar="sd", palette="dark", alpha=.6, ax=ax)
plt.ylabel("Firing Rate [Hz]")
plt.xlabel('Network Size')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('SI-COBAHH-Net-Firing-Rate.pdf')
plt.show()

