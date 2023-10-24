# -*- coding: utf-8 -*-


import sys
import jax

sys.path.append('../')
import argparse
import time
import json
import brainpy as bp
import brainpy.math as bm
import numpy as np

from src.decision_making_network2 import DecisionMakingNet, Tool


def simulate_a_trial(scale, platform='cpu', x64=False, monitor=True):
  bm.set_platform(platform)
  bm.random.seed()
  if x64:
    bm.enable_x64()

  # simulation tools
  pre_stimulus_period = 100.  # time before the external simuli are given
  stimulus_period = 1000.  # time within which the external simuli are given
  delay_period = 500.  # time after the external simuli are removed
  tool = Tool(pre_stimulus_period, stimulus_period, delay_period)

  # stimulus
  mu0 = 40.
  coherence = 25.6
  IA_freqs = tool.generate_freqs(mu0 + mu0 / 100. * coherence)
  IB_freqs = tool.generate_freqs(mu0 - mu0 / 100. * coherence)

  # network
  net = DecisionMakingNet(scale)

  def run(i):
    bp.share.save(i=i, t=bm.get_dt() * i)
    net.IA.freqs.value = IA_freqs[i]
    net.IB.freqs.value = IB_freqs[i]
    net.update()
    if monitor:
      return {'A.spike': net.A.spike.value, 'B.spike': net.B.spike.value}

  @bm.jit
  def jit_run(indices):
    return bm.for_loop(run, indices)

  # first running
  n_step = int(tool.total_period / bm.get_dt())
  indices = np.arange(n_step)
  # net.reset_state()
  t0 = time.time()
  mon = jax.block_until_ready(jit_run(indices))
  t1 = time.time()

  # # second running
  # net.reset_state()
  # t2 = time.time()
  # mon = jax.block_until_ready(jit_run(indices))
  # t3 = time.time()

  # mon['ts'] = indices * bm.get_dt()
  # tool.visualize_results(mon, IA_freqs, IB_freqs)

  print(f'platform = {platform}, x64 = {x64}, scale = {scale}, '
        f'first run = {t1 - t0} s')
  # print(f'platform = {platform}, x64 = {x64}, scale = {scale}, '
  #       f'first run = {t1 - t0} s, second run = {t3- t2} s')

  # post
  bm.disable_x64()
  bm.clear_buffer_memory(platform)
  return {'num': net.num,
          'exe_time': t1 - t0,
          'run_time': t1 - t0,
          # 'fr': rate
          }


def benchmark1(platform='cpu', x64=True):
  final_results = dict()
  # for scale in [200, 400, 800, 1000]:
  for scale in [1, 4, 8, 10, ]:
    for _ in range(4):
      r = simulate_a_trial(scale=scale, platform=platform, x64=x64, monitor=True)
      if r['num'] not in final_results:
        final_results[r['num']] = {'exetime': [], 'runtime': []}
      final_results[r['num']]['exetime'].append(r['exe_time'])
      final_results[r['num']]['runtime'].append(r['run_time'])
      # final_results[r['num']]['firing_rate'].append(r['fr'])

  postfix = 'x64' if x64 else 'x32'
  with open(f'speed_results/brainpy-DM-{platform}-{postfix}.json', 'w') as fout:
    json.dump(final_results, fout, indent=2)


def benchmark2(devices, platform='cpu', x64=True):
  final_results = dict()
  scales = [1, 4, 8, 10, 20, 40, 60, 80, 100]
  scales = [1, 4, 8, 10, ]

  for scale in scales:
    for _ in range(4):
      with bm.sharding.device_mesh(devices, [bm.sharding.NEU_AXIS]):
        r = simulate_a_trial(scale=scale, platform=platform, x64=x64, monitor=True)
      if r['num'] not in final_results:
        final_results[r['num']] = {'exetime': [], 'runtime': []}
      final_results[r['num']]['exetime'].append(r['exe_time'])
      final_results[r['num']]['runtime'].append(r['run_time'])
      # final_results[r['num']]['firing_rate'].append(r['fr'])

  postfix = 'x64' if x64 else 'x32'
  with open(f'speed_results/brainpy-DM-{platform}-{postfix}.json', 'w') as fout:
    json.dump(final_results, fout, indent=2)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-platform', default='gpu', help='platform')
  parser.add_argument('-x64', action='store_true')
  args = parser.parse_args()
  benchmark1(platform=args.platform, x64=args.x64)
  # bm.set_host_device_count(8)
  # benchmark(jax.devices(), platform='cpu', x64=False)
