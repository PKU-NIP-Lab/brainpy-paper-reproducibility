# -*- coding: utf-8 -*-


import brainpy as bp

from src.decision_making_network import DecisionMakingNet

bp.check.turn_off()


def build_and_run(scale, file=None,
                  pre_stimulus_period=500.,
                  stimulus_period=1000.,
                  delay_period=500.,
                  progress_bar=False):
    net = DecisionMakingNet(scale=scale, pre_stimulus_period=pre_stimulus_period, stimulus_period=stimulus_period)

    runner = bp.DSRunner(net,
                         # monitors=['A.spike', 'B.spike', 'IA.freq', 'IB.freq'],
                         progress_bar=progress_bar)
    total_period = pre_stimulus_period + stimulus_period + delay_period
    t = runner(total_period, eval_time=True)[0]

    if file is not None:
        file.write(f'scale={scale}, num={net.num}, time={t}\n')
        file.flush()
    print(f'Used time: {t} s')


if __name__ == '__main__':
    platform = 'cpu'

    bp.math.set_platform(platform)
    name = f'brainpy-v2-{platform}'

    # for scale in [a / 4 for a in range(1, 41, 2)]:
    #   build_and_run(scale=scale)

    build_and_run(scale=1., file=None, progress_bar=True)

    # with open(f'speed_results/{name}.txt', 'w') as fout:
    #   for size in [1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 2e6]:
    #     scale = size / 2e3
    #     build_and_run(scale=scale, file=fout)
