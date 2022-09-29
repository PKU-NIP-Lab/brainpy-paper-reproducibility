# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm

import matplotlib.pyplot as plt


def get_data():
  lorenz = bp.datasets.lorenz_series(100)
  data = bm.hstack([lorenz['x'], lorenz['y'], lorenz['z']])
  X, Y = data[:-5], data[5:]

  # here batch size is 1
  X = bm.expand_dims(X, axis=0)
  Y = bm.expand_dims(Y, axis=0)

  return X, Y


def visualize(predict1, method=''):
  plt.rcParams.update({"font.size": 15})
  bp.visualize.get_figure(1, 1, 4.5, 6.)
  plt.plot(predict1[0, :, 0], predict1[0, :, 2], )
  plt.xticks([])
  plt.yticks([])
  plt.xlabel('x')
  plt.ylabel('z')
  ax = plt.gca()
  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')
  plt.title(f'Trained with {method}')
  plt.show()


def offline_training():
  bm.enable_x64()

  model = bp.dyn.Sequential(bp.layers.Reservoir(3, 100, mode=bp.modes.batching,),
                            bp.layers.Dense(100, 3, mode=bp.modes.training))
  trainer = bp.train.OfflineTrainer(
    model, fit_method=bp.algorithms.RidgeRegression(alpha=1e-6)
  )

  # fitting
  X, Y = get_data()
  trainer.fit([X, Y])

  # prediction
  predict = trainer.predict(X, reset_state=True)
  predict1 = bm.as_numpy(predict)

  # visualization
  visualize(predict1, method='Ridge Regression')


def online_training():
  bm.enable_x64()

  model = bp.dyn.Sequential(bp.layers.Reservoir(3, 100, mode=bp.modes.batching),
                            bp.layers.Dense(100, 3, mode=bp.modes.training))
  trainer = bp.train.OnlineTrainer(
    model, fit_method=bp.algorithms.RLS()
  )

  # fitting
  X, Y = get_data()
  trainer.fit([X, Y])

  # prediction
  predict = trainer.predict(X, reset_state=True)
  predict1 = bm.as_numpy(predict)

  # visualization
  visualize(predict1, method='FORCE learning')


def bptt():
  bm.enable_x64()

  reservoir = bp.layers.Reservoir(3, 100, mode=bp.modes.batching)
  readout = bp.layers.Dense(100, 3, mode=bp.modes.training)

  X, Y = get_data()

  runner = bp.train.DSTrainer(target=reservoir)
  projections = runner.predict(X)

  # For linear readout node, it is not a recurrent node.
  # There is no need to keep time axis.
  # Therefore, we make the original time step as the sample size.
  projections = projections[0]
  targets = Y[0]

  trainer = bp.train.BPFF(target=readout,
                          loss_fun=bp.losses.mean_squared_error,
                          optimizer=bp.optim.Adam(1e-3))
  trainer.fit([projections, targets],
              num_report=2000,
              batch_size=64,
              num_epoch=60)

  # prediction
  model = bp.dyn.Sequential(reservoir, readout)
  runner = bp.train.DSTrainer(model)
  predict = runner.predict(X, reset_state=True)
  predict1 = bm.as_numpy(predict)

  # visualization
  visualize(predict1, method='BPTT')


if __name__ == '__main__':
  offline_training()
  online_training()
  bptt()

