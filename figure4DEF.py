# -*- coding: utf-8 -*-
from jax import core

import brainpy as bp
import brainpy.math as bm
import brainpy_datasets as bd

import matplotlib.pyplot as plt


def get_data():
  lorenz = bd.chaos.LorenzEq(100)
  data = bm.hstack([lorenz.xs, lorenz.ys, lorenz.zs])
  X, Y = data[:-5], data[5:]

  # here batch size is 1
  X = bm.expand_dims(X, axis=0)
  Y = bm.expand_dims(Y, axis=0)

  return X, Y


def visualize(predict1, method='', save_fn=None):
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
  if save_fn:
    plt.savefig(save_fn, transparent=True, dpi=1000)
  plt.show()


def offline_training():
  bm.enable_x64()

  with bm.batching_environment():
    model = bp.Sequential(bp.dyn.Reservoir(3, 100),
                          bp.dnn.Dense(100, 3, mode=bm.training_mode))
  trainer = bp.OfflineTrainer(model, fit_method=bp.algorithms.RidgeRegression(alpha=1e-6))

  # fitting
  X, Y = get_data()
  trainer.fit([X, Y])

  # prediction
  predict = trainer.predict(X, reset_state=True)
  predict = bm.as_numpy(predict)

  print('MSE of offline training algorithm: ', bp.losses.mean_absolute_error(predict, Y))

  # visualization
  visualize(predict, method='Ridge Regression', save_fn='ridge.pdf')


def online_training():
  bm.enable_x64()

  with bm.batching_environment():
    model = bp.Sequential(bp.dyn.Reservoir(3, 100),
                          bp.dnn.Dense(100, 3, mode=bm.training_mode))
  trainer = bp.OnlineTrainer(model, fit_method=bp.algorithms.RLS())

  # fitting
  X, Y = get_data()
  trainer.fit([X, Y])

  # prediction
  predict = trainer.predict(X, reset_state=True)
  predict = bm.as_numpy(predict)

  print('MSE of online training algorithm: ', bp.losses.mean_absolute_error(predict, Y))

  # visualization
  visualize(predict, method='FORCE learning', save_fn='force.pdf')


def bp_training():
  bm.enable_x64()

  with bm.batching_environment():
    reservoir = bp.dyn.Reservoir(3, 100)
    readout = bp.dnn.Dense(100, 3, mode=bm.training_mode)

  X, Y = get_data()

  runner = bp.DSTrainer(target=reservoir)
  projections = runner.predict(X)

  # For linear readout node, it is not a recurrent node.
  # There is no need to keep time axis.
  # Therefore, we make the original time step as the sample size.
  def get_batch_data():
    for i in range(0, projections.shape[1], 64):
      yield projections[0, i: i + 64], Y[0, i: i + 64]

  trainer = bp.BPFF(target=readout,
                    loss_fun=bp.losses.mean_squared_error,
                    optimizer=bp.optim.Adam(1e-3))
  trainer.fit(get_batch_data, num_epoch=100)

  # prediction
  model = bp.Sequential(reservoir, readout, mode=bm.batching_mode)
  runner = bp.DSTrainer(model)
  predict = runner.predict(X, reset_state=True)
  predict = bm.as_numpy(predict)

  print('MSE of backpropagation algorithm: ', bp.losses.mean_absolute_error(predict, Y))

  # visualization
  visualize(predict, method='Backpropagation', save_fn='bp.pdf')


if __name__ == '__main__':
  # offline_training()
  # online_training()
  bp_training()
