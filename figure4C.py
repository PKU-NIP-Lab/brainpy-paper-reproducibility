# -*- coding: utf-8 -*-

import numpy as np

import brainpy as bp
import brainpy.math as bm
import brainpy_datasets as bd
import matplotlib.pyplot as plt

bm.enable_x64()


def get_subset(data, start, end):
  res = {'x': data.xs[start: end], 'y': data.ys[start: end], 'z': data.zs[start: end]}
  X = bm.hstack([res['x'], res['y']])
  X = X.reshape((1,) + X.shape)
  Y = res['z']
  Y = Y.reshape((1,) + Y.shape)
  return X, Y


def plot_lorenz(x, y, true_z, predict_z, linewidth=None):
  plt.rcParams.update({"font.size": 15})

  fig, gs = bp.visualize.get_figure(3, 1, 1.5, 6)
  t_all = t_warmup + t_train + t_test
  ts = np.arange(0, t_all, dt)

  ax1 = fig.add_subplot(gs[0, 0])
  ax1.plot(ts[num_warmup + num_train:num_warmup + num_train + num_test],
           x[num_warmup + num_train:num_warmup + num_train + num_test],
           color='b', linewidth=linewidth)
  ax1.set_ylabel('x')
  ax1.axes.xaxis.set_ticklabels([])
  ax1.axes.yaxis.set_ticklabels([])
  ax1.axes.set_ybound(-21., 21.)
  ax1.axes.set_xbound(t_warmup + t_train - .5, t_all + .5)
  ax1.set_title('Reservoir Model')
  ax1.spines['right'].set_color('none')
  ax1.spines['top'].set_color('none')
  ax1.set_xticks([])
  ax1.set_yticks([])

  # testing phase y
  ax2 = fig.add_subplot(gs[1, 0])
  ax2.plot(ts[num_warmup + num_train:num_warmup + num_train + num_test],
           y[num_warmup + num_train:num_warmup + num_train + num_test],
           color='b', linewidth=linewidth)
  ax2.set_ylabel('y')
  ax2.axes.xaxis.set_ticklabels([])
  ax2.axes.yaxis.set_ticklabels([])
  ax2.axes.set_ybound(-26., 26.)
  ax2.axes.set_xbound(t_warmup + t_train - .5, t_all + .5)
  ax2.spines['right'].set_color('none')
  ax2.spines['top'].set_color('none')
  ax2.set_xticks([])
  ax2.set_yticks([])

  # testing phose z
  ax3 = fig.add_subplot(gs[2, 0])
  ax3.plot(ts[num_warmup + num_train:num_warmup + num_train + num_test],
           true_z[num_warmup + num_train:num_warmup + num_train + num_test],
           color='b', linewidth=linewidth)
  ax3.plot(ts[num_warmup + num_train:num_warmup + num_train + num_test],
           predict_z[num_warmup + num_train:num_warmup + num_train + num_test],
           color='r', linewidth=linewidth)
  ax3.set_ylabel('z')
  ax3.set_xlabel('Time [ms]')
  ax3.axes.yaxis.set_ticklabels([])
  ax3.axes.set_ybound(3., 48.)
  ax3.axes.set_xbound(t_warmup + t_train - .5, t_all + .5)
  ax3.set_xticks([])
  ax3.set_yticks([])
  ax3.spines['right'].set_color('none')
  ax3.spines['top'].set_color('none')

  # plt.savefig(f'reservoir-lorenz-training.pdf', dpi=1000, transparent=True)
  plt.show()


dt = 0.02
t_warmup = 10.  # ms
t_train = 20.  # ms
t_test = 50.  # ms
num_warmup = int(t_warmup / dt)  # warm up NVAR
num_train = int(t_train / dt)
num_test = int(t_test / dt)

# Datasets #
# -------- #
lorenz_series = bd.chaos.LorenzEq(t_warmup + t_train + t_test, dt=dt,
                                  inits={'x': 17.67715816276679,
                                         'y': 12.931379185960404,
                                         'z': 43.91404334248268})
X_warmup, Y_warmup = get_subset(lorenz_series, 0, num_warmup)
X_train, Y_train = get_subset(lorenz_series, num_warmup, num_warmup + num_train)
X_test, Y_test = get_subset(lorenz_series, 0, num_warmup + num_train + num_test)

# Model #
# ----- #

class NGRC(bp.DynamicalSystem):
  def __init__(self, num_in):
    super(NGRC, self).__init__()
    self.r = bp.layers.NVAR(num_in, delay=4, order=2, stride=5)
    self.o = bp.layers.Dense(self.r.num_out, 1, mode=bm.training_mode)

  def update(self, x):
    return self.o(self.r(x))


with bm.batching_environment():
    model = NGRC(2)


# Training #
# -------- #

trainer = bp.RidgeTrainer(model, alpha=0.05)

# warm-up
outputs = trainer.predict(X_warmup)
print('Warmup NMS: ', bp.losses.mean_squared_error(outputs, Y_warmup))

# training
trainer.fit([X_train, Y_train])

# prediction
outputs = trainer.predict(X_test, reset_state=True)
print('Prediction NMS: ', bp.losses.mean_squared_error(outputs, Y_test))

plot_lorenz(x=bm.as_numpy(lorenz_series.xs.flatten()),
            y=bm.as_numpy(lorenz_series.ys.flatten()),
            true_z=bm.as_numpy(lorenz_series.zs.flatten()),
            predict_z=bm.as_numpy(outputs.flatten()))
