# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import neurogym as ngym
import numpy as np

# bp.math.set_platform('cpu')
task = 'PerceptualDecisionMaking-v0'
timing = {
  'fixation': ('choice', (50, 100, 200, 400)),
  'stimulus': ('choice', (100, 200, 400, 800)),
}
kwargs = {'dt': 20, 'timing': timing}
# Make supervised dataset
dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=16, seq_len=100)
# A sample environment from dataset
env = dataset.env

# data size
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
batch_size = dataset.batch_size
print(f'Input size = {input_size}')
print(f'Output size = {output_size}')
print(f'Batch size = {batch_size}')


class RNN(bp.Base):
  def __init__(self, num_input, num_hidden, num_output, num_batch,
               dt=None, e_ratio=0.8, sigma_rec=0., seed=None,
               w_ir=bp.init.KaimingUniform(scale=1.),
               w_rr=bp.init.KaimingUniform(scale=1.),
               w_ro=bp.init.KaimingUniform(scale=1.)):
    super(RNN, self).__init__()

    # parameters
    self.tau = 100
    self.num_batch = num_batch
    self.num_input = num_input
    self.num_hidden = num_hidden
    self.num_output = num_output
    self.e_size = int(num_hidden * e_ratio)
    self.i_size = num_hidden - self.e_size
    self.alpha = 1 if (dt is None) else dt / self.tau
    self.sigma_rec = (2 * self.alpha) ** 0.5 * sigma_rec  # Recurrent noise
    self.rng = bm.random.RandomState(seed=seed)

    # hidden mask
    mask = np.tile([1] * self.e_size + [-1] * self.i_size, (num_hidden, 1))
    np.fill_diagonal(mask, 0)
    self.mask = bm.asarray(mask, dtype=bm.float_)

    # input weight
    self.w_ir = bm.TrainVar(bp.init.parameter(w_ir, (num_input, num_hidden)))

    # recurrent weight
    bound = 1 / num_hidden ** 0.5
    self.w_rr = bm.TrainVar(bp.init.parameter(w_rr, (num_hidden, num_hidden)))
    self.w_rr[:, :self.e_size] /= (self.e_size / self.i_size)
    self.b_rr = bm.TrainVar(self.rng.uniform(-bound, bound, num_hidden))

    # readout weight
    bound = 1 / self.e_size ** 0.5
    self.w_ro = bm.TrainVar(bp.init.parameter(w_ro, (self.e_size, num_output)))
    self.b_ro = bm.TrainVar(self.rng.uniform(-bound, bound, num_output))

    # variables
    self.h = bm.Variable(bm.zeros((num_batch, num_hidden)))
    self.o = bm.Variable(bm.zeros((num_batch, num_output)))

  def cell(self, x, h):
    ins = x @ self.w_ir + h @ (bm.abs(self.w_rr) * self.mask) + self.b_rr
    state = h * (1 - self.alpha) + ins * self.alpha
    state += self.sigma_rec * self.rng.randn(self.num_hidden)
    return bm.relu(state)

  def readout(self, h):
    return h @ self.w_ro + self.b_ro

  def make_update(self, h: bm.JaxArray, o: bm.JaxArray):
    def f(x):
      h.value = self.cell(x, h.value)
      o.value = self.readout(h.value[:, :self.e_size])
      return h.value, o.value

    return f

  def predict(self, xs):
    self.h[:] = 0.
    return bm.for_loop(self.make_update(self.h, self.o), xs, dyn_vars=self.vars())

  def loss(self, xs, ys):
    hs, os = self.predict(xs)
    os = os.reshape((-1, os.shape[-1]))
    return bp.losses.cross_entropy_loss(os, ys.flatten())


# Instantiate the network and print information
hidden_size = 50
with bm.training_environment():
  net = RNN(num_input=input_size,
            num_hidden=hidden_size,
            num_output=output_size,
            num_batch=batch_size,
            dt=env.dt,
            sigma_rec=0.15)

# Adam optimizer
opt = bp.optim.Adam(lr=0.001, train_vars=net.train_vars().unique())

# gradient function
grad_f = bm.grad(net.loss,
                 dyn_vars=net.vars(),
                 grad_vars=net.train_vars().unique(),
                 return_value=True)


@bm.jit
@bm.to_object(child_objs=(net, opt))
def train(xs, ys):
  grads, loss = grad_f(xs, ys)
  opt.update(grads)
  return loss


# training #
# -------- #
running_loss = 0
print_step = 200
for i in range(5000):
  inputs, labels = dataset()
  inputs = bm.asarray(inputs)
  labels = bm.asarray(labels)
  loss = train(inputs, labels)
  running_loss += loss
  if i % print_step == (print_step - 1):
    running_loss /= print_step
    print('Step {}, Loss {:0.4f}'.format(i + 1, running_loss))
    running_loss = 0


# prediction #
# ---------- #
predict = bm.jit(net.predict, dyn_vars=net.vars())
env.reset(no_step=True)
env.timing.update({'fixation': ('constant', 500), 'stimulus': ('constant', 500)})
perf = 0
num_trial = 500
activity_dict = {}
trial_infos = {}
stim_activity = [[], []]  # response for ground-truth 0 and 1
for i in range(num_trial):
  env.new_trial()
  ob, gt = env.ob, env.gt
  inputs = bm.asarray(ob[:, np.newaxis, :])
  rnn_activity, action_pred = predict(inputs)

  # Compute performance
  action_pred = bm.as_numpy(action_pred)
  choice = np.argmax(action_pred[-1, 0, :])
  correct = choice == gt[-1]

  # Log trial info
  trial_info = env.trial
  trial_info.update({'correct': correct, 'choice': choice})
  trial_infos[i] = trial_info

  # Log stimulus period activity
  rnn_activity = bm.as_numpy(rnn_activity)[:, 0, :]
  activity_dict[i] = rnn_activity

  # Compute stimulus selectivity for all units
  # Compute each neuron's response in trials where ground_truth=0 and 1 respectively
  rnn_activity = rnn_activity[env.start_ind['stimulus']: env.end_ind['stimulus']]
  stim_activity[env.trial['ground_truth']].append(rnn_activity)

print('Average performance', np.mean([val['correct'] for val in trial_infos.values()]))

# visualization #
# ------------- #
mean_activity = []
std_activity = []
for ground_truth in [0, 1]:
  activity = np.concatenate(stim_activity[ground_truth], axis=0)
  mean_activity.append(np.mean(activity, axis=0))
  std_activity.append(np.std(activity, axis=0))
# Compute d'
selectivity = (mean_activity[0] - mean_activity[1])
selectivity /= np.sqrt((std_activity[0] ** 2 + std_activity[1] ** 2 + 1e-7) / 2)
# Sort index for selectivity, separately for E and I
ind_sort = np.concatenate((np.argsort(selectivity[:net.e_size]),
                           np.argsort(selectivity[net.e_size:]) + net.e_size))

trial = 2
W = bm.as_numpy(bm.abs(net.w_rr) * net.mask)
# Sort by selectivity
W = W[:, ind_sort][ind_sort, :]
wlim = np.max(np.abs(W))
plt.rcParams.update({"font.size": 15})
fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
ax1 = fig.add_subplot(gs[0, 0])
lines1 = ax1.plot(activity_dict[trial][:, :net.e_size], color='blue', label='Excitatory')
lines2 = ax1.plot(activity_dict[trial][:, net.e_size:], color='red', label='Inhibitory')
ax1.set_xlabel('Time step')
ax1.set_ylabel('Recurrent Activity')
plt.xticks([])
plt.ylim([-0.1, activity_dict[trial].max()])
plt.xlim([0, activity_dict[trial].shape[0]])
plt.title('Recurrent Neural Network')
lg = ax1.legend(handles=[lines1[0], lines2[0]], fontsize=12)
lg.get_frame().set_facecolor('none')
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
plt.show()

