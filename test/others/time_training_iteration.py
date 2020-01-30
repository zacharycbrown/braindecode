"""This script performs a timing analysis of a single training iteration.

For the training to efficiently use GPUs, we will need the loading time to be
less than the time taken to run one training iteration.

Steps: Measure the wall clock time of the following:
- Forward one dummy batch through a network
- Compute loss with dummy targets
- Do the backward pass
"""

import time

import torch
from torch import nn, optim
import numpy as np
from braindecode.models import ShallowFBCSPNet, Deep4Net


USE_CUDA = True


minibatch_size = 64
n_minibatches = 10
loss = nn.CrossEntropyLoss()

n_channels = 22
n_times = 1000
n_classes = 4

model_type = 'shallow'
# Create fake data
X = torch.from_numpy(
    np.random.rand(minibatch_size, n_channels, n_times, 1).astype(np.float32))
y = torch.from_numpy(np.random.randint(4, size=minibatch_size))

# Instantiate model and optimizer
if model_type == 'shallow':
    model = ShallowFBCSPNet(
        n_channels, n_classes, input_time_length=n_times, n_filters_time=40,
        filter_time_length=25, n_filters_spat=40, pool_time_length=75, 
        pool_time_stride=15, final_conv_length=30, split_first_layer=True,
        batch_norm=True, batch_norm_alpha=0.1, drop_prob=0.5)
elif model_type == 'deep':
    # TODO: Fix arguments so that output is same as shallow
    model = Deep4Net(
        n_channels, n_classes, input_time_length=n_times, final_conv_length=7,
        n_filters_time=25, n_filters_spat=25, filter_time_length=10,
        pool_time_length=3, pool_time_stride=3, n_filters_2=50,
        filter_length_2=10, n_filters_3=100, filter_length_3=10, 
        n_filters_4=200, filter_length_4=10, first_pool_mode="max", 
        later_pool_mode="max", drop_prob=0.5, double_time_convs=False,
        split_first_layer=True, batch_norm=True, batch_norm_alpha=0.1,
        stride_before_pool=False)
else:
    raise ValueError

optimizer = optim.Adam(model.parameters())
if USE_CUDA:
    model.cuda()
    X, y = X.cuda(), y.cuda()

# Train model on fake data
start = time.time()
for _ in range(n_minibatches):
    model.train()
    model.zero_grad()

    y_hat = torch.sum(model(X), axis=-1)
    loss_val = loss(y_hat, y)
    # print(loss_val)

    loss_val.backward()
    optimizer.step()

duration = (time.time() - start) * 1e3 / n_minibatches  # in ms

print(f'Took {duration} ms per minibatch.')
