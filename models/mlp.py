"""Chainer example: train a multi-layer perceptron on MNIST
This is a minimal example to write a feed-forward net. It requires scikit-learn
to load MNIST dataset.
"""
import argparse

import numpy as np
import six

import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

import bloscpack as bp
import pdb


parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

batchsize = 100
n_epoch = 20
n_units = 1000

# Prepare dataset
print 'Loading training data...'
train_data = bp.unpack_ndarray_file('../data/pickles/train_data.pickle').astype(np.float32) / 255.0
train_labels = bp.unpack_ndarray_file('../data/pickles/train_labels.pickle').astype(np.int32)
print '\tDone'

print 'Loading testing data...'
test_data = bp.unpack_ndarray_file('../data/pickles/test_data.pickle').astype(np.float32) / 255.0
print '\tDone'

N = int(train_data.shape[0] * 0.8)

validation_data = train_data[N:, :, :, :]
validation_data = validation_data.reshape((validation_data.shape[0], 200*200*3))
validation_labels = train_labels[N:]

train_data = train_data[:N, :, :, :]
train_data = train_data.reshape((train_data.shape[0], 200*200*3))
train_labels = train_labels[:N]

pdb.set_trace()

N_validation = validation_labels.shape[0]

# Prepare multi-layer perceptron model
model = chainer.FunctionSet(l1=F.Linear(200*200*3, n_units),
                            l2=F.Linear(n_units, n_units),
                            l3=F.Linear(n_units, 2))

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

def forward(x_data, y_data, train=True):
    # Neural net architecture
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    # h1 = F.dropout(F.relu(model.l1(x)),  train=train)
    h1 = F.relu(model.l1(x))
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    y = model.l3(h2)
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N, batchsize):
        x_batch = xp.asarray(train_data[perm[i:i + batchsize]])
        y_batch = xp.asarray(train_labels[perm[i:i + batchsize]])

        optimizer.zero_grads()
        loss, acc = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        if epoch == 1 and i == 0:
            with open("graph.dot", "w") as o:
                o.write(c.build_computational_graph((loss, )).dump())
            with open("graph.wo_split.dot", "w") as o:
                g = c.build_computational_graph((loss, ),
                                                remove_split=True)
                o.write(g.dump())
            print('graph generated')

        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)

    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_validation, batchsize):
        x_batch = xp.asarray(validation_data[i:i + batchsize])
        y_batch = xp.asarray(validation_labels[i:i + batchsize])

        loss, acc = forward(x_batch, y_batch, train=False)

        sum_loss += float(loss.data) * len(y_batch)
        sum_accuracy += float(acc.data) * len(y_batch)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_validation, sum_accuracy / N_validation))