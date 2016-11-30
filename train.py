#!/usr/bin/env python
from __future__ import print_function
import argparse

import paramiko
import numpy
import socket
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset
import random


# Network definition
class MLP(chainer.Chain):
    def __init__(self, n_units, n_out, ll1, ll3):
        super(MLP, self).__init__(
            l1=ll1,
            l3=ll3,
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        return self.l3(h1)


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=2,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=784,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    ll1 = L.Linear(args.unit, args.unit)
    ll3 = L.Linear(args.unit, 10)


    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = socket.gethostname()
    port = 8080
    s.connect((host, port))

    network2_delta = numpy.zeros((10, 784), dtype='float32')
    network2 = numpy.zeros((10, 784), dtype='float32')

    view = memoryview(network2).cast('B')
    while len(view):
        nrecv = s.recv_into(view)
        view = view[nrecv:]


    for i in range(10):
     for j in range(args.unit):
         ll3.W.data[i][j] = network2[i][j];

    print("start network:")
    for i in range(0, 9):
        print(ll3.W.data[0][i] , end="  ")
    print()
    print()

    mlp = MLP(args.unit, 10, ll1, ll3)

    model = L.Classifier(mlp)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('t2.technion.ac.il', username='sfirasss', password='Firspn3#')
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command('cat ~/in.txt')
    read_image = str(ssh_stdout.read(), 'utf-8')
    ssh.close()
    read_image = read_image[:-1] # we delete the ladt line
    read_image = read_image[1:-1] # we delete the '[' ']'
    f = open('output_real', 'w')
    f.write(numpy.array_str(test._datasets[0][0]))  # python will convert \n to os.linesep
    f.close()
    f = open('output', 'w')
    f.write(read_image)  # python will convert \n to os.linesep
    f.close()

    print('type read image: ',type(numpy.fromstring(read_image,dtype=numpy.float32, sep=' ')[0]),'   type test.datasets[0][0][0]: ',type(test._datasets[0][0][0]))
    print('length read image: ', len(read_image), '   length test.datasets[0][0]: ', len(test._datasets[0][0]))
    print('real string length test.datasets[0][0]: ', len(numpy.array_str(test._datasets[0][0])))
    print('real string length read_image: ', len(numpy.fromstring(read_image,dtype=numpy.float32, sep=' ')))
    test2 = tuple_dataset.TupleDataset([numpy.fromstring(read_image,dtype=numpy.float32, sep=' ')], [test._datasets[1][0]])
    train2 = tuple_dataset.TupleDataset([train._datasets[0][0]], [train._datasets[1][0]])


    train_iter = chainer.iterators.SerialIterator(train2, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test2, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    for i in range(10):
     for j in range(args.unit):
          network2_delta[i][j] = ll3.W.data[i][j] - network2[i][j] ;

    print()
    print("New delta network:")
    for i in range(0, 9):
        print(network2_delta[0][i], end="  ")
    print()
    print()
    print("New network:")
    for i in range(0, 9):
        print(ll3.W.data[0][i], end="  ")
    print()

    view = memoryview(network2_delta).cast('B')
    while len(view):
        nsent = s.send(view)
        view = view[nsent:]

   # print('SERVER RESPONSE:')
   # print(urllib.request.urlopen('http://127.0.0.1:9000/').read())

    s.close()

if __name__ == '__main__':
    main()