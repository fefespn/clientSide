# server.py
from __future__ import print_function
import argparse

import urllib.request
import socket
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset
import random
import threading
import _thread
import socket
import numpy as np
import pickle

# create a socket object
serversocket = socket.socket(
    socket.AF_INET, socket.SOCK_STREAM)
# get local machine name
host = socket.gethostname()
port = 8080
# bind to the port
serversocket.bind((host, port))
# queue up to 5 requests
serversocket.listen(5)

# generate the sending data
convNeuralNetwork_1 = L.Linear(784, 300)
convNeuralNetwork_2 = L.Linear(300, 10)

sentNetwork_1 = convNeuralNetwork_1.W.data;
sentNetwork_2 = convNeuralNetwork_2.W.data;
returnedDeltaNetwork_1 = np.zeros((300, 784), dtype='float32')
returnedDeltaNetwork_2 = np.zeros((10, 300), dtype='float32')
network1_lock = threading.Lock()
network2_lock = threading.Lock()

print()
print("Started network:")
for i in range(0, 9):
    print(sentNetwork_2[0][i], end="  ")
print()

image_label_set = 1;


def clientthread(clientsocket):
    print("Got a connection from %s" % str(addr))
    # clientsocket.send(pickle.dumps(a))

    # sends the number of images and label file e.g.  1 or 7 or 16 ..
    global image_label_set
    global returnedDeltaNetwork_1
    global returnedDeltaNetwork_2
    global sentNetwork_1
    global sentNetwork_2
    print('image_label_set = ' + str(image_label_set))
    clientsocket.send(str(image_label_set).encode())
    image_label_set = image_label_set % 20 + 1;

    # send the first Convolutional neural network
    with network1_lock:
        view = memoryview(sentNetwork_1).cast('B')
        while len(view):
            nsent = clientsocket.send(view)
            view = view[nsent:]

    # send second Convolutional neural network
    with network2_lock:
        view = memoryview(sentNetwork_2).cast('B')
        while len(view):
            nsent = clientsocket.send(view)
            view = view[nsent:]

    # receive the first Convolutional neural network delta from the client
    view = memoryview(returnedDeltaNetwork_1).cast('B')
    while len(view):
        nrecv = clientsocket.recv_into(view)
        view = view[nrecv:]
    # receive the second Convolutional neural network delta from the client
    view = memoryview(returnedDeltaNetwork_2).cast('B')
    while len(view):
        nrecv = clientsocket.recv_into(view)
        view = view[nrecv:]

    print()
    print("New delta network:")
    for i in range(0, 9):
        print(returnedDeltaNetwork_2[0][i], end="  ")
    print()
    print()
    print("New network:")
    for i in range(0, 9):
        print(returnedDeltaNetwork_2[0][i] + sentNetwork_2[0][i], end="  ")
    print()

    with network1_lock:
        for i in range(300):
            for j in range(784):
                sentNetwork_1[i][j] = returnedDeltaNetwork_1[i][j] + sentNetwork_1[i][j]

    with network2_lock:
        for i in range(10):
            for j in range(300):
                sentNetwork_2[i][j] = returnedDeltaNetwork_2[i][j] + sentNetwork_2[i][j]


while True:
    # establish a connection
    clientsocket, addr = serversocket.accept()
    -_thread.start_new_thread(clientthread, (clientsocket,))

clientsocket.close()
serversocket.close()
