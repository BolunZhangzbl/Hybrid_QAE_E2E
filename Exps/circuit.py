"""
Circuits

Define Fundamental Modules for circuits
"""


# -- Public Imports
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F

import pennylane as qml
from pennylane import numpy as np

import matplotlib.pyplot as plt


# --Private Imports
import circuit_templates as ct

# -- Global Variables

dev = qml.device("default.qubit", wires=4)

num_qubits = 4


# -- Functions

def tx_layer(W):
    for i in range(num_qubits):
        qml.RX(W[i, 0], wires=i)
        qml.RY(W[i, 1], wires=i)
        qml.RZ(W[i, 2], wires=i)


# Transmitter Circuit
@qml.qnode(dev, interface='torch')
def tx_circuit(W, s):
    """
    :param W: Layer Variable Parameters
    :param s: State Variable
    :return:
    """
    for i in range(num_qubits):
        qml.RY((np.pi)*s[i], wires=i)

    # Variational Quantum Circuit
    tx_layer(W[0])
    for i in range(num_qubits-1):
        qml.CNOT(wires=[i, i+1])

    tx_layer(W[1])
    for i in range(num_qubits-1):
        qml.CNOT(wires=[i, i+1])

    tx_layer(W[2])
    for i in range(num_qubits-1):
        qml.CNOT(wires=[i, i+1])

    tx_layer(W[3])
    qml.CNOT(wires=[0, 2])
    qml.CNOT(wires=[1, 3])

    return [qml.expval(qml.PauliY(ind)) for ind in range(2, 4)]


def rx_layer(layer_W, y):
    for i in range(num_qubits):
        qml.RX(layer_W[i, 0]*y[i%2], wires=i)

    for i in range(num_qubits):
        qml.Rot(*layer_W[i], wires=i)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])


@qml.qnode(dev, interface='torch')
def rx_circuit_reuploading(W, y):
    """
    Receiver Circuit
    :param W: Layer Variable Parameters
    :param y: Received Signal
    :return:
    """
    num_layer = W.shape[0]

    for layer in range(num_layer):
        rx_layer(W[layer], y)

    return qml.probs(wires=range(num_qubits))


qml.draw_mpl(rx_circuit_reuploading, show_all_wires=True)(np.random.rand(4, 4, 3), np.random.rand(2))
plt.show()
# dict_ct = {str(i): getattr(ct, f"circuit{i}") for i in range(20)}
