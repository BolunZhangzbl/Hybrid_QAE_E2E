"""
Template of Variational Classifier


"""


# -- Public Imports
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import StronglyEntanglingLayers, BasicEntanglerLayers
from pennylane.optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt


# --Private Imports


# -- Global Variables
dev = qml.device("default.qubit")

num_qubits = 4   # number of wires
num_layers = 2
weights_init = 0.01 * np.random.rand(num_layers, num_qubits, 3, requires_grad=True)

# -- Functions


def layer(layer_weights):
    for wire in range(4):
        qml.Rot(*layer_weights[wire], wires=wire)

    for wires in ([0, 1], [1, 2], [2, 3], [3, 0]):
        qml.CNOT(wires)


def state_preparation(x):
    qml.BasisState(x, wires=range(num_qubits))


def state_embedding(x):
    qml.BasisEmbedding(x, wires=range(num_qubits))


def state_hadamard(x):
    for wire in range(num_qubits):
        qml.Hadamard(wires=wire)


@qml.qnode(dev)
def circuit(weights, x):
    # state_hadamard(x)
    qml.AngleEmbedding(x, wires=range(num_qubits))

    for layer_weights in weights:
        layer(layer_weights)

    return qml.expval(qml.PauliZ(wires=0))


def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias


@qml.qnode(dev)
def str_entangle_circuit(weights, x):
    state_preparation(x)

    StronglyEntanglingLayers(weights, wires=range(num_qubits))

    return qml.expval(qml.PauliZ(wires=0))



# print(qml.draw_mpl(circuit, show_all_wires=True)(weights_init, [0, 0, 0, 1]))

print(qml.draw(circuit, expansion_strategy='device')(weights_init, np.random.rand(4)))
print("\n\n")
print(qml.draw(str_entangle_circuit, expansion_strategy='device')(weights_init, [0, 1, 0, 0]))