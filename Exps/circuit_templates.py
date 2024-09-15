"""
Circuits Templates

Cited from "Expressibility and entangling capability of parameterized quantum circuits for hybrid
            quantum-classical algorithms"
"""

# -- Public Imports
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import StronglyEntanglingLayers, BasicEntanglerLayers
from pennylane.optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt


# --Private Imports


# -- Global Variables

num_qubits = 4


# -- Functions

def circuit1(W, num_qubits=num_qubits):
    for wire in range(num_qubits):
        qml.RX(W[wire, 0], wires=wire)
        qml.RZ(W[wire, 2], wires=wire)


def circuit2(W, num_qubits=num_qubits):
    for wire in range(num_qubits):
        qml.RX(W[wire, 0], wires=wire)
        qml.RZ(W[wire, 2], wires=wire)

    for wire in ([3, 2], [2, 1], [1, 0]):
        qml.CNOT(wires=wire)


def circuit3(W, num_qubits=num_qubits):
    for wire in range(num_qubits):
        qml.RX(W[wire, 0], wires=wire)
        qml.RZ(W[wire, 2], wires=wire)

    for wire in ([3, 2], [2, 1], [1, 0]):
        qml.CRZ(W[wire[1], 2], wires=wire)


def circuit4(W, num_qubits=num_qubits):
    for wire in range(num_qubits):
        qml.RX(W[wire, 0], wires=wire)
        qml.RZ(W[wire, 2], wires=wire)

    for wire in ([3, 2], [2, 1], [1, 0]):
        qml.CRX(W[wire[1], 0], wires=wire)


def get_layer56():
    return [[i, j] for i in range(3, -1, -1) for j in range(3, -1,-1) if (i!=j)]


def circuit5(W, num_qubits=num_qubits):
    for wire in range(num_qubits):
        qml.RX(W[wire, 0], wires=wire)
        qml.RZ(W[wire, 2], wires=wire)

    for wire in get_layer56():
        qml.CRZ(W[wire[1], 2], wires=wire)

    for wire in range(num_qubits):
        qml.RX(W[wire, 0], wires=wire)
        qml.RZ(W[wire, 2], wires=wire)


def circuit6(W, num_qubits=num_qubits):
    for wire in range(num_qubits):
        qml.RX(W[wire, 0], wires=wire)
        qml.RZ(W[wire, 2], wires=wire)

    for wire in get_layer56():
        qml.CRX(W[wire[1], 0], wires=wire)

    for wire in range(num_qubits):
        qml.RX(W[wire, 0], wires=wire)
        qml.RZ(W[wire, 2], wires=wire)


def circuit7(W, num_qubits=num_qubits):
    for wire in range(num_qubits):
        qml.RX(W[wire, 0], wires=wire)
        qml.RZ(W[wire, 2], wires=wire)

    for wire in ([3, 2], [1, 0]):
        qml.CRZ(W[wire[1], 2], wires=wire)

    for wire in range(num_qubits):
        qml.RX(W[wire, 0], wires=wire)

    qml.CRZ(W[1, 2], wires=[2, 1])


def circuit8(W, num_qubits=num_qubits):
    for wire in range(num_qubits):
        qml.RX(W[wire, 0], wires=wire)
        qml.RZ(W[wire, 2], wires=wire)

    for wire in ([3, 2], [1, 0]):
        qml.CRX(W[wire[1], 0], wires=wire)

    for wire in range(num_qubits):
        qml.RX(W[wire, 0], wires=wire)

    qml.CRX(W[1, 0], wires=[2, 1])


def circuit9(W, num_qubits=num_qubits):
    for wire in range(num_qubits):
        qml.Hadamard(wires=wire)

    for wire in ([3, 2], [2, 1], [1, 0]):
        qml.CZ(wires=wire)

    for wire in range(num_qubits):
        qml.RX(W[wire, 0], wires=wire)


def circuit10(W, num_qubits=num_qubits):
    for wire in range(num_qubits):
        qml.RY(W[wire, 1], wires=wire)

    for wire in ([3, 2], [2, 1], [1, 0], [0, 3]):
        qml.CZ(wires=wire)

    for wire in range(num_qubits):
        qml.RX(W[wire, 0], wires=wire)


def circuit11(W, num_qubits=num_qubits):
    for wire in range(num_qubits):
        qml.RY(W[wire, 1], wires=wire)
        qml.RZ(W[wire, 2], wires=wire)

    for wire in ([3, 2], [1, 0]):
        qml.CNOT(wires=wire)

    for wire in range(2, 4):
        qml.RY(W[wire, 1], wires=wire)
        qml.RZ(W[wire, 2], wires=wire)

    qml.CNOT(wires=[2, 1])


def circuit12(W, num_qubits=num_qubits):
    for wire in range(num_qubits):
        qml.RY(W[wire, 1], wires=wire)
        qml.RZ(W[wire, 2], wires=wire)

    for wire in ([3, 2], [1, 0]):
        qml.CZ(wires=wire)

    for wire in range(2, 4):
        qml.RY(W[wire, 1], wires=wire)
        qml.RZ(W[wire, 2], wires=wire)

    qml.CZ(wires=[2, 1])


def circuit13(W, num_qubits=num_qubits):
    for wire in range(num_qubits):
        qml.RY(W[wire, 1], wires=wire)

    for wire in ([3, 0], [2, 3], [1, 2], [0, 1]):
        qml.CRZ(W[wire, 2], wires=wire)

    for wire in range(num_qubits):
        qml.RY(W[wire, 1], wires=wire)

    for wire in ([3, 2], [0, 3], [1, 0], [2, 1]):
        qml.CRZ(W[wire, 2], wires=wire)


def circuit14(W, num_qubits=num_qubits):
    for wire in range(num_qubits):
        qml.RY(W[wire, 1], wires=wire)

    for wire in ([3, 0], [2, 3], [1, 2], [0, 1]):
        qml.CRX(W[wire, 0], wires=wire)

    for wire in range(num_qubits):
        qml.RY(W[wire, 1], wires=wire)

    for wire in ([3, 2], [0, 3], [1, 0], [2, 1]):
        qml.CRX(W[wire, 0], wires=wire)


def circuit15(W, num_qubits=num_qubits):
    for wire in range(num_qubits):
        qml.RY(W[wire, 1], wires=wire)

    for wire in ([3, 0], [2, 3], [1, 2], [0, 1]):
        qml.CNOT(wires=wire)

    for wire in range(num_qubits):
        qml.RY(W[wire, 1], wires=wire)

    for wire in ([3, 2], [0, 3], [1, 0], [2, 1]):
        qml.CNOT(wires=wire)


def circuit16(W, num_qubits=num_qubits):
    for wire in range(num_qubits):
        qml.RX(W[wire, 0], wires=wire)
        qml.RZ(W[wire, 2], wires=wire)

    for wire in ([3, 2], [1, 0], [2, 1]):
        qml.CRZ(W[wire, 2], wires=wire)


def circuit17(W, num_qubits=num_qubits):
    for wire in range(num_qubits):
        qml.RX(W[wire, 0], wires=wire)
        qml.RZ(W[wire, 2], wires=wire)

    for wire in ([3, 2], [1, 0], [2, 1]):
        qml.CRX(W[wire, 0], wires=wire)


def circuit18(W, num_qubits=num_qubits):
    for wire in range(num_qubits):
        qml.RX(W[wire, 0], wires=wire)
        qml.RZ(W[wire, 2], wires=wire)

    for wire in ([3, 0], [2, 3], [1, 2], [0, 1]):
        qml.CRZ(W[wire, 2], wires=wire)


def circuit19(W, num_qubits=num_qubits):
    for wire in range(num_qubits):
        qml.RX(W[wire, 0], wires=wire)
        qml.RZ(W[wire, 2], wires=wire)

    for wire in ([3, 0], [2, 3], [1, 2], [0, 1]):
        qml.CRX(W[wire, 0], wires=wire)